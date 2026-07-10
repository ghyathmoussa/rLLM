use anyhow::Result;
use candle_core::Device;
use candle_core::{D, DType, Tensor};
use candle_transformers::models::deepseek2::DeepSeekV2RopeScaling;
use serde::Deserialize;

use crate::loader::WeightMap;

#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekV3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    #[serde(default = "default_moe_frequency")]
    pub moe_layer_freq: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_group: usize,
    pub topk_group: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scale")]
    pub routed_scaling_factor: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub rms_norm_eps: f64,
}

fn default_moe_frequency() -> usize {
    1
}

fn default_routed_scale() -> f64 {
    1.0
}

impl DeepseekV3Config {
    pub fn from_json(content: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            anyhow::bail!("DeepSeek V3 hidden size and attention heads must be non-zero");
        }
        if self.n_group == 0 || self.n_routed_experts % self.n_group != 0 {
            anyhow::bail!("DeepSeek V3 routed experts must be divisible by n_group");
        }
        if self.topk_group == 0 || self.topk_group > self.n_group {
            anyhow::bail!("DeepSeek V3 topk_group must be in 1..=n_group");
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            anyhow::bail!("DeepSeek V3 num_experts_per_tok is invalid");
        }
        if self.qk_nope_head_dim + self.qk_rope_head_dim == 0 || self.v_head_dim == 0 {
            anyhow::bail!("DeepSeek V3 MLA head dimensions must be non-zero");
        }
        Ok(())
    }
}

/// DeepSeek V3 `noaux_tc` router. Selection uses corrected sigmoid scores,
/// while expert contribution weights use the uncorrected sigmoid scores.
pub struct NoAuxTcRouter {
    weight: Tensor,
    correction_bias: Tensor,
    top_k: usize,
    num_groups: usize,
    topk_groups: usize,
    normalize: bool,
    routed_scaling_factor: f64,
}

impl NoAuxTcRouter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: Tensor,
        correction_bias: Tensor,
        top_k: usize,
        num_groups: usize,
        topk_groups: usize,
        normalize: bool,
        routed_scaling_factor: f64,
    ) -> Result<Self> {
        let (num_experts, hidden_size) = weight.dims2()?;
        if correction_bias.dims() != [num_experts] {
            anyhow::bail!("router correction bias must have {num_experts} elements");
        }
        if num_groups == 0 || num_experts % num_groups != 0 {
            anyhow::bail!("router experts must be divisible by num_groups");
        }
        if top_k == 0 || top_k > num_experts {
            anyhow::bail!("router top_k must be in 1..={num_experts}");
        }
        if topk_groups == 0 || topk_groups > num_groups {
            anyhow::bail!("router topk_groups must be in 1..={num_groups}");
        }
        let _ = hidden_size;
        Ok(Self {
            weight,
            correction_bias,
            top_k,
            num_groups,
            topk_groups,
            normalize,
            routed_scaling_factor,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let hidden_size = self.weight.dim(1)?;
        let num_experts = self.weight.dim(0)?;
        let tokens: usize = hidden_states.elem_count() / hidden_size;
        let hidden = hidden_states.reshape((tokens, hidden_size))?.to_dtype(DType::F32)?;
        let logits = hidden.matmul(&self.weight.to_dtype(DType::F32)?.t()?)?;
        let raw_scores = candle_nn::ops::sigmoid(&logits)?;
        let selection_scores = raw_scores.broadcast_add(
            &self.correction_bias.to_dtype(DType::F32)?.reshape((1, num_experts))?,
        )?;

        let experts_per_group = num_experts / self.num_groups;
        let grouped = selection_scores.reshape((tokens, self.num_groups, experts_per_group))?;
        let top_two = grouped
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, 2.min(experts_per_group))?
            .contiguous()?;
        let group_scores = grouped.gather(&top_two, D::Minus1)?.sum(D::Minus1)?;
        let selected_groups = group_scores
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.topk_groups)?
            .contiguous()?;
        let group_mask =
            Tensor::zeros((tokens, self.num_groups), DType::F32, hidden_states.device())?
                .scatter_add(
                    &selected_groups,
                    &Tensor::ones(selected_groups.shape(), DType::F32, hidden_states.device())?,
                    1,
                )?;
        let expert_mask = group_mask
            .unsqueeze(D::Minus1)?
            .expand((tokens, self.num_groups, experts_per_group))?
            .reshape((tokens, num_experts))?;
        let negative_inf = Tensor::new(f32::NEG_INFINITY, hidden_states.device())?
            .broadcast_as(selection_scores.shape())?;
        let masked_scores = expert_mask.eq(0f32)?.where_cond(&negative_inf, &selection_scores)?;
        let expert_ids = masked_scores
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.top_k)?
            .contiguous()?;

        let mut expert_weights = raw_scores.gather(&expert_ids, D::Minus1)?;
        if self.normalize && self.top_k > 1 {
            expert_weights =
                expert_weights.broadcast_div(&(expert_weights.sum_keepdim(D::Minus1)? + 1e-20)?)?;
        }
        expert_weights = (expert_weights * self.routed_scaling_factor)?;
        Ok((expert_ids, expert_weights))
    }
}

/// Linear weight using DeepSeek's two-dimensional block-scaled FP8 format.
pub struct BlockFp8Linear {
    weight: Tensor,
    scales: Tensor,
    block_size: usize,
}

/// Stacked FP8 matrices for routed experts.
pub struct SelectedBlockFp8Experts {
    weights: Tensor,
    scales: Tensor,
    block_size: usize,
}

impl SelectedBlockFp8Experts {
    pub fn new(weights: Tensor, scales: Tensor, block_size: usize) -> Result<Self> {
        let (num_experts, out_features, in_features) = weights.dims3()?;
        let expected =
            [num_experts, out_features.div_ceil(block_size), in_features.div_ceil(block_size)];
        if weights.dtype() != DType::F8E4M3 || scales.dims() != expected {
            anyhow::bail!(
                "selected expert FP8 tensors have invalid dtype/shape: weights={:?} scales={:?}, expected scales={expected:?}",
                weights.dtype(),
                scales.dims()
            );
        }
        Ok(Self { weights, scales, block_size })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
        shared_input: bool,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(input.device(), Device::Cuda(_)) {
            return self.forward_cuda(input, expert_ids, shared_input);
        }
        self.forward_reference(input, expert_ids, shared_input)
    }

    fn forward_reference(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
        shared_input: bool,
    ) -> Result<Tensor> {
        let (num_experts, out_features, in_features) = self.weights.dims3()?;
        let (tokens, top_k) = expert_ids.dims2()?;
        let input = input.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let input_values = input.flatten_all()?.to_vec1::<f32>()?;
        let ids = expert_ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let weights = self.weights.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let weight_values = weights.flatten_all()?.to_vec1::<f32>()?;
        let scale_values = self
            .scales
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let in_blocks = in_features.div_ceil(self.block_size);
        let out_blocks = out_features.div_ceil(self.block_size);
        let mut output = vec![0f32; tokens * top_k * out_features];
        for token in 0..tokens {
            for route in 0..top_k {
                let expert = ids[token][route] as usize;
                if expert >= num_experts {
                    anyhow::bail!("expert id {expert} is out of range");
                }
                let input_row = if shared_input { token } else { token * top_k + route };
                for out in 0..out_features {
                    let mut acc = 0f32;
                    for col in 0..in_features {
                        let weight_index = ((expert * out_features + out) * in_features) + col;
                        let scale_index = expert * out_blocks * in_blocks
                            + (out / self.block_size) * in_blocks
                            + col / self.block_size;
                        acc += input_values[input_row * in_features + col]
                            * weight_values[weight_index]
                            * scale_values[scale_index];
                    }
                    output[(token * top_k + route) * out_features + out] = acc;
                }
            }
        }
        Tensor::from_vec(output, (tokens, top_k, out_features), input.device())?
            .to_device(expert_ids.device())
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
        shared_input: bool,
    ) -> Result<Tensor> {
        let (num_experts, out_features, in_features) = self.weights.dims3()?;
        let (tokens, top_k) = expert_ids.dims2()?;
        let input = input.to_dtype(DType::F16)?.contiguous()?;
        let ids = expert_ids.to_dtype(DType::U32)?.contiguous()?;
        let weights = self.weights.contiguous()?;
        let scales = self.scales.to_dtype(DType::F32)?.contiguous()?;
        let output = Tensor::zeros((tokens, top_k, out_features), DType::F16, input.device())?;
        let stream = match input.device() {
            Device::Cuda(device) => device.cuda_stream().cu_stream() as usize,
            _ => unreachable!(),
        };
        unsafe {
            rllm_kernels::deepseek_v3::fp8_selected_expert_matmul_f16(
                cuda_ptr::<half::f16>(&input)? as *const u16,
                cuda_ptr::<u32>(&ids)?,
                cuda_ptr::<float8::F8E4M3>(&weights)? as *const u8,
                cuda_ptr::<f32>(&scales)?,
                cuda_ptr::<half::f16>(&output)? as *mut u16,
                tokens as i64,
                top_k as i64,
                num_experts as i64,
                out_features as i64,
                in_features as i64,
                self.block_size as i64,
                shared_input,
                stream,
            )?;
        }
        Ok(output)
    }
}

pub struct DeepseekV3Moe {
    router: NoAuxTcRouter,
    gate_experts: SelectedBlockFp8Experts,
    up_experts: SelectedBlockFp8Experts,
    down_experts: SelectedBlockFp8Experts,
    shared_gate: Option<BlockFp8Linear>,
    shared_up: Option<BlockFp8Linear>,
    shared_down: Option<BlockFp8Linear>,
}

impl DeepseekV3Moe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router: NoAuxTcRouter,
        gate_experts: SelectedBlockFp8Experts,
        up_experts: SelectedBlockFp8Experts,
        down_experts: SelectedBlockFp8Experts,
        shared_gate: Option<BlockFp8Linear>,
        shared_up: Option<BlockFp8Linear>,
        shared_down: Option<BlockFp8Linear>,
    ) -> Result<Self> {
        let shared_count = usize::from(shared_gate.is_some())
            + usize::from(shared_up.is_some())
            + usize::from(shared_down.is_some());
        if shared_count != 0 && shared_count != 3 {
            anyhow::bail!("shared expert gate/up/down projections must be provided together");
        }
        Ok(Self {
            router,
            gate_experts,
            up_experts,
            down_experts,
            shared_gate,
            shared_up,
            shared_down,
        })
    }

    pub fn from_weights(
        config: &DeepseekV3Config,
        weights: &mut WeightMap,
        prefix: &str,
        block_size: usize,
    ) -> Result<Self> {
        let router_weight = take_tensor(weights, &format!("{prefix}.gate.weight"))?;
        let correction_bias =
            take_tensor(weights, &format!("{prefix}.gate.e_score_correction_bias"))?;
        let router = NoAuxTcRouter::new(
            router_weight,
            correction_bias,
            config.num_experts_per_tok,
            config.n_group,
            config.topk_group,
            config.norm_topk_prob,
            config.routed_scaling_factor,
        )?;

        let gate_experts = take_stacked_experts(
            weights,
            prefix,
            "gate_proj",
            config.n_routed_experts,
            block_size,
        )?;
        let up_experts =
            take_stacked_experts(weights, prefix, "up_proj", config.n_routed_experts, block_size)?;
        let down_experts = take_stacked_experts(
            weights,
            prefix,
            "down_proj",
            config.n_routed_experts,
            block_size,
        )?;
        let shared_prefix = format!("{prefix}.shared_experts");
        let shared_gate = Some(take_block_fp8_linear(
            weights,
            &format!("{shared_prefix}.gate_proj"),
            block_size,
        )?);
        let shared_up =
            Some(take_block_fp8_linear(weights, &format!("{shared_prefix}.up_proj"), block_size)?);
        let shared_down = Some(take_block_fp8_linear(
            weights,
            &format!("{shared_prefix}.down_proj"),
            block_size,
        )?);
        Self::new(
            router,
            gate_experts,
            up_experts,
            down_experts,
            shared_gate,
            shared_up,
            shared_down,
        )
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let original_shape = hidden_states.shape().clone();
        let hidden_size = *hidden_states
            .dims()
            .last()
            .ok_or_else(|| anyhow::anyhow!("MoE input must have a hidden dimension"))?;
        let tokens = hidden_states.elem_count() / hidden_size;
        let hidden = hidden_states.reshape((tokens, hidden_size))?;
        let (expert_ids, expert_weights) = self.router.forward(&hidden)?;
        let gate = self.gate_experts.forward(&hidden, &expert_ids, true)?.silu()?;
        let up = self.up_experts.forward(&hidden, &expert_ids, true)?;
        let intermediate = gate.broadcast_mul(&up)?;
        let routed = self.down_experts.forward(&intermediate, &expert_ids, false)?;
        let mut output = routed
            .broadcast_mul(&expert_weights.to_dtype(routed.dtype())?.unsqueeze(D::Minus1)?)?
            .sum(1)?;
        if let (Some(gate), Some(up), Some(down)) =
            (&self.shared_gate, &self.shared_up, &self.shared_down)
        {
            let shared = gate.forward(&hidden)?.silu()?.broadcast_mul(&up.forward(&hidden)?)?;
            output = (output + down.forward(&shared)?)?;
        }
        output.reshape(original_shape).map_err(Into::into)
    }
}

fn take_tensor(weights: &mut WeightMap, name: &str) -> Result<Tensor> {
    weights.weights.remove(name).ok_or_else(|| anyhow::anyhow!("missing DeepSeek V3 tensor {name}"))
}

fn take_block_fp8_linear(
    weights: &mut WeightMap,
    prefix: &str,
    block_size: usize,
) -> Result<BlockFp8Linear> {
    let weight = take_tensor(weights, &format!("{prefix}.weight"))?;
    let scales = take_tensor(weights, &format!("{prefix}.weight_scale_inv"))?;
    BlockFp8Linear::new(weight, scales, block_size)
}

fn take_stacked_experts(
    weights: &mut WeightMap,
    prefix: &str,
    projection: &str,
    num_experts: usize,
    block_size: usize,
) -> Result<SelectedBlockFp8Experts> {
    let mut expert_weights = Vec::with_capacity(num_experts);
    let mut expert_scales = Vec::with_capacity(num_experts);
    for expert in 0..num_experts {
        let expert_prefix = format!("{prefix}.experts.{expert}.{projection}");
        expert_weights.push(take_tensor(weights, &format!("{expert_prefix}.weight"))?);
        expert_scales.push(take_tensor(weights, &format!("{expert_prefix}.weight_scale_inv"))?);
    }
    let weight_refs: Vec<&Tensor> = expert_weights.iter().collect();
    let scale_refs: Vec<&Tensor> = expert_scales.iter().collect();
    SelectedBlockFp8Experts::new(
        Tensor::stack(&weight_refs, 0)?,
        Tensor::stack(&scale_refs, 0)?,
        block_size,
    )
}

impl BlockFp8Linear {
    pub fn new(weight: Tensor, scales: Tensor, block_size: usize) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let expected = [out_features.div_ceil(block_size), in_features.div_ceil(block_size)];
        if weight.dtype() != DType::F8E4M3 {
            anyhow::bail!("DeepSeek FP8 weight must use F8E4M3, got {:?}", weight.dtype());
        }
        if scales.dims() != expected {
            anyhow::bail!("DeepSeek FP8 scale shape must be {expected:?}, got {:?}", scales.dims());
        }
        Ok(Self { weight, scales, block_size })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(input.device(), Device::Cuda(_)) {
            return self.forward_cuda(input);
        }
        self.forward_reference(input)
    }

    pub fn forward_reference(&self, input: &Tensor) -> Result<Tensor> {
        let (out_features, in_features) = self.weight.dims2()?;
        let out_blocks = out_features.div_ceil(self.block_size);
        let in_blocks = in_features.div_ceil(self.block_size);
        let expanded_scales = self
            .scales
            .to_dtype(DType::F32)?
            .reshape((out_blocks, 1, in_blocks, 1))?
            .repeat((1, self.block_size, 1, self.block_size))?
            .reshape((out_blocks * self.block_size, in_blocks * self.block_size))?
            .narrow(0, 0, out_features)?
            .narrow(1, 0, in_features)?;
        let weight = self.weight.to_dtype(DType::F32)?.broadcast_mul(&expanded_scales)?;
        let input_shape = input.dims();
        let rows: usize = input_shape[..input_shape.len() - 1].iter().product();
        let output =
            input.reshape((rows, in_features))?.to_dtype(DType::F32)?.matmul(&weight.t()?)?;
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);
        output.reshape(output_shape)?.to_dtype(input.dtype()).map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, input: &Tensor) -> Result<Tensor> {
        let (out_features, in_features) = self.weight.dims2()?;
        let original_dtype = input.dtype();
        let input_shape = input.dims();
        let rows: usize = input_shape[..input_shape.len() - 1].iter().product();
        let input = input.to_dtype(DType::F16)?.reshape((rows, in_features))?.contiguous()?;
        let weight = self.weight.contiguous()?;
        let scales = self.scales.to_dtype(DType::F32)?.contiguous()?;
        let output = Tensor::zeros((rows, out_features), DType::F16, input.device())?;
        let stream = match input.device() {
            Device::Cuda(device) => device.cuda_stream().cu_stream() as usize,
            _ => unreachable!(),
        };
        unsafe {
            rllm_kernels::deepseek_v3::fp8_block_matmul_f16(
                cuda_ptr::<half::f16>(&input)? as *const u16,
                cuda_ptr::<float8::F8E4M3>(&weight)? as *const u8,
                cuda_ptr::<f32>(&scales)?,
                cuda_ptr::<half::f16>(&output)? as *mut u16,
                rows as i64,
                out_features as i64,
                in_features as i64,
                self.block_size as i64,
                stream,
            )?;
        }
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(out_features);
        output.reshape(output_shape)?.to_dtype(original_dtype).map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
fn cuda_ptr<T: candle_core::cuda_backend::CudaDType>(tensor: &Tensor) -> Result<*const T> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(storage) => {
            let slice = storage.as_cuda_slice::<T>()?;
            let stream = storage.device.cuda_stream();
            let (pointer, _guard) = slice.device_ptr(&stream);
            Ok(pointer as *const T)
        }
        _ => anyhow::bail!("tensor is not on a CUDA device"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router_on(device: &Device) -> Result<()> {
        let weight = Tensor::new(&[[1f32, 0.], [0., 1.], [-1., 0.], [0., -1.]], device)?;
        let bias = Tensor::new(&[0f32, 0., 3., 0.], device)?;
        let router = NoAuxTcRouter::new(weight, bias, 1, 2, 1, false, 1.0)?;
        let hidden = Tensor::new(&[[1f32, 0.]], device)?;
        let (ids, weights) = router.forward(&hidden)?;
        assert_eq!(ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?, vec![vec![2]]);
        let value = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?[0][0];
        let expected = 1.0 / (1.0 + 1.0f32.exp());
        assert!((value - expected).abs() < 1e-5, "{value} != {expected}");
        Ok(())
    }

    #[test]
    fn parses_v3_config_contract() -> Result<()> {
        let config = DeepseekV3Config::from_json(
            r#"{
                "vocab_size": 128,
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4,
                "v_head_dim": 4,
                "n_group": 2,
                "topk_group": 1,
                "routed_scaling_factor": 2.5,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "rms_norm_eps": 0.000001
            }"#,
        )?;
        assert_eq!(config.n_routed_experts, 4);
        assert_eq!(config.moe_layer_freq, 1);
        Ok(())
    }

    fn fp8_linear_on(device: &Device) -> Result<()> {
        // Production FP8 weights arrive pre-encoded from SafeTensors. Construct
        // the fixture on CPU and upload it to avoid requiring a GPU FP32->FP8 cast.
        let weight = Tensor::ones((4, 4), DType::F32, &Device::Cpu)?
            .to_dtype(DType::F8E4M3)?
            .to_device(device)?;
        let scales = Tensor::new(&[[1f32, 2.], [3., 4.]], device)?;
        let linear = BlockFp8Linear::new(weight, scales, 2)?;
        let input = Tensor::ones((1, 4), DType::F16, device)?;
        let output = linear.forward(&input)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let values = output.to_vec2::<f32>()?;
        for value in &values[0][..2] {
            assert!((*value - 6.0).abs() < 0.05, "{value}");
        }
        for value in &values[0][2..] {
            assert!((*value - 14.0).abs() < 0.05, "{value}");
        }
        Ok(())
    }

    fn fp8_experts(
        device: &Device,
        num_experts: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<SelectedBlockFp8Experts> {
        let weights =
            Tensor::ones((num_experts, out_features, in_features), DType::F32, &Device::Cpu)?
                .to_dtype(DType::F8E4M3)?
                .to_device(device)?;
        let scales = Tensor::ones(
            (num_experts, out_features.div_ceil(2), in_features.div_ceil(2)),
            DType::F32,
            device,
        )?;
        SelectedBlockFp8Experts::new(weights, scales, 2)
    }

    fn v3_moe_on(device: &Device) -> Result<()> {
        let router = NoAuxTcRouter::new(
            Tensor::zeros((2, 4), DType::F32, device)?,
            Tensor::new(&[1f32, 0.], device)?,
            1,
            1,
            1,
            false,
            1.0,
        )?;
        let moe = DeepseekV3Moe::new(
            router,
            fp8_experts(device, 2, 4, 4)?,
            fp8_experts(device, 2, 4, 4)?,
            fp8_experts(device, 2, 4, 4)?,
            None,
            None,
            None,
        )?;
        let input = Tensor::ones((1, 1, 4), DType::F16, device)?;
        let output = moe.forward(&input)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        assert_eq!(output.dims(), &[1, 1, 4]);
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = 0.5 * 4.0 * (4.0 / (1.0 + (-4.0f32).exp())) * 4.0;
        for value in values {
            assert!((value - expected).abs() < 0.2, "{value} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn noaux_tc_uses_uncorrected_contribution_weight_cpu() -> Result<()> {
        router_on(&Device::Cpu)
    }

    #[test]
    fn block_fp8_linear_reference_cpu() -> Result<()> {
        fp8_linear_on(&Device::Cpu)
    }

    #[test]
    fn v3_moe_reference_cpu() -> Result<()> {
        v3_moe_on(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn noaux_tc_cuda() -> Result<()> {
        router_on(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn block_fp8_linear_cuda() -> Result<()> {
        fp8_linear_on(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn v3_moe_cuda() -> Result<()> {
        v3_moe_on(&Device::new_cuda(0)?)
    }
}
