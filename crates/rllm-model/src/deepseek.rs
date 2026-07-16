use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_transformers::models::deepseek2::{DeepSeekV2RopeScaling, ScaledRopeType};
use rllm_core::config::ModelConfig;

use crate::{
    deepseek_v3::{BlockFp8Linear, DeepseekV3Config, DeepseekV3Moe},
    layers::{Linear, RmsNorm},
    loader::WeightMap,
    registry::{CausalLM, Model},
};

/// Native DeepSeek V2/V3/R1 decoder.
///
/// MLA is expanded after the latent projections and stored in rLLM's global
/// paged cache. Values are zero-padded to the Q/K width for the cache kernel,
/// then narrowed back to `v_head_dim` before the output projection.
pub struct DeepseekForCausalLM {
    model: DeepseekModel,
    config: ModelConfig,
}

impl DeepseekForCausalLM {
    pub fn factory(_config: &ModelConfig) -> Result<Box<dyn CausalLM>> {
        anyhow::bail!("DeepseekForCausalLM::factory requires loaded weights")
    }

    pub fn parse_config(path: &Path) -> Result<DeepseekV3Config> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading DeepSeek config from {}", path.display()))?;
        DeepseekV3Config::from_json(&content)
            .with_context(|| format!("parsing DeepSeek config from {}", path.display()))
    }

    pub fn from_weights(
        config: ModelConfig,
        deepseek_config: DeepseekV3Config,
        weights: WeightMap,
    ) -> Result<Self> {
        if !weights.gguf_weights.is_empty() {
            anyhow::bail!("native DeepSeek MLA does not support GGUF checkpoints");
        }
        let model = DeepseekModel::new(&config, &deepseek_config, weights)?;
        Ok(Self { model, config })
    }
}

impl Model for DeepseekForCausalLM {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn quantized_layer_count(&self) -> usize {
        self.model.quantized_layer_count
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        self.model.forward(input_ids, positions, kv_cache)
    }

    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        self.model.forward_paged(input_ids, positions, gpu_kv_cache, attn_meta)
    }
}

impl CausalLM for DeepseekForCausalLM {
    fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            anyhow::bail!("cannot generate from an empty prompt");
        }
        let mut tokens = prompt.to_vec();
        let mut cache = (0..self.config.num_layers).map(|_| None).collect::<Vec<_>>();
        let input = Tensor::new(prompt, self.model.device())?.reshape((1, prompt.len()))?;
        let positions = (0..prompt.len()).collect::<Vec<_>>();
        let mut logits = self.forward(&input, &positions, &mut cache)?;
        while tokens.len() < max_tokens {
            let last = logits.narrow(1, logits.dim(1)? - 1, 1)?;
            let next = argmax(&last)?;
            tokens.push(next);
            if tokens.len() == max_tokens {
                break;
            }
            let pos = tokens.len() - 1;
            let input = Tensor::new(&[next], self.model.device())?.reshape((1, 1))?;
            logits = self.forward(&input, &[pos], &mut cache)?;
        }
        Ok(tokens)
    }
}

enum Projection {
    Linear(Linear),
    BlockFp8(BlockFp8Linear),
}

impl Projection {
    fn load(
        prefix: &str,
        weights: &mut WeightMap,
        config: &ModelConfig,
        device: &Device,
        block_size: usize,
    ) -> Result<Self> {
        let scale_name = format!("{prefix}.weight_scale_inv");
        if weights.weights.contains_key(&scale_name) {
            let weight = take_tensor(weights, &format!("{prefix}.weight"), device)?;
            let scales = take_tensor(weights, &scale_name, device)?;
            Ok(Self::BlockFp8(BlockFp8Linear::new(weight, scales, block_size)?))
        } else {
            Ok(Self::Linear(crate::llama::load_linear(prefix, weights, config, device)?))
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(linear) => linear.forward(input).map_err(Into::into),
            Self::BlockFp8(linear) => linear.forward(input),
        }
    }

    fn is_quantized(&self) -> bool {
        match self {
            Self::Linear(linear) => linear.is_quantized(),
            Self::BlockFp8(_) => true,
        }
    }
}

struct DeepseekRope {
    cos: Tensor,
    sin: Tensor,
}

impl DeepseekRope {
    fn new(config: &DeepseekV3Config, device: &Device) -> Result<Self> {
        let dim = config.qk_rope_head_dim;
        let half = dim / 2;
        if dim == 0 || dim % 2 != 0 {
            anyhow::bail!("DeepSeek rotary head dimension must be positive and even");
        }

        let base = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let (inv_freq, mscale) = match &config.rope_scaling {
            Some(DeepSeekV2RopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
                factor,
                ..
            }) => {
                let correction = |rotations: f32| {
                    (dim as f32
                        * (*original_max_position_embeddings as f32
                            / (rotations * 2.0 * std::f32::consts::PI))
                            .ln())
                        / (2.0 * config.rope_theta.ln())
                };
                let low = correction(*beta_fast).floor().max(0.0);
                let high = correction(*beta_slow).ceil().min((half - 1) as f32);
                let ramp_denominator = (high - low).max(0.001);
                let inv = base
                    .iter()
                    .enumerate()
                    .map(|(i, &extra)| {
                        let ramp = ((i as f32 - low) / ramp_denominator).clamp(0.0, 1.0);
                        let mask = 1.0 - ramp;
                        let interpolated = extra / factor;
                        interpolated * (1.0 - mask) + extra * mask
                    })
                    .collect();
                let yarn_scale = |value: f32| {
                    if *factor <= 1.0 { 1.0 } else { 0.1 * value * factor.ln() + 1.0 }
                };
                (inv, yarn_scale(*mscale) / yarn_scale(*mscale_all_dim))
            }
            Some(DeepSeekV2RopeScaling::LinearOrDynamic { scaling_type, factor }) => {
                let divisor = match scaling_type {
                    ScaledRopeType::Linear | ScaledRopeType::Dynamic => *factor as f32,
                    _ => *factor as f32,
                };
                (base.iter().map(|v| v / divisor).collect(), 1.0)
            }
            None => (base, 1.0),
        };

        let inv = Tensor::from_vec(inv_freq, (1, half), device)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let freqs = positions.matmul(&inv)?;
        Ok(Self { cos: (freqs.cos()? * mscale as f64)?, sin: (freqs.sin()? * mscale as f64)? })
    }

    fn apply(&self, input: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (batch, heads, tokens, dim) = input.dims4()?;
        if positions.len() != tokens {
            anyhow::bail!(
                "DeepSeek RoPE received {} positions for {tokens} tokens",
                positions.len()
            );
        }
        let indices =
            Tensor::from_iter(positions.iter().map(|&position| position as u32), input.device())?;
        let cos = self.cos.index_select(&indices, 0)?.to_dtype(input.dtype())?.reshape((
            1,
            1,
            tokens,
            dim / 2,
        ))?;
        let sin = self.sin.index_select(&indices, 0)?.to_dtype(input.dtype())?.reshape((
            1,
            1,
            tokens,
            dim / 2,
        ))?;
        let pairs = input.reshape((batch, heads, tokens, dim / 2, 2))?;
        let even = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let odd = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
        let out_even = (even.broadcast_mul(&cos)? - odd.broadcast_mul(&sin)?)?;
        let out_odd = (even.broadcast_mul(&sin)? + odd.broadcast_mul(&cos)?)?;
        Tensor::stack(&[&out_even, &out_odd], D::Minus1)?
            .reshape((batch, heads, tokens, dim))
            .map_err(Into::into)
    }
}

enum QueryProjection {
    Plain(Projection),
    Lora { a: Projection, norm: RmsNorm, b: Projection },
}

impl QueryProjection {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(projection) => projection.forward(input),
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(input)?)?),
        }
    }

    fn quantized_count(&self) -> usize {
        match self {
            Self::Plain(p) => usize::from(p.is_quantized()),
            Self::Lora { a, b, .. } => {
                usize::from(a.is_quantized()) + usize::from(b.is_quantized())
            }
        }
    }
}

struct MlaAttention {
    query: QueryProjection,
    kv_a: Projection,
    kv_a_norm: RmsNorm,
    kv_b: Projection,
    output: Projection,
    rope: Arc<DeepseekRope>,
    num_heads: usize,
    qk_nope_dim: usize,
    qk_rope_dim: usize,
    value_dim: usize,
    scale: f32,
}

impl MlaAttention {
    fn project(&self, hidden: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, tokens, _) = hidden.dims3()?;
        let head_dim = self.qk_nope_dim + self.qk_rope_dim;
        let query = self
            .query
            .forward(hidden)?
            .reshape((batch, tokens, self.num_heads, head_dim))?
            .transpose(1, 2)?;
        let q_nope = query.narrow(D::Minus1, 0, self.qk_nope_dim)?;
        let q_rope = query.narrow(D::Minus1, self.qk_nope_dim, self.qk_rope_dim)?;

        let compressed = self.kv_a.forward(hidden)?;
        let latent = compressed.narrow(D::Minus1, 0, self.kv_a_norm.weight_dim()?)?;
        let k_rope = compressed
            .narrow(D::Minus1, self.kv_a_norm.weight_dim()?, self.qk_rope_dim)?
            .reshape((batch, tokens, 1, self.qk_rope_dim))?
            .transpose(1, 2)?;
        let expanded = self
            .kv_b
            .forward(&self.kv_a_norm.forward(&latent)?)?
            .reshape((batch, tokens, self.num_heads, self.qk_nope_dim + self.value_dim))?
            .transpose(1, 2)?;
        let k_nope = expanded.narrow(D::Minus1, 0, self.qk_nope_dim)?;
        let value = expanded.narrow(D::Minus1, self.qk_nope_dim, self.value_dim)?;
        let q_rope = self.rope.apply(&q_rope, positions)?;
        let k_rope = self.rope.apply(&k_rope, positions)?.repeat((1, self.num_heads, 1, 1))?;
        Ok((
            Tensor::cat(&[&q_nope, &q_rope], D::Minus1)?,
            Tensor::cat(&[&k_nope, &k_rope], D::Minus1)?,
            value,
        ))
    }

    fn forward(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (query, new_key, new_value) = self.project(hidden, positions)?;
        let (key, value) = match cache.take() {
            Some((key, value)) => {
                (Tensor::cat(&[&key, &new_key], 2)?, Tensor::cat(&[&value, &new_value], 2)?)
            }
            None => (new_key, new_value),
        };
        *cache = Some((key.clone(), value.clone()));
        let query_len = query.dim(2)?;
        let key_len = key.dim(2)?;
        let scores = (query.contiguous()?.matmul(&key.t()?.contiguous()?)? * self.scale as f64)?;
        let mask = (0..query_len)
            .flat_map(|query_index| {
                let position = positions[query_index];
                (0..key_len).map(
                    move |key_index| {
                        if key_index > position { f32::NEG_INFINITY } else { 0.0 }
                    },
                )
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, query_len, key_len), hidden.device())?
            .to_dtype(scores.dtype())?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let output = probabilities.matmul(&value.contiguous()?)?;
        let output =
            output.transpose(1, 2)?.reshape((1, query_len, self.num_heads * self.value_dim))?;
        self.output.forward(&output)
    }

    fn forward_paged(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        gpu_cache: &rllm_kernels::cache_ops::GpuKVCache,
        metadata: &rllm_kernels::AttentionMetadata,
        layer: usize,
    ) -> Result<Tensor> {
        let (query, key, value) = self.project(hidden, positions)?;
        #[cfg(has_cuda)]
        {
            use crate::layers::PagedAttentionOp;
            let tokens = positions.len();
            let head_dim = self.qk_nope_dim + self.qk_rope_dim;
            let query = query
                .transpose(1, 2)?
                .reshape((tokens, self.num_heads, head_dim))?
                .to_dtype(DType::F16)?
                .contiguous()?;
            let key = key
                .transpose(1, 2)?
                .reshape((tokens, self.num_heads, head_dim))?
                .to_dtype(DType::F16)?
                .contiguous()?;
            let value = if self.value_dim < head_dim {
                let padding = Tensor::zeros(
                    (1, self.num_heads, tokens, head_dim - self.value_dim),
                    value.dtype(),
                    value.device(),
                )?;
                Tensor::cat(&[&value, &padding], D::Minus1)?
            } else {
                value
            };
            let value = value
                .transpose(1, 2)?
                .reshape((tokens, self.num_heads, head_dim))?
                .to_dtype(DType::F16)?
                .contiguous()?;
            let op = PagedAttentionOp {
                key_cache: gpu_cache.key_ptr(layer) as usize,
                value_cache: gpu_cache.value_ptr(layer) as usize,
                cache_dtype: gpu_cache.dtype(),
                k_scale: gpu_cache.k_scale(layer),
                v_scale: gpu_cache.v_scale(layer),
                num_blocks: gpu_cache.num_blocks() as i64,
                block_size: gpu_cache.block_size() as i64,
                num_q_heads: self.num_heads as i64,
                num_kv_heads: self.num_heads as i64,
                head_dim: head_dim as i64,
                num_tokens: tokens as i64,
                num_seqs: metadata.num_seqs() as i64,
                max_num_blocks_per_seq: metadata.max_num_blocks_per_seq as i64,
                scale: self.scale,
                is_prefill: metadata.num_prefill_tokens > 0,
                slot_mapping: metadata.slot_mapping.clone(),
                block_tables_flat: metadata.flatten_block_tables(),
                seq_lens: metadata.seq_lens.iter().map(|&v| v as i32).collect(),
                query_start_loc: metadata.query_start_loc.iter().map(|&v| v as i32).collect(),
            };
            let output = query.apply_op3_no_bwd(&key, &value, &op)?;
            let output = output
                .narrow(D::Minus1, 0, self.value_dim)?
                .reshape((1, tokens, self.num_heads * self.value_dim))?
                .to_dtype(hidden.dtype())?;
            return self.output.forward(&output);
        }
        #[cfg(not(has_cuda))]
        {
            let _ = (gpu_cache, metadata, layer, query, key, value);
            anyhow::bail!("DeepSeek paged MLA requires a CUDA build")
        }
    }

    fn quantized_count(&self) -> usize {
        self.query.quantized_count()
            + usize::from(self.kv_a.is_quantized())
            + usize::from(self.kv_b.is_quantized())
            + usize::from(self.output.is_quantized())
    }
}

struct DenseMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl DenseMlp {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&self.gate.forward(hidden)?.silu()?.broadcast_mul(&self.up.forward(hidden)?)?)
    }

    fn quantized_count(&self) -> usize {
        usize::from(self.gate.is_quantized())
            + usize::from(self.up.is_quantized())
            + usize::from(self.down.is_quantized())
    }
}

struct DenseMoe {
    gate_weight: Tensor,
    experts: Vec<DenseMlp>,
    shared: Option<DenseMlp>,
    top_k: usize,
    num_groups: usize,
    topk_groups: usize,
    group_limited: bool,
    normalize: bool,
    scaling: f64,
}

impl DenseMoe {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let shape = hidden.shape().clone();
        let width = hidden.dim(D::Minus1)?;
        let tokens = hidden.elem_count() / width;
        let flat = hidden.reshape((tokens, width))?;
        let scores = candle_nn::ops::softmax_last_dim(
            &flat.to_dtype(DType::F32)?.matmul(&self.gate_weight.to_dtype(DType::F32)?.t()?)?,
        )?;
        let selection = if self.group_limited {
            let experts_per_group = self.experts.len() / self.num_groups;
            let grouped = scores.reshape((tokens, self.num_groups, experts_per_group))?;
            let group_scores = grouped.max(D::Minus1)?;
            let groups = group_scores
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.topk_groups)?
                .contiguous()?;
            let group_mask = Tensor::zeros((tokens, self.num_groups), DType::F32, hidden.device())?
                .scatter_add(
                    &groups,
                    &Tensor::ones(groups.shape(), DType::F32, hidden.device())?,
                    1,
                )?;
            let expert_mask = group_mask
                .unsqueeze(D::Minus1)?
                .expand((tokens, self.num_groups, experts_per_group))?
                .reshape((tokens, self.experts.len()))?;
            let negative =
                Tensor::new(f32::NEG_INFINITY, hidden.device())?.broadcast_as(scores.shape())?;
            expert_mask.eq(0f32)?.where_cond(&negative, &scores)?
        } else {
            scores.clone()
        };
        let ids =
            selection.arg_sort_last_dim(false)?.narrow(D::Minus1, 0, self.top_k)?.contiguous()?;
        let mut route_weights = scores.gather(&ids, D::Minus1)?;
        if self.normalize && self.top_k > 1 {
            route_weights =
                route_weights.broadcast_div(&(route_weights.sum_keepdim(D::Minus1)? + 1e-20)?)?;
        } else {
            route_weights = (route_weights * self.scaling)?;
        }

        // Selection metadata is small. Expert activations and GEMMs stay on GPU.
        let cpu_ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let cpu_weights =
            route_weights.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut output = flat.zeros_like()?;
        for (expert_id, expert) in self.experts.iter().enumerate() {
            let mut token_indices = Vec::new();
            let mut weights = Vec::new();
            for token in 0..tokens {
                for route in 0..self.top_k {
                    if cpu_ids[token][route] as usize == expert_id {
                        token_indices.push(token as u32);
                        weights.push(cpu_weights[token][route]);
                    }
                }
            }
            if token_indices.is_empty() {
                continue;
            }
            let indices = Tensor::from_vec(token_indices, weights.len(), hidden.device())?;
            let selected = expert.forward(&flat.index_select(&indices, 0)?)?;
            let weights = Tensor::from_vec(weights, (indices.elem_count(), 1), hidden.device())?
                .to_dtype(selected.dtype())?;
            output = output.index_add(&indices, &selected.broadcast_mul(&weights)?, 0)?;
        }
        if let Some(shared) = &self.shared {
            output = (output + shared.forward(&flat)?)?;
        }
        output.reshape(shape).map_err(Into::into)
    }

    fn quantized_count(&self) -> usize {
        self.experts.iter().map(DenseMlp::quantized_count).sum::<usize>()
            + self.shared.as_ref().map(DenseMlp::quantized_count).unwrap_or(0)
    }
}

enum FeedForward {
    Dense(DenseMlp),
    DenseMoe(DenseMoe),
    Fp8Moe(DeepseekV3Moe),
}

impl FeedForward {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden),
            Self::DenseMoe(moe) => moe.forward(hidden),
            Self::Fp8Moe(moe) => moe.forward(hidden),
        }
    }

    fn quantized_count(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.quantized_count(),
            Self::DenseMoe(moe) => moe.quantized_count(),
            Self::Fp8Moe(_) => 1,
        }
    }
}

struct DecoderLayer {
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    attention: MlaAttention,
    feed_forward: FeedForward,
}

impl DecoderLayer {
    fn forward(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let attention =
            self.attention.forward(&self.input_norm.forward(hidden)?, positions, cache)?;
        let hidden = (hidden + attention)?;
        let mlp = self.feed_forward.forward(&self.post_attention_norm.forward(&hidden)?)?;
        (hidden + mlp).map_err(Into::into)
    }

    fn forward_paged(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        cache: &rllm_kernels::cache_ops::GpuKVCache,
        metadata: &rllm_kernels::AttentionMetadata,
        layer: usize,
    ) -> Result<Tensor> {
        let attention = self.attention.forward_paged(
            &self.input_norm.forward(hidden)?,
            positions,
            cache,
            metadata,
            layer,
        )?;
        let hidden = (hidden + attention)?;
        let mlp = self.feed_forward.forward(&self.post_attention_norm.forward(&hidden)?)?;
        (hidden + mlp).map_err(Into::into)
    }
}

struct DeepseekModel {
    embedding: Tensor,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Projection,
    device: Device,
    quantized_layer_count: usize,
}

impl DeepseekModel {
    fn new(
        model_config: &ModelConfig,
        config: &DeepseekV3Config,
        mut weights: WeightMap,
    ) -> Result<Self> {
        let device = weights.device.clone();
        let block_size = config.fp8_block_size()?;
        let embedding = take_tensor(&mut weights, "model.embed_tokens.weight", &device)?;
        let lm_head = if has_projection(&weights, "lm_head") {
            Projection::load("lm_head", &mut weights, model_config, &device, block_size)?
        } else if config.tie_word_embeddings {
            Projection::Linear(Linear::new(embedding.clone()))
        } else {
            anyhow::bail!("missing lm_head.weight for untied DeepSeek checkpoint");
        };
        let norm = RmsNorm::new(
            take_tensor(&mut weights, "model.norm.weight", &device)?,
            config.rms_norm_eps,
        );
        let is_v3 = matches!(
            model_config.architecture.as_str(),
            "DeepseekV3ForCausalLM" | "DeepseekR1ForCausalLM"
        );
        let rope = Arc::new(DeepseekRope::new(config, &device)?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut quantized = usize::from(lm_head.is_quantized());
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_index}");
            let attention = load_attention(
                &format!("{prefix}.self_attn"),
                model_config,
                config,
                &mut weights,
                &device,
                block_size,
                rope.clone(),
            )?;
            let input_norm = RmsNorm::new(
                take_tensor(&mut weights, &format!("{prefix}.input_layernorm.weight"), &device)?,
                config.rms_norm_eps,
            );
            let post_attention_norm = RmsNorm::new(
                take_tensor(
                    &mut weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &device,
                )?,
                config.rms_norm_eps,
            );
            let is_moe = layer_index >= config.first_k_dense_replace
                && layer_index % config.moe_layer_freq == 0;
            let feed_forward = if is_moe && is_v3 {
                FeedForward::Fp8Moe(DeepseekV3Moe::from_weights(
                    config,
                    &mut weights,
                    &format!("{prefix}.mlp"),
                    block_size,
                )?)
            } else if is_moe {
                FeedForward::DenseMoe(load_dense_moe(
                    &format!("{prefix}.mlp"),
                    model_config,
                    config,
                    &mut weights,
                    &device,
                    block_size,
                )?)
            } else {
                FeedForward::Dense(load_mlp(
                    &format!("{prefix}.mlp"),
                    model_config,
                    &mut weights,
                    &device,
                    block_size,
                )?)
            };
            quantized += attention.quantized_count() + feed_forward.quantized_count();
            layers.push(DecoderLayer { input_norm, post_attention_norm, attention, feed_forward });
        }
        let leftovers = weights.unconsumed_names();
        if !leftovers.is_empty() {
            tracing::debug!(
                count = leftovers.len(),
                first = ?leftovers.iter().take(8).collect::<Vec<_>>(),
                "unconsumed DeepSeek checkpoint tensors"
            );
        }
        Ok(Self { embedding, layers, norm, lm_head, device, quantized_layer_count: quantized })
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        if caches.len() != self.layers.len() {
            anyhow::bail!("DeepSeek cache layer count does not match the decoder");
        }
        let mut hidden = embedding_lookup(&self.embedding, input_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer
                .forward(&hidden, positions, &mut caches[index])
                .with_context(|| format!("DeepSeek layer {index}"))?;
        }
        self.lm_head.forward(&self.norm.forward(&hidden)?)
    }

    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        cache: &rllm_kernels::cache_ops::GpuKVCache,
        metadata: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        if cache.num_layers() != self.layers.len() {
            anyhow::bail!(
                "DeepSeek paged cache has {} layers, expected {}",
                cache.num_layers(),
                self.layers.len()
            );
        }
        let mut hidden = embedding_lookup(&self.embedding, input_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer
                .forward_paged(&hidden, positions, cache, metadata, index)
                .with_context(|| format!("DeepSeek paged layer {index}"))?;
        }
        self.lm_head.forward(&self.norm.forward(&hidden)?)
    }
}

fn load_attention(
    prefix: &str,
    model: &ModelConfig,
    config: &DeepseekV3Config,
    weights: &mut WeightMap,
    device: &Device,
    block_size: usize,
    rope: Arc<DeepseekRope>,
) -> Result<MlaAttention> {
    let query = if let Some(_rank) = config.q_lora_rank {
        QueryProjection::Lora {
            a: Projection::load(&format!("{prefix}.q_a_proj"), weights, model, device, block_size)?,
            norm: RmsNorm::new(
                take_tensor(weights, &format!("{prefix}.q_a_layernorm.weight"), device)?,
                config.rms_norm_eps,
            ),
            b: Projection::load(&format!("{prefix}.q_b_proj"), weights, model, device, block_size)?,
        }
    } else {
        QueryProjection::Plain(Projection::load(
            &format!("{prefix}.q_proj"),
            weights,
            model,
            device,
            block_size,
        )?)
    };
    let _ = config.q_lora_rank;
    let kv_a = Projection::load(
        &format!("{prefix}.kv_a_proj_with_mqa"),
        weights,
        model,
        device,
        block_size,
    )?;
    let kv_a_norm = RmsNorm::new(
        take_tensor(weights, &format!("{prefix}.kv_a_layernorm.weight"), device)?,
        config.rms_norm_eps,
    );
    let mut scale = 1.0f32 / ((config.qk_nope_head_dim + config.qk_rope_head_dim) as f32).sqrt();
    if let Some(DeepSeekV2RopeScaling::Yarn { mscale_all_dim, factor, .. }) = &config.rope_scaling {
        let mscale = if *factor <= 1.0 { 1.0 } else { 0.1 * mscale_all_dim * factor.ln() + 1.0 };
        scale *= mscale * mscale;
    }
    Ok(MlaAttention {
        query,
        kv_a,
        kv_a_norm,
        kv_b: Projection::load(&format!("{prefix}.kv_b_proj"), weights, model, device, block_size)?,
        output: Projection::load(&format!("{prefix}.o_proj"), weights, model, device, block_size)?,
        rope,
        num_heads: config.num_attention_heads,
        qk_nope_dim: config.qk_nope_head_dim,
        qk_rope_dim: config.qk_rope_head_dim,
        value_dim: config.v_head_dim,
        scale,
    })
}

fn load_mlp(
    prefix: &str,
    model: &ModelConfig,
    weights: &mut WeightMap,
    device: &Device,
    block_size: usize,
) -> Result<DenseMlp> {
    Ok(DenseMlp {
        gate: Projection::load(&format!("{prefix}.gate_proj"), weights, model, device, block_size)?,
        up: Projection::load(&format!("{prefix}.up_proj"), weights, model, device, block_size)?,
        down: Projection::load(&format!("{prefix}.down_proj"), weights, model, device, block_size)?,
    })
}

fn load_dense_moe(
    prefix: &str,
    model: &ModelConfig,
    config: &DeepseekV3Config,
    weights: &mut WeightMap,
    device: &Device,
    block_size: usize,
) -> Result<DenseMoe> {
    let gate_weight = take_tensor(weights, &format!("{prefix}.gate.weight"), device)?;
    let mut experts = Vec::with_capacity(config.n_routed_experts);
    for expert in 0..config.n_routed_experts {
        experts.push(load_mlp(
            &format!("{prefix}.experts.{expert}"),
            model,
            weights,
            device,
            block_size,
        )?);
    }
    let shared = if config.n_shared_experts > 0 {
        Some(load_mlp(&format!("{prefix}.shared_experts"), model, weights, device, block_size)?)
    } else {
        None
    };
    Ok(DenseMoe {
        gate_weight,
        experts,
        shared,
        top_k: config.num_experts_per_tok,
        num_groups: config.n_group,
        topk_groups: config.topk_group,
        group_limited: config.topk_method == "group_limited_greedy",
        normalize: config.norm_topk_prob,
        scaling: config.routed_scaling_factor,
    })
}

fn take_tensor(weights: &mut WeightMap, name: &str, device: &Device) -> Result<Tensor> {
    let tensor =
        weights.weights.remove(name).ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
    if tensor.device().is_cpu() && !device.is_cpu() {
        tensor.to_device(device).map_err(Into::into)
    } else {
        Ok(tensor)
    }
}

fn has_projection(weights: &WeightMap, prefix: &str) -> bool {
    weights.weights.contains_key(&format!("{prefix}.weight"))
        || weights.weights.contains_key(&format!("{prefix}.qweight"))
        || weights.quantized.contains_key(&format!("{prefix}.weight"))
        || weights.gguf_weights.contains_key(&format!("{prefix}.weight"))
}

fn embedding_lookup(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let values = ids.flatten_all()?.to_vec1::<u32>()?;
    let batch = ids.dim(0)?;
    let tokens = ids.dim(1)?;
    let hidden = weight.dim(D::Minus1)?;
    let indices = Tensor::from_vec(values, batch * tokens, ids.device())?;
    weight.index_select(&indices, 0)?.reshape((batch, tokens, hidden)).map_err(Into::into)
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let values =
        logits.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    Ok(values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .unwrap_or(0))
}

trait RmsNormExt {
    fn weight_dim(&self) -> Result<usize>;
}

impl RmsNormExt for RmsNorm {
    fn weight_dim(&self) -> Result<usize> {
        self.hidden_size().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rllm_core::config::TokenizerMode;

    use super::*;

    fn toy_config() -> DeepseekV3Config {
        DeepseekV3Config::from_json(
            r#"{
                "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
                "moe_intermediate_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "n_shared_experts": 1,
                "n_routed_experts": 4, "num_experts_per_tok": 2,
                "first_k_dense_replace": 1, "q_lora_rank": 8,
                "kv_lora_rank": 8, "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4, "v_head_dim": 4,
                "n_group": 2, "topk_group": 1,
                "max_position_embeddings": 32, "rope_theta": 10000.0,
                "rms_norm_eps": 0.000001
            }"#,
        )
        .unwrap()
    }

    fn core_config(architecture: &str, dtype: rllm_core::dtype::DType) -> ModelConfig {
        ModelConfig {
            model_id: "toy".into(),
            architecture: architecture.into(),
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            max_model_len: 32,
            rope_theta: 10000.0,
            rope_scaling: None,
            dtype,
            quantization: None,
            tokenizer_mode: TokenizerMode::Auto,
        }
    }

    fn toy_weights(device: &Device, dtype: DType) -> Result<WeightMap> {
        let mut weights = HashMap::new();
        let mut insert = |name: &str, shape: &[usize]| -> Result<()> {
            weights.insert(name.to_string(), Tensor::zeros(shape, dtype, device)?);
            Ok(())
        };
        insert("model.embed_tokens.weight", &[32, 16])?;
        insert("lm_head.weight", &[32, 16])?;
        insert("model.norm.weight", &[16])?;
        insert("model.layers.0.input_layernorm.weight", &[16])?;
        insert("model.layers.0.post_attention_layernorm.weight", &[16])?;
        insert("model.layers.0.self_attn.q_a_proj.weight", &[8, 16])?;
        insert("model.layers.0.self_attn.q_a_layernorm.weight", &[8])?;
        insert("model.layers.0.self_attn.q_b_proj.weight", &[16, 8])?;
        insert("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", &[12, 16])?;
        insert("model.layers.0.self_attn.kv_a_layernorm.weight", &[8])?;
        insert("model.layers.0.self_attn.kv_b_proj.weight", &[16, 8])?;
        insert("model.layers.0.self_attn.o_proj.weight", &[16, 8])?;
        insert("model.layers.0.mlp.gate_proj.weight", &[24, 16])?;
        insert("model.layers.0.mlp.up_proj.weight", &[24, 16])?;
        insert("model.layers.0.mlp.down_proj.weight", &[16, 24])?;
        Ok(WeightMap {
            weights,
            quantized: HashMap::new(),
            gguf_weights: HashMap::new(),
            quant_schema: None,
            device: device.clone(),
        })
    }

    fn toy_v3_config() -> DeepseekV3Config {
        DeepseekV3Config::from_json(
            r#"{
                "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
                "moe_intermediate_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "n_shared_experts": 1,
                "n_routed_experts": 4, "num_experts_per_tok": 2,
                "first_k_dense_replace": 0, "q_lora_rank": 8,
                "kv_lora_rank": 8, "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4, "v_head_dim": 4,
                "n_group": 2, "topk_group": 1, "topk_method": "noaux_tc",
                "scoring_func": "sigmoid", "max_position_embeddings": 32,
                "rope_theta": 10000.0, "rms_norm_eps": 0.000001,
                "quantization_config": {"weight_block_size": [2, 2]}
            }"#,
        )
        .unwrap()
    }

    fn toy_v3_weights(device: &Device) -> Result<WeightMap> {
        let mut weights = HashMap::new();
        let mut dense = |name: &str, shape: &[usize]| -> Result<()> {
            weights.insert(name.to_string(), Tensor::zeros(shape, DType::F32, device)?);
            Ok(())
        };
        dense("model.embed_tokens.weight", &[32, 16])?;
        dense("model.norm.weight", &[16])?;
        dense("model.layers.0.input_layernorm.weight", &[16])?;
        dense("model.layers.0.post_attention_layernorm.weight", &[16])?;
        dense("model.layers.0.self_attn.q_a_layernorm.weight", &[8])?;
        dense("model.layers.0.self_attn.kv_a_layernorm.weight", &[8])?;
        dense("model.layers.0.mlp.gate.weight", &[4, 16])?;
        dense("model.layers.0.mlp.gate.e_score_correction_bias", &[4])?;

        let fp8_shapes = [
            ("lm_head", 32, 16),
            ("model.layers.0.self_attn.q_a_proj", 8, 16),
            ("model.layers.0.self_attn.q_b_proj", 16, 8),
            ("model.layers.0.self_attn.kv_a_proj_with_mqa", 12, 16),
            ("model.layers.0.self_attn.kv_b_proj", 16, 8),
            ("model.layers.0.self_attn.o_proj", 16, 8),
            ("model.layers.0.mlp.shared_experts.gate_proj", 8, 16),
            ("model.layers.0.mlp.shared_experts.up_proj", 8, 16),
            ("model.layers.0.mlp.shared_experts.down_proj", 16, 8),
        ];
        for (prefix, output, input) in fp8_shapes {
            weights.insert(
                format!("{prefix}.weight"),
                Tensor::zeros((output, input), DType::F32, &Device::Cpu)?
                    .to_dtype(DType::F8E4M3)?
                    .to_device(device)?,
            );
            weights.insert(
                format!("{prefix}.weight_scale_inv"),
                Tensor::ones((output.div_ceil(2), input.div_ceil(2)), DType::F32, device)?,
            );
        }
        for expert in 0..4 {
            for (projection, output, input) in
                [("gate_proj", 8, 16), ("up_proj", 8, 16), ("down_proj", 16, 8)]
            {
                let prefix = format!("model.layers.0.mlp.experts.{expert}.{projection}");
                weights.insert(
                    format!("{prefix}.weight"),
                    Tensor::zeros((output, input), DType::F32, &Device::Cpu)?
                        .to_dtype(DType::F8E4M3)?
                        .to_device(device)?,
                );
                weights.insert(
                    format!("{prefix}.weight_scale_inv"),
                    Tensor::ones((output.div_ceil(2), input.div_ceil(2)), DType::F32, device)?,
                );
            }
        }
        Ok(WeightMap {
            weights,
            quantized: HashMap::new(),
            gguf_weights: HashMap::new(),
            quant_schema: None,
            device: device.clone(),
        })
    }

    fn run_toy_legacy(device: &Device, dtype: DType) -> Result<()> {
        let model = DeepseekForCausalLM::from_weights(
            core_config("DeepseekV2ForCausalLM", rllm_core::dtype::DType::F16),
            toy_config(),
            toy_weights(device, dtype)?,
        )?;
        let mut cache = vec![None];
        let input = Tensor::new(&[[1u32, 2]], device)?;
        let logits = model.forward(&input, &[0, 1], &mut cache)?;
        assert_eq!(logits.dims(), &[1, 2, 32]);
        let decode = Tensor::new(&[[3u32]], device)?;
        let logits = model.forward(&decode, &[2], &mut cache)?;
        assert_eq!(logits.dims(), &[1, 1, 32]);
        Ok(())
    }

    #[test]
    fn parses_v2_v3_shared_config() -> Result<()> {
        let config = DeepseekV3Config::from_json(
            r#"{
                "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                "moe_intermediate_size": 8, "num_hidden_layers": 2,
                "num_attention_heads": 2, "n_shared_experts": 1,
                "n_routed_experts": 4, "num_experts_per_tok": 2,
                "first_k_dense_replace": 1, "q_lora_rank": 8,
                "kv_lora_rank": 8, "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4, "v_head_dim": 4,
                "n_group": 2, "topk_group": 1,
                "max_position_embeddings": 32, "rope_theta": 10000.0,
                "rms_norm_eps": 0.000001
            }"#,
        )?;
        assert_eq!(config.fp8_block_size()?, 128);
        assert_eq!(config.topk_method, "greedy");
        Ok(())
    }

    #[test]
    fn interleaved_rope_position_zero_is_identity() -> Result<()> {
        let config = DeepseekV3Config::from_json(
            r#"{
                "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
                "moe_intermediate_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 1, "n_shared_experts": 1,
                "n_routed_experts": 2, "num_experts_per_tok": 1,
                "first_k_dense_replace": 1, "q_lora_rank": null,
                "kv_lora_rank": 4, "qk_nope_head_dim": 2,
                "qk_rope_head_dim": 4, "v_head_dim": 2,
                "n_group": 1, "topk_group": 1,
                "max_position_embeddings": 8, "rope_theta": 10000.0,
                "rms_norm_eps": 0.000001
            }"#,
        )?;
        let rope = DeepseekRope::new(&config, &Device::Cpu)?;
        let input = Tensor::new(&[[[[1f32, 2., 3., 4.]]]], &Device::Cpu)?;
        let output = rope.apply(&input, &[0])?;
        assert_eq!(input.flatten_all()?.to_vec1::<f32>()?, output.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn toy_v2_decoder_prefill_and_decode_cpu() -> Result<()> {
        run_toy_legacy(&Device::Cpu, DType::F32)
    }

    #[test]
    fn toy_v3_full_fp8_decoder_cpu() -> Result<()> {
        let model = DeepseekForCausalLM::from_weights(
            core_config("DeepseekV3ForCausalLM", rllm_core::dtype::DType::FP8E4M3),
            toy_v3_config(),
            toy_v3_weights(&Device::Cpu)?,
        )?;
        let mut cache = vec![None];
        let logits =
            model.forward(&Tensor::new(&[[1u32, 2]], &Device::Cpu)?, &[0, 1], &mut cache)?;
        assert_eq!(logits.dims(), &[1, 2, 32]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn toy_v2_decoder_prefill_and_decode_cuda() -> Result<()> {
        run_toy_legacy(&Device::new_cuda(0)?, DType::F16)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn toy_r1_full_fp8_decoder_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let model = DeepseekForCausalLM::from_weights(
            core_config("DeepseekR1ForCausalLM", rllm_core::dtype::DType::FP8E4M3),
            toy_v3_config(),
            toy_v3_weights(&device)?,
        )?;
        let cache =
            rllm_kernels::cache_ops::GpuKVCache::new(1, 1, 2, 8, 4, rllm_core::dtype::DType::F16)?;
        let metadata = rllm_kernels::AttentionMetadata {
            seq_lens: vec![1],
            query_start_loc: vec![0, 1],
            block_tables: vec![vec![0]],
            slot_mapping: vec![0],
            num_prefill_tokens: 1,
            num_decode_tokens: 0,
            max_num_blocks_per_seq: 1,
            common_prefix_blocks: 0,
            sliding_window: None,
        };
        let logits =
            model.forward_paged(&Tensor::new(&[[1u32]], &device)?, &[0], &cache, &metadata)?;
        assert_eq!(logits.dims(), &[1, 1, 32]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn toy_v2_paged_mla_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let model = DeepseekForCausalLM::from_weights(
            core_config("DeepseekV2ForCausalLM", rllm_core::dtype::DType::F16),
            toy_config(),
            toy_weights(&device, DType::F16)?,
        )?;
        let cache =
            rllm_kernels::cache_ops::GpuKVCache::new(2, 1, 2, 8, 4, rllm_core::dtype::DType::F16)?;
        let prefill = rllm_kernels::AttentionMetadata {
            seq_lens: vec![2],
            query_start_loc: vec![0, 2],
            block_tables: vec![vec![0]],
            slot_mapping: vec![0, 1],
            num_prefill_tokens: 2,
            num_decode_tokens: 0,
            max_num_blocks_per_seq: 1,
            common_prefix_blocks: 0,
            sliding_window: None,
        };
        let logits =
            model.forward_paged(&Tensor::new(&[[1u32, 2]], &device)?, &[0, 1], &cache, &prefill)?;
        assert_eq!(logits.dims(), &[1, 2, 32]);

        let decode = rllm_kernels::AttentionMetadata {
            seq_lens: vec![3],
            query_start_loc: vec![0, 1],
            block_tables: vec![vec![0]],
            slot_mapping: vec![2],
            num_prefill_tokens: 0,
            num_decode_tokens: 1,
            max_num_blocks_per_seq: 1,
            common_prefix_blocks: 0,
            sliding_window: None,
        };
        let logits =
            model.forward_paged(&Tensor::new(&[[3u32]], &device)?, &[2], &cache, &decode)?;
        assert_eq!(logits.dims(), &[1, 1, 32]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn toy_v2_paged_mla_multi_sequence_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let model = DeepseekForCausalLM::from_weights(
            core_config("DeepseekV2ForCausalLM", rllm_core::dtype::DType::F16),
            toy_config(),
            toy_weights(&device, DType::F16)?,
        )?;
        let cache =
            rllm_kernels::cache_ops::GpuKVCache::new(2, 1, 2, 8, 4, rllm_core::dtype::DType::F16)?;
        let prefill = rllm_kernels::AttentionMetadata {
            seq_lens: vec![2, 2],
            query_start_loc: vec![0, 2, 4],
            block_tables: vec![vec![0], vec![1]],
            slot_mapping: vec![0, 1, 4, 5],
            num_prefill_tokens: 4,
            num_decode_tokens: 0,
            max_num_blocks_per_seq: 1,
            common_prefix_blocks: 0,
            sliding_window: None,
        };
        let logits = model.forward_paged(
            &Tensor::new(&[[1u32, 2, 3, 4]], &device)?,
            &[0, 1, 0, 1],
            &cache,
            &prefill,
        )?;
        assert_eq!(logits.dims(), &[1, 4, 32]);

        let decode = rllm_kernels::AttentionMetadata {
            seq_lens: vec![3, 3],
            query_start_loc: vec![0, 1, 2],
            block_tables: vec![vec![0], vec![1]],
            slot_mapping: vec![2, 6],
            num_prefill_tokens: 0,
            num_decode_tokens: 2,
            max_num_blocks_per_seq: 1,
            common_prefix_blocks: 0,
            sliding_window: None,
        };
        let logits =
            model.forward_paged(&Tensor::new(&[[5u32, 6]], &device)?, &[2, 2], &cache, &decode)?;
        assert_eq!(logits.dims(), &[1, 2, 32]);
        Ok(())
    }
}
