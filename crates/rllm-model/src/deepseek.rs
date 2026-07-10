use std::{path::Path, sync::Mutex};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::deepseek2::{DeepSeekV2, DeepSeekV2Config};
use rllm_core::config::ModelConfig;

use crate::{
    loader::WeightMap,
    registry::{CausalLM, Model},
};

/// Native DeepSeek V2 model adapter.
///
/// Candle's implementation executes MLA and routed/shared MoE on the tensor's
/// device. Its expanded K/V cache is sequential and model-owned, so this adapter
/// deliberately accepts one active sequence until rLLM has a paged MLA cache.
pub struct DeepseekForCausalLM {
    model: Mutex<DeepSeekV2>,
    config: ModelConfig,
}

impl DeepseekForCausalLM {
    pub fn factory(_config: &ModelConfig) -> Result<Box<dyn CausalLM>> {
        anyhow::bail!(
            "DeepseekForCausalLM::factory requires loaded weights; use from_weights() instead"
        )
    }

    pub fn parse_config(path: &Path) -> Result<DeepSeekV2Config> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading DeepSeek config from {}", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("parsing DeepSeek V2 config from {}", path.display()))
    }

    pub fn from_weights(
        config: ModelConfig,
        deepseek_config: DeepSeekV2Config,
        weights: WeightMap,
    ) -> Result<Self> {
        if !weights.quantized.is_empty() || !weights.gguf_weights.is_empty() {
            anyhow::bail!(
                "native DeepSeek MLA/MoE currently requires unquantized SafeTensors; \
                 quantized expert weights are not supported by this model path"
            );
        }

        let dtype = model_dtype(&weights)?;
        let device = weights.device.clone();
        let vb = VarBuilder::from_tensors(weights.weights, dtype, &device);
        let model = DeepSeekV2::new(&deepseek_config, vb)
            .context("building native DeepSeek MLA/MoE model")?;

        Ok(Self { model: Mutex::new(model), config })
    }

    fn lock_model(&self) -> Result<std::sync::MutexGuard<'_, DeepSeekV2>> {
        self.model.lock().map_err(|_| anyhow::anyhow!("DeepSeek model lock poisoned"))
    }

    fn forward_sequential(&self, input_ids: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2().map_err(|e| anyhow::anyhow!("{e}"))?;
        if batch != 1 {
            anyhow::bail!(
                "native DeepSeek MLA/MoE currently supports one active sequence, got batch {batch}"
            );
        }
        if positions.len() != seq_len {
            anyhow::bail!(
                "DeepSeek positions length {} does not match input length {seq_len}",
                positions.len()
            );
        }
        let offset = positions.first().copied().unwrap_or(0);
        if positions.iter().enumerate().any(|(i, &position)| position != offset + i) {
            anyhow::bail!("DeepSeek sequential cache requires contiguous positions");
        }

        let mut model = self.lock_model()?;
        if offset == 0 {
            model.clear_kv_cache();
        }
        let logits = model.forward(input_ids, offset).map_err(|e| anyhow::anyhow!("{e}"))?;
        logits.unsqueeze(1).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

impl Model for DeepseekForCausalLM {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        _kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        self.forward_sequential(input_ids, positions)
    }

    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        _gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        if attn_meta.seq_lens.len() > 1 {
            anyhow::bail!(
                "native DeepSeek MLA/MoE paged batches are not implemented; set --max-num-seqs 1"
            );
        }
        self.forward_sequential(input_ids, positions)
    }
}

impl CausalLM for DeepseekForCausalLM {
    fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let _ = (prompt, max_tokens);
        anyhow::bail!("direct DeepSeek generation is not available; use the engine forward path")
    }
}

fn model_dtype(weights: &WeightMap) -> Result<DType> {
    weights
        .weights
        .values()
        .find_map(|tensor| match tensor.dtype() {
            dtype @ (DType::F16 | DType::BF16 | DType::F32) => Some(dtype),
            _ => None,
        })
        .ok_or_else(|| {
            anyhow::anyhow!("DeepSeek checkpoint contains no supported floating weights")
        })
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn toy_config() -> DeepSeekV2Config {
        serde_json::from_str(
            r#"{
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 32,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "attention_bias": false,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 4,
                "kv_lora_rank": 8,
                "v_head_dim": 4,
                "qk_nope_head_dim": 4,
                "n_group": 1,
                "topk_group": 1
            }"#,
        )
        .unwrap()
    }

    fn run_toy_forward(device: &Device, dtype: DType) -> Result<()> {
        let config = toy_config();
        let vb = VarBuilder::zeros(dtype, device);
        let mut model = DeepSeekV2::new(&config, vb)?;
        let input = Tensor::new(&[[1u32, 2, 3]], device)?;
        let logits = model.forward(&input, 0)?;
        assert_eq!(logits.dims(), &[1, 32]);

        let decode = Tensor::new(&[[4u32]], device)?;
        let logits = model.forward(&decode, 3)?;
        assert_eq!(logits.dims(), &[1, 32]);
        Ok(())
    }

    #[test]
    fn toy_mla_moe_forward_cpu() -> Result<()> {
        run_toy_forward(&Device::Cpu, DType::F32)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn toy_mla_moe_forward_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        run_toy_forward(&device, DType::F16)
    }
}
