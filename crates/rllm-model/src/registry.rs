use std::collections::HashMap;

use anyhow::Result;
#[cfg(feature = "candle-backend")]
use candle_core::Tensor;
use rllm_core::config::ModelConfig;

#[cfg(feature = "candle-backend")]
pub trait Model: Send + Sync {
    fn config(&self) -> &ModelConfig;

    /// Legacy forward pass using per-request Candle KV cache tensors.
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor>;

    /// Paged forward pass using global GPU KV cache and PagedAttention kernels.
    ///
    /// Uses `GpuKVCache` for block-addressed K/V storage and `AttentionMetadata`
    /// for batched attention computation. Falls back to `forward()` when CUDA
    /// is not available.
    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        // Default: fall back to legacy forward (no paged attention).
        // Models that support PagedAttention should override this.
        let _ = (gpu_kv_cache, attn_meta);
        let mut kv_cache: Vec<Option<(Tensor, Tensor)>> = vec![None; self.config().num_layers];
        self.forward(input_ids, positions, &mut kv_cache)
    }
}

#[cfg(not(feature = "candle-backend"))]
pub trait Model: Send + Sync {
    fn config(&self) -> &ModelConfig;
}

#[cfg(feature = "candle-backend")]
pub trait CausalLM: Model {
    fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>>;
}

#[cfg(not(feature = "candle-backend"))]
pub trait CausalLM: Model {}

pub type ModelFactory = fn(&ModelConfig) -> Result<Box<dyn CausalLM>>;

pub struct ModelRegistry {
    factories: HashMap<String, ModelFactory>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self { factories: HashMap::new() }
    }

    pub fn register(&mut self, architecture: &str, factory: ModelFactory) {
        self.factories.insert(architecture.to_string(), factory);
    }

    pub fn create(&self, architecture: &str, config: &ModelConfig) -> Result<Box<dyn CausalLM>> {
        let factory = self
            .factories
            .get(architecture)
            .ok_or_else(|| anyhow::anyhow!("unsupported architecture: {architecture}"))?;
        factory(config)
    }

    pub fn default_registry() -> Self {
        let mut reg = Self::new();
        #[cfg(feature = "candle-backend")]
        {
            reg.register("LlamaForCausalLM", crate::llama::LlamaForCausalLM::factory);
            reg.register("MistralForCausalLM", crate::llama::LlamaForCausalLM::factory);
        }
        reg
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
