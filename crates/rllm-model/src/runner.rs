#[cfg(feature = "candle-backend")]
use anyhow::{Context, Result};
#[cfg(feature = "candle-backend")]
use candle_core::Device;
#[cfg(feature = "candle-backend")]
use rllm_core::config::{ModelConfig, QuantizationKind};

#[cfg(feature = "candle-backend")]
use crate::hf_config;
#[cfg(feature = "candle-backend")]
use crate::llama::LlamaForCausalLM;
#[cfg(feature = "candle-backend")]
use crate::loader;
#[cfg(feature = "candle-backend")]
use crate::registry::CausalLM;

/// Simple model runner for single-prompt greedy decode.
#[cfg(feature = "candle-backend")]
pub struct ModelRunner {
    model: Box<dyn CausalLM>,
    device: Device,
}

#[cfg(feature = "candle-backend")]
impl ModelRunner {
    /// Load a model from a local directory or Hugging Face model ID.
    pub fn from_model_ref(model_ref: &str) -> Result<Self> {
        let model_dir = loader::resolve_model_dir(model_ref)
            .with_context(|| format!("resolving model reference {model_ref}"))?;
        Self::from_dir_path_with_config(&model_dir, None)
    }

    /// Load a model from a reference while preserving server-side config overrides.
    pub fn from_model_ref_with_config(model_ref: &str, mut config: ModelConfig) -> Result<Self> {
        let model_dir = loader::resolve_model_dir(model_ref)
            .with_context(|| format!("resolving model reference {model_ref}"))?;
        config.model_id = model_dir.to_string_lossy().to_string();
        Self::from_dir_path_with_config(&model_dir, Some(config))
    }

    /// Load a model from a local directory.
    pub fn from_dir(model_dir: &str) -> Result<Self> {
        Self::from_dir_path_with_config(std::path::Path::new(model_dir), None)
    }

    fn from_dir_path_with_config(
        model_dir: &std::path::Path,
        config: Option<ModelConfig>,
    ) -> Result<Self> {
        let is_gguf = model_dir.is_file() && model_dir.extension().is_some_and(|ext| ext == "gguf");
        let gguf_file_path = if is_gguf {
            Some(model_dir.to_path_buf())
        } else if model_dir.is_dir() {
            let mut gguf_path = None;
            if let Ok(entries) = std::fs::read_dir(model_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
                        gguf_path = Some(path);
                        break;
                    }
                }
            }
            gguf_path
        } else {
            None
        };

        let device =
            Device::cuda_if_available(0).map_err(|e| anyhow::anyhow!("device init: {e}"))?;

        let (config, weight_map) = if let Some(ref path) = gguf_file_path {
            let config = match config {
                Some(config) => config,
                None => crate::gguf_loader::parse_gguf_config(path)
                    .context("parsing GGUF model config")?,
            };
            tracing::info!(
                model_path = %path.display(),
                architecture = %config.architecture,
                device = ?device,
                "loading GGUF model"
            );
            let weight_map = crate::gguf_loader::load_gguf_weights(path, &device)
                .context("loading GGUF weights")?;
            (config, weight_map)
        } else {
            let config = match config {
                Some(config) => config,
                None => {
                    let config_path = model_dir.join("config.json");
                    hf_config::parse_hf_config(&config_path).context("parsing model config")?
                }
            };

            let is_int8 =
                config.quantization.as_ref().is_some_and(|q| q.kind == QuantizationKind::Int8);
            let load_device = if is_int8 && matches!(device, Device::Cuda(_)) {
                tracing::info!(
                    model_dir = %model_dir.display(),
                    "INT8 quantization active — loading weights to CPU first to save GPU memory"
                );
                Some(&Device::Cpu)
            } else {
                None
            };

            tracing::info!(
                model_dir = %model_dir.display(),
                architecture = %config.architecture,
                device = ?device,
                load_to_cpu = load_device.is_some(),
                "loading model"
            );

            let (weight_map, _tied) =
                loader::load_weights_with_tied_detection(model_dir, &device, load_device)
                    .context("loading weights")?;
            (config, weight_map)
        };

        tracing::debug!(
            model_dir = %model_dir.display(),
            tensors = weight_map.weights.len(),
            quantized_tensors = weight_map.quantized.len(),
            gguf_tensors = weight_map.gguf_weights.len(),
            "model weights loaded into memory"
        );

        let model = match config.architecture.as_str() {
            "LlamaForCausalLM" | "MistralForCausalLM" | "DeepseekForCausalLM" => {
                LlamaForCausalLM::from_weights(config, weight_map)?
            }
            arch => anyhow::bail!("unsupported architecture: {arch}"),
        };

        Ok(Self { model: Box::new(model), device })
    }

    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    pub fn quantized_layer_count(&self) -> usize {
        self.model.quantized_layer_count()
    }

    pub fn device_description(&self) -> String {
        format!("{:?}", self.device)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Run greedy generation on a tokenized prompt.
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        self.model.generate(prompt, max_tokens)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(
        &self,
        input_ids: &candle_core::Tensor,
        positions: &[usize],
        kv_cache: &mut [Option<(candle_core::Tensor, candle_core::Tensor)>],
    ) -> Result<candle_core::Tensor> {
        self.model.forward(input_ids, positions, kv_cache)
    }

    /// Paged forward pass using global GPU KV cache and PagedAttention kernels.
    pub fn forward_paged(
        &self,
        input_ids: &candle_core::Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<candle_core::Tensor> {
        self.model.forward_paged(input_ids, positions, gpu_kv_cache, attn_meta)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_runner_placeholder() {
        // ModelRunner requires actual model files to instantiate.
        // Integration tests with real models are in the examples directory.
    }
}
