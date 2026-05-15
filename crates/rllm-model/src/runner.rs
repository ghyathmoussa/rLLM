#[cfg(feature = "candle-backend")]
use anyhow::{Context, Result};
#[cfg(feature = "candle-backend")]
use candle_core::Device;
#[cfg(feature = "candle-backend")]
use rllm_core::config::ModelConfig;

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
}

#[cfg(feature = "candle-backend")]
impl ModelRunner {
    /// Load a model from a local directory.
    pub fn from_dir(model_dir: &str) -> Result<Self> {
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let config = hf_config::parse_hf_config(&config_path)
            .context("parsing model config")?;

        let device = Device::cuda_if_available(0)
            .map_err(|e| anyhow::anyhow!("device init: {e}"))?;
        tracing::info!("loading model on device: {:?}", device);

        let (weight_map, _tied) = loader::load_weights_with_tied_detection(
            std::path::Path::new(model_dir),
            &device,
        )
        .context("loading weights")?;

        let model = match config.architecture.as_str() {
            "LlamaForCausalLM" | "MistralForCausalLM" => {
                LlamaForCausalLM::from_weights(config, weight_map)?
            }
            arch => anyhow::bail!("unsupported architecture: {arch}"),
        };

        Ok(Self {
            model: Box::new(model),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    /// Run greedy generation on a tokenized prompt.
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        self.model.generate(prompt, max_tokens)
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
