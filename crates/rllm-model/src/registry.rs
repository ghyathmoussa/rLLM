use anyhow::Result;
use std::collections::HashMap;

use rllm_core::config::ModelConfig;

pub trait Model: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn forward(&self, token_ids: &[u32], positions: &[usize]) -> Result<()>;
}

pub trait CausalLM: Model {
    fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>>;
}

pub type ModelFactory = fn(&ModelConfig) -> Result<Box<dyn CausalLM>>;

pub struct ModelRegistry {
    factories: HashMap<String, ModelFactory>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
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
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
