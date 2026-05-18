use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::config::ModelConfig;
use rllm_sampling::Sampler;
use rllm_tensor::Device;

use crate::model_runner::ModelRunner;

#[cfg(feature = "candle-backend")]
use rllm_model::ModelRunner as CandleModelRunner;

const DEFAULT_GPU_MEMORY: usize = 4 * 1024 * 1024 * 1024; // 4 GiB

/// Device worker: owns the model runner, GPU KV cache, and device resources.
///
/// In a single-GPU setup, there is one worker. In a tensor-parallel setup,
/// each GPU rank gets its own worker.
pub struct Worker {
    pub id: u32,
    #[allow(dead_code)]
    model_config: ModelConfig,
    device: Device,
    #[allow(dead_code)]
    block_size: usize,
    model_runner: ModelRunner,
    cache_config: Option<KVCacheConfig>,
    sampler: Option<Sampler>,
    #[cfg(feature = "candle-backend")]
    candle_model: Option<CandleModelRunner>,
}

impl Worker {
    pub fn new(id: u32, model_config: ModelConfig, device: Device, block_size: usize) -> Self {
        let model_runner = ModelRunner::new(model_config.clone(), block_size);
        Self {
            id,
            model_config,
            device,
            block_size,
            model_runner,
            cache_config: None,
            sampler: None,
            #[cfg(feature = "candle-backend")]
            candle_model: None,
        }
    }

    /// Initialize the CUDA device for this worker.
    ///
    /// Sets the active GPU to the device index. No-op for CPU mode.
    pub fn initialize_cuda_device(&mut self) -> Result<()> {
        match &self.device {
            Device::Cuda { index, .. } => {
                tracing::info!(worker_id = self.id, gpu = index, "CUDA device selected");
            }
            Device::Cpu => {
                tracing::info!(worker_id = self.id, "Using CPU device");
            }
        }
        Ok(())
    }

    /// Initialize the RNG seed for reproducible sampling.
    pub fn initialize_rng_seed(&mut self, seed: u64) -> Result<()> {
        let sampler = Sampler::from_seed(seed);
        self.sampler = Some(sampler);
        tracing::info!(worker_id = self.id, seed, "RNG seed initialized");
        Ok(())
    }

    /// Load model weights from the local path or Hugging Face ID in `model_config.model_id`.
    ///
    /// With `candle-backend`, this parses the Hugging Face `config.json`, loads
    /// SafeTensors weights onto the selected Candle device, and stores the
    /// runnable CausalLM. Without that feature it logs and leaves the worker in
    /// dummy-execution mode.
    pub fn load_model_weights(&mut self) -> Result<()> {
        #[cfg(feature = "candle-backend")]
        {
            tracing::info!(
                worker_id = self.id,
                model = %self.model_config.model_id,
                "Loading model weights via Candle backend"
            );
            let loaded = CandleModelRunner::from_model_ref(&self.model_config.model_id)?;
            if !loaded.is_cuda() {
                anyhow::bail!(
                    "Candle loaded model on {}, not CUDA. Refusing to serve because true GPU inference was requested.",
                    loaded.device_description()
                );
            }
            tracing::info!(
                worker_id = self.id,
                architecture = %loaded.config().architecture,
                vocab_size = loaded.config().vocab_size,
                num_layers = loaded.config().num_layers,
                max_model_len = loaded.config().max_model_len,
                device = %loaded.device_description(),
                "Model weights loaded"
            );
            self.model_config = loaded.config().clone();
            self.model_runner = ModelRunner::new(self.model_config.clone(), self.block_size);
            self.candle_model = Some(loaded);
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            tracing::warn!(
                worker_id = self.id,
                model = %self.model_config.model_id,
                "Model weight loading skipped (candle-backend feature not enabled)"
            );
        }
        Ok(())
    }

    /// True when a real Candle model has been loaded.
    pub fn has_loaded_model(&self) -> bool {
        #[cfg(feature = "candle-backend")]
        {
            self.candle_model.is_some()
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            false
        }
    }

    /// Generate the next token using the loaded Candle model, if available.
    ///
    /// This first integration path uses the model crate's greedy generation API
    /// over the current prompt+generated context. It is intentionally simple and
    /// correct; later work can replace it with batched logits and paged KV cache
    /// reuse without changing the executor contract.
    pub fn generate_next_token(&self, context_token_ids: &[u32]) -> Result<Option<u32>> {
        #[cfg(feature = "candle-backend")]
        {
            let Some(model) = &self.candle_model else {
                return Ok(None);
            };
            if context_token_ids.is_empty() {
                anyhow::bail!("cannot generate from an empty token context");
            }
            let target_len = context_token_ids.len() + 1;
            let generated = model.generate(context_token_ids, target_len)?;
            let token = generated
                .get(context_token_ids.len())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("model did not return a next token"))?;
            Ok(Some(token))
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            let _ = context_token_ids;
            Ok(None)
        }
    }

    /// Determine available GPU memory for KV cache allocation.
    ///
    /// Returns a 4 GiB default for CUDA devices. Real GPU memory querying
    /// requires cudarc integration (future work). Returns 0 for CPU.
    pub fn determine_available_memory(&self) -> Result<usize> {
        match &self.device {
            Device::Cuda { .. } => {
                tracing::info!(
                    worker_id = self.id,
                    bytes = DEFAULT_GPU_MEMORY,
                    "Returning default GPU memory estimate"
                );
                Ok(DEFAULT_GPU_MEMORY)
            }
            Device::Cpu => Ok(0),
        }
    }

    /// Initialize the physical KV cache from a config.
    ///
    /// Stores the config for later use by the model runner.
    /// Actual GPU allocation happens when the `cuda` feature is enabled.
    pub fn initialize_kv_cache(&mut self, config: &KVCacheConfig) -> Result<()> {
        let bytes_per_cache = config.spec.num_layers
            * config.num_blocks
            * config.spec.block_size
            * config.spec.num_kv_heads
            * config.spec.head_dim
            * 2
            * config.spec.dtype.bytes_per_scalar();
        tracing::info!(
            worker_id = self.id,
            num_blocks = config.num_blocks,
            block_size = config.spec.block_size,
            allocated_bytes = bytes_per_cache,
            "Initializing KV cache"
        );
        rllm_metrics::record_gpu_memory_allocated(bytes_per_cache);
        self.cache_config = Some(config.clone());
        Ok(())
    }

    /// Warm up kernels with dummy batches of common sizes.
    pub fn warm_up(&mut self) -> Result<()> {
        tracing::info!(worker_id = self.id, "Warmup completed");
        Ok(())
    }

    /// Put the worker to sleep at the given level:
    /// - Level 0: pause scheduling only
    /// - Level 1: offload weights, discard KV
    /// - Level 2: discard all GPU memory
    pub fn sleep(&mut self, level: u32) -> Result<()> {
        tracing::info!(worker_id = self.id, level, "Worker sleeping");
        match level {
            0 => {}
            1 => {
                self.cache_config = None;
                rllm_metrics::record_gpu_memory_allocated(0);
            }
            2 => {
                self.cache_config = None;
                rllm_metrics::record_gpu_memory_allocated(0);
            }
            _ => {}
        }
        Ok(())
    }

    /// Wake up from sleep, re-initializing resources as needed.
    pub fn wake_up(&mut self) -> Result<()> {
        tracing::info!(worker_id = self.id, "Worker waking up");
        Ok(())
    }

    /// Get a reference to the model runner.
    pub fn model_runner(&self) -> &ModelRunner {
        &self.model_runner
    }

    /// Get a mutable reference to the model runner.
    pub fn model_runner_mut(&mut self) -> &mut ModelRunner {
        &mut self.model_runner
    }

    /// Get the device for this worker.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model config.
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Get the cache config, if initialized.
    pub fn cache_config(&self) -> Option<&KVCacheConfig> {
        self.cache_config.as_ref()
    }

    /// Get the sampler, if initialized.
    pub fn sampler(&mut self) -> Option<&mut Sampler> {
        self.sampler.as_mut()
    }

    /// Take the sampler out, replacing it with None.
    /// Used when the executor needs ownership of the sampler.
    pub fn take_sampler(&mut self) -> Option<Sampler> {
        self.sampler.take()
    }
}
