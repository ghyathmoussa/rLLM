use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::config::ModelConfig;
use rllm_tensor::Device;

use crate::model_runner::ModelRunner;

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
    pub fn initialize_rng_seed(&mut self, _seed: u64) -> Result<()> {
        // Will be wired to the sampler in Phase 10.
        Ok(())
    }

    /// Load model weights from the path specified in `model_config.model_id`.
    ///
    /// Actual weight loading requires the `candle-backend` feature.
    pub fn load_model_weights(&mut self) -> Result<()> {
        tracing::info!(
            worker_id = self.id,
            model = %self.model_config.model_id,
            "Model weight loading requested (requires candle-backend)"
        );
        Ok(())
    }

    /// Determine available GPU memory for KV cache allocation.
    ///
    /// Returns the available bytes after accounting for model weights
    /// and activation peak. Returns 0 if the device is CPU.
    pub fn determine_available_memory(&self) -> Result<usize> {
        match &self.device {
            Device::Cuda { .. } => {
                // TODO: query actual GPU memory via cudarc when available.
                tracing::warn!(worker_id = self.id, "GPU memory query not yet implemented, returning 0");
                Ok(0)
            }
            Device::Cpu => Ok(0),
        }
    }

    /// Initialize the physical KV cache from a config.
    ///
    /// Stores the config for later use by the model runner.
    /// Actual GPU allocation happens when the `cuda` feature is enabled.
    pub fn initialize_kv_cache(&mut self, config: &KVCacheConfig) -> Result<()> {
        tracing::info!(
            worker_id = self.id,
            num_blocks = config.num_blocks,
            block_size = config.spec.block_size,
            "Initializing KV cache"
        );
        self.cache_config = Some(config.clone());
        Ok(())
    }

    /// Warm up kernels with dummy batches of common sizes.
    pub fn warm_up(&mut self) -> Result<()> {
        tracing::info!(worker_id = self.id, "Warmup requested");
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
            }
            2 => {
                self.cache_config = None;
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
}
