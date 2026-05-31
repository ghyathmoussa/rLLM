use anyhow::Result;
use rllm_cache::spec::{KVCacheConfig, KVCacheSpec};
use rllm_core::config::ModelConfig;
use rllm_kernels::cache_ops::GpuKVCache;
#[cfg(feature = "candle-backend")]
use rllm_model::ModelRunner as CandleModelRunner;
use rllm_sampling::Sampler;
use rllm_tensor::Device;

use crate::model_runner::ModelRunner;

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
    /// Global GPU KV cache for PagedAttention.
    /// Allocated during `initialize_kv_cache()` when CUDA is available.
    gpu_kv_cache: Option<GpuKVCache>,
    /// CUDA graph replay/capture manager.
    pub cuda_graphs: crate::model_runner::CudaGraphCapture,
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
            gpu_kv_cache: None,
            cuda_graphs: crate::model_runner::CudaGraphCapture::new(),
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
            let loaded = CandleModelRunner::from_model_ref_with_config(
                &self.model_config.model_id,
                self.model_config.clone(),
            )?;
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

    /// Execute one stateful forward step on the model for a single request.
    pub fn execute_model_step(
        &mut self,
        request_id: &rllm_core::ids::RequestId,
        input_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Option<candle_core::Tensor>> {
        #[cfg(feature = "candle-backend")]
        {
            let Some(model) = &self.candle_model else {
                return Ok(None);
            };
            let kv_cache = self
                .model_runner
                .get_kv_cache_mut(request_id)
                .ok_or_else(|| anyhow::anyhow!("request state not found for {:?}", request_id))?;
            let device = model.device();
            let input_ids =
                candle_core::Tensor::new(input_tokens, device)?.reshape((1, input_tokens.len()))?;
            let logits = model.forward(&input_ids, positions, kv_cache)?;
            Ok(Some(logits))
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            let _ = (request_id, input_tokens, positions);
            Ok(None)
        }
    }

    /// Determine free GPU memory (bytes) available right now.
    ///
    /// With the `cuda` feature this queries the driver (`cuMemGetInfo`) against
    /// the current context — call it *after* `load_model_weights` so the model
    /// weights are already resident and excluded from the free figure. Falls
    /// back to a fixed estimate if the query fails or CUDA is unavailable.
    /// Returns 0 for CPU.
    pub fn determine_available_memory(&self) -> Result<usize> {
        match &self.device {
            Device::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    match cudarc::driver::result::mem_get_info() {
                        Ok((free, total)) => {
                            tracing::info!(
                                worker_id = self.id,
                                free_bytes = free,
                                total_bytes = total,
                                "Queried GPU memory"
                            );
                            Ok(free)
                        }
                        Err(e) => {
                            tracing::warn!(
                                worker_id = self.id,
                                error = %e,
                                fallback_bytes = DEFAULT_GPU_MEMORY,
                                "cuMemGetInfo failed; using default GPU memory estimate"
                            );
                            Ok(DEFAULT_GPU_MEMORY)
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    tracing::info!(
                        worker_id = self.id,
                        bytes = DEFAULT_GPU_MEMORY,
                        "Returning default GPU memory estimate (cuda feature disabled)"
                    );
                    Ok(DEFAULT_GPU_MEMORY)
                }
            }
            Device::Cpu => Ok(0),
        }
    }

    /// Compute how many KV-cache blocks fit in available GPU memory.
    ///
    /// vLLM-style profiling: weights are already resident, so `free` excludes
    /// them. We keep the process under `gpu_memory_utilization * total`, reserve
    /// what is already used (weights + context), apply a safety margin for
    /// activations/fragmentation, then divide by the per-block byte cost. The
    /// result is capped at `max_blocks` (the worst-case need, so we never
    /// allocate more than the model could ever address).
    ///
    /// Call **after** `load_model_weights`. Errors if not even a minimal cache
    /// fits, with guidance on how to proceed.
    pub fn fit_kv_blocks(
        &self,
        spec: &KVCacheSpec,
        gpu_memory_utilization: f32,
        max_blocks: usize,
    ) -> Result<usize> {
        // Per-block cost in bytes (K + V, all layers).
        let bytes_per_block = spec.bytes_per_block();
        if bytes_per_block == 0 {
            anyhow::bail!("KV cache spec has zero bytes per block");
        }

        // Safety margin held back for activation peaks and allocator fragmentation.
        const KV_SAFETY_FRACTION: f64 = 0.95;
        // Always keep at least this many blocks or refuse to start.
        const MIN_GPU_BLOCKS: usize = 256;

        let free = self.determine_available_memory()?;

        // When we can read the total too, bound the whole process to
        // `utilization * total`; otherwise fall back to `utilization * free`.
        #[cfg(feature = "cuda")]
        let (free, total) = match &self.device {
            Device::Cuda { .. } => cudarc::driver::result::mem_get_info().unwrap_or((free, free)),
            Device::Cpu => (free, free),
        };
        #[cfg(not(feature = "cuda"))]
        let total = free;

        let util = gpu_memory_utilization.clamp(0.01, 1.0) as f64;
        let budget = (total as f64 * util) as usize; // total process cap
        let already_used = total.saturating_sub(free); // weights + context
        let kv_budget = budget.saturating_sub(already_used);
        let kv_budget = (kv_budget as f64 * KV_SAFETY_FRACTION) as usize;

        let mem_blocks = kv_budget / bytes_per_block;
        let num_blocks = mem_blocks.min(max_blocks);

        tracing::info!(
            worker_id = self.id,
            free_bytes = free,
            total_bytes = total,
            gpu_memory_utilization = util,
            bytes_per_block,
            mem_blocks,
            max_blocks,
            chosen_blocks = num_blocks,
            "Profiled KV cache size"
        );

        if num_blocks < MIN_GPU_BLOCKS {
            anyhow::bail!(
                "insufficient GPU memory for KV cache: only {num_blocks} block(s) fit \
                 (need >= {MIN_GPU_BLOCKS}). Free={free} bytes, total={total} bytes, \
                 bytes/block={bytes_per_block}. Try a smaller model, a shorter \
                 --max-model-len, or a higher --gpu-memory-utilization."
            );
        }

        Ok(num_blocks)
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

        // Allocate physical GPU KV cache when CUDA is available.
        #[cfg(has_cuda)]
        {
            let cache = GpuKVCache::new(
                config.num_blocks,
                config.spec.num_layers,
                config.spec.num_kv_heads,
                config.spec.head_dim,
                config.spec.block_size,
                config.spec.dtype,
            )
            .map_err(|e| anyhow::anyhow!("GPU KV cache allocation failed: {e}"))?;
            tracing::info!(
                worker_id = self.id,
                gpu_cache_bytes = cache.total_bytes(),
                num_layers = cache.num_layers(),
                "GPU KV cache allocated"
            );
            self.gpu_kv_cache = Some(cache);
        }

        self.cache_config = Some(config.clone());
        Ok(())
    }

    /// Warm up kernels with dummy batches of common sizes and capture CUDA graphs.
    pub fn warm_up(&mut self) -> Result<()> {
        // CUDA graph capture is not yet implemented (it depends on the paged
        // forward kernel path, which is also pending), so warmup does not capture
        // graphs. Decode runs on the eager forward path.
        tracing::info!(worker_id = self.id, "Warmup completed (CUDA graph capture disabled)");
        Ok(())
    }

    /// Execute a dummy decode step for CUDA graph capture.
    #[cfg(feature = "candle-backend")]
    pub fn execute_dummy_decode(&self, batch_size: usize) -> Result<()> {
        let device =
            self.worker_model_device().ok_or_else(|| anyhow::anyhow!("model device not found"))?;

        // 1. Create dummy input tensors
        let token_ids = vec![0u32; batch_size];
        let input_ids =
            candle_core::Tensor::new(&token_ids[..], device)?.reshape((1, batch_size))?;

        let positions = vec![0usize; batch_size];

        // 2. Build dummy AttentionMetadata for decode
        let seq_lens = vec![1u32; batch_size];
        // Map each sequence to physical block 0
        let block_tables = vec![vec![0i32]; batch_size];
        let mut attn_meta = rllm_kernels::AttentionMetadata::for_decode(
            seq_lens,
            block_tables,
            1, // max_num_blocks_per_seq
        );

        // Map tokens to block 0 offsets
        let slot_mapping: Vec<i64> =
            (0..batch_size).map(|i| (i % self.block_size) as i64).collect();
        attn_meta.slot_mapping = slot_mapping;

        // 3. Run forward pass
        let _logits = self.forward_paged_batch(&input_ids, &positions, &attn_meta)?;
        Ok(())
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn execute_dummy_decode(&self, _batch_size: usize) -> Result<()> {
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

    /// Get a reference to the GPU KV cache, if allocated.
    pub fn gpu_kv_cache(&self) -> Option<&GpuKVCache> {
        self.gpu_kv_cache.as_ref()
    }

    /// Get the Candle device from the loaded model.
    #[cfg(feature = "candle-backend")]
    pub fn worker_model_device(&self) -> Option<&candle_core::Device> {
        self.candle_model.as_ref().map(|m| m.device())
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn worker_model_device(&self) -> Option<&()> {
        None
    }

    /// Execute a batched paged forward pass using the global GPU KV cache.
    ///
    /// This is the high-performance path: all requests in the batch share a
    /// single kernel launch through `model.forward_paged()`, with K/V data
    /// scatter-written to and read from the block-addressed `GpuKVCache`.
    #[cfg(feature = "candle-backend")]
    pub fn forward_paged_batch(
        &self,
        input_ids: &candle_core::Tensor,
        positions: &[usize],
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<candle_core::Tensor> {
        let gpu_cache = self
            .gpu_kv_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU KV cache not initialized"))?;
        let model =
            self.candle_model.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;

        let logits = model.forward_paged(input_ids, positions, gpu_cache, attn_meta)?;
        Ok(logits)
    }
}
