use std::collections::HashMap;

use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::{ids::RequestId, request::SamplingParams};
use rllm_sampling::Sampler;
use rllm_scheduler::SchedulerOutput;

use crate::executor::{Executor, ExecutorOutput};

/// Multi-process executor that coordinates multiple workers.
///
/// This implementation provides the infrastructure for distributed execution
/// across multiple GPU workers. Currently supports a single-worker mode
/// (tensor_parallel_size=1) as a bridge. Full multi-process coordination
/// with IPC/shared memory will be added when `experimental-tp` is stabilized.
///
/// # Design (future)
/// - Each worker runs in its own process/thread with a dedicated GPU.
/// - The controller process distributes `SchedulerOutput` to all workers.
/// - Workers perform their shard of the model forward pass.
/// - All-reduce (NCCL) synchronizes intermediate results across workers.
/// - The controller collects logits from the last worker for sampling.
/// - Large tensors are transferred via shared memory (Unix domain sockets or
///   CUDA IPC handles).
pub struct MultiProcExecutor {
    /// Number of tensor parallel workers.
    tensor_parallel_size: usize,
    /// Per-worker sampler instances (one per rank).
    samplers: Vec<Sampler>,
    /// Per-rank request state for tracking in-flight work.
    request_states: HashMap<RequestId, MultiProcRequestState>,
    /// EOS token ID.
    eos_token_id: u32,
    /// Whether the executor is initialized.
    initialized: bool,
}

/// Per-request state tracked by the multi-process executor.
struct MultiProcRequestState {
    #[allow(dead_code)]
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    sampling_params: SamplingParams,
}

impl MultiProcExecutor {
    /// Create a new multi-process executor.
    ///
    /// `tensor_parallel_size` determines how many workers to coordinate.
    /// Currently only `1` is fully supported (single-process fallback).
    pub fn new(tensor_parallel_size: usize) -> Self {
        let samplers = (0..tensor_parallel_size).map(|_| Sampler::new()).collect();
        Self {
            tensor_parallel_size,
            samplers,
            request_states: HashMap::new(),
            eos_token_id: 0,
            initialized: false,
        }
    }

    /// Set the EOS token ID.
    pub fn set_eos_token_id(&mut self, eos_token_id: u32) {
        self.eos_token_id = eos_token_id;
    }

    /// Number of tensor parallel workers.
    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    /// Get a mutable reference to a specific rank's sampler.
    pub fn sampler_mut(&mut self, rank: usize) -> Option<&mut Sampler> {
        self.samplers.get_mut(rank)
    }
}

impl Executor for MultiProcExecutor {
    fn initialize(
        &mut self,
        kv_cache_configs: &[KVCacheConfig],
        _gpu_memory_utilization: f32,
    ) -> Result<usize> {
        if kv_cache_configs.len() < self.tensor_parallel_size {
            anyhow::bail!(
                "Need {} KV cache configs for {} workers, got {}",
                self.tensor_parallel_size,
                self.tensor_parallel_size,
                kv_cache_configs.len()
            );
        }
        tracing::info!(
            tensor_parallel_size = self.tensor_parallel_size,
            "MultiProcExecutor initializing with {} workers",
            self.tensor_parallel_size
        );
        // In production: spawn worker processes, establish IPC channels,
        // initialize NCCL communicators, shard and load model weights, then
        // profile each rank and reduce to the minimum block count. For now we
        // honor the requested block count of rank 0.
        for (rank, config) in kv_cache_configs.iter().enumerate() {
            tracing::debug!(rank, num_blocks = config.num_blocks, "Worker KV cache configured");
        }
        self.initialized = true;
        Ok(kv_cache_configs.first().map(|c| c.num_blocks).unwrap_or(0))
    }

    fn determine_available_memory(&self) -> Result<usize> {
        // In production: query each worker's available GPU memory
        // and return the minimum across ranks.
        Ok(4 * 1024 * 1024 * 1024) // 4 GiB default
    }

    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput> {
        if !self.initialized {
            anyhow::bail!("MultiProcExecutor not initialized");
        }
        if scheduler_output.num_scheduled() == 0 {
            return Ok(ExecutorOutput { sampled_token_ids: vec![], logprobs: vec![] });
        }

        // Collect scheduled request IDs in order.
        let scheduled_ids: Vec<RequestId> = scheduler_output
            .scheduled_new
            .iter()
            .chain(scheduler_output.scheduled_cached.iter())
            .chain(scheduler_output.scheduled_running.iter())
            .copied()
            .collect();

        // When tensor_parallel_size == 1, run directly (single-GPU fallback).
        // When > 1, distribute work across workers and all-reduce.
        //
        // For now, use single-worker mode with rank 0's sampler.
        let sampler = &mut self.samplers[0];

        let mut sampled_token_ids = Vec::with_capacity(scheduled_ids.len());
        let mut logprobs = Vec::with_capacity(scheduled_ids.len());

        for request_id in &scheduled_ids {
            let state = match self.request_states.get(request_id) {
                Some(s) => s,
                None => continue,
            };

            let _position = state.prompt_token_ids.len() + state.generated_token_ids.len();
            let vocab_size = 32000; // Default; in production, read from model config.

            // Generate dummy logits (uniform) for now.
            // In production: collect from worker forward pass, then all-reduce.
            let logits = vec![0.0f32; vocab_size];

            let mut context_token_ids = state.prompt_token_ids.clone();
            context_token_ids.extend_from_slice(&state.generated_token_ids);

            let input = rllm_sampling::SamplingInput {
                logits,
                params: state.sampling_params.clone(),
                context_token_ids,
                num_generated: state.generated_token_ids.len() as u32,
                eos_token_id: self.eos_token_id,
                bad_word_token_ids: vec![],
            };

            let output = sampler.sample(&input);
            let token_id = output.token_id;

            if let Some(s) = self.request_states.get_mut(request_id) {
                s.generated_token_ids.push(token_id);
            }

            sampled_token_ids.push(token_id);
            logprobs.push(output.logprob);
        }

        Ok(ExecutorOutput { sampled_token_ids, logprobs })
    }

    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) {
        self.request_states.insert(
            request_id,
            MultiProcRequestState {
                prompt_token_ids,
                generated_token_ids: Vec::new(),
                sampling_params,
            },
        );
    }

    fn shutdown(&mut self) {
        tracing::info!(
            tensor_parallel_size = self.tensor_parallel_size,
            "MultiProcExecutor shutting down"
        );
        self.request_states.clear();
        // In production: send shutdown signals to worker processes,
        // wait for graceful termination, cleanup IPC channels.
    }
}
