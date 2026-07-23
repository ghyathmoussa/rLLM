use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::{ids::RequestId, request::SamplingParams};
use rllm_scheduler::SchedulerOutput;

pub trait Executor: Send + Sync {
    /// Initialize the device, load weights, and allocate the KV cache.
    ///
    /// The requested `kv_cache_configs[*].num_blocks` is treated as an upper
    /// bound; the executor profiles real GPU memory and may allocate fewer
    /// blocks to fit. Returns the **actual** number of GPU blocks allocated,
    /// which the caller must use to size the scheduler's block manager.
    fn initialize(
        &mut self,
        kv_cache_configs: &[KVCacheConfig],
        gpu_memory_utilization: f32,
    ) -> Result<usize>;
    fn determine_available_memory(&self) -> Result<usize>;
    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput>;
    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    );
    fn shutdown(&mut self);
}

#[derive(Debug)]
pub struct ExecutorOutput {
    /// Per-request outputs. In the base case (no speculative decoding),
    /// each entry contains exactly 1 token. With speculative decoding,
    /// a single forward pass can produce multiple accepted tokens.
    pub per_request_outputs: Vec<PerRequestOutput>,
}

impl ExecutorOutput {
    pub fn empty() -> Self {
        Self { per_request_outputs: Vec::new() }
    }

    /// Returns true when no outputs were produced.
    pub fn is_empty(&self) -> bool {
        self.per_request_outputs.is_empty()
    }

    /// Total number of output tokens across all requests.
    pub fn num_tokens(&self) -> usize {
        self.per_request_outputs.iter().map(|o| o.token_ids.len()).sum()
    }

    /// Convenience: flat list of all sampled token IDs.
    pub fn all_token_ids(&self) -> Vec<u32> {
        self.per_request_outputs.iter().flat_map(|o| o.token_ids.iter().copied()).collect()
    }

    /// Convenience: flat list of all logprobs.
    pub fn all_logprobs(&self) -> Vec<Option<f32>> {
        self.per_request_outputs.iter().flat_map(|o| o.logprobs.iter().copied()).collect()
    }
}

#[derive(Debug)]
pub struct PerRequestOutput {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub logprobs: Vec<Option<f32>>,
}
