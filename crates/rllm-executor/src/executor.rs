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
    pub sampled_token_ids: Vec<u32>,
    /// Per-request logprobs, parallel array with sampled_token_ids.
    pub logprobs: Vec<Option<f32>>,
}
