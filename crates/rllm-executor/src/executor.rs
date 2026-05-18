use anyhow::Result;

use rllm_cache::spec::KVCacheConfig;
use rllm_core::ids::RequestId;
use rllm_core::request::SamplingParams;
use rllm_scheduler::SchedulerOutput;

pub trait Executor: Send + Sync {
    fn initialize(&mut self, kv_cache_configs: &[KVCacheConfig]) -> Result<()>;
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
