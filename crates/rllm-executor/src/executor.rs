use anyhow::Result;

use rllm_cache::spec::KVCacheConfig;
use rllm_scheduler::SchedulerOutput;

pub trait Executor: Send + Sync {
    fn initialize(&mut self, kv_cache_configs: &[KVCacheConfig]) -> Result<()>;
    fn determine_available_memory(&self) -> Result<usize>;
    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput>;
    fn shutdown(&mut self);
}

#[derive(Debug)]
pub struct ExecutorOutput {
    pub sampled_token_ids: Vec<u32>,
}
