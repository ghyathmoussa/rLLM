use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_scheduler::SchedulerOutput;
use rllm_worker::Worker;

use crate::executor::{Executor, ExecutorOutput};

/// Single-process executor that owns one worker in the same process.
///
/// Delegates all calls directly to the worker. In the future, a
/// `MultiProcExecutor` will use IPC to coordinate multiple workers.
pub struct UniProcExecutor {
    worker: Worker,
}

impl UniProcExecutor {
    pub fn new(worker: Worker) -> Self {
        Self { worker }
    }

    /// Get a reference to the underlying worker.
    pub fn worker(&self) -> &Worker {
        &self.worker
    }

    /// Get a mutable reference to the underlying worker.
    pub fn worker_mut(&mut self) -> &mut Worker {
        &mut self.worker
    }
}

impl Executor for UniProcExecutor {
    fn initialize(&mut self, kv_cache_configs: &[KVCacheConfig]) -> Result<()> {
        if let Some(config) = kv_cache_configs.first() {
            self.worker.initialize_kv_cache(config)?;
        }
        Ok(())
    }

    fn determine_available_memory(&self) -> Result<usize> {
        self.worker.determine_available_memory()
    }

    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput> {
        // Build input tensors from scheduler output.
        let _batch = self.worker.model_runner_mut().build_tensors(scheduler_output)?;

        // TODO (Phase 10-11): actual model forward pass + sampling.
        // For now, return empty output. The model runner has updated its
        // internal state (positions, slot mappings, block tables) which
        // will be consumed when the full execution pipeline is wired up.
        Ok(ExecutorOutput {
            sampled_token_ids: vec![],
        })
    }

    fn shutdown(&mut self) {
        tracing::info!(worker_id = self.worker.id, "Executor shutting down");
    }
}
