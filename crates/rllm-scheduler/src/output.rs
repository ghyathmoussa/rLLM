use std::collections::HashMap;

use rllm_core::ids::{BlockId, RequestId};

/// Per-request block table snapshot and scheduling info produced by a scheduler step.
#[derive(Debug, Clone)]
pub struct SchedulerOutput {
    /// Requests scheduled for the first time (prefill).
    pub scheduled_new: Vec<RequestId>,
    /// Requests scheduled with prefix cache hits.
    pub scheduled_cached: Vec<RequestId>,
    /// Requests continuing from a previous step (decode).
    pub scheduled_running: Vec<RequestId>,
    /// Number of tokens scheduled per request this step.
    pub num_scheduled_tokens: HashMap<RequestId, usize>,
    /// Per-request block tables (logical block IDs).
    pub block_tables: HashMap<RequestId, Vec<BlockId>>,
    /// Total token budget consumed this step.
    pub token_budget_used: usize,
    /// Requests preempted this step (moved from running back to waiting).
    pub preempted: Vec<RequestId>,
    /// Requests that finished this step.
    pub finished: Vec<RequestId>,
    /// Scheduler statistics snapshot.
    pub stats: SchedulerStats,
}

impl SchedulerOutput {
    pub fn empty() -> Self {
        Self {
            scheduled_new: Vec::new(),
            scheduled_cached: Vec::new(),
            scheduled_running: Vec::new(),
            num_scheduled_tokens: HashMap::new(),
            block_tables: HashMap::new(),
            token_budget_used: 0,
            preempted: Vec::new(),
            finished: Vec::new(),
            stats: SchedulerStats::default(),
        }
    }

    /// Total number of scheduled requests (new + cached + running).
    pub fn num_scheduled(&self) -> usize {
        self.scheduled_new.len() + self.scheduled_cached.len() + self.scheduled_running.len()
    }
}

/// Scheduler statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Number of requests currently in the waiting queue.
    pub num_waiting: usize,
    /// Number of requests currently running.
    pub num_running: usize,
    /// Number of requests finished (cumulative this step).
    pub num_finished: usize,
    /// Total preemptions across all steps.
    pub total_preemptions: usize,
    /// Prefill tokens scheduled this step.
    pub prefill_tokens: usize,
    /// Decode tokens scheduled this step.
    pub decode_tokens: usize,
}
