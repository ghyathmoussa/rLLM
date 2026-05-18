use std::collections::VecDeque;

use rllm_core::ids::RequestId;
use rllm_tensor::{AsyncPinnedCopy, async_copy_token_ids};

use crate::InputBatch;

/// Batch slices that can be submitted to separate prefill/decode streams.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DualBatchPlan {
    pub prefill_request_ids: Vec<RequestId>,
    pub decode_request_ids: Vec<RequestId>,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
}

impl DualBatchPlan {
    pub fn has_overlap(&self) -> bool {
        self.prefill_tokens > 0 && self.decode_tokens > 0
    }
}

/// Builds overlap plans from mixed batches.
///
/// The actual CUDA stream submission lives in the runner/executor layer; this
/// helper keeps the split deterministic and easy to test.
#[derive(Debug, Clone)]
pub struct DualBatchOverlap {
    enabled: bool,
}

impl DualBatchOverlap {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn plan(&self, batch: &InputBatch) -> DualBatchPlan {
        if !self.enabled || batch.num_seqs == 0 {
            return DualBatchPlan::default();
        }

        let mut plan = DualBatchPlan::default();
        for i in 0..batch.num_seqs {
            if batch.is_prefill[i] {
                plan.prefill_request_ids.push(batch.request_ids[i]);
                plan.prefill_tokens += batch.tokens_per_seq[i];
            } else {
                plan.decode_request_ids.push(batch.request_ids[i]);
                plan.decode_tokens += batch.tokens_per_seq[i];
            }
        }
        plan
    }
}

impl Default for DualBatchOverlap {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Output-copy queue that lets token staging complete while the next GPU batch
/// is already being built or launched.
pub struct AsyncTokenOutputQueue {
    pending: VecDeque<AsyncPinnedCopy<u32>>,
}

impl AsyncTokenOutputQueue {
    pub fn new() -> Self {
        Self { pending: VecDeque::new() }
    }

    pub fn submit(&mut self, token_ids: &[u32]) {
        self.pending.push_back(async_copy_token_ids(token_ids));
    }

    pub fn drain_ready(&mut self) -> Vec<Vec<u32>> {
        let mut out = Vec::with_capacity(self.pending.len());
        while let Some(copy) = self.pending.pop_front() {
            out.push(copy.wait());
        }
        out
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Default for AsyncTokenOutputQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_batch_plan_splits_prefill_and_decode() {
        let r1 = RequestId::new();
        let r2 = RequestId::new();
        let mut batch = InputBatch::empty();
        batch.request_ids = vec![r1, r2];
        batch.tokens_per_seq = vec![4, 1];
        batch.is_prefill = vec![true, false];
        batch.num_seqs = 2;

        let plan = DualBatchOverlap::default().plan(&batch);
        assert_eq!(plan.prefill_request_ids, vec![r1]);
        assert_eq!(plan.decode_request_ids, vec![r2]);
        assert_eq!(plan.prefill_tokens, 4);
        assert_eq!(plan.decode_tokens, 1);
        assert!(plan.has_overlap());
    }

    #[test]
    fn async_output_queue_stages_tokens() {
        let mut queue = AsyncTokenOutputQueue::new();
        queue.submit(&[1, 2]);
        queue.submit(&[3]);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.drain_ready(), vec![vec![1, 2], vec![3]]);
        assert!(queue.is_empty());
    }
}
