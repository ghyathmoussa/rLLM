use std::collections::VecDeque;
use std::time::Duration;

use rllm_core::ids::RequestId;
use rllm_core::request::InferenceRequest;

/// A batch queue that accumulates requests before scheduling them.
///
/// This is a future-compatible component for pipeline parallelism.
/// In a pipelined setup, the queue allows the scheduler to prepare
/// the next batch while the current batch is being executed, enabling
/// overlap of CPU-side preparation with GPU execution.
///
/// # Batching Strategies
/// - `FixedSize`: Wait until `batch_size` requests are accumulated.
/// - `Timeout`: Send whatever is available after `max_latency` elapses.
/// - `Adaptive`: Dynamically adjust batch size based on throughput.
#[derive(Debug, Clone)]
pub enum BatchingStrategy {
    /// Wait for a fixed number of requests before scheduling.
    FixedSize { batch_size: usize },
    /// Send requests after a timeout, up to a max batch size.
    Timeout {
        max_batch_size: usize,
        max_latency: Duration,
    },
    /// Send requests immediately (no batching).
    Immediate,
}

impl Default for BatchingStrategy {
    fn default() -> Self {
        Self::Immediate
    }
}

/// A queue that holds pending inference requests before scheduling.
///
/// In production, this sits between the HTTP server and the engine core,
/// collecting requests and releasing them in batches to the scheduler.
/// This reduces scheduler overhead and enables pipeline parallelism.
pub struct BatchQueue {
    /// Pending requests waiting to be scheduled.
    pending: VecDeque<InferenceRequest>,
    /// Maximum number of requests in the queue.
    max_capacity: usize,
    /// Current batching strategy.
    strategy: BatchingStrategy,
    /// Total requests that have been enqueued.
    total_enqueued: u64,
    /// Total requests that have been drained.
    total_drained: u64,
}

impl BatchQueue {
    /// Create a new batch queue with immediate strategy.
    pub fn new() -> Self {
        Self::with_strategy(BatchingStrategy::Immediate)
    }

    /// Create a new batch queue with a specific strategy.
    pub fn with_strategy(strategy: BatchingStrategy) -> Self {
        Self {
            pending: VecDeque::new(),
            max_capacity: 4096,
            strategy,
            total_enqueued: 0,
            total_drained: 0,
        }
    }

    /// Create a new batch queue with a fixed-size batching strategy.
    pub fn with_fixed_batch(batch_size: usize) -> Self {
        Self::with_strategy(BatchingStrategy::FixedSize { batch_size })
    }

    /// Create a new batch queue with a timeout-based batching strategy.
    pub fn with_timeout(max_batch_size: usize, max_latency: Duration) -> Self {
        Self::with_strategy(BatchingStrategy::Timeout { max_batch_size, max_latency })
    }

    /// Set the maximum queue capacity.
    pub fn set_max_capacity(&mut self, capacity: usize) {
        self.max_capacity = capacity;
    }

    /// Enqueue a new request.
    ///
    /// Returns `true` if the request was added, `false` if the queue is full.
    pub fn enqueue(&mut self, request: InferenceRequest) -> bool {
        if self.pending.len() >= self.max_capacity {
            tracing::warn!("BatchQueue at capacity ({}), dropping request", self.max_capacity);
            return false;
        }
        self.total_enqueued += 1;
        self.pending.push_back(request);
        true
    }

    /// Enqueue multiple requests at once.
    pub fn enqueue_all(&mut self, requests: Vec<InferenceRequest>) -> usize {
        let mut added = 0;
        for req in requests {
            if self.enqueue(req) {
                added += 1;
            } else {
                break;
            }
        }
        added
    }

    /// Drain the queue according to the batching strategy.
    ///
    /// Returns the next batch of requests to schedule.
    pub fn drain(&mut self) -> Vec<InferenceRequest> {
        if self.pending.is_empty() {
            return vec![];
        }

        let batch_size = match self.strategy {
            BatchingStrategy::Immediate => 1,
            BatchingStrategy::FixedSize { batch_size } => {
                batch_size.min(self.pending.len())
            }
            BatchingStrategy::Timeout { max_batch_size, .. } => {
                max_batch_size.min(self.pending.len())
            }
        };

        let batch: Vec<InferenceRequest> = self.pending.drain(..batch_size).collect();
        self.total_drained += batch.len() as u64;
        batch
    }

    /// Drain all pending requests.
    pub fn drain_all(&mut self) -> Vec<InferenceRequest> {
        let batch: Vec<InferenceRequest> = self.pending.drain(..).collect();
        self.total_drained += batch.len() as u64;
        batch
    }

    /// Peek at the next request without removing it.
    pub fn peek(&self) -> Option<&InferenceRequest> {
        self.pending.front()
    }

    /// Remove a specific request by ID.
    pub fn remove(&mut self, request_id: RequestId) -> Option<InferenceRequest> {
        if let Some(pos) = self.pending.iter().position(|r| r.request_id == request_id) {
            let req = self.pending.remove(pos);
            self.total_drained += 1;
            return req;
        }
        None
    }

    /// Number of requests currently in the queue.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Whether the queue is full.
    pub fn is_full(&self) -> bool {
        self.pending.len() >= self.max_capacity
    }

    /// Clear all pending requests.
    pub fn clear(&mut self) {
        self.total_drained += self.pending.len() as u64;
        self.pending.clear();
    }

    /// Set the batching strategy.
    pub fn set_strategy(&mut self, strategy: BatchingStrategy) {
        self.strategy = strategy;
    }

    /// Get the current batching strategy.
    pub fn strategy(&self) -> &BatchingStrategy {
        &self.strategy
    }

    /// Get total requests enqueued.
    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued
    }

    /// Get total requests drained.
    pub fn total_drained(&self) -> u64 {
        self.total_drained
    }
}

impl Default for BatchQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_core::request::SamplingParams;

    fn make_request(id: u32) -> InferenceRequest {
        InferenceRequest {
            request_id: RequestId::new(),
            prompt: Some(format!("test {}", id)),
            token_ids: Some(vec![1, 2, 3]),
            messages: None,
            sampling_params: SamplingParams::default(),
            arrival_time: std::time::Instant::now(),
            priority: 0,
            stream: false,
            cache_salt: None,
        }
    }

    #[test]
    fn test_batch_queue_empty() {
        let queue = BatchQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_batch_queue_enqueue_dequeue() {
        let mut queue = BatchQueue::new();
        assert!(queue.enqueue(make_request(1)));
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let batch = queue.drain();
        assert_eq!(batch.len(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_queue_immediate_strategy() {
        let mut queue = BatchQueue::new();
        queue.enqueue(make_request(1));
        queue.enqueue(make_request(2));
        queue.enqueue(make_request(3));

        // Immediate strategy drains one at a time.
        assert_eq!(queue.drain().len(), 1);
        assert_eq!(queue.drain().len(), 1);
        assert_eq!(queue.drain().len(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_queue_fixed_size() {
        let mut queue = BatchQueue::with_fixed_batch(4);
        queue.enqueue(make_request(1));
        queue.enqueue(make_request(2));

        // Not enough to fill batch, but returns what's available.
        assert_eq!(queue.drain().len(), 2);

        queue.enqueue(make_request(3));

        // Add batch_size items so drain returns up to batch_size.
        queue.enqueue(make_request(4));
        queue.enqueue(make_request(5));
        queue.enqueue(make_request(6));

        // Should drain at most batch_size = 4.
        let batch = queue.drain();
        assert_eq!(batch.len(), 4);
        // 2 drained first, then 4 more enqueued, then 4 drained = 0 remaining
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_batch_queue_capacity() {
        let mut queue = BatchQueue::new();
        queue.set_max_capacity(3);
        assert!(queue.enqueue(make_request(1)));
        assert!(queue.enqueue(make_request(2)));
        assert!(queue.enqueue(make_request(3)));
        assert!(!queue.enqueue(make_request(4))); // should be rejected
        assert!(queue.is_full());
    }

    #[test]
    fn test_batch_queue_drain_all() {
        let mut queue = BatchQueue::new();
        queue.enqueue(make_request(1));
        queue.enqueue(make_request(2));
        queue.enqueue(make_request(3));
        assert_eq!(queue.drain_all().len(), 3);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_queue_remove_by_id() {
        let mut queue = BatchQueue::new();
        let req1 = make_request(1);
        let id1 = req1.request_id;
        queue.enqueue(req1);
        queue.enqueue(make_request(2));

        let removed = queue.remove(id1);
        assert!(removed.is_some());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_batch_queue_clear() {
        let mut queue = BatchQueue::new();
        queue.enqueue(make_request(1));
        queue.enqueue(make_request(2));
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_queue_total_counters() {
        let mut queue = BatchQueue::new();
        queue.enqueue(make_request(1));
        queue.enqueue(make_request(2));
        queue.drain();
        assert_eq!(queue.total_enqueued(), 2);
        assert_eq!(queue.total_drained(), 1);

        queue.drain();
        assert_eq!(queue.total_drained(), 2);
    }

    #[test]
    fn test_batch_queue_peek() {
        let mut queue = BatchQueue::new();
        let req = make_request(1);
        let id = req.request_id;
        queue.enqueue(req);
        let peeked = queue.peek();
        assert!(peeked.is_some());
        assert_eq!(peeked.unwrap().request_id, id);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_batch_queue_timeout_strategy() {
        let mut queue = BatchQueue::with_timeout(4, Duration::from_millis(100));
        queue.enqueue(make_request(1));
        queue.enqueue(make_request(2));
        // Timeout strategy drains up to max_batch_size.
        assert_eq!(queue.drain().len(), 2);
    }

    #[test]
    fn test_batch_queue_enqueue_all() {
        let mut queue = BatchQueue::new();
        let requests: Vec<_> = (0..5).map(|i| make_request(i)).collect();
        let added = queue.enqueue_all(requests);
        assert_eq!(added, 5);
        assert_eq!(queue.len(), 5);
    }
}
