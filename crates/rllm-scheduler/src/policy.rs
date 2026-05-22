use indexmap::IndexMap;
use rllm_core::{config::SchedulingPolicy, ids::RequestId};

/// Metadata tracked for each queued request.
#[derive(Debug, Clone)]
struct QueueEntry {
    priority: i32,
    arrival_time: std::time::Instant,
}

/// Ordered request queue supporting both FCFS and priority scheduling.
///
/// For FCFS, requests are served in arrival order (natural `IndexMap` ordering).
/// For Priority, requests are sorted by priority (descending) then arrival time.
#[derive(Debug)]
pub struct RequestQueue {
    entries: IndexMap<RequestId, QueueEntry>,
    policy: SchedulingPolicy,
}

impl RequestQueue {
    pub fn new(policy: SchedulingPolicy) -> Self {
        Self { entries: IndexMap::new(), policy }
    }

    /// Push a request into the queue.
    ///
    /// For FCFS, appends at the end (insertion order = arrival order).
    /// For Priority, inserts in sorted position.
    pub fn push(&mut self, request_id: RequestId, priority: i32, arrival_time: std::time::Instant) {
        let entry = QueueEntry { priority, arrival_time };
        if self.policy == SchedulingPolicy::FCFS {
            self.entries.insert(request_id, entry);
        } else {
            // For priority: insert then sort
            self.entries.insert(request_id, entry);
            self.sort_by_priority();
        }
    }

    /// Pop the next request to schedule.
    ///
    /// FCFS: front of queue (earliest arrival).
    /// Priority: front of queue (highest priority, earliest arrival among ties).
    pub fn pop(&mut self) -> Option<RequestId> {
        self.entries.shift_remove_index(0).map(|(id, _)| id)
    }

    /// Peek at the front of the queue without removing.
    pub fn peek(&self) -> Option<RequestId> {
        self.entries.get_index(0).map(|(id, _)| *id)
    }

    /// Remove a specific request from the queue.
    pub fn remove(&mut self, request_id: RequestId) -> bool {
        self.entries.shift_remove(&request_id).is_some()
    }

    /// Pop a specific request while preserving the order of the remaining queue.
    pub fn pop_request(&mut self, request_id: RequestId) -> Option<RequestId> {
        self.entries.shift_remove(&request_id).map(|_| request_id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over request IDs in queue order.
    pub fn iter(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.entries.keys().copied()
    }

    /// Check if the queue contains a request.
    pub fn contains(&self, request_id: RequestId) -> bool {
        self.entries.contains_key(&request_id)
    }

    /// Get the priority of a queued request.
    pub fn priority_of(&self, request_id: RequestId) -> Option<i32> {
        self.entries.get(&request_id).map(|e| e.priority)
    }

    /// Sort entries by priority descending, then arrival time ascending.
    fn sort_by_priority(&mut self) {
        let mut entries: Vec<(RequestId, QueueEntry)> = self.entries.drain(..).collect();
        entries.sort_by(|a, b| {
            b.1.priority.cmp(&a.1.priority).then_with(|| a.1.arrival_time.cmp(&b.1.arrival_time))
        });
        self.entries.extend(entries);
    }
}

#[cfg(test)]
mod tests {
    use rllm_core::ids::RequestId;

    use super::*;

    fn instant(offset_ms: u64) -> std::time::Instant {
        std::time::Instant::now() - std::time::Duration::from_millis(1000 - offset_ms)
    }

    #[test]
    fn fcfs_preserves_arrival_order() {
        let mut q = RequestQueue::new(SchedulingPolicy::FCFS);
        let r1 = RequestId::new();
        let r2 = RequestId::new();
        let r3 = RequestId::new();

        q.push(r1, 0, instant(100));
        q.push(r2, 0, instant(200));
        q.push(r3, 0, instant(300));

        assert_eq!(q.pop(), Some(r1));
        assert_eq!(q.pop(), Some(r2));
        assert_eq!(q.pop(), Some(r3));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn priority_orders_by_priority_desc() {
        let mut q = RequestQueue::new(SchedulingPolicy::Priority);
        let r1 = RequestId::new();
        let r2 = RequestId::new();
        let r3 = RequestId::new();

        q.push(r1, 1, instant(100));
        q.push(r2, 5, instant(200));
        q.push(r3, 3, instant(300));

        assert_eq!(q.pop(), Some(r2)); // priority 5
        assert_eq!(q.pop(), Some(r3)); // priority 3
        assert_eq!(q.pop(), Some(r1)); // priority 1
    }

    #[test]
    fn priority_ties_broken_by_arrival() {
        let mut q = RequestQueue::new(SchedulingPolicy::Priority);
        let r1 = RequestId::new();
        let r2 = RequestId::new();

        q.push(r1, 5, instant(100));
        q.push(r2, 5, instant(50)); // earlier arrival

        assert_eq!(q.pop(), Some(r2)); // same priority, earlier arrival
        assert_eq!(q.pop(), Some(r1));
    }

    #[test]
    fn remove_from_middle() {
        let mut q = RequestQueue::new(SchedulingPolicy::FCFS);
        let r1 = RequestId::new();
        let r2 = RequestId::new();
        let r3 = RequestId::new();

        q.push(r1, 0, instant(100));
        q.push(r2, 0, instant(200));
        q.push(r3, 0, instant(300));

        assert!(q.remove(r2));
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop(), Some(r1));
        assert_eq!(q.pop(), Some(r3));
    }

    #[test]
    fn peek_does_not_remove() {
        let mut q = RequestQueue::new(SchedulingPolicy::FCFS);
        let r1 = RequestId::new();
        q.push(r1, 0, instant(100));

        assert_eq!(q.peek(), Some(r1));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn empty_queue_operations() {
        let mut q = RequestQueue::new(SchedulingPolicy::FCFS);
        assert!(q.is_empty());
        assert_eq!(q.pop(), None);
        assert_eq!(q.peek(), None);
        assert!(!q.remove(RequestId::new()));
    }
}
