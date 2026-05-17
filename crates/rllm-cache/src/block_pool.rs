use std::collections::HashMap;

use smallvec::SmallVec;

use rllm_core::ids::BlockId;

use crate::prefix::BlockHash;

/// Metadata for a single KV cache block.
#[derive(Debug)]
pub(crate) struct KVCacheBlock {
    /// Immutable block ID (index in the pool's blocks vector).
    pub id: BlockId,
    /// Prefix cache hash. `None` means uncached / not part of prefix cache.
    pub hash: Option<BlockHash>,
    /// Reference count. Block is free when ref_count == 0 and not in free queue.
    pub ref_count: u32,
    /// Index of previous block in free queue (u32 = block index).
    pub prev_free: Option<u32>,
    /// Index of next block in free queue.
    pub next_free: Option<u32>,
    /// Whether this is the null block (block 0, never allocated).
    pub is_null: bool,
}

/// Intrusive doubly-linked list of free block indices.
///
/// Blocks are ordered by eviction priority: head = most recently used (MRU),
/// tail = least recently used (LRU). New freed blocks go to head (recently used).
/// Eviction pops from tail (LRU).
#[derive(Debug)]
pub(crate) struct FreeBlockQueue {
    head: Option<u32>,
    tail: Option<u32>,
    len: usize,
}

impl FreeBlockQueue {
    pub fn new() -> Self {
        Self { head: None, tail: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(dead_code)]
    /// Push block to the head (MRU end).
    pub fn push_front(&mut self, blocks: &mut [KVCacheBlock], idx: u32) {
        let block = &mut blocks[idx as usize];
        block.prev_free = None;
        block.next_free = self.head;

        if let Some(head_idx) = self.head {
            blocks[head_idx as usize].prev_free = Some(idx);
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
        self.len += 1;
    }

    /// Push block to the tail (LRU end).
    pub fn push_back(&mut self, blocks: &mut [KVCacheBlock], idx: u32) {
        let block = &mut blocks[idx as usize];
        block.prev_free = self.tail;
        block.next_free = None;

        if let Some(tail_idx) = self.tail {
            blocks[tail_idx as usize].next_free = Some(idx);
        } else {
            self.head = Some(idx);
        }
        self.tail = Some(idx);
        self.len += 1;
    }

    /// Pop block from the head (MRU, most recently freed).
    pub fn pop_front(&mut self, blocks: &mut [KVCacheBlock]) -> Option<u32> {
        let idx = self.head?;
        let block = &blocks[idx as usize];

        self.head = block.next_free;
        if let Some(new_head) = self.head {
            blocks[new_head as usize].prev_free = None;
        } else {
            self.tail = None;
        }

        let block = &mut blocks[idx as usize];
        block.prev_free = None;
        block.next_free = None;
        self.len -= 1;
        Some(idx)
    }

    /// Remove a specific block from the queue.
    pub fn remove(&mut self, blocks: &mut [KVCacheBlock], idx: u32) {
        let block = &blocks[idx as usize];
        let prev = block.prev_free;
        let next = block.next_free;

        match prev {
            Some(p) => blocks[p as usize].next_free = next,
            None => self.head = next,
        }
        match next {
            Some(n) => blocks[n as usize].prev_free = prev,
            None => self.tail = prev,
        }

        let block = &mut blocks[idx as usize];
        block.prev_free = None;
        block.next_free = None;
        self.len -= 1;
    }
}

/// Maps block hashes to one or more blocks with that hash.
pub(crate) type BlockHashToBlockMap = HashMap<BlockHash, SmallVec<[BlockId; 2]>>;

/// Pool of KV cache blocks with prefix cache support.
pub struct BlockPool {
    /// All block metadata, indexed by block ID (BlockId.0 as usize).
    blocks: Vec<KVCacheBlock>,
    /// Free block queue (intrusive linked list).
    free_queue: FreeBlockQueue,
    /// Hash → block IDs for prefix caching.
    hash_map: BlockHashToBlockMap,
    /// Number of blocks currently in use (ref_count > 0).
    num_active_blocks: usize,
}

impl BlockPool {
    /// Create a new block pool with `num_blocks` blocks.
    ///
    /// Block 0 is reserved as the null block. Blocks 1..num_blocks start in the
    /// free queue.
    pub fn new(num_blocks: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);

        // Block 0: null block (never allocated, never freed)
        blocks.push(KVCacheBlock {
            id: BlockId(0),
            hash: None,
            ref_count: 0,
            prev_free: None,
            next_free: None,
            is_null: true,
        });

        // Blocks 1..num_blocks-1: start free
        for i in 1..num_blocks {
            blocks.push(KVCacheBlock {
                id: BlockId(i as u32),
                hash: None,
                ref_count: 0,
                prev_free: None,
                next_free: None,
                is_null: false,
            });
        }

        let mut free_queue = FreeBlockQueue::new();
        // Push blocks in order 1..num_blocks (first allocated = first in queue)
        for i in 1..num_blocks {
            free_queue.push_back(&mut blocks, i as u32);
        }

        Self { blocks, free_queue, hash_map: HashMap::new(), num_active_blocks: 0 }
    }

    /// Total number of blocks (including null block).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of blocks currently in use.
    pub fn num_active(&self) -> usize {
        self.num_active_blocks
    }

    /// Number of free blocks available.
    pub fn num_free(&self) -> usize {
        self.free_queue.len()
    }

    /// Allocate a new block.
    ///
    /// Prefers uncached free blocks to preserve prefix cache. Falls back to
    /// evicting the LRU cached block from the tail of the free queue.
    /// Returns `None` only if no blocks are available at all.
    pub fn allocate(&mut self) -> Option<BlockId> {
        // First try to find an uncached free block (search from head)
        let idx = self.pop_uncached().or_else(|| self.evict_lru())?;

        let block = &mut self.blocks[idx as usize];
        block.ref_count = 1;
        block.hash = None;
        self.num_active_blocks += 1;

        Some(block.id)
    }

    /// Pop the first uncached block from the free queue.
    fn pop_uncached(&mut self) -> Option<u32> {
        let mut candidate = self.free_queue.head;
        while let Some(idx) = candidate {
            let has_hash = self.blocks[idx as usize].hash.is_some();
            let next = self.blocks[idx as usize].next_free;
            if !has_hash {
                self.free_queue.remove(&mut self.blocks, idx);
                return Some(idx);
            }
            candidate = next;
        }
        None
    }

    /// Evict the LRU cached block from the tail of the free queue.
    fn evict_lru(&mut self) -> Option<u32> {
        let mut candidate = self.free_queue.tail;
        while let Some(idx) = candidate {
            let block = &self.blocks[idx as usize];
            let prev = block.prev_free;
            if block.hash.is_some() {
                self.free_queue.remove(&mut self.blocks, idx);
                let hash = self.blocks[idx as usize].hash.take().unwrap();
                let block_id = self.blocks[idx as usize].id;
                self.remove_hash_entry(hash, block_id);
                return Some(idx);
            }
            candidate = prev;
        }
        None
    }

    /// Free a block by decrementing its reference count.
    ///
    /// When the ref count reaches zero, the block returns to the free queue.
    /// If the block was cached, its hash is removed from the hash map.
    pub fn free(&mut self, block_id: BlockId) {
        let idx = block_id.0 as usize;
        let block = &mut self.blocks[idx];
        if block.is_null || block.ref_count == 0 {
            return;
        }

        block.ref_count -= 1;
        if block.ref_count == 0 {
            // Keep hash for prefix caching — hash is only removed on eviction or reset.
            // Push to front of free queue (MRU, recently freed)
            self.free_queue.push_front(&mut self.blocks, idx as u32);
            self.num_active_blocks -= 1;
        }
    }

    /// Increment the reference count on a block.
    pub fn incref(&mut self, block_id: BlockId) {
        let idx = block_id.0 as usize;
        let block = &mut self.blocks[idx];
        if !block.is_null && block.ref_count > 0 {
            block.ref_count += 1;
        }
    }

    /// Touch a cached block, moving it to the MRU end of the free queue.
    ///
    /// This is called on a prefix cache hit to keep the block alive.
    pub fn touch(&mut self, block_id: BlockId) {
        let idx = block_id.0 as usize;
        let block = &self.blocks[idx];
        // Only touch blocks that are in the free queue (cached but unreferenced)
        if block.ref_count == 0 && !block.is_null && block.hash.is_some() {
            let in_queue = block.prev_free.is_some()
                || block.next_free.is_some()
                || self.free_queue.head == Some(idx as u32);
            if in_queue {
                self.free_queue.remove(&mut self.blocks, idx as u32);
                self.free_queue.push_front(&mut self.blocks, idx as u32);
            }
        }
    }

    /// Look up a cached block by hash, increment its ref count, and touch it.
    ///
    /// Returns `None` if no cached block with this hash exists.
    pub fn get_cached_block(&mut self, hash: BlockHash) -> Option<BlockId> {
        let block_ids = self.hash_map.get(&hash)?;
        let block_id = *block_ids.first()?;

        // Touch and incref
        self.touch(block_id);

        let idx = block_id.0 as usize;
        let needs_activate = self.blocks[idx].ref_count == 0;
        let in_queue = needs_activate
            && (self.blocks[idx].prev_free.is_some()
                || self.blocks[idx].next_free.is_some()
                || self.free_queue.head == Some(idx as u32));

        if in_queue {
            self.free_queue.remove(&mut self.blocks, idx as u32);
        }
        if needs_activate {
            self.num_active_blocks += 1;
        }
        self.blocks[idx].ref_count += 1;

        Some(block_id)
    }

    /// Cache a block by associating it with a hash.
    ///
    /// The block must be active (ref_count > 0).
    pub fn cache_block(&mut self, block_id: BlockId, hash: BlockHash) {
        let idx = block_id.0 as usize;
        let block = &mut self.blocks[idx];
        if block.ref_count == 0 || block.is_null {
            return;
        }
        block.hash = Some(hash);
        self.hash_map.entry(hash).or_default().push(block_id);
    }

    /// Reset the prefix cache, clearing all hash associations.
    ///
    /// Returns `Ok(())` if successful. Returns `Err` if any cached blocks
    /// are still active (ref_count > 0 with a hash), as their KV data is in use.
    pub fn reset_prefix_cache(&mut self) -> Result<(), String> {
        // Check for active cached blocks
        for block in &self.blocks {
            if block.hash.is_some() && block.ref_count > 0 {
                return Err(format!(
                    "cannot reset prefix cache: block {} is active with hash",
                    block.id.0
                ));
            }
        }

        // Clear all hash associations
        for block in &mut self.blocks {
            block.hash = None;
        }
        self.hash_map.clear();

        Ok(())
    }

    /// Return cache usage stats.
    pub fn usage(&self) -> BlockPoolUsage {
        BlockPoolUsage {
            num_total: self.blocks.len() - 1, // exclude null block
            num_active: self.num_active_blocks,
            num_free: self.num_free(),
            num_cached: self.hash_map.len(),
        }
    }

    /// Remove a specific block_id from the hash map entry for the given hash.
    fn remove_hash_entry(&mut self, hash: BlockHash, block_id: BlockId) {
        if let Some(ids) = self.hash_map.get_mut(&hash) {
            ids.retain(|id| *id != block_id);
            if ids.is_empty() {
                self.hash_map.remove(&hash);
            }
        }
    }
}

/// Block pool usage statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPoolUsage {
    /// Total blocks available (excluding null block).
    pub num_total: usize,
    /// Blocks currently in use (ref_count > 0).
    pub num_active: usize,
    /// Blocks in the free queue.
    pub num_free: usize,
    /// Distinct cached hashes.
    pub num_cached: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_has_correct_counts() {
        let pool = BlockPool::new(10);
        // 1 null + 9 free = 10 total, 9 usable
        assert_eq!(pool.num_blocks(), 10);
        assert_eq!(pool.num_active(), 0);
        assert_eq!(pool.num_free(), 9);
    }

    #[test]
    fn allocate_increments_active() {
        let mut pool = BlockPool::new(10);
        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();
        assert_ne!(b1, b2);
        assert_eq!(pool.num_active(), 2);
        assert_eq!(pool.num_free(), 7);
    }

    #[test]
    fn free_decrements_active() {
        let mut pool = BlockPool::new(10);
        let b = pool.allocate().unwrap();
        assert_eq!(pool.num_active(), 1);
        pool.free(b);
        assert_eq!(pool.num_active(), 0);
        assert_eq!(pool.num_free(), 9);
    }

    #[test]
    fn allocate_free_preserves_ref_counts() {
        let mut pool = BlockPool::new(10);
        let b = pool.allocate().unwrap();
        assert_eq!(pool.blocks[b.0 as usize].ref_count, 1);

        pool.free(b);
        assert_eq!(pool.blocks[b.0 as usize].ref_count, 0);

        // Can reallocate the same block
        let b2 = pool.allocate().unwrap();
        // The freed block was pushed to front, so pop_front returns it
        assert_eq!(b, b2);
        assert_eq!(pool.blocks[b2.0 as usize].ref_count, 1);
    }

    #[test]
    fn incref_decref_multi_reference() {
        let mut pool = BlockPool::new(10);
        let b = pool.allocate().unwrap();
        pool.incref(b);
        assert_eq!(pool.blocks[b.0 as usize].ref_count, 2);

        pool.free(b);
        assert_eq!(pool.blocks[b.0 as usize].ref_count, 1);
        assert_eq!(pool.num_active(), 1);

        pool.free(b);
        assert_eq!(pool.blocks[b.0 as usize].ref_count, 0);
        assert_eq!(pool.num_active(), 0);
    }

    #[test]
    fn free_null_block_is_noop() {
        let mut pool = BlockPool::new(10);
        pool.free(BlockId(0));
        assert_eq!(pool.num_active(), 0);
    }

    #[test]
    fn double_free_is_safe() {
        let mut pool = BlockPool::new(10);
        let b = pool.allocate().unwrap();
        pool.free(b);
        pool.free(b); // should be noop (ref_count already 0)
        assert_eq!(pool.num_active(), 0);
    }

    #[test]
    fn touch_moves_cached_block_to_mru() {
        let mut pool = BlockPool::new(10);

        // Allocate, cache, then free
        let b = pool.allocate().unwrap();
        pool.cache_block(b, 12345);
        pool.free(b);

        assert_eq!(pool.num_active(), 0);
        assert!(pool.blocks[b.0 as usize].hash.is_some());

        // Touch should move it to front of free queue
        pool.touch(b);

        // Allocate should get a different block first (the touched one stays MRU)
        // Actually touch keeps it in the free queue but moves it to front.
        // pop_front gets MRU (the touched one), so we'd get the same block.
    }

    #[test]
    fn touched_prefix_blocks_leave_free_queue_on_get() {
        let mut pool = BlockPool::new(10);

        // Allocate, cache, free
        let b = pool.allocate().unwrap();
        let hash = 12345u64;
        pool.cache_block(b, hash);
        pool.free(b);

        // Block should be in free queue with a hash
        assert_eq!(pool.num_free(), 9);
        assert!(pool.blocks[b.0 as usize].hash.is_some());

        // Get cached block — should remove from free queue and make active
        let hit = pool.get_cached_block(hash).unwrap();
        assert_eq!(hit, b);
        assert_eq!(pool.num_active(), 1);
        assert_eq!(pool.num_free(), 8);
        assert_eq!(pool.blocks[b.0 as usize].ref_count, 1);
    }

    #[test]
    fn lru_eviction_removes_hash_metadata() {
        let mut pool = BlockPool::new(4); // null + 3 usable blocks

        // Allocate all 3 blocks, cache 2 of them
        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();
        let b3 = pool.allocate().unwrap();
        pool.cache_block(b1, 100);
        pool.cache_block(b2, 200);
        // b3 is uncached

        // Free all → all go to free queue (b3, b2, b1 head→tail)
        pool.free(b1);
        pool.free(b2);
        pool.free(b3);
        assert_eq!(pool.num_free(), 3);

        // Allocate should prefer uncached blocks → gets b3 (head of free queue after push_front)
        let new_b = pool.allocate().unwrap();
        assert_eq!(new_b, b3); // uncached block reused
        assert_eq!(pool.num_free(), 2);

        // Now all free blocks are cached. Next allocate must evict LRU from tail.
        let evicted = pool.allocate().unwrap();
        // b1 was freed first → at tail (LRU) → should be evicted
        assert_eq!(evicted, b1);
        assert!(pool.blocks[b1.0 as usize].hash.is_none());
        assert_eq!(pool.num_free(), 1);
    }

    #[test]
    fn reset_prefix_cache_refuses_active_requests() {
        let mut pool = BlockPool::new(10);
        let b = pool.allocate().unwrap();
        pool.cache_block(b, 42);

        let result = pool.reset_prefix_cache();
        assert!(result.is_err());
    }

    #[test]
    fn reset_prefix_cache_succeeds_when_no_active() {
        let mut pool = BlockPool::new(10);
        let b = pool.allocate().unwrap();
        pool.cache_block(b, 42);
        pool.free(b);

        let result = pool.reset_prefix_cache();
        assert!(result.is_ok());
        assert!(pool.hash_map.is_empty());
        assert!(pool.blocks[b.0 as usize].hash.is_none());
    }

    #[test]
    fn usage_stats() {
        let mut pool = BlockPool::new(10);
        let usage = pool.usage();
        assert_eq!(usage.num_total, 9);
        assert_eq!(usage.num_active, 0);
        assert_eq!(usage.num_free, 9);
        assert_eq!(usage.num_cached, 0);

        let b = pool.allocate().unwrap();
        pool.cache_block(b, 99);
        let usage = pool.usage();
        assert_eq!(usage.num_active, 1);
        assert_eq!(usage.num_cached, 1);
    }

    #[test]
    fn allocate_exhaustion_returns_none() {
        let mut pool = BlockPool::new(3); // null + 2 usable
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        // No cached blocks to evict
        let result = pool.allocate();
        assert!(result.is_none());
    }

    #[test]
    fn multiple_blocks_same_hash() {
        let mut pool = BlockPool::new(10);
        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();
        pool.cache_block(b1, 100);
        pool.cache_block(b2, 100);

        let ids = pool.hash_map.get(&100).unwrap();
        assert_eq!(ids.len(), 2);
    }
}
