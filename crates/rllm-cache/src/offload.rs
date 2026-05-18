use std::collections::HashMap;

use rllm_core::ids::{BlockId, RequestId};
use rllm_tensor::PinnedBuffer;

/// Pinned CPU storage for KV blocks evicted from GPU memory.
pub struct CpuKvOffloadStore {
    block_size_bytes: usize,
    capacity_blocks: usize,
    free_slots: Vec<usize>,
    slots: HashMap<(RequestId, BlockId), usize>,
    storage: PinnedBuffer,
}

impl CpuKvOffloadStore {
    pub fn new(capacity_blocks: usize, block_size_bytes: usize) -> Self {
        let free_slots = (0..capacity_blocks).rev().collect();
        Self {
            block_size_bytes,
            capacity_blocks,
            free_slots,
            slots: HashMap::new(),
            storage: PinnedBuffer::alloc(capacity_blocks * block_size_bytes),
        }
    }

    pub fn put(
        &mut self,
        request_id: RequestId,
        block_id: BlockId,
        bytes: &[u8],
    ) -> Result<(), String> {
        if bytes.len() != self.block_size_bytes {
            return Err(format!(
                "offloaded block has {} bytes, expected {}",
                bytes.len(),
                self.block_size_bytes
            ));
        }
        let key = (request_id, block_id);
        let slot = match self.slots.get(&key).copied() {
            Some(slot) => slot,
            None => self.free_slots.pop().ok_or("CPU KV offload store is full")?,
        };

        let offset = slot * self.block_size_bytes;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.storage.as_mut_ptr().add(offset),
                self.block_size_bytes,
            );
        }
        self.slots.insert(key, slot);
        Ok(())
    }

    pub fn get(&self, request_id: RequestId, block_id: BlockId) -> Option<Vec<u8>> {
        let slot = *self.slots.get(&(request_id, block_id))?;
        let offset = slot * self.block_size_bytes;
        let src = unsafe {
            std::slice::from_raw_parts(self.storage.as_ptr().add(offset), self.block_size_bytes)
        };
        Some(src.to_vec())
    }

    pub fn remove(&mut self, request_id: RequestId, block_id: BlockId) -> bool {
        match self.slots.remove(&(request_id, block_id)) {
            Some(slot) => {
                self.free_slots.push(slot);
                true
            }
            None => false,
        }
    }

    pub fn usage(&self) -> CpuKvOffloadUsage {
        CpuKvOffloadUsage {
            capacity_blocks: self.capacity_blocks,
            used_blocks: self.slots.len(),
            free_blocks: self.free_slots.len(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuKvOffloadUsage {
    pub capacity_blocks: usize,
    pub used_blocks: usize,
    pub free_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_remove_roundtrip() {
        let mut store = CpuKvOffloadStore::new(2, 4);
        let rid = RequestId::new();
        store.put(rid, BlockId(7), &[1, 2, 3, 4]).unwrap();
        assert_eq!(store.get(rid, BlockId(7)), Some(vec![1, 2, 3, 4]));
        assert_eq!(store.usage().used_blocks, 1);
        assert!(store.remove(rid, BlockId(7)));
        assert_eq!(store.get(rid, BlockId(7)), None);
    }
}
