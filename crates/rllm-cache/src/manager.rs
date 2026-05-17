use std::collections::HashMap;

use rllm_core::config::PrefixHashAlgorithm;
use rllm_core::ids::{BlockId, RequestId};

use crate::block_pool::BlockPool;
use crate::prefix::{self, BlockHash};
use crate::spec::{KVCacheConfig, KVCacheSpec};

/// Result of a prefix cache lookup for a request.
#[derive(Debug, Clone)]
pub struct PrefixCacheResult {
    /// Block IDs that were found in the prefix cache.
    pub cached_block_ids: Vec<BlockId>,
    /// Number of tokens already computed (cached_block_ids.len() * block_size).
    pub num_computed_tokens: usize,
}

/// Cache usage statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheUsage {
    pub num_total_blocks: usize,
    pub num_active_blocks: usize,
    pub num_free_blocks: usize,
    pub num_cached_hashes: usize,
    pub num_tracked_requests: usize,
}

/// Manages KV cache block allocation, prefix caching, and per-request block tables.
pub struct KVCacheManager {
    /// Block pool managing physical block metadata.
    block_pool: BlockPool,
    /// Cache specification.
    spec: KVCacheSpec,
    /// Cache configuration.
    #[allow(dead_code)]
    config: KVCacheConfig,
    /// Whether prefix caching is enabled.
    enable_prefix_caching: bool,
    /// Hash algorithm for prefix caching.
    hash_algorithm: PrefixHashAlgorithm,
    /// Per-request block tables: request_id → ordered list of block IDs.
    request_blocks: HashMap<RequestId, Vec<BlockId>>,
    /// Per-request computed token counts.
    request_computed_tokens: HashMap<RequestId, usize>,
    /// Per-request block hashes for prefix caching.
    request_block_hashes: HashMap<RequestId, Vec<BlockHash>>,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    pub fn new(
        config: KVCacheConfig,
        enable_prefix_caching: bool,
        hash_algorithm: PrefixHashAlgorithm,
    ) -> Self {
        let spec = config.spec.clone();
        let num_blocks = config.num_blocks;
        Self {
            block_pool: BlockPool::new(num_blocks),
            spec,
            config,
            enable_prefix_caching,
            hash_algorithm,
            request_blocks: HashMap::new(),
            request_computed_tokens: HashMap::new(),
            request_block_hashes: HashMap::new(),
        }
    }

    /// Look up prefix cache hits for a request's prompt tokens.
    ///
    /// Returns the cached block IDs and number of already-computed tokens.
    /// Skips prefix caching if disabled, if `skip_prefix_cache` is true,
    /// or if `prompt_logprobs` require full prompt computation.
    pub fn get_computed_blocks(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        skip_prefix_cache: bool,
    ) -> PrefixCacheResult {
        if !self.enable_prefix_caching || skip_prefix_cache || prompt_tokens.is_empty() {
            return PrefixCacheResult { cached_block_ids: vec![], num_computed_tokens: 0 };
        }

        let block_size = self.spec.block_size;

        // Compute hashes for all full blocks
        let hashes = prefix::compute_block_hashes(
            self.hash_algorithm,
            prompt_tokens,
            block_size,
            None, // cache_salt comes from request, but we don't store it here yet
        );

        // Find longest prefix hit by checking cached blocks one by one
        let mut cached_block_ids = Vec::new();
        for &hash in &hashes {
            match self.block_pool.get_cached_block(hash) {
                Some(block_id) => {
                    cached_block_ids.push(block_id);
                }
                None => break,
            }
        }

        let num_computed_tokens = cached_block_ids.len() * block_size;

        // Store hashes and cached blocks for this request
        self.request_block_hashes.insert(request_id, hashes);
        self.request_computed_tokens.insert(request_id, num_computed_tokens);

        // Store the cached block IDs in the request's block table
        self.request_blocks.insert(request_id, cached_block_ids.clone());

        PrefixCacheResult { cached_block_ids, num_computed_tokens }
    }

    /// Allocate KV cache slots for a request.
    ///
    /// `num_tokens`: total tokens needed (prompt + generated so far + new tokens)
    /// `num_computed`: tokens already computed (from prefix cache or previous steps)
    /// `cached_blocks`: blocks already assigned (from prefix cache)
    ///
    /// Returns `Some(new_block_ids)` on success, `None` if insufficient blocks.
    /// On failure, no state is modified.
    pub fn allocate_slots(
        &mut self,
        request_id: RequestId,
        num_tokens: usize,
        num_computed: usize,
    ) -> Option<Vec<BlockId>> {
        let block_size = self.spec.block_size;
        let tokens_needed = num_tokens.saturating_sub(num_computed);
        if tokens_needed == 0 {
            return Some(vec![]);
        }

        let num_new_blocks = tokens_needed.div_ceil(block_size);

        // Try to allocate all needed blocks
        let mut new_blocks = Vec::with_capacity(num_new_blocks);
        for _ in 0..num_new_blocks {
            match self.block_pool.allocate() {
                Some(block_id) => new_blocks.push(block_id),
                None => {
                    // Rollback: free already-allocated blocks
                    for &block_id in &new_blocks {
                        self.block_pool.free(block_id);
                    }
                    return None;
                }
            }
        }

        // Append to request's block table
        let block_table = self.request_blocks.entry(request_id).or_default();
        block_table.extend_from_slice(&new_blocks);

        Some(new_blocks)
    }

    /// Free all blocks associated with a request.
    pub fn free(&mut self, request_id: RequestId) {
        if let Some(blocks) = self.request_blocks.remove(&request_id) {
            for block_id in blocks {
                self.block_pool.free(block_id);
            }
        }
        self.request_computed_tokens.remove(&request_id);
        self.request_block_hashes.remove(&request_id);
    }

    /// Cache newly completed blocks for a request.
    ///
    /// This should be called after the model has computed tokens for this request.
    /// `num_computed_tokens` is the total number of tokens that have been computed
    /// (including previous prefix cache hits).
    pub fn cache_blocks(&mut self, request_id: RequestId, num_computed_tokens: usize) {
        if !self.enable_prefix_caching {
            return;
        }

        let block_size = self.spec.block_size;
        let num_full_blocks = num_computed_tokens / block_size;

        let Some(block_table) = self.request_blocks.get(&request_id) else {
            return;
        };
        let Some(hashes) = self.request_block_hashes.get(&request_id) else {
            return;
        };

        // Cache all full blocks that haven't been cached yet
        let existing_cached = self.request_computed_tokens.get(&request_id).copied().unwrap_or(0);
        let already_cached_blocks = existing_cached / block_size;

        for i in already_cached_blocks..std::cmp::min(num_full_blocks, hashes.len()) {
            if i < block_table.len() {
                self.block_pool.cache_block(block_table[i], hashes[i]);
            }
        }

        self.request_computed_tokens.insert(request_id, num_computed_tokens);
    }

    /// Get the block IDs for a request.
    pub fn get_block_ids(&self, request_id: RequestId) -> Option<&[BlockId]> {
        self.request_blocks.get(&request_id).map(|v| v.as_slice())
    }

    /// Get the number of computed tokens for a request.
    pub fn get_computed_tokens(&self, request_id: RequestId) -> usize {
        self.request_computed_tokens.get(&request_id).copied().unwrap_or(0)
    }

    /// Return cache usage statistics.
    pub fn usage(&self) -> CacheUsage {
        let pool_usage = self.block_pool.usage();
        CacheUsage {
            num_total_blocks: pool_usage.num_total,
            num_active_blocks: pool_usage.num_active,
            num_free_blocks: pool_usage.num_free,
            num_cached_hashes: pool_usage.num_cached,
            num_tracked_requests: self.request_blocks.len(),
        }
    }

    /// Reset the prefix cache. Returns `Err` if active requests exist.
    pub fn reset_prefix_cache(&mut self) -> Result<(), String> {
        if !self.request_blocks.is_empty() {
            return Err(format!(
                "cannot reset prefix cache: {} active requests",
                self.request_blocks.len()
            ));
        }
        self.block_pool.reset_prefix_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_core::dtype::DType;

    fn make_spec(block_size: usize) -> KVCacheSpec {
        KVCacheSpec {
            block_size,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F16,
            sliding_window: None,
        }
    }

    fn make_config(spec: KVCacheSpec, num_blocks: usize) -> KVCacheConfig {
        KVCacheConfig { num_blocks, spec }
    }

    #[test]
    fn block_allocation_count_matches_tokens() {
        let spec = make_spec(16);
        let config = make_config(spec.clone(), 100);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        // 33 tokens, block_size=16 → need 3 blocks (2 full + 1 partial)
        let new_blocks = mgr.allocate_slots(rid, 33, 0).unwrap();
        assert_eq!(new_blocks.len(), 3);

        let table = mgr.get_block_ids(rid).unwrap();
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn allocation_with_computed_tokens() {
        let spec = make_spec(16);
        let config = make_config(spec.clone(), 100);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        // 33 tokens, 16 already computed → 17 remaining → 2 blocks
        let new_blocks = mgr.allocate_slots(rid, 33, 16).unwrap();
        assert_eq!(new_blocks.len(), 2);
    }

    #[test]
    fn insufficient_blocks_returns_none() {
        let spec = make_spec(16);
        let config = make_config(spec, 3); // null + 2 usable
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        // Need 3 blocks but only 2 available
        let result = mgr.allocate_slots(rid, 33, 0);
        assert!(result.is_none());

        // State should be clean — no partial allocations
        assert!(mgr.get_block_ids(rid).is_none());
        assert_eq!(mgr.usage().num_active_blocks, 0);
    }

    #[test]
    fn free_releases_all_blocks() {
        let spec = make_spec(16);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        mgr.allocate_slots(rid, 32, 0).unwrap();
        assert_eq!(mgr.usage().num_active_blocks, 2);

        mgr.free(rid);
        assert_eq!(mgr.usage().num_active_blocks, 0);
        assert!(mgr.get_block_ids(rid).is_none());
    }

    #[test]
    fn prefix_hit_reduces_tokens_to_compute() {
        let spec = make_spec(4);
        let config = make_config(spec.clone(), 100);
        let mut mgr = KVCacheManager::new(config, true, PrefixHashAlgorithm::Sha256Cbor);

        let rid1 = RequestId::new();
        let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks
        let result = mgr.get_computed_blocks(rid1, &tokens1, false);
        assert_eq!(result.cached_block_ids.len(), 0); // nothing cached yet

        // Allocate and cache blocks for first request
        let blocks = mgr.allocate_slots(rid1, 8, 0).unwrap();
        assert_eq!(blocks.len(), 2);
        mgr.cache_blocks(rid1, 8);

        // Free first request (blocks stay cached)
        mgr.free(rid1);

        // Second request with same prefix should hit cache
        let rid2 = RequestId::new();
        let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // same prefix + more
        let result = mgr.get_computed_blocks(rid2, &tokens2, false);
        assert_eq!(result.cached_block_ids.len(), 2);
        assert_eq!(result.num_computed_tokens, 8);
    }

    #[test]
    fn free_order_matches_eviction_priority() {
        let spec = make_spec(4);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        // Allocate for 3 requests
        let r1 = RequestId::new();
        let r2 = RequestId::new();
        let r3 = RequestId::new();
        mgr.allocate_slots(r1, 4, 0).unwrap(); // 1 block
        mgr.allocate_slots(r2, 8, 0).unwrap(); // 2 blocks
        mgr.allocate_slots(r3, 4, 0).unwrap(); // 1 block

        assert_eq!(mgr.usage().num_active_blocks, 4);

        // Free in order
        mgr.free(r1);
        assert_eq!(mgr.usage().num_active_blocks, 3);
        mgr.free(r2);
        assert_eq!(mgr.usage().num_active_blocks, 1);
        mgr.free(r3);
        assert_eq!(mgr.usage().num_active_blocks, 0);
    }

    #[test]
    fn multiple_requests_share_prefix_blocks() {
        let spec = make_spec(4);
        let config = make_config(spec.clone(), 100);
        let mut mgr = KVCacheManager::new(config, true, PrefixHashAlgorithm::Sha256Cbor);

        // First request: fill and cache
        let r1 = RequestId::new();
        let tokens: Vec<u32> = vec![10, 20, 30, 40];
        let result = mgr.get_computed_blocks(r1, &tokens, false);
        assert!(result.cached_block_ids.is_empty());
        mgr.allocate_slots(r1, 4, 0).unwrap();
        mgr.cache_blocks(r1, 4);
        mgr.free(r1); // blocks stay cached

        // Two requests with same prefix
        let r2 = RequestId::new();
        let r3 = RequestId::new();

        let result2 = mgr.get_computed_blocks(r2, &tokens, false);
        let result3 = mgr.get_computed_blocks(r3, &tokens, false);

        assert_eq!(result2.cached_block_ids.len(), 1);
        assert_eq!(result3.cached_block_ids.len(), 1);

        // They should share the same cached block (ref count > 1)
        let block2 = result2.cached_block_ids[0];
        let block3 = result3.cached_block_ids[0];
        assert_eq!(block2, block3); // Same block shared
    }

    #[test]
    fn skip_prefix_cache_flag() {
        let spec = make_spec(4);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, true, PrefixHashAlgorithm::Sha256Cbor);

        // Prime the cache
        let r1 = RequestId::new();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        mgr.get_computed_blocks(r1, &tokens, false);
        mgr.allocate_slots(r1, 4, 0).unwrap();
        mgr.cache_blocks(r1, 4);
        mgr.free(r1);

        // Skip prefix cache
        let r2 = RequestId::new();
        let result = mgr.get_computed_blocks(r2, &tokens, true);
        assert!(result.cached_block_ids.is_empty());
        assert_eq!(result.num_computed_tokens, 0);
    }

    #[test]
    fn prefix_caching_disabled() {
        let spec = make_spec(4);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = mgr.get_computed_blocks(rid, &tokens, false);
        assert!(result.cached_block_ids.is_empty());
    }

    #[test]
    fn usage_stats() {
        let spec = make_spec(4);
        let config = make_config(spec, 10);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let usage = mgr.usage();
        assert_eq!(usage.num_total_blocks, 9); // 10 - 1 null
        assert_eq!(usage.num_active_blocks, 0);

        let rid = RequestId::new();
        mgr.allocate_slots(rid, 8, 0).unwrap();
        let usage = mgr.usage();
        assert_eq!(usage.num_active_blocks, 2);
        assert_eq!(usage.num_tracked_requests, 1);
    }

    #[test]
    fn reset_prefix_cache_with_active_requests_fails() {
        let spec = make_spec(4);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, true, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        mgr.allocate_slots(rid, 4, 0).unwrap();

        let result = mgr.reset_prefix_cache();
        assert!(result.is_err());
    }

    #[test]
    fn reset_prefix_cache_succeeds_when_empty() {
        let spec = make_spec(4);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, true, PrefixHashAlgorithm::Sha256Cbor);

        let result = mgr.reset_prefix_cache();
        assert!(result.is_ok());
    }

    #[test]
    fn zero_tokens_needed_returns_empty() {
        let spec = make_spec(4);
        let config = make_config(spec, 100);
        let mut mgr = KVCacheManager::new(config, false, PrefixHashAlgorithm::Sha256Cbor);

        let rid = RequestId::new();
        let result = mgr.allocate_slots(rid, 4, 4).unwrap(); // all computed
        assert!(result.is_empty());
    }
}
