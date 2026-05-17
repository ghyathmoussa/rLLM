use std::collections::HashMap;

use rllm_core::config::PrefixHashAlgorithm;

/// Block hash: 64-bit hash identifying the content of a full KV cache block.
pub type BlockHash = u64;

/// Input to the block hash function.
#[derive(Debug, Clone)]
pub struct BlockHashInput<'a> {
    /// Hash of the parent block (0 for the first block).
    pub parent_hash: BlockHash,
    /// Token IDs in this block.
    pub token_ids: &'a [u32],
    /// Optional cache salt for tenant/privacy separation.
    pub cache_salt: Option<&'a str>,
    /// Extra keys for future use (LoRA ID, multimodal hash, etc.).
    pub extra_keys: &'a [(String, Vec<u8>)],
}

/// Compute the hash for a single block.
///
/// Only full blocks (token count == block_size) should be hashed.
/// Returns `None` if the block is not full.
pub fn compute_block_hash(
    algo: PrefixHashAlgorithm,
    input: &BlockHashInput<'_>,
    block_size: usize,
) -> Option<BlockHash> {
    if input.token_ids.len() < block_size {
        return None;
    }
    Some(match algo {
        PrefixHashAlgorithm::Sha256Cbor => sha256_cbor_hash(input),
        PrefixHashAlgorithm::XxHash => xxhash_hash(input),
    })
}

/// Compute block hashes for a sequence of tokens, one per full block.
///
/// Returns hashes for each full block in order, using chained parent hashes.
/// Returns `(hashes, num_full_blocks)`.
pub fn compute_block_hashes(
    algo: PrefixHashAlgorithm,
    token_ids: &[u32],
    block_size: usize,
    cache_salt: Option<&str>,
) -> Vec<BlockHash> {
    let num_full_blocks = token_ids.len() / block_size;
    let mut hashes = Vec::with_capacity(num_full_blocks);
    let mut parent_hash: BlockHash = 0;

    for i in 0..num_full_blocks {
        let start = i * block_size;
        let end = start + block_size;
        let input = BlockHashInput {
            parent_hash,
            token_ids: &token_ids[start..end],
            cache_salt,
            extra_keys: &[],
        };
        let hash = match algo {
            PrefixHashAlgorithm::Sha256Cbor => sha256_cbor_hash(&input),
            PrefixHashAlgorithm::XxHash => xxhash_hash(&input),
        };
        hashes.push(hash);
        parent_hash = hash;
    }

    hashes
}

/// Check if a sequence fully hits the prefix cache.
///
/// Returns `(num_cached_blocks, needs_one_token_recompute)`.
/// When the entire prompt is cached, we recompute the last token so logits are available.
pub fn full_cache_hit_info(
    prompt_len: usize,
    num_cached_blocks: usize,
    block_size: usize,
) -> (usize, bool) {
    let cached_tokens = num_cached_blocks * block_size;
    let full_hit = cached_tokens >= prompt_len && num_cached_blocks > 0;
    // On full hit, recompute the last token for logits
    (num_cached_blocks, full_hit)
}

/// SHA-256 of CBOR-encoded input.
fn sha256_cbor_hash(input: &BlockHashInput<'_>) -> BlockHash {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();

    // Deterministic encoding: parent_hash as u64 BE bytes
    hasher.update(input.parent_hash.to_be_bytes());

    // Token IDs as packed u32 LE
    for &tid in input.token_ids {
        hasher.update(tid.to_le_bytes());
    }

    // Cache salt if present
    if let Some(salt) = input.cache_salt {
        // Length-prefixed to prevent collisions
        hasher.update((salt.len() as u64).to_le_bytes());
        hasher.update(salt.as_bytes());
    }

    // Extra keys
    for (key, value) in input.extra_keys {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    }

    let result = hasher.finalize();
    // Take first 8 bytes as u64
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&result[..8]);
    u64::from_be_bytes(bytes)
}

/// Fast xxh3 hash of concatenated input bytes.
fn xxhash_hash(input: &BlockHashInput<'_>) -> BlockHash {
    use xxhash_rust::xxh3::xxh3_64;

    let mut data = Vec::new();
    data.extend_from_slice(&input.parent_hash.to_be_bytes());
    for &tid in input.token_ids {
        data.extend_from_slice(&tid.to_le_bytes());
    }
    if let Some(salt) = input.cache_salt {
        data.extend_from_slice(&(salt.len() as u64).to_be_bytes());
        data.extend_from_slice(salt.as_bytes());
    }
    for (key, value) in input.extra_keys {
        data.extend_from_slice(&(key.len() as u64).to_be_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(value.len() as u64).to_be_bytes());
        data.extend_from_slice(value);
    }

    xxh3_64(&data)
}

/// Find the longest prefix cache hit for a sequence of tokens.
///
/// Compares computed block hashes against the cached hash map.
/// Returns the number of matching full blocks from the start.
pub fn find_prefix_hit(
    computed_hashes: &[BlockHash],
    cached_hashes: &HashMap<BlockHash, usize>,
) -> usize {
    for (i, &hash) in computed_hashes.iter().enumerate() {
        if !cached_hashes.contains_key(&hash) {
            return i;
        }
    }
    computed_hashes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_prefixes_produce_identical_hashes() {
        let tokens: Vec<u32> = (0..32).collect();
        let hashes_a = compute_block_hashes(PrefixHashAlgorithm::Sha256Cbor, &tokens, 16, None);
        let hashes_b = compute_block_hashes(PrefixHashAlgorithm::Sha256Cbor, &tokens, 16, None);
        assert_eq!(hashes_a, hashes_b);
    }

    #[test]
    fn different_cache_salts_prevent_reuse() {
        let tokens: Vec<u32> = (0..16).collect();
        let hash_a = compute_block_hash(
            PrefixHashAlgorithm::Sha256Cbor,
            &BlockHashInput {
                parent_hash: 0,
                token_ids: &tokens,
                cache_salt: Some("tenant_a"),
                extra_keys: &[],
            },
            16,
        );
        let hash_b = compute_block_hash(
            PrefixHashAlgorithm::Sha256Cbor,
            &BlockHashInput {
                parent_hash: 0,
                token_ids: &tokens,
                cache_salt: Some("tenant_b"),
                extra_keys: &[],
            },
            16,
        );
        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn partial_final_block_is_not_cached() {
        let tokens: Vec<u32> = (0..10).collect();
        let result = compute_block_hash(
            PrefixHashAlgorithm::Sha256Cbor,
            &BlockHashInput {
                parent_hash: 0,
                token_ids: &tokens,
                cache_salt: None,
                extra_keys: &[],
            },
            16,
        );
        assert!(result.is_none());
    }

    #[test]
    fn full_block_produces_hash() {
        let tokens: Vec<u32> = (0..16).collect();
        let result = compute_block_hash(
            PrefixHashAlgorithm::Sha256Cbor,
            &BlockHashInput {
                parent_hash: 0,
                token_ids: &tokens,
                cache_salt: None,
                extra_keys: &[],
            },
            16,
        );
        assert!(result.is_some());
    }

    #[test]
    fn sha256_and_xxhash_differ() {
        let tokens: Vec<u32> = (0..16).collect();
        let sha = compute_block_hash(
            PrefixHashAlgorithm::Sha256Cbor,
            &BlockHashInput {
                parent_hash: 0,
                token_ids: &tokens,
                cache_salt: None,
                extra_keys: &[],
            },
            16,
        );
        let xxh = compute_block_hash(
            PrefixHashAlgorithm::XxHash,
            &BlockHashInput {
                parent_hash: 0,
                token_ids: &tokens,
                cache_salt: None,
                extra_keys: &[],
            },
            16,
        );
        assert!(sha.is_some());
        assert!(xxh.is_some());
        assert_ne!(sha, xxh);
    }

    #[test]
    fn chained_hashes_differ_per_block() {
        let tokens: Vec<u32> = (0..48).collect();
        let hashes = compute_block_hashes(PrefixHashAlgorithm::Sha256Cbor, &tokens, 16, None);
        assert_eq!(hashes.len(), 3);
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn parent_hash_affects_output() {
        let tokens: Vec<u32> = (0..16).collect();
        let h1 = sha256_cbor_hash(&BlockHashInput {
            parent_hash: 0,
            token_ids: &tokens,
            cache_salt: None,
            extra_keys: &[],
        });
        let h2 = sha256_cbor_hash(&BlockHashInput {
            parent_hash: 999,
            token_ids: &tokens,
            cache_salt: None,
            extra_keys: &[],
        });
        assert_ne!(h1, h2);
    }

    #[test]
    fn full_cache_hit_detection() {
        // 32 tokens, block_size=16 → 2 full blocks
        let (blocks, recompute) = full_cache_hit_info(32, 2, 16);
        assert_eq!(blocks, 2);
        assert!(recompute);

        // 33 tokens, 2 cached blocks (32 tokens cached) → not full hit
        let (blocks, recompute) = full_cache_hit_info(33, 2, 16);
        assert_eq!(blocks, 2);
        assert!(!recompute);

        // 0 cached blocks → not full hit
        let (blocks, recompute) = full_cache_hit_info(16, 0, 16);
        assert_eq!(blocks, 0);
        assert!(!recompute);
    }

    #[test]
    fn find_prefix_hit_returns_correct_count() {
        let tokens: Vec<u32> = (0..48).collect();
        let hashes = compute_block_hashes(PrefixHashAlgorithm::Sha256Cbor, &tokens, 16, None);
        assert_eq!(hashes.len(), 3);

        // All cached
        let mut cached = HashMap::new();
        for &h in &hashes {
            cached.insert(h, 0);
        }
        assert_eq!(find_prefix_hit(&hashes, &cached), 3);

        // Only first two cached
        cached.remove(&hashes[2]);
        assert_eq!(find_prefix_hit(&hashes, &cached), 2);

        // None cached
        cached.clear();
        assert_eq!(find_prefix_hit(&hashes, &cached), 0);
    }

    #[test]
    fn extra_keys_affect_hash() {
        let tokens: Vec<u32> = (0..16).collect();
        let h1 = sha256_cbor_hash(&BlockHashInput {
            parent_hash: 0,
            token_ids: &tokens,
            cache_salt: None,
            extra_keys: &[],
        });
        let h2 = sha256_cbor_hash(&BlockHashInput {
            parent_hash: 0,
            token_ids: &tokens,
            cache_salt: None,
            extra_keys: &[(String::from("lora"), vec![1, 2, 3])],
        });
        assert_ne!(h1, h2);
    }
}
