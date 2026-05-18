//! PagedAttention kernel interface for decode and prefill paths.
//!
//! Provides:
//! - `AttentionMetadata`: metadata for attention computation (seq_lens, block_tables, etc.)
//! - `AttentionParams`: kernel parameters (head counts, dimensions, scale)
//! - FFI wrappers for CUDA PagedAttention kernels
//! - Non-CUDA stubs

use crate::cuda::CudaKernelError;

// ── Attention Parameters ──────────────────────────────────────────────────

/// Static parameters for attention computation.
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub scale: f32,
}

impl AttentionParams {
    pub fn new(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { num_q_heads, num_kv_heads, head_dim, block_size, scale }
    }

    /// GQA ratio: number of Q heads per KV head.
    pub fn gqa_ratio(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    /// Map a Q head index to its corresponding KV head index.
    pub fn kv_head_for_q_head(&self, q_head: usize) -> usize {
        q_head * self.num_kv_heads / self.num_q_heads
    }

    /// Number of blocks needed for a given sequence length.
    pub fn num_blocks_for_seq_len(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_size)
    }
}

// ── Attention Metadata ────────────────────────────────────────────────────

/// Metadata for a batch of attention computations.
///
/// Supports mixed prefill/decode batches via `num_prefill_tokens` and
/// `num_decode_tokens`.
#[derive(Debug, Clone)]
pub struct AttentionMetadata {
    /// Sequence length per request (total KV length).
    pub seq_lens: Vec<u32>,
    /// Cumulative token start index per request (prefix sum).
    /// Length = num_seqs + 1. `query_start_loc[0] = 0`.
    pub query_start_loc: Vec<u32>,
    /// Per-request block tables: logical to physical block mapping.
    /// `block_tables[seq_idx][block_idx]` is the physical block ID (-1 = unused).
    pub block_tables: Vec<Vec<i32>>,
    /// Flat slot mapping for cache writes: token position to physical slot.
    pub slot_mapping: Vec<i64>,
    /// Number of prefill tokens in this batch.
    pub num_prefill_tokens: usize,
    /// Number of decode tokens in this batch (one per decode sequence).
    pub num_decode_tokens: usize,
    /// Maximum number of blocks per sequence.
    pub max_num_blocks_per_seq: usize,
    /// Number of common prefix blocks shared across all sequences.
    ///
    /// When all sequences share a common prefix (via prefix caching), the
    /// attention kernel can skip computing attention for these blocks since
    /// the result is identical. Set to 0 when no common prefix exists.
    pub common_prefix_blocks: usize,
    /// Optional sliding window size for windowed attention.
    ///
    /// When `Some(window_size)`, attention is restricted to the last
    /// `window_size` tokens. This reduces KV cache usage for long sequences.
    /// Set to `None` for full attention.
    pub sliding_window: Option<usize>,
}

impl AttentionMetadata {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self {
            seq_lens: Vec::new(),
            query_start_loc: vec![0],
            block_tables: Vec::new(),
            slot_mapping: Vec::new(),
            num_prefill_tokens: 0,
            num_decode_tokens: 0,
            max_num_blocks_per_seq: 0,
            common_prefix_blocks: 0,
            sliding_window: None,
        }
    }

    /// Create metadata for a decode-only batch.
    ///
    /// Each sequence contributes one token. Block tables and sequence lengths
    /// are provided.
    pub fn for_decode(
        seq_lens: Vec<u32>,
        block_tables: Vec<Vec<i32>>,
        max_num_blocks_per_seq: usize,
    ) -> Self {
        let num_seqs = seq_lens.len();
        let query_start_loc: Vec<u32> = (0..=num_seqs).map(|i| i as u32).collect();

        Self {
            seq_lens,
            query_start_loc,
            block_tables,
            slot_mapping: Vec::new(),
            num_prefill_tokens: 0,
            num_decode_tokens: num_seqs,
            max_num_blocks_per_seq,
            common_prefix_blocks: 0,
            sliding_window: None,
        }
    }

    /// Create metadata for a decode-only batch with sliding window and common prefix.
    pub fn for_decode_with_options(
        seq_lens: Vec<u32>,
        block_tables: Vec<Vec<i32>>,
        max_num_blocks_per_seq: usize,
        common_prefix_blocks: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        let mut meta = Self::for_decode(seq_lens, block_tables, max_num_blocks_per_seq);
        meta.common_prefix_blocks = common_prefix_blocks;
        meta.sliding_window = sliding_window;
        meta
    }

    /// Create metadata for a prefill-only batch.
    ///
    /// `prompt_tokens_per_seq`: number of new tokens to prefill per sequence.
    pub fn for_prefill(
        seq_lens: Vec<u32>,
        prompt_tokens_per_seq: Vec<u32>,
        block_tables: Vec<Vec<i32>>,
        max_num_blocks_per_seq: usize,
    ) -> Self {
        let num_seqs = seq_lens.len();
        let mut query_start_loc = Vec::with_capacity(num_seqs + 1);
        query_start_loc.push(0);
        let mut cumulative = 0u32;
        for &count in &prompt_tokens_per_seq {
            cumulative += count;
            query_start_loc.push(cumulative);
        }
        let num_prefill_tokens = cumulative as usize;

        Self {
            seq_lens,
            query_start_loc,
            block_tables,
            slot_mapping: Vec::new(),
            num_prefill_tokens,
            num_decode_tokens: 0,
            max_num_blocks_per_seq,
            common_prefix_blocks: 0,
            sliding_window: None,
        }
    }

    /// Flatten block tables into a contiguous array for GPU transfer.
    ///
    /// Returns a flat vector of shape `[num_seqs * max_num_blocks_per_seq]`,
    /// padded with -1 for unused entries.
    pub fn flatten_block_tables(&self) -> Vec<i32> {
        let mut flat = vec![-1i32; self.seq_lens.len() * self.max_num_blocks_per_seq];
        for (seq_idx, bt) in self.block_tables.iter().enumerate() {
            let start = seq_idx * self.max_num_blocks_per_seq;
            let len = bt.len().min(self.max_num_blocks_per_seq);
            flat[start..start + len].copy_from_slice(&bt[..len]);
        }
        flat
    }

    /// Number of sequences in this batch.
    pub fn num_seqs(&self) -> usize {
        self.seq_lens.len()
    }

    /// Total number of tokens (prefill + decode).
    pub fn num_tokens(&self) -> usize {
        self.num_prefill_tokens + self.num_decode_tokens
    }

    /// Detect the number of common prefix blocks shared by all sequences.
    ///
    /// Compares block tables across all sequences. The common prefix count
    /// is the number of leading blocks that are identical across all sequences.
    /// This can be used by attention kernels to skip recomputing the prefix.
    pub fn detect_common_prefix_blocks(&mut self) {
        if self.block_tables.is_empty() || self.block_tables.len() < 2 {
            self.common_prefix_blocks = 0;
            return;
        }

        let first = &self.block_tables[0];
        let mut common = first.len();

        for bt in self.block_tables[1..].iter() {
            let mut count = 0;
            for (a, b) in first.iter().zip(bt.iter()) {
                if a == b {
                    count += 1;
                } else {
                    break;
                }
            }
            common = common.min(count);
        }

        self.common_prefix_blocks = common;
    }

    /// Compute a sliding window mask for the attention computation.
    ///
    /// Returns a tuple of `(start_positions, end_positions)` for each sequence
    /// indicating the range of KV positions to attend to.
    /// When `sliding_window` is `None`, the full range is returned.
    pub fn sliding_window_ranges(&self) -> Vec<(u32, u32)> {
        let window = match self.sliding_window {
            Some(w) => w as u32,
            None => return self.seq_lens.iter().map(|&len| (0, len)).collect(),
        };

        self.seq_lens
            .iter()
            .map(|&len| {
                if len <= window {
                    (0, len)
                } else {
                    (len - window, len)
                }
            })
            .collect()
    }

    /// Number of tokens that are within the sliding window for each sequence.
    /// Returns 0 for all sequences when no sliding window is configured.
    pub fn num_window_tokens(&self) -> usize {
        match self.sliding_window {
            Some(w) => self.seq_lens.iter().map(|&len| (len as usize).min(w)).sum(),
            None => self.seq_lens.iter().map(|&len| len as usize).sum(),
        }
    }
}

impl Default for AttentionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// ── FFI declarations ──────────────────────────────────────────────────────

#[cfg(has_cuda)]
mod ffi {
    use std::os::raw::c_int;

    extern "C" {
        pub fn rllm_paged_attention_decode_f16(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u16,
            value_cache: *const u16,
            block_tables: *const i32,
            seq_lens: *const i32,
            num_seqs: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_paged_attention_decode_f16_sync(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u16,
            value_cache: *const u16,
            block_tables: *const i32,
            seq_lens: *const i32,
            num_seqs: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
        ) -> c_int;

        pub fn rllm_paged_attention_prefill_f16(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u16,
            value_cache: *const u16,
            block_tables: *const i32,
            seq_lens: *const i32,
            query_start_loc: *const i32,
            num_seqs: i64,
            num_tokens: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_paged_attention_prefill_f16_sync(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u16,
            value_cache: *const u16,
            block_tables: *const i32,
            seq_lens: *const i32,
            query_start_loc: *const i32,
            num_seqs: i64,
            num_tokens: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
        ) -> c_int;
    }
}

#[cfg(has_cuda)]
fn check(rc: i32) -> Result<(), CudaKernelError> {
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
}

// ── Decode PagedAttention ─────────────────────────────────────────────────

/// Launch async decode PagedAttention (FP16).
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes.
/// - `block_tables` must have `num_seqs * max_num_blocks_per_seq` elements.
/// - `seq_lens` must have `num_seqs` elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_f16(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u16,
    value_cache: *const u16,
    block_tables: *const i32,
    seq_lens: *const i32,
    num_seqs: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_decode_f16(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks_per_seq,
            scale,
            stream,
        )
    };
    check(rc)
}

/// Synchronous decode PagedAttention for testing.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_f16_sync(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u16,
    value_cache: *const u16,
    block_tables: *const i32,
    seq_lens: *const i32,
    num_seqs: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_decode_f16_sync(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks_per_seq,
            scale,
        )
    };
    check(rc)
}

// ── Prefill PagedAttention ────────────────────────────────────────────────

/// Launch async prefill PagedAttention (FP16).
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes.
/// - `query_start_loc` must have `num_seqs + 1` elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_prefill_f16(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u16,
    value_cache: *const u16,
    block_tables: *const i32,
    seq_lens: *const i32,
    query_start_loc: *const i32,
    num_seqs: i64,
    num_tokens: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_prefill_f16(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            query_start_loc,
            num_seqs,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks_per_seq,
            scale,
            stream,
        )
    };
    check(rc)
}

/// Synchronous prefill PagedAttention for testing.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_prefill_f16_sync(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u16,
    value_cache: *const u16,
    block_tables: *const i32,
    seq_lens: *const i32,
    query_start_loc: *const i32,
    num_seqs: i64,
    num_tokens: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_prefill_f16_sync(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            query_start_loc,
            num_seqs,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks_per_seq,
            scale,
        )
    };
    check(rc)
}

// ── Non-CUDA stubs ────────────────────────────────────────────────────────

#[cfg(not(has_cuda))]
pub use stubs::*;

#[cfg(not(has_cuda))]
mod stubs {
    use super::CudaKernelError;

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_decode_f16(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u16,
        _value_cache: *const u16,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _num_seqs: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_decode_f16_sync(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u16,
        _value_cache: *const u16,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _num_seqs: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_prefill_f16(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u16,
        _value_cache: *const u16,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _query_start_loc: *const i32,
        _num_seqs: i64,
        _num_tokens: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_prefill_f16_sync(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u16,
        _value_cache: *const u16,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _query_start_loc: *const i32,
        _num_seqs: i64,
        _num_tokens: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_params_scale() {
        let params = AttentionParams::new(32, 8, 128, 16);
        assert!((params.scale - (1.0 / 128.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn attention_params_gqa() {
        let params = AttentionParams::new(32, 8, 128, 16);
        assert_eq!(params.gqa_ratio(), 4);
        assert_eq!(params.kv_head_for_q_head(0), 0);
        assert_eq!(params.kv_head_for_q_head(3), 0);
        assert_eq!(params.kv_head_for_q_head(4), 1);
        assert_eq!(params.kv_head_for_q_head(31), 7);
    }

    #[test]
    fn attention_params_no_gqa() {
        let params = AttentionParams::new(32, 32, 128, 16);
        assert_eq!(params.gqa_ratio(), 1);
        assert_eq!(params.kv_head_for_q_head(15), 15);
    }

    #[test]
    fn attention_params_num_blocks() {
        let params = AttentionParams::new(32, 8, 128, 16);
        assert_eq!(params.num_blocks_for_seq_len(1), 1);
        assert_eq!(params.num_blocks_for_seq_len(16), 1);
        assert_eq!(params.num_blocks_for_seq_len(17), 2);
        assert_eq!(params.num_blocks_for_seq_len(32), 2);
    }

    #[test]
    fn metadata_decode_construction() {
        let meta = AttentionMetadata::for_decode(
            vec![16, 32, 8],
            vec![vec![0, -1], vec![1, 2], vec![3, -1]],
            2,
        );
        assert_eq!(meta.num_seqs(), 3);
        assert_eq!(meta.num_decode_tokens, 3);
        assert_eq!(meta.num_prefill_tokens, 0);
        assert_eq!(meta.num_tokens(), 3);
        assert_eq!(meta.query_start_loc, vec![0, 1, 2, 3]);
    }

    #[test]
    fn metadata_prefill_construction() {
        let meta = AttentionMetadata::for_prefill(
            vec![16, 32],
            vec![16, 8],
            vec![vec![0, -1], vec![1, 2]],
            2,
        );
        assert_eq!(meta.num_seqs(), 2);
        assert_eq!(meta.num_prefill_tokens, 24);
        assert_eq!(meta.num_decode_tokens, 0);
        assert_eq!(meta.query_start_loc, vec![0, 16, 24]);
    }

    #[test]
    fn metadata_flatten_block_tables() {
        let meta = AttentionMetadata::for_decode(vec![16, 32], vec![vec![0, -1], vec![1, 2]], 4);
        let flat = meta.flatten_block_tables();
        assert_eq!(flat.len(), 2 * 4);
        assert_eq!(flat[0], 0);
        assert_eq!(flat[1], -1);
        assert_eq!(flat[2], -1);
        assert_eq!(flat[3], -1);
        assert_eq!(flat[4], 1);
        assert_eq!(flat[5], 2);
        assert_eq!(flat[6], -1);
        assert_eq!(flat[7], -1);
    }

    #[test]
    fn metadata_default() {
        let meta = AttentionMetadata::default();
        assert_eq!(meta.num_seqs(), 0);
        assert_eq!(meta.num_tokens(), 0);
    }

    #[test]
    fn metadata_new() {
        let meta = AttentionMetadata::new();
        assert_eq!(meta.query_start_loc, vec![0]);
    }

    #[test]
    fn detect_common_prefix_blocks_empty() {
        let meta = AttentionMetadata::new();
        let mut m = meta.clone();
        m.detect_common_prefix_blocks();
        assert_eq!(m.common_prefix_blocks, 0);
    }

    #[test]
    fn detect_common_prefix_blocks_single_seq() {
        let meta = AttentionMetadata::for_decode(
            vec![10],
            vec![vec![0, 1, 2]],
            3,
        );
        let mut m = meta;
        m.detect_common_prefix_blocks();
        assert_eq!(m.common_prefix_blocks, 0);
    }

    #[test]
    fn detect_common_prefix_blocks_shared() {
        let meta = AttentionMetadata::for_decode(
            vec![16, 16],
            vec![vec![0, 1, 2], vec![0, 1, 3]],
            3,
        );
        let mut m = meta;
        m.detect_common_prefix_blocks();
        // First two blocks (0, 1) are shared
        assert_eq!(m.common_prefix_blocks, 2);
    }

    #[test]
    fn detect_common_prefix_blocks_none_shared() {
        let meta = AttentionMetadata::for_decode(
            vec![16, 16],
            vec![vec![0, 1], vec![5, 6]],
            2,
        );
        let mut m = meta;
        m.detect_common_prefix_blocks();
        assert_eq!(m.common_prefix_blocks, 0);
    }

    #[test]
    fn test_sliding_window_ranges_none() {
        let meta = AttentionMetadata::for_decode(
            vec![10, 20],
            vec![vec![0], vec![1]],
            1,
        );
        let ranges = meta.sliding_window_ranges();
        assert_eq!(ranges, vec![(0, 10), (0, 20)]);
    }

    #[test]
    fn test_sliding_window_ranges_with_window() {
        let mut meta = AttentionMetadata::for_decode(
            vec![10, 200, 50],
            vec![vec![0], vec![1], vec![2]],
            1,
        );
        meta.sliding_window = Some(64);
        let ranges = meta.sliding_window_ranges();
        // seq 0: len=10 <= 64, full range (0, 10)
        assert_eq!(ranges[0], (0, 10));
        // seq 1: len=200 > 64, last 64 tokens (136, 200)
        assert_eq!(ranges[1], (136, 200));
        // seq 2: len=50 <= 64, full range (0, 50)
        assert_eq!(ranges[2], (0, 50));
    }

    #[test]
    fn test_num_window_tokens() {
        let mut meta = AttentionMetadata::for_decode(
            vec![10, 200, 50],
            vec![vec![0], vec![1], vec![2]],
            1,
        );
        meta.sliding_window = Some(64);
        // windowed: min(10,64) + min(200,64) + min(50,64) = 10 + 64 + 50 = 124
        assert_eq!(meta.num_window_tokens(), 124);
    }

    #[test]
    fn test_num_window_tokens_no_window() {
        let meta = AttentionMetadata::for_decode(
            vec![10, 200, 50],
            vec![vec![0], vec![1], vec![2]],
            1,
        );
        // without window: sum of all lengths
        assert_eq!(meta.num_window_tokens(), 260);
    }

    #[test]
    fn test_for_decode_with_options() {
        let meta = AttentionMetadata::for_decode_with_options(
            vec![10],
            vec![vec![0, 1]],
            2,
            1,
            Some(32),
        );
        assert_eq!(meta.common_prefix_blocks, 1);
        assert_eq!(meta.sliding_window, Some(32));
    }

    #[cfg(not(has_cuda))]
    mod no_cuda {
        use super::*;

        #[test]
        fn decode_returns_not_available() {
            let result = paged_attention_decode_f16(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn prefill_returns_not_available() {
            let result = paged_attention_prefill_f16(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }
    }
}
