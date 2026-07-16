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
            .map(|&len| if len <= window { (0, len) } else { (len - window, len) })
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

// ── INT8 dequant-attention CPU reference ──────────────────────────────────

/// CPU reference for single-(sequence, head) paged attention over an INT8 KV
/// cache, mirroring the math of `paged_attention_{decode,prefill}_i8`.
///
/// Inputs are dense (not block-addressed) so the oracle validates the
/// dequant + softmax math, independent of the cache's block layout:
/// - `query`: `head_dim` f32 elements (the Q vector, already in f32).
/// - `key_i8` / `value_i8`: `kv_len * head_dim` int8 elements, row-major
///   `[position, dim]`, as written by `cache_write_i8`.
/// - `k_scale` / `v_scale`: per-tensor dequant scales (`x ~= q * scale`).
/// - `scale`: softmax scale (typically `1/sqrt(head_dim)`).
///
/// Computes `logit[t] = scale * Σ_d q[d] * (key_i8[t,d] * k_scale)`, a numerically
/// stable softmax over `t in 0..kv_len`, then
/// `out[d] = Σ_t softmax[t] * (value_i8[t,d] * v_scale)`. Returns `head_dim`
/// f32 outputs. Used as a numeric oracle so the kernel algorithm is verifiable
/// without a GPU.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_i8_reference(
    query: &[f32],
    key_i8: &[i8],
    value_i8: &[i8],
    kv_len: usize,
    head_dim: usize,
    k_scale: f32,
    v_scale: f32,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(query.len(), head_dim);
    assert_eq!(key_i8.len(), kv_len * head_dim);
    assert_eq!(value_i8.len(), kv_len * head_dim);

    if kv_len == 0 {
        return vec![0.0; head_dim];
    }

    // Logits with dequantized keys.
    let mut logits = vec![0.0f32; kv_len];
    for (t, logit) in logits.iter_mut().enumerate() {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            let k = key_i8[t * head_dim + d] as f32 * k_scale;
            dot += query[d] * k;
        }
        *logit = dot * scale;
    }

    // Numerically stable softmax.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_logit).exp();
        exp_sum += *logit;
    }

    // Weighted sum of dequantized values.
    let mut out = vec![0.0f32; head_dim];
    for (t, &w) in logits.iter().enumerate() {
        let weight = if exp_sum > 0.0 { w / exp_sum } else { 0.0 };
        for (d, o) in out.iter_mut().enumerate() {
            let v = value_i8[t * head_dim + d] as f32 * v_scale;
            *o += weight * v;
        }
    }
    out
}

// ── FFI declarations ──────────────────────────────────────────────────────

#[cfg(has_cuda)]
mod ffi {
    use std::os::raw::c_int;

    unsafe extern "C" {
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

        pub fn rllm_paged_attention_decode_fp8(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u8,
            value_cache: *const u8,
            block_tables: *const i32,
            seq_lens: *const i32,
            num_seqs: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
            is_e5m2: c_int,
            stream: usize,
        ) -> c_int;

        pub fn rllm_paged_attention_decode_fp8_sync(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u8,
            value_cache: *const u8,
            block_tables: *const i32,
            seq_lens: *const i32,
            num_seqs: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
            is_e5m2: c_int,
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

        pub fn rllm_paged_attention_prefill_fp8(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u8,
            value_cache: *const u8,
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
            is_e5m2: c_int,
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

        pub fn rllm_paged_attention_prefill_fp8_sync(
            output: *mut u16,
            query: *const u16,
            key_cache: *const u8,
            value_cache: *const u8,
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
            is_e5m2: c_int,
        ) -> c_int;

        pub fn rllm_paged_attention_decode_i8(
            output: *mut u16,
            query: *const u16,
            key_cache: *const i8,
            value_cache: *const i8,
            block_tables: *const i32,
            seq_lens: *const i32,
            num_seqs: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
            k_scale: f32,
            v_scale: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_paged_attention_decode_i8_sync(
            output: *mut u16,
            query: *const u16,
            key_cache: *const i8,
            value_cache: *const i8,
            block_tables: *const i32,
            seq_lens: *const i32,
            num_seqs: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            max_num_blocks_per_seq: i64,
            scale: f32,
            k_scale: f32,
            v_scale: f32,
        ) -> c_int;

        pub fn rllm_paged_attention_prefill_i8(
            output: *mut u16,
            query: *const u16,
            key_cache: *const i8,
            value_cache: *const i8,
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
            k_scale: f32,
            v_scale: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_paged_attention_prefill_i8_sync(
            output: *mut u16,
            query: *const u16,
            key_cache: *const i8,
            value_cache: *const i8,
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
            k_scale: f32,
            v_scale: f32,
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

/// Launch async decode PagedAttention (FP8).
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes.
/// - `block_tables` must have `num_seqs * max_num_blocks_per_seq` elements.
/// - `seq_lens` must have `num_seqs` elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_fp8(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u8,
    value_cache: *const u8,
    block_tables: *const i32,
    seq_lens: *const i32,
    num_seqs: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    is_e5m2: bool,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_decode_fp8(
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
            if is_e5m2 { 1 } else { 0 },
            stream,
        )
    };
    check(rc)
}

/// Synchronous decode PagedAttention for testing (FP8).
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_fp8_sync(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u8,
    value_cache: *const u8,
    block_tables: *const i32,
    seq_lens: *const i32,
    num_seqs: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    is_e5m2: bool,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_decode_fp8_sync(
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
            if is_e5m2 { 1 } else { 0 },
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

/// Launch async prefill PagedAttention (FP8).
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes.
/// - `query_start_loc` must have `num_seqs + 1` elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_prefill_fp8(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u8,
    value_cache: *const u8,
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
    is_e5m2: bool,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_prefill_fp8(
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
            if is_e5m2 { 1 } else { 0 },
            stream,
        )
    };
    check(rc)
}

/// Synchronous prefill PagedAttention for testing (FP8).
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_prefill_fp8_sync(
    output: *mut u16,
    query: *const u16,
    key_cache: *const u8,
    value_cache: *const u8,
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
    is_e5m2: bool,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_prefill_fp8_sync(
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
            if is_e5m2 { 1 } else { 0 },
        )
    };
    check(rc)
}

// ── INT8 PagedAttention ───────────────────────────────────────────────────

/// Launch async decode PagedAttention (INT8 KV cache).
///
/// The K/V cache holds symmetric int8 written by `cache_write_i8`; each element
/// is dequantized on read as `q * k_scale` / `q * v_scale`. Pass the scales
/// carried by [`crate::cache_ops::GpuKVCache`] (default `1.0` until calibrated).
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes.
/// - `block_tables` must have `num_seqs * max_num_blocks_per_seq` elements.
/// - `seq_lens` must have `num_seqs` elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_i8(
    output: *mut u16,
    query: *const u16,
    key_cache: *const i8,
    value_cache: *const i8,
    block_tables: *const i32,
    seq_lens: *const i32,
    num_seqs: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    k_scale: f32,
    v_scale: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_decode_i8(
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
            k_scale,
            v_scale,
            stream,
        )
    };
    check(rc)
}

/// Synchronous decode PagedAttention for testing (INT8).
///
/// # Safety
/// Same invariants as [`paged_attention_decode_i8`].
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_i8_sync(
    output: *mut u16,
    query: *const u16,
    key_cache: *const i8,
    value_cache: *const i8,
    block_tables: *const i32,
    seq_lens: *const i32,
    num_seqs: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    k_scale: f32,
    v_scale: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_decode_i8_sync(
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
            k_scale,
            v_scale,
        )
    };
    check(rc)
}

/// Launch async prefill PagedAttention (INT8 KV cache). See
/// [`paged_attention_decode_i8`] for the dequant scheme and scale semantics.
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes.
/// - `query_start_loc` must have `num_seqs + 1` elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_prefill_i8(
    output: *mut u16,
    query: *const u16,
    key_cache: *const i8,
    value_cache: *const i8,
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
    k_scale: f32,
    v_scale: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_prefill_i8(
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
            k_scale,
            v_scale,
            stream,
        )
    };
    check(rc)
}

/// Synchronous prefill PagedAttention for testing (INT8).
///
/// # Safety
/// Same invariants as [`paged_attention_prefill_i8`].
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_prefill_i8_sync(
    output: *mut u16,
    query: *const u16,
    key_cache: *const i8,
    value_cache: *const i8,
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
    k_scale: f32,
    v_scale: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_paged_attention_prefill_i8_sync(
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
            k_scale,
            v_scale,
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
    pub fn paged_attention_decode_fp8(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u8,
        _value_cache: *const u8,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _num_seqs: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
        _is_e5m2: bool,
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
    pub fn paged_attention_decode_fp8_sync(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u8,
        _value_cache: *const u8,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _num_seqs: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
        _is_e5m2: bool,
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
    pub fn paged_attention_prefill_fp8(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u8,
        _value_cache: *const u8,
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
        _is_e5m2: bool,
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

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_prefill_fp8_sync(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const u8,
        _value_cache: *const u8,
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
        _is_e5m2: bool,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_decode_i8(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const i8,
        _value_cache: *const i8,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _num_seqs: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_decode_i8_sync(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const i8,
        _value_cache: *const i8,
        _block_tables: *const i32,
        _seq_lens: *const i32,
        _num_seqs: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _max_num_blocks_per_seq: i64,
        _scale: f32,
        _k_scale: f32,
        _v_scale: f32,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_prefill_i8(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const i8,
        _value_cache: *const i8,
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
        _k_scale: f32,
        _v_scale: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_prefill_i8_sync(
        _output: *mut u16,
        _query: *const u16,
        _key_cache: *const i8,
        _value_cache: *const i8,
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
        _k_scale: f32,
        _v_scale: f32,
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
        let meta = AttentionMetadata::for_decode(vec![10], vec![vec![0, 1, 2]], 3);
        let mut m = meta;
        m.detect_common_prefix_blocks();
        assert_eq!(m.common_prefix_blocks, 0);
    }

    #[test]
    fn detect_common_prefix_blocks_shared() {
        let meta =
            AttentionMetadata::for_decode(vec![16, 16], vec![vec![0, 1, 2], vec![0, 1, 3]], 3);
        let mut m = meta;
        m.detect_common_prefix_blocks();
        // First two blocks (0, 1) are shared
        assert_eq!(m.common_prefix_blocks, 2);
    }

    #[test]
    fn detect_common_prefix_blocks_none_shared() {
        let meta = AttentionMetadata::for_decode(vec![16, 16], vec![vec![0, 1], vec![5, 6]], 2);
        let mut m = meta;
        m.detect_common_prefix_blocks();
        assert_eq!(m.common_prefix_blocks, 0);
    }

    #[test]
    fn test_sliding_window_ranges_none() {
        let meta = AttentionMetadata::for_decode(vec![10, 20], vec![vec![0], vec![1]], 1);
        let ranges = meta.sliding_window_ranges();
        assert_eq!(ranges, vec![(0, 10), (0, 20)]);
    }

    #[test]
    fn test_sliding_window_ranges_with_window() {
        let mut meta =
            AttentionMetadata::for_decode(vec![10, 200, 50], vec![vec![0], vec![1], vec![2]], 1);
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
        let mut meta =
            AttentionMetadata::for_decode(vec![10, 200, 50], vec![vec![0], vec![1], vec![2]], 1);
        meta.sliding_window = Some(64);
        // windowed: min(10,64) + min(200,64) + min(50,64) = 10 + 64 + 50 = 124
        assert_eq!(meta.num_window_tokens(), 124);
    }

    #[test]
    fn test_num_window_tokens_no_window() {
        let meta =
            AttentionMetadata::for_decode(vec![10, 200, 50], vec![vec![0], vec![1], vec![2]], 1);
        // without window: sum of all lengths
        assert_eq!(meta.num_window_tokens(), 260);
    }

    #[test]
    fn test_for_decode_with_options() {
        let meta =
            AttentionMetadata::for_decode_with_options(vec![10], vec![vec![0, 1]], 2, 1, Some(32));
        assert_eq!(meta.common_prefix_blocks, 1);
        assert_eq!(meta.sliding_window, Some(32));
    }

    #[test]
    fn i8_reference_single_position_returns_dequant_value() {
        // With one KV position, softmax is 1.0 regardless of the logit, so the
        // output is exactly the dequantized value vector.
        let head_dim = 4;
        let query = vec![0.5, -0.5, 1.0, 0.0];
        let key_i8 = vec![10i8, -20, 30, 5];
        let value_i8 = vec![4i8, 8, -16, 32];
        let v_scale = 0.1;
        let out = paged_attention_i8_reference(
            &query, &key_i8, &value_i8, 1, head_dim, 0.05, v_scale, 0.5,
        );
        for d in 0..head_dim {
            assert!((out[d] - value_i8[d] as f32 * v_scale).abs() < 1e-6, "d={d}");
        }
    }

    #[test]
    fn i8_reference_equal_logits_average_values() {
        // Two positions with an all-zero query → both logits 0 → softmax 0.5/0.5,
        // output is the mean of the two dequantized value vectors.
        let head_dim = 2;
        let query = vec![0.0, 0.0];
        let key_i8 = vec![1i8, 2, 3, 4]; // irrelevant when query is zero
        let value_i8 = vec![10i8, 20, 30, 40];
        let v_scale = 0.5;
        let out = paged_attention_i8_reference(
            &query, &key_i8, &value_i8, 2, head_dim, 0.1, v_scale, 1.0,
        );
        // mean of (10,20) and (30,40) scaled by 0.5 → (20*0.5, 30*0.5) = (10, 15)
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn i8_reference_matches_hand_computed_softmax() {
        // Two positions, head_dim 1, easy to compute by hand.
        let head_dim = 1;
        let query = vec![2.0];
        let k_scale = 0.5;
        let v_scale = 1.0;
        let soft_scale = 1.0;
        let key_i8 = vec![1i8, 3]; // dequant keys: 0.5, 1.5
        let value_i8 = vec![10i8, 20];
        // logits = scale * q * (k*k_scale): l0 = 2*0.5 = 1.0, l1 = 2*1.5 = 3.0
        let out = paged_attention_i8_reference(
            &query, &key_i8, &value_i8, 2, head_dim, k_scale, v_scale, soft_scale,
        );
        let (l0, l1) = (1.0f32, 3.0f32);
        let (e0, e1) = ((l0 - l1).exp(), 1.0f32); // stable: subtract max=l1
        let sum = e0 + e1;
        let expected = (e0 / sum) * 10.0 + (e1 / sum) * 20.0;
        assert!((out[0] - expected).abs() < 1e-5, "out={} expected={}", out[0], expected);
    }

    #[test]
    fn i8_reference_empty_is_zero() {
        let out = paged_attention_i8_reference(&[1.0, 2.0], &[], &[], 0, 2, 1.0, 1.0, 1.0);
        assert_eq!(out, vec![0.0, 0.0]);
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

    #[cfg(has_cuda)]
    mod with_cuda {
        use super::*;
        use crate::cache_ops::{gpu_alloc, gpu_free, gpu_memcpy_d2h, gpu_memcpy_h2d};

        #[test]
        fn prefill_f16_attends_cached_prefix() {
            // One new query token follows two cached tokens. With an all-zero
            // query the attention weights are uniform, so values [2, 4, 6]
            // must produce 4. The old kernel used query-chunk-relative length
            // and returned only the first cached value.
            let half_bits = |value: f32| half::f16::from_f32(value).to_bits();
            let key_host = vec![half_bits(0.0); 4];
            let value_host = vec![half_bits(2.0), half_bits(4.0), half_bits(6.0), half_bits(0.0)];
            let query_host = [half_bits(0.0)];
            let block_tables = [0i32];
            let seq_lens = [3i32];
            let query_start = [0i32, 1i32];

            unsafe {
                let key = gpu_alloc(8).unwrap() as *mut u16;
                let value = gpu_alloc(8).unwrap() as *mut u16;
                let query = gpu_alloc(2).unwrap() as *mut u16;
                let block_table = gpu_alloc(4).unwrap() as *mut i32;
                let seq_len = gpu_alloc(4).unwrap() as *mut i32;
                let query_loc = gpu_alloc(8).unwrap() as *mut i32;
                let output = gpu_alloc(2).unwrap() as *mut u16;
                gpu_memcpy_h2d(key as *mut u8, key_host.as_ptr() as *const u8, 8).unwrap();
                gpu_memcpy_h2d(value as *mut u8, value_host.as_ptr() as *const u8, 8).unwrap();
                gpu_memcpy_h2d(query as *mut u8, query_host.as_ptr() as *const u8, 2).unwrap();
                gpu_memcpy_h2d(block_table as *mut u8, block_tables.as_ptr() as *const u8, 4)
                    .unwrap();
                gpu_memcpy_h2d(seq_len as *mut u8, seq_lens.as_ptr() as *const u8, 4).unwrap();
                gpu_memcpy_h2d(query_loc as *mut u8, query_start.as_ptr() as *const u8, 8).unwrap();
                paged_attention_prefill_f16_sync(
                    output,
                    query,
                    key,
                    value,
                    block_table,
                    seq_len,
                    query_loc,
                    1,
                    1,
                    1,
                    1,
                    1,
                    4,
                    1,
                    1.0,
                )
                .unwrap();
                let mut result = [0u16];
                gpu_memcpy_d2h(result.as_mut_ptr() as *mut u8, output as *const u8, 2).unwrap();
                assert!((half::f16::from_bits(result[0]).to_f32() - 4.0).abs() < 0.01);
                for pointer in [
                    key as *mut u8,
                    value as *mut u8,
                    query as *mut u8,
                    block_table as *mut u8,
                    seq_len as *mut u8,
                    query_loc as *mut u8,
                    output as *mut u8,
                ] {
                    gpu_free(pointer).unwrap();
                }
            }
        }

        #[test]
        fn decode_i8_matches_cpu_reference() {
            // 1 seq, 1 q head, 1 kv head, head_dim 4, block_size 4, seq_len 2,
            // 1 physical block. Validates the int8 decode kernel against the CPU
            // dequant-attention oracle.
            let head_dim = 4usize;
            let block_size = 4usize;
            let kv_len = 2usize;
            let num_blocks = 1i64;
            let k_scale = 0.05f32;
            let v_scale = 0.1f32;
            let soft_scale = 1.0 / (head_dim as f32).sqrt();

            // f16 query (kernel reads f16); use the f16-rounded value for the oracle.
            let q_f32 = [0.5f32, -0.3, 1.2, 0.1];
            let q_u16: Vec<u16> = q_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let q_oracle: Vec<f32> =
                q_f32.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();

            // Dense int8 K/V, row-major [pos, d].
            let key_i8: Vec<i8> = vec![10, -20, 30, 5, -7, 40, -3, 12];
            let value_i8: Vec<i8> = vec![4, 8, -16, 32, -10, 22, 6, -2];

            // Scatter dense [pos,d] into NHD cache buffer: idx = d*block_size + pos
            // (block 0, kv_head 0). Unused slots stay 0.
            let cache_elems = (num_blocks as usize) * head_dim * block_size;
            let mut key_cache_host = vec![0i8; cache_elems];
            let mut val_cache_host = vec![0i8; cache_elems];
            for pos in 0..kv_len {
                for d in 0..head_dim {
                    let idx = d * block_size + pos;
                    key_cache_host[idx] = key_i8[pos * head_dim + d];
                    val_cache_host[idx] = value_i8[pos * head_dim + d];
                }
            }

            let block_tables: Vec<i32> = vec![0];
            let seq_lens: Vec<i32> = vec![kv_len as i32];

            // Device buffers.
            let key_cache = unsafe { gpu_alloc(cache_elems).unwrap() } as *mut i8;
            let value_cache = unsafe { gpu_alloc(cache_elems).unwrap() } as *mut i8;
            let query = unsafe { gpu_alloc(head_dim * 2).unwrap() } as *mut u16;
            let bt_dev = unsafe { gpu_alloc(block_tables.len() * 4).unwrap() } as *mut i32;
            let sl_dev = unsafe { gpu_alloc(seq_lens.len() * 4).unwrap() } as *mut i32;
            let out_dev = unsafe { gpu_alloc(head_dim * 2).unwrap() } as *mut u16;

            unsafe {
                gpu_memcpy_h2d(
                    key_cache as *mut u8,
                    key_cache_host.as_ptr() as *const u8,
                    cache_elems,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    value_cache as *mut u8,
                    val_cache_host.as_ptr() as *const u8,
                    cache_elems,
                )
                .unwrap();
                gpu_memcpy_h2d(query as *mut u8, q_u16.as_ptr() as *const u8, head_dim * 2)
                    .unwrap();
                gpu_memcpy_h2d(
                    bt_dev as *mut u8,
                    block_tables.as_ptr() as *const u8,
                    block_tables.len() * 4,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    sl_dev as *mut u8,
                    seq_lens.as_ptr() as *const u8,
                    seq_lens.len() * 4,
                )
                .unwrap();

                paged_attention_decode_i8_sync(
                    out_dev,
                    query as *const u16,
                    key_cache as *const i8,
                    value_cache as *const i8,
                    bt_dev as *const i32,
                    sl_dev as *const i32,
                    1, // num_seqs
                    1, // num_q_heads
                    1, // num_kv_heads
                    head_dim as i64,
                    block_size as i64,
                    1, // max_num_blocks_per_seq
                    soft_scale,
                    k_scale,
                    v_scale,
                )
                .expect("decode_i8_sync failed");
            }

            let mut out_u16 = vec![0u16; head_dim];
            unsafe {
                gpu_memcpy_d2h(out_u16.as_mut_ptr() as *mut u8, out_dev as *const u8, head_dim * 2)
                    .unwrap();
            }
            let out_gpu: Vec<f32> =
                out_u16.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();

            let out_ref = paged_attention_i8_reference(
                &q_oracle, &key_i8, &value_i8, kv_len, head_dim, k_scale, v_scale, soft_scale,
            );

            // f16 output + fp32 accumulation: allow a small tolerance.
            for d in 0..head_dim {
                assert!(
                    (out_gpu[d] - out_ref[d]).abs() < 5e-3,
                    "d={d} gpu={} ref={}",
                    out_gpu[d],
                    out_ref[d]
                );
            }

            unsafe {
                gpu_free(key_cache as *mut u8).unwrap();
                gpu_free(value_cache as *mut u8).unwrap();
                gpu_free(query as *mut u8).unwrap();
                gpu_free(bt_dev as *mut u8).unwrap();
                gpu_free(sl_dev as *mut u8).unwrap();
                gpu_free(out_dev as *mut u8).unwrap();
            }
        }

        #[test]
        fn prefill_i8_matches_cpu_reference() {
            // 1 seq, 2 tokens (prefill), 1 q head, 1 kv head, head_dim 4, block_size 4, seq_len 2,
            // 1 physical block. Validates the int8 prefill kernel against the CPU
            // dequant-attention oracle.
            let head_dim = 4usize;
            let block_size = 4usize;
            let kv_len = 2usize;
            let num_tokens = 2usize;
            let num_blocks = 1i64;
            let k_scale = 0.05f32;
            let v_scale = 0.1f32;
            let soft_scale = 1.0 / (head_dim as f32).sqrt();

            // f16 query (kernel reads f16); use the f16-rounded value for the oracle.
            // 2 tokens, head_dim 4.
            let q_f32 = [0.5f32, -0.3, 1.2, 0.1, 0.2, 0.8, -0.5, 0.4];
            let q_u16: Vec<u16> = q_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let q_oracle: Vec<f32> =
                q_f32.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();

            // Dense int8 K/V, row-major [pos, d].
            let key_i8: Vec<i8> = vec![10, -20, 30, 5, -7, 40, -3, 12];
            let value_i8: Vec<i8> = vec![4, 8, -16, 32, -10, 22, 6, -2];

            // Scatter dense [pos,d] into NHD cache buffer: idx = d*block_size + pos
            // (block 0, kv_head 0). Unused slots stay 0.
            let cache_elems = (num_blocks as usize) * head_dim * block_size;
            let mut key_cache_host = vec![0i8; cache_elems];
            let mut val_cache_host = vec![0i8; cache_elems];
            for pos in 0..kv_len {
                for d in 0..head_dim {
                    let idx = d * block_size + pos;
                    key_cache_host[idx] = key_i8[pos * head_dim + d];
                    val_cache_host[idx] = value_i8[pos * head_dim + d];
                }
            }

            let block_tables: Vec<i32> = vec![0];
            let seq_lens: Vec<i32> = vec![kv_len as i32];
            let query_start_loc: Vec<i32> = vec![0, num_tokens as i32];

            // Device buffers.
            let key_cache = unsafe { gpu_alloc(cache_elems).unwrap() } as *mut i8;
            let value_cache = unsafe { gpu_alloc(cache_elems).unwrap() } as *mut i8;
            let query = unsafe { gpu_alloc(num_tokens * head_dim * 2).unwrap() } as *mut u16;
            let bt_dev = unsafe { gpu_alloc(block_tables.len() * 4).unwrap() } as *mut i32;
            let sl_dev = unsafe { gpu_alloc(seq_lens.len() * 4).unwrap() } as *mut i32;
            let qsl_dev = unsafe { gpu_alloc(query_start_loc.len() * 4).unwrap() } as *mut i32;
            let out_dev = unsafe { gpu_alloc(num_tokens * head_dim * 2).unwrap() } as *mut u16;

            unsafe {
                gpu_memcpy_h2d(
                    key_cache as *mut u8,
                    key_cache_host.as_ptr() as *const u8,
                    cache_elems,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    value_cache as *mut u8,
                    val_cache_host.as_ptr() as *const u8,
                    cache_elems,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    query as *mut u8,
                    q_u16.as_ptr() as *const u8,
                    num_tokens * head_dim * 2,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    bt_dev as *mut u8,
                    block_tables.as_ptr() as *const u8,
                    block_tables.len() * 4,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    sl_dev as *mut u8,
                    seq_lens.as_ptr() as *const u8,
                    seq_lens.len() * 4,
                )
                .unwrap();
                gpu_memcpy_h2d(
                    qsl_dev as *mut u8,
                    query_start_loc.as_ptr() as *const u8,
                    query_start_loc.len() * 4,
                )
                .unwrap();

                paged_attention_prefill_i8_sync(
                    out_dev,
                    query as *const u16,
                    key_cache as *const i8,
                    value_cache as *const i8,
                    bt_dev as *const i32,
                    sl_dev as *const i32,
                    qsl_dev as *const i32,
                    1, // num_seqs
                    num_tokens as i64,
                    1, // num_q_heads
                    1, // num_kv_heads
                    head_dim as i64,
                    block_size as i64,
                    1, // max_num_blocks_per_seq
                    soft_scale,
                    k_scale,
                    v_scale,
                )
                .expect("prefill_i8_sync failed");
            }

            let mut out_u16 = vec![0u16; num_tokens * head_dim];
            unsafe {
                gpu_memcpy_d2h(
                    out_u16.as_mut_ptr() as *mut u8,
                    out_dev as *const u8,
                    num_tokens * head_dim * 2,
                )
                .unwrap();
            }
            let out_gpu: Vec<f32> =
                out_u16.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();

            // Oracle output for token 0 (kv_len = 1)
            let out_ref_0 = paged_attention_i8_reference(
                &q_oracle[0..4],
                &key_i8[0..4],
                &value_i8[0..4],
                1,
                head_dim,
                k_scale,
                v_scale,
                soft_scale,
            );

            // Oracle output for token 1 (kv_len = 2)
            let out_ref_1 = paged_attention_i8_reference(
                &q_oracle[4..8],
                &key_i8[0..8],
                &value_i8[0..8],
                2,
                head_dim,
                k_scale,
                v_scale,
                soft_scale,
            );

            // Verify token 0
            for d in 0..head_dim {
                assert!(
                    (out_gpu[d] - out_ref_0[d]).abs() < 5e-3,
                    "token 0, d={d} gpu={} ref={}",
                    out_gpu[d],
                    out_ref_0[d]
                );
            }

            // Verify token 1
            for d in 0..head_dim {
                assert!(
                    (out_gpu[head_dim + d] - out_ref_1[d]).abs() < 5e-3,
                    "token 1, d={d} gpu={} ref={}",
                    out_gpu[head_dim + d],
                    out_ref_1[d]
                );
            }

            unsafe {
                gpu_free(key_cache as *mut u8).unwrap();
                gpu_free(value_cache as *mut u8).unwrap();
                gpu_free(query as *mut u8).unwrap();
                gpu_free(bt_dev as *mut u8).unwrap();
                gpu_free(sl_dev as *mut u8).unwrap();
                gpu_free(qsl_dev as *mut u8).unwrap();
                gpu_free(out_dev as *mut u8).unwrap();
            }
        }
    }
}
