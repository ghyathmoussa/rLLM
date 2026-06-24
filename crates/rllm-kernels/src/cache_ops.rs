//! KV cache kernels and GPU memory management.
//!
//! Provides:
//! - Cache write: writes new K/V data into physical cache at slot-mapped positions
//! - Cache block copy: copies full blocks for prefix sharing/forking
//! - Cache zero: zeroes out cache memory
//! - GPU memory alloc/free wrappers
//! - `GpuKVCache` type for per-layer K/V tensor allocation

use rllm_core::ids::BlockId;

use crate::cuda::CudaKernelError;

// ── FFI declarations ──────────────────────────────────────────────────────

#[cfg(has_cuda)]
mod ffi {
    use std::os::raw::c_int;

    unsafe extern "C" {
        // Cache write (FP16)
        pub fn rllm_cache_write_f16(
            key_cache: *mut u16,      // __half*
            value_cache: *mut u16,    // __half*
            new_key: *const u16,      // const __half*
            new_value: *const u16,    // const __half*
            slot_mapping: *const i64, // const int64_t*
            num_tokens: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            num_blocks: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_cache_write_fp8(
            key_cache: *mut u8,
            value_cache: *mut u8,
            new_key: *const u16,
            new_value: *const u16,
            slot_mapping: *const i64,
            num_tokens: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            num_blocks: i64,
            is_e5m2: c_int,
            stream: usize,
        ) -> c_int;

        pub fn rllm_cache_write_f16_sync(
            key_cache: *mut u16,
            value_cache: *mut u16,
            new_key: *const u16,
            new_value: *const u16,
            slot_mapping: *const i64,
            num_tokens: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            num_blocks: i64,
        ) -> c_int;

        pub fn rllm_cache_write_fp8_sync(
            key_cache: *mut u8,
            value_cache: *mut u8,
            new_key: *const u16,
            new_value: *const u16,
            slot_mapping: *const i64,
            num_tokens: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            num_blocks: i64,
            is_e5m2: c_int,
        ) -> c_int;

        pub fn rllm_cache_write_i8(
            key_cache: *mut i8,
            value_cache: *mut i8,
            new_key: *const u16,
            new_value: *const u16,
            slot_mapping: *const i64,
            num_tokens: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            num_blocks: i64,
            k_scale: f32,
            v_scale: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_cache_write_i8_sync(
            key_cache: *mut i8,
            value_cache: *mut i8,
            new_key: *const u16,
            new_value: *const u16,
            slot_mapping: *const i64,
            num_tokens: i64,
            num_kv_heads: i64,
            head_dim: i64,
            block_size: i64,
            num_blocks: i64,
            k_scale: f32,
            v_scale: f32,
        ) -> c_int;

        // Cache block copy
        pub fn rllm_cache_block_copy(
            src: *const u8,
            dst: *mut u8,
            block_nbytes: i64,
            num_blocks: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_cache_block_copy_sync(
            src: *const u8,
            dst: *mut u8,
            block_nbytes: i64,
            num_blocks: i64,
        ) -> c_int;

        // Cache zero
        pub fn rllm_cache_zero(ptr: *mut u8, nbytes: i64, stream: usize) -> c_int;
        pub fn rllm_cache_zero_sync(ptr: *mut u8, nbytes: i64) -> c_int;

        // GPU memory management
        pub fn rllm_gpu_alloc(ptr: *mut *mut std::ffi::c_void, nbytes: i64) -> c_int;
        pub fn rllm_gpu_free(ptr: *mut std::ffi::c_void) -> c_int;
        pub fn rllm_gpu_alloc_host(ptr: *mut *mut std::ffi::c_void, nbytes: i64) -> c_int;
        pub fn rllm_gpu_free_host(ptr: *mut std::ffi::c_void) -> c_int;

        // Host <-> device memory copies (host memcpy cannot touch device memory).
        pub fn rllm_gpu_memcpy_h2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, nbytes: i64) -> c_int;
        pub fn rllm_gpu_memcpy_d2h(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, nbytes: i64) -> c_int;
    }
}

#[cfg(has_cuda)]
fn check(rc: i32) -> Result<(), CudaKernelError> {
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
}

/// CPU reference for the INT8 KV cache quantization, mirroring the CUDA
/// `float_to_i8_scaled` device function used by `cache_write_i8`.
///
/// Symmetric per-tensor quantization: `q = round(x / scale)` clamped to
/// `[-127, 127]`. Dequantization is `q * scale`. Used as a numeric oracle in
/// tests so the algorithm is verifiable without a GPU.
pub fn quantize_kv_i8_reference(val: f32, scale: f32) -> i8 {
    let q = (val / scale).round();
    q.clamp(-127.0, 127.0) as i8
}

// ── Cache Write ───────────────────────────────────────────────────────────

/// Launch async cache write (FP16): writes new K/V into physical cache.
///
/// # Safety
/// - All pointers must be valid device pointers.
/// - `slot_mapping` must have `num_tokens` entries.
/// - `new_key` and `new_value` must have `num_tokens * num_kv_heads * head_dim` elements.
/// - `key_cache` and `value_cache` must be large enough for the layout.
/// Launch async cache write (FP16): writes new K/V into physical cache.
///
/// # Safety
/// - All pointers must be valid device pointers.
/// - `slot_mapping` must have `num_tokens` entries.
/// - `new_key` and `new_value` must have `num_tokens * num_kv_heads * head_dim` elements.
/// - `key_cache` and `value_cache` must be large enough for the layout.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cache_write_f16(
    key_cache: *mut u16,
    value_cache: *mut u16,
    new_key: *const u16,
    new_value: *const u16,
    slot_mapping: *const i64,
    num_tokens: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_cache_write_f16(
            key_cache,
            value_cache,
            new_key,
            new_value,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            stream,
        )
    };
    check(rc)
}

/// Synchronous cache write for testing.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cache_write_f16_sync(
    key_cache: *mut u16,
    value_cache: *mut u16,
    new_key: *const u16,
    new_value: *const u16,
    slot_mapping: *const i64,
    num_tokens: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_cache_write_f16_sync(
            key_cache,
            value_cache,
            new_key,
            new_value,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
        )
    };
    check(rc)
}

/// Launch async cache write (FP8): writes new K/V into physical cache, converting from FP16.
///
/// # Safety
/// - All pointers must be valid device pointers.
/// - `slot_mapping` must have `num_tokens` entries.
/// - `new_key` and `new_value` must have `num_tokens * num_kv_heads * head_dim` elements.
/// - `key_cache` and `value_cache` must be large enough for the layout.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cache_write_fp8(
    key_cache: *mut u8,
    value_cache: *mut u8,
    new_key: *const u16,
    new_value: *const u16,
    slot_mapping: *const i64,
    num_tokens: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
    is_e5m2: bool,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_cache_write_fp8(
            key_cache,
            value_cache,
            new_key,
            new_value,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            if is_e5m2 { 1 } else { 0 },
            stream,
        )
    };
    check(rc)
}

/// Synchronous cache write for testing (FP8).
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cache_write_fp8_sync(
    key_cache: *mut u8,
    value_cache: *mut u8,
    new_key: *const u16,
    new_value: *const u16,
    slot_mapping: *const i64,
    num_tokens: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
    is_e5m2: bool,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_cache_write_fp8_sync(
            key_cache,
            value_cache,
            new_key,
            new_value,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            if is_e5m2 { 1 } else { 0 },
        )
    };
    check(rc)
}

/// Launch async cache write (INT8): writes new K/V into physical cache, converting from FP16.
///
/// Uses symmetric per-tensor quantization `round(x / scale)` clamped to
/// `[-127, 127]` (mirrors vLLM's `scaled_int8_quant`), storing one byte per
/// scalar. `k_scale`/`v_scale` are the key/value dequantization scales; pass the
/// values carried by [`GpuKVCache`] (default `1.0` until calibration lands — see
/// [`GpuKVCache::k_scale`]). The matching paged-attention INT8 read kernel
/// (Layer 2) must dequantize with the same scales.
///
/// # Safety
/// - All pointers must be valid device pointers.
/// - `slot_mapping` must have `num_tokens` entries.
/// - `new_key` and `new_value` must have `num_tokens * num_kv_heads * head_dim` elements.
/// - `key_cache` and `value_cache` must be large enough for the layout.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cache_write_i8(
    key_cache: *mut i8,
    value_cache: *mut i8,
    new_key: *const u16,
    new_value: *const u16,
    slot_mapping: *const i64,
    num_tokens: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
    k_scale: f32,
    v_scale: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_cache_write_i8(
            key_cache,
            value_cache,
            new_key,
            new_value,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            k_scale,
            v_scale,
            stream,
        )
    };
    check(rc)
}

/// Synchronous cache write for testing (INT8). See [`cache_write_i8`] for the
/// quantization scheme and scale semantics.
///
/// # Safety
/// Same invariants as [`cache_write_i8`].
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cache_write_i8_sync(
    key_cache: *mut i8,
    value_cache: *mut i8,
    new_key: *const u16,
    new_value: *const u16,
    slot_mapping: *const i64,
    num_tokens: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
    k_scale: f32,
    v_scale: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_cache_write_i8_sync(
            key_cache,
            value_cache,
            new_key,
            new_value,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            k_scale,
            v_scale,
        )
    };
    check(rc)
}

// ── Cache Block Copy ──────────────────────────────────────────────────────

/// Launch async cache block copy for prefix sharing/forking.
///
/// # Safety
/// - `src` and `dst` must be valid device pointers with `block_nbytes * num_blocks` bytes.
#[cfg(has_cuda)]
pub unsafe fn cache_block_copy(
    src: *const u8,
    dst: *mut u8,
    block_nbytes: i64,
    num_blocks: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_cache_block_copy(src, dst, block_nbytes, num_blocks, stream) };
    check(rc)
}

/// Synchronous cache block copy for testing.
#[cfg(has_cuda)]
pub unsafe fn cache_block_copy_sync(
    src: *const u8,
    dst: *mut u8,
    block_nbytes: i64,
    num_blocks: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_cache_block_copy_sync(src, dst, block_nbytes, num_blocks) };
    check(rc)
}

// ── Cache Zero ────────────────────────────────────────────────────────────

/// Launch async cache zero.
///
/// # Safety
/// - `ptr` must be a valid device pointer with at least `nbytes` bytes.
#[cfg(has_cuda)]
pub unsafe fn cache_zero(ptr: *mut u8, nbytes: i64, stream: usize) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_cache_zero(ptr, nbytes, stream) };
    check(rc)
}

/// Synchronous cache zero for testing.
#[cfg(has_cuda)]
pub unsafe fn cache_zero_sync(ptr: *mut u8, nbytes: i64) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_cache_zero_sync(ptr, nbytes) };
    check(rc)
}

// ── GPU Memory Management ─────────────────────────────────────────────────

/// Allocate GPU device memory.
///
/// # Safety
/// - Returns a raw device pointer. Caller must free with [`gpu_free`].
#[cfg(has_cuda)]
pub unsafe fn gpu_alloc(nbytes: usize) -> Result<*mut u8, CudaKernelError> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let rc = unsafe { ffi::rllm_gpu_alloc(&mut ptr, nbytes as i64) };
    check(rc)?;
    Ok(ptr as *mut u8)
}

/// Free GPU device memory.
///
/// # Safety
/// - `ptr` must have been allocated by [`gpu_alloc`] and not already freed.
#[cfg(has_cuda)]
pub unsafe fn gpu_free(ptr: *mut u8) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_gpu_free(ptr as *mut std::ffi::c_void) };
    check(rc)
}

/// Copy `nbytes` from host memory to a device pointer (cudaMemcpy H2D).
///
/// # Safety
/// - `dst` must be a valid device pointer with at least `nbytes` allocated.
/// - `src` must be a valid host pointer with at least `nbytes` readable.
#[cfg(has_cuda)]
pub unsafe fn gpu_memcpy_h2d(dst: *mut u8, src: *const u8, nbytes: usize) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_gpu_memcpy_h2d(
            dst as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            nbytes as i64,
        )
    };
    check(rc)
}

/// Copy `nbytes` from a device pointer to host memory (cudaMemcpy D2H).
///
/// # Safety
/// - `src` must be a valid device pointer with at least `nbytes` readable.
/// - `dst` must be a valid host pointer with at least `nbytes` writable.
#[cfg(has_cuda)]
pub unsafe fn gpu_memcpy_d2h(dst: *mut u8, src: *const u8, nbytes: usize) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_gpu_memcpy_d2h(
            dst as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            nbytes as i64,
        )
    };
    check(rc)
}

// ── GpuKVCache ────────────────────────────────────────────────────────────

/// Physical GPU KV cache holding per-layer key and value tensors.
///
/// Each layer has its own K and V tensor pair. The layout is NHD:
/// - K shape: `[num_blocks, num_kv_heads, head_dim, block_size]`
/// - V shape: `[num_blocks, num_kv_heads, head_dim, block_size]`
pub struct GpuKVCache {
    /// Per-layer (key_ptr, value_ptr) pairs.
    layer_ptrs: Vec<(*mut u8, *mut u8)>,
    /// Per-layer (key_nbytes, value_nbytes).
    layer_sizes: Vec<(usize, usize)>,
    /// Number of blocks.
    num_blocks: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Dimension of each attention head.
    head_dim: usize,
    /// Tokens per block.
    block_size: usize,
    /// Bytes per scalar element.
    #[allow(dead_code)]
    element_size: usize,
    /// Cache element data type.
    dtype: rllm_core::dtype::DType,
    /// Per-layer key dequantization scale for INT8 caches (`x ~= q * k_scale`).
    /// Defaults to `1.0` per layer (uncalibrated). Ignored for non-INT8 dtypes.
    k_scales: Vec<f32>,
    /// Per-layer value dequantization scale for INT8 caches (`x ~= q * v_scale`).
    /// Defaults to `1.0` per layer (uncalibrated). Ignored for non-INT8 dtypes.
    v_scales: Vec<f32>,
}

/// Warn once (process-wide) that the INT8 KV cache is running uncalibrated.
#[cfg(has_cuda)]
fn warn_uncalibrated_int8_kv_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "INT8 KV cache is using uncalibrated scales (k_scale=v_scale=1.0 per layer); \
             K/V activations are quantized as round(x) clamped to [-127, 127], \
             which clips magnitudes > 127. Provide calibrated k_scale/v_scale via \
             GpuKVCache::set_all_kv_scales for accurate results."
        );
    });
}

unsafe impl Send for GpuKVCache {}
unsafe impl Sync for GpuKVCache {}

impl GpuKVCache {
    /// Create a new GPU KV cache, allocating device memory for all layers.
    ///
    /// Only available when CUDA is present.
    #[cfg(has_cuda)]
    pub fn new(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: rllm_core::dtype::DType,
    ) -> Result<Self, CudaKernelError> {
        let element_size = dtype.bytes_per_scalar();
        let kv_bytes_per_layer = num_blocks * num_kv_heads * head_dim * block_size * element_size;
        let mut layer_ptrs = Vec::with_capacity(num_layers);
        let layer_sizes = vec![(kv_bytes_per_layer, kv_bytes_per_layer); num_layers];

        for _ in 0..num_layers {
            unsafe {
                let key_ptr = gpu_alloc(kv_bytes_per_layer)?;
                let value_ptr = gpu_alloc(kv_bytes_per_layer)?;
                // Zero-initialize
                cache_zero(key_ptr, kv_bytes_per_layer as i64, 0)?;
                cache_zero(value_ptr, kv_bytes_per_layer as i64, 0)?;
                layer_ptrs.push((key_ptr, value_ptr));
            }
        }

        // INT8 caches start with uncalibrated unit scales (vLLM's default). Warn
        // once so the operator knows the cache is lossy until scales are set.
        if dtype == rllm_core::dtype::DType::INT8 {
            warn_uncalibrated_int8_kv_once();
        }

        Ok(Self {
            layer_ptrs,
            layer_sizes,
            num_blocks,
            num_kv_heads,
            head_dim,
            block_size,
            element_size,
            dtype,
            k_scales: vec![1.0; num_layers],
            v_scales: vec![1.0; num_layers],
        })
    }

    /// Get the key tensor device pointer for a layer.
    pub fn key_ptr(&self, layer: usize) -> *const u8 {
        self.layer_ptrs[layer].0
    }

    /// Get the value tensor device pointer for a layer.
    pub fn value_ptr(&self, layer: usize) -> *const u8 {
        self.layer_ptrs[layer].1
    }

    /// Key tensor shape: `[num_blocks, num_kv_heads, head_dim, block_size]`.
    pub fn key_shape(&self) -> [usize; 4] {
        [self.num_blocks, self.num_kv_heads, self.head_dim, self.block_size]
    }

    /// Value tensor shape: `[num_blocks, num_kv_heads, head_dim, block_size]`.
    pub fn value_shape(&self) -> [usize; 4] {
        self.key_shape()
    }

    /// Data type of the cache.
    pub fn dtype(&self) -> rllm_core::dtype::DType {
        self.dtype
    }

    /// Per-layer key dequantization scale for INT8 caches (`x ~= q * k_scale`).
    /// `1.0` when uncalibrated. Meaningless for non-INT8 dtypes.
    pub fn k_scale(&self, layer: usize) -> f32 {
        self.k_scales[layer]
    }

    /// Per-layer value dequantization scale for INT8 caches (`x ~= q * v_scale`).
    /// `1.0` when uncalibrated. Meaningless for non-INT8 dtypes.
    pub fn v_scale(&self, layer: usize) -> f32 {
        self.v_scales[layer]
    }

    /// Set the INT8 key/value dequantization scales for a single layer.
    /// Both must be finite and `> 0`; invalid values are ignored and the
    /// previous scale is kept. The matching INT8 write and read kernels
    /// consume these scales — keep them in sync.
    pub fn set_kv_scales(&mut self, layer: usize, k_scale: f32, v_scale: f32) {
        if k_scale.is_finite() && k_scale > 0.0 {
            self.k_scales[layer] = k_scale;
        }
        if v_scale.is_finite() && v_scale > 0.0 {
            self.v_scales[layer] = v_scale;
        }
    }

    /// Set the INT8 key/value dequantization scales for all layers at once.
    /// Values must be finite and `> 0`; invalid entries are skipped.
    /// The vectors must have length equal to the number of layers.
    pub fn set_all_kv_scales(&mut self, k_scales: Vec<f32>, v_scales: Vec<f32>) {
        if k_scales.len() != self.k_scales.len() || v_scales.len() != self.v_scales.len() {
            tracing::warn!(
                expected = self.k_scales.len(),
                got_k = k_scales.len(),
                got_v = v_scales.len(),
                "ignoring set_all_kv_scales: length mismatch"
            );
            return;
        }
        for (i, (&k, &v)) in k_scales.iter().zip(v_scales.iter()).enumerate() {
            self.set_kv_scales(i, k, v);
        }
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layer_ptrs.len()
    }

    /// Total GPU memory used in bytes.
    pub fn total_bytes(&self) -> usize {
        self.layer_sizes.iter().map(|(k, v)| k + v).sum()
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Size of cache element in bytes.
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Compute slot mapping for a list of (block_id, block_offset) pairs.
    ///
    /// Each slot = block_id * block_size + offset.
    pub fn compute_slots(&self, positions: &[(BlockId, usize)]) -> Vec<i64> {
        positions
            .iter()
            .map(|(block_id, offset)| {
                (block_id.0 as i64) * (self.block_size as i64) + (*offset as i64)
            })
            .collect()
    }
}

#[cfg(has_cuda)]
impl Drop for GpuKVCache {
    fn drop(&mut self) {
        for (key_ptr, value_ptr) in &self.layer_ptrs {
            unsafe {
                let _ = gpu_free(*key_ptr);
                let _ = gpu_free(*value_ptr);
            }
        }
    }
}

// ── Non-CUDA stubs ────────────────────────────────────────────────────────

#[cfg(not(has_cuda))]
pub use stubs::*;

#[cfg(not(has_cuda))]
mod stubs {
    use super::CudaKernelError;

    #[allow(clippy::too_many_arguments)]
    pub fn cache_write_f16(
        _key_cache: *mut u16,
        _value_cache: *mut u16,
        _new_key: *const u16,
        _new_value: *const u16,
        _slot_mapping: *const i64,
        _num_tokens: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _num_blocks: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cache_write_fp8(
        _key_cache: *mut u8,
        _value_cache: *mut u8,
        _new_key: *const u16,
        _new_value: *const u16,
        _slot_mapping: *const i64,
        _num_tokens: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _num_blocks: i64,
        _is_e5m2: bool,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cache_write_f16_sync(
        _key_cache: *mut u16,
        _value_cache: *mut u16,
        _new_key: *const u16,
        _new_value: *const u16,
        _slot_mapping: *const i64,
        _num_tokens: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _num_blocks: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cache_write_fp8_sync(
        _key_cache: *mut u8,
        _value_cache: *mut u8,
        _new_key: *const u16,
        _new_value: *const u16,
        _slot_mapping: *const i64,
        _num_tokens: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _num_blocks: i64,
        _is_e5m2: bool,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cache_write_i8(
        _key_cache: *mut i8,
        _value_cache: *mut i8,
        _new_key: *const u16,
        _new_value: *const u16,
        _slot_mapping: *const i64,
        _num_tokens: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _num_blocks: i64,
        _k_scale: f32,
        _v_scale: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cache_write_i8_sync(
        _key_cache: *mut i8,
        _value_cache: *mut i8,
        _new_key: *const u16,
        _new_value: *const u16,
        _slot_mapping: *const i64,
        _num_tokens: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _block_size: i64,
        _num_blocks: i64,
        _k_scale: f32,
        _v_scale: f32,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn cache_block_copy(
        _src: *const u8,
        _dst: *mut u8,
        _block_nbytes: i64,
        _num_blocks: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn cache_block_copy_sync(
        _src: *const u8,
        _dst: *mut u8,
        _block_nbytes: i64,
        _num_blocks: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn cache_zero(_ptr: *mut u8, _nbytes: i64, _stream: usize) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn cache_zero_sync(_ptr: *mut u8, _nbytes: i64) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn gpu_memcpy_to_device(
        _dst: *mut u8,
        _src: *const u8,
        _nbytes: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn gpu_memcpy_to_host(
        _dst: *mut u8,
        _src: *const u8,
        _nbytes: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(has_cuda))]
    mod no_cuda {
        use super::*;

        #[test]
        fn cache_write_returns_not_available() {
            let result = cache_write_f16(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn cache_write_i8_returns_not_available() {
            let result = cache_write_i8(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                1.0,
                1.0,
                0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn cache_block_copy_returns_not_available() {
            let result = cache_block_copy(std::ptr::null(), std::ptr::null_mut(), 0, 0, 0);
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn cache_zero_returns_not_available() {
            let result = cache_zero(std::ptr::null_mut(), 0, 0);
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }
    }

    // Pure-CPU coverage of the INT8 KV quantization math (the oracle the GPU
    // kernel mirrors). Runs on every platform regardless of CUDA.
    #[test]
    fn quantize_kv_i8_unit_scale_rounds_and_clamps() {
        // Unit scale (uncalibrated default): round-to-nearest, clamp to ±127.
        assert_eq!(quantize_kv_i8_reference(0.0, 1.0), 0);
        assert_eq!(quantize_kv_i8_reference(0.4, 1.0), 0);
        assert_eq!(quantize_kv_i8_reference(0.6, 1.0), 1);
        assert_eq!(quantize_kv_i8_reference(-0.6, 1.0), -1);
        assert_eq!(quantize_kv_i8_reference(200.0, 1.0), 127);
        assert_eq!(quantize_kv_i8_reference(-200.0, 1.0), -127);
    }

    #[test]
    fn quantize_kv_i8_with_scale_round_trips() {
        // A calibrated scale maps the tensor range into [-127, 127] with low
        // error: dequant(quant(x)) is within half a step (scale/2).
        let scale = 0.05_f32; // max magnitude ~6.35
        for &x in &[-6.0_f32, -1.3, -0.02, 0.0, 0.02, 1.3, 6.0] {
            let q = quantize_kv_i8_reference(x, scale);
            let dq = q as f32 * scale;
            assert!((dq - x).abs() <= scale / 2.0 + 1e-6, "x={x} q={q} dq={dq}");
        }
        // Out-of-range still clamps at the int8 limit.
        assert_eq!(quantize_kv_i8_reference(100.0, scale), 127);
    }

    #[cfg(has_cuda)]
    mod with_cuda {
        use super::*;

        #[test]
        fn cache_zero_clears_memory() {
            let nbytes = 256;
            let ptr = unsafe { gpu_alloc(nbytes).expect("gpu_alloc failed") };
            // Initialize with non-zero
            let initial = vec![42u8; nbytes];
            unsafe {
                gpu_memcpy_h2d(ptr, initial.as_ptr(), nbytes).unwrap();
                cache_zero_sync(ptr, nbytes as i64).expect("cache_zero_sync failed");
            }
            // Verify zeroed — copy to host
            let mut host_buf = vec![0u8; nbytes];
            unsafe {
                gpu_memcpy_d2h(host_buf.as_mut_ptr(), ptr, nbytes).unwrap();
            }
            assert!(host_buf.iter().all(|&b| b == 0));
            unsafe { gpu_free(ptr).expect("gpu_free failed") };
        }

        #[test]
        fn cache_block_copy_is_exact() {
            let block_nbytes = 128;
            let num_blocks = 2;
            let total = block_nbytes * num_blocks;

            let src = unsafe { gpu_alloc(total).expect("gpu_alloc failed") };
            let dst = unsafe { gpu_alloc(total).expect("gpu_alloc failed") };

            // Initialize src
            let src_initial = (0..total).map(|i| (i % 256) as u8).collect::<Vec<_>>();
            unsafe {
                gpu_memcpy_h2d(src, src_initial.as_ptr(), total).unwrap();
            }

            // Zero dst first
            unsafe { cache_zero_sync(dst, total as i64).expect("cache_zero failed") };

            // Copy
            unsafe {
                cache_block_copy_sync(src, dst, block_nbytes as i64, num_blocks as i64)
                    .expect("cache_block_copy_sync failed");
            }

            // Verify src == dst
            let mut src_host = vec![0u8; total];
            let mut dst_host = vec![0u8; total];
            unsafe {
                gpu_memcpy_d2h(src_host.as_mut_ptr(), src, total).unwrap();
                gpu_memcpy_d2h(dst_host.as_mut_ptr(), dst, total).unwrap();
            }
            assert_eq!(src_host, dst_host);

            unsafe { gpu_free(src).expect("gpu_free failed") };
            unsafe { gpu_free(dst).expect("gpu_free failed") };
        }

        #[test]
        fn gpu_kv_cache_allocation() {
            let cache = GpuKVCache::new(
                10, // num_blocks
                2,  // num_layers
                4,  // num_kv_heads
                64, // head_dim
                16, // block_size
                rllm_core::dtype::DType::F16,
            )
            .expect("GpuKVCache::new failed");

            assert_eq!(cache.num_layers(), 2);
            assert_eq!(cache.num_blocks(), 10);
            assert_eq!(cache.block_size(), 16);
            assert_eq!(cache.key_shape(), [10, 4, 64, 16]);

            let expected_per_layer = 10 * 4 * 64 * 16 * 2; // K + V
            let expected_total = expected_per_layer * 2 * 2; // 2 layers * K+V
            assert_eq!(cache.total_bytes(), expected_total);
        }

        #[test]
        fn slot_mapping_computation() {
            let cache = GpuKVCache::new(10, 1, 4, 64, 16, rllm_core::dtype::DType::F16)
                .expect("GpuKVCache::new failed");

            let slots = cache.compute_slots(&[
                (BlockId(0), 0),
                (BlockId(0), 1),
                (BlockId(1), 0),
                (BlockId(5), 15),
            ]);
            assert_eq!(slots, vec![0, 1, 16, 95]);
        }

        #[test]
        fn kv_scales_default_to_one_and_set() {
            let mut cache = GpuKVCache::new(4, 2, 1, 8, 4, rllm_core::dtype::DType::INT8)
                .expect("GpuKVCache::new failed");
            // Per-layer defaults
            assert_eq!(cache.k_scale(0), 1.0);
            assert_eq!(cache.v_scale(0), 1.0);
            assert_eq!(cache.k_scale(1), 1.0);
            assert_eq!(cache.v_scale(1), 1.0);

            // Set per-layer scales
            cache.set_kv_scales(0, 0.05, 0.1);
            assert_eq!(cache.k_scale(0), 0.05);
            assert_eq!(cache.v_scale(0), 0.1);
            assert_eq!(cache.k_scale(1), 1.0); // layer 1 unchanged

            // Set all layers at once
            cache.set_all_kv_scales(vec![0.02, 0.03], vec![0.04, 0.05]);
            assert_eq!(cache.k_scale(0), 0.02);
            assert_eq!(cache.v_scale(0), 0.04);
            assert_eq!(cache.k_scale(1), 0.03);
            assert_eq!(cache.v_scale(1), 0.05);

            // Invalid scales are rejected; previous values are kept.
            cache.set_kv_scales(0, 0.0, -1.0);
            assert_eq!(cache.k_scale(0), 0.02);
            assert_eq!(cache.v_scale(0), 0.04);
            cache.set_kv_scales(1, f32::NAN, f32::INFINITY);
            assert_eq!(cache.k_scale(1), 0.03);
            assert_eq!(cache.v_scale(1), 0.05);

            // Length mismatch is rejected
            cache.set_all_kv_scales(vec![1.0], vec![1.0]);
            assert_eq!(cache.k_scale(0), 0.02); // unchanged
        }

        #[test]
        fn cache_write_i8_matches_reference_oracle() {
            // 2 tokens, 1 kv head, head_dim 4, block_size 4, 2 blocks.
            let num_tokens = 2i64;
            let num_kv_heads = 1i64;
            let head_dim = 4i64;
            let block_size = 4i64;
            let num_blocks = 2i64;
            let elems = (num_tokens * num_kv_heads * head_dim) as usize;
            let cache_elems = (num_blocks * num_kv_heads * head_dim * block_size) as usize;

            let k_scale = 0.05_f32;
            let v_scale = 0.1_f32;

            // Host f16 K/V values (as raw u16 bit patterns) and slot mapping.
            let k_vals = [0.10_f32, -0.30, 6.0, 0.02, -0.07, 0.20, -100.0, 0.0];
            let v_vals = [0.50_f32, -0.05, 0.0, 12.0, -0.20, 0.30, 0.40, -0.50];
            let k_u16: Vec<u16> =
                k_vals.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let v_u16: Vec<u16> =
                v_vals.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            // token 0 -> slot 0 (block 0, off 0); token 1 -> slot 5 (block 1, off 1)
            let slots: Vec<i64> = vec![0, 5];

            // Device buffers.
            let key_cache = unsafe { gpu_alloc(cache_elems).expect("alloc key") } as *mut i8;
            let value_cache = unsafe { gpu_alloc(cache_elems).expect("alloc value") } as *mut i8;
            let new_key = unsafe { gpu_alloc(elems * 2).expect("alloc new_key") } as *mut u16;
            let new_value = unsafe { gpu_alloc(elems * 2).expect("alloc new_value") } as *mut u16;
            let slot_dev = unsafe { gpu_alloc(slots.len() * 8).expect("alloc slots") } as *mut i64;

            unsafe {
                cache_zero_sync(key_cache as *mut u8, cache_elems as i64).unwrap();
                cache_zero_sync(value_cache as *mut u8, cache_elems as i64).unwrap();
                gpu_memcpy_to_device(new_key as *mut u8, k_u16.as_ptr() as *const u8, elems * 2)
                    .unwrap();
                gpu_memcpy_to_device(new_value as *mut u8, v_u16.as_ptr() as *const u8, elems * 2)
                    .unwrap();
                gpu_memcpy_to_device(
                    slot_dev as *mut u8,
                    slots.as_ptr() as *const u8,
                    slots.len() * 8,
                )
                .unwrap();

                cache_write_i8_sync(
                    key_cache,
                    value_cache,
                    new_key as *const u16,
                    new_value as *const u16,
                    slot_dev as *const i64,
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    num_blocks,
                    k_scale,
                    v_scale,
                )
                .expect("cache_write_i8_sync failed");
            }

            // Read back the int8 caches.
            let mut key_host = vec![0i8; cache_elems];
            let mut val_host = vec![0i8; cache_elems];
            unsafe {
                gpu_memcpy_to_host(
                    key_host.as_mut_ptr() as *mut u8,
                    key_cache as *const u8,
                    cache_elems,
                )
                .unwrap();
                gpu_memcpy_to_host(
                    val_host.as_mut_ptr() as *mut u8,
                    value_cache as *const u8,
                    cache_elems,
                )
                .unwrap();
            }

            // Verify each written element against the CPU oracle. NHD cache layout:
            // idx = ((block_id*num_kv_heads + kv_head)*head_dim + d)*block_size + off
            for token in 0..num_tokens as usize {
                let slot = slots[token];
                let block_id = slot / block_size;
                let off = slot % block_size;
                for d in 0..head_dim as usize {
                    let cache_idx = (((block_id * num_kv_heads) * head_dim + d as i64) * block_size
                        + off) as usize;
                    let src = token * head_dim as usize + d;
                    // f16 rounding happens before quantization, so quantize the f16 value.
                    let kf = half::f16::from_f32(k_vals[src]).to_f32();
                    let vf = half::f16::from_f32(v_vals[src]).to_f32();
                    assert_eq!(
                        key_host[cache_idx],
                        quantize_kv_i8_reference(kf, k_scale),
                        "key token={token} d={d}"
                    );
                    assert_eq!(
                        val_host[cache_idx],
                        quantize_kv_i8_reference(vf, v_scale),
                        "val token={token} d={d}"
                    );
                }
            }

            unsafe {
                gpu_free(key_cache as *mut u8).unwrap();
                gpu_free(value_cache as *mut u8).unwrap();
                gpu_free(new_key as *mut u8).unwrap();
                gpu_free(new_value as *mut u8).unwrap();
                gpu_free(slot_dev as *mut u8).unwrap();
            }
        }
    }
}
