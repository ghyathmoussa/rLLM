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
    }
}

#[cfg(has_cuda)]
fn check(rc: i32) -> Result<(), CudaKernelError> {
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
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

        Ok(Self {
            layer_ptrs,
            layer_sizes,
            num_blocks,
            num_kv_heads,
            head_dim,
            block_size,
            element_size,
            dtype,
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

    #[cfg(has_cuda)]
    mod with_cuda {
        use super::*;

        #[test]
        fn cache_zero_clears_memory() {
            let nbytes = 256;
            let ptr = unsafe { gpu_alloc(nbytes).expect("gpu_alloc failed") };
            // Write non-zero
            unsafe { cache_zero_sync(ptr, nbytes as i64).expect("cache_zero_sync failed") };
            // Verify zeroed — copy to host
            let mut host_buf = vec![0u8; nbytes];
            unsafe {
                libc::memcpy(
                    host_buf.as_mut_ptr() as *mut libc::c_void,
                    ptr as *const libc::c_void,
                    nbytes,
                );
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
                libc::memcpy(
                    src_host.as_mut_ptr() as *mut libc::c_void,
                    src as *const libc::c_void,
                    total,
                );
                libc::memcpy(
                    dst_host.as_mut_ptr() as *mut libc::c_void,
                    dst as *const libc::c_void,
                    total,
                );
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
            let cache = GpuKVCache::new(10, 1, 4, 64, 16, rllm_core::dtype::DType::F16).expect("GpuKVCache::new failed");


            let slots = cache.compute_slots(&[
                (BlockId(0), 0),
                (BlockId(0), 1),
                (BlockId(1), 0),
                (BlockId(5), 15),
            ]);
            assert_eq!(slots, vec![0, 1, 16, 95]);
        }
    }
}
