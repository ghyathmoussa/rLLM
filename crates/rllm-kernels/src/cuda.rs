//! CUDA kernel FFI wrappers and launch conventions.
//!
//! # Conventions
//!
//! - All CUDA kernels return `i32` error codes (0 = success).
//! - Production kernels accept a `stream` parameter for async execution.
//! - `_sync` suffixed variants synchronize after launch (debug/test only).
//! - Non-CUDA builds compile without these symbols — the `has_cuda` cfg gate
//!   controls availability.

#[cfg(has_cuda)]
mod ffi {
    use std::os::raw::c_int;

    unsafe extern "C" {
        // Vector add (float32)
        pub fn rllm_vector_add_f32(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            n: i64,
            stream: usize, // cudaStream_t — 0 = default stream
        ) -> c_int;

        pub fn rllm_vector_add_f32_sync(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            n: i64,
        ) -> c_int;

        // Block copy (byte-wise)
        pub fn rllm_block_copy(src: *const u8, dst: *mut u8, nbytes: i64, stream: usize) -> c_int;

        pub fn rllm_block_copy_sync(src: *const u8, dst: *mut u8, nbytes: i64) -> c_int;

        // GPTQ GEMM (FP16 activations, FP16 output, FP16 scales)
        pub fn rllm_gptq_gemm_f16(
            x: *const u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            g_idx: *const u32,
            out: *mut u16,
            m: i64,
            in_features: i64,
            out_features: i64,
            num_groups: i64,
            group_size: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_gptq_gemm_f16_sync(
            x: *const u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            g_idx: *const u32,
            out: *mut u16,
            m: i64,
            in_features: i64,
            out_features: i64,
            num_groups: i64,
            group_size: i64,
        ) -> c_int;

        pub fn rllm_awq_gemm_f16(
            x: *const u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            out: *mut u16,
            m: i64,
            in_features: i64,
            out_features: i64,
            num_groups: i64,
            group_size: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_awq_gemm_f16_sync(
            x: *const u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            out: *mut u16,
            m: i64,
            in_features: i64,
            out_features: i64,
            num_groups: i64,
            group_size: i64,
        ) -> c_int;
    }
}

/// Error returned by CUDA kernel launches.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaKernelError {
    #[error("CUDA kernel returned error code {code}")]
    KernelError { code: i32 },
    #[error("CUDA is not available on this build")]
    NotAvailable,
}

/// Check a CUDA kernel return code and convert to Result.
#[cfg(has_cuda)]
fn check(rc: i32) -> Result<(), CudaKernelError> {
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
}

// ── Vector Add ──────────────────────────────────────────────────────────

/// Launch async vector add: `out[i] = a[i] + b[i]` on the given stream.
///
/// # Safety
/// - `a`, `b`, `out` must be valid device pointers with at least `n` elements.
/// - The stream must be a valid CUDA stream (0 for default).
#[cfg(has_cuda)]
pub unsafe fn vector_add_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    n: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_vector_add_f32(a, b, out, n, stream) };
    check(rc)
}

/// Synchronous vector add for testing.
///
/// # Safety
/// Same as [`vector_add_f32`].
#[cfg(has_cuda)]
pub unsafe fn vector_add_f32_sync(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    n: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_vector_add_f32_sync(a, b, out, n) };
    check(rc)
}

// ── Block Copy ──────────────────────────────────────────────────────────

/// Launch async block copy on the given stream.
///
/// # Safety
/// - `src` and `dst` must be valid device pointers with at least `nbytes` bytes.
/// - The stream must be a valid CUDA stream.
#[cfg(has_cuda)]
pub unsafe fn block_copy(
    src: *const u8,
    dst: *mut u8,
    nbytes: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_block_copy(src, dst, nbytes, stream) };
    check(rc)
}

/// Synchronous block copy for testing.
///
/// # Safety
/// Same as [`block_copy`].
#[cfg(has_cuda)]
pub unsafe fn block_copy_sync(
    src: *const u8,
    dst: *mut u8,
    nbytes: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_block_copy_sync(src, dst, nbytes) };
    check(rc)
}

// ── GPTQ GEMM ────────────────────────────────────────────────────────────

/// Launch async GPTQ GEMM with FP16 activations/output and packed INT4 weights.
///
/// # Safety
/// - All pointers must be valid CUDA device pointers.
/// - `x` must contain `m * in_features` FP16 elements.
/// - `qweight` must contain `(in_features / 8) * out_features` packed i32 words.
/// - `qzeros` must contain `num_groups * (out_features / 8)` packed i32 words.
/// - `scales` must contain `num_groups * out_features` FP16 elements.
/// - `g_idx` must contain `in_features` entries.
/// - `out` must contain `m * out_features` FP16 elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gptq_gemm_f16(
    x: *const u16,
    qweight: *const i32,
    qzeros: *const i32,
    scales: *const u16,
    g_idx: *const u32,
    out: *mut u16,
    m: i64,
    in_features: i64,
    out_features: i64,
    num_groups: i64,
    group_size: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_gptq_gemm_f16(
            x,
            qweight,
            qzeros,
            scales,
            g_idx,
            out,
            m,
            in_features,
            out_features,
            num_groups,
            group_size,
            stream,
        )
    };
    check(rc)
}

/// Synchronous GPTQ GEMM for testing/debugging.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gptq_gemm_f16_sync(
    x: *const u16,
    qweight: *const i32,
    qzeros: *const i32,
    scales: *const u16,
    g_idx: *const u32,
    out: *mut u16,
    m: i64,
    in_features: i64,
    out_features: i64,
    num_groups: i64,
    group_size: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_gptq_gemm_f16_sync(
            x,
            qweight,
            qzeros,
            scales,
            g_idx,
            out,
            m,
            in_features,
            out_features,
            num_groups,
            group_size,
        )
    };
    check(rc)
}

// ── AWQ GEMM ────────────────────────────────────────────────────────────

/// Launch async AWQ GEMM with FP16 activations/output and packed INT4 weights.
///
/// # Safety
/// - All pointers must be valid CUDA device pointers.
/// - `x` must contain `m * in_features` FP16 elements.
/// - `qweight` must contain `in_features * (out_features / 8)` packed i32 words.
/// - `qzeros` must contain `num_groups * (out_features / 8)` packed i32 words.
/// - `scales` must contain `num_groups * out_features` FP16 elements.
/// - `out` must contain `m * out_features` FP16 elements.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn awq_gemm_f16(
    x: *const u16,
    qweight: *const i32,
    qzeros: *const i32,
    scales: *const u16,
    out: *mut u16,
    m: i64,
    in_features: i64,
    out_features: i64,
    num_groups: i64,
    group_size: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_awq_gemm_f16(
            x,
            qweight,
            qzeros,
            scales,
            out,
            m,
            in_features,
            out_features,
            num_groups,
            group_size,
            stream,
        )
    };
    check(rc)
}

/// Synchronous AWQ GEMM for testing/debugging.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn awq_gemm_f16_sync(
    x: *const u16,
    qweight: *const i32,
    qzeros: *const i32,
    scales: *const u16,
    out: *mut u16,
    m: i64,
    in_features: i64,
    out_features: i64,
    num_groups: i64,
    group_size: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_awq_gemm_f16_sync(
            x,
            qweight,
            qzeros,
            scales,
            out,
            m,
            in_features,
            out_features,
            num_groups,
            group_size,
        )
    };
    check(rc)
}

// ── Non-CUDA stubs ──────────────────────────────────────────────────────

#[cfg(not(has_cuda))]
pub use stubs::*;

#[cfg(not(has_cuda))]
mod stubs {
    use super::CudaKernelError;

    /// Returns an error indicating CUDA is not available.
    pub fn vector_add_f32(
        _a: *const f32,
        _b: *const f32,
        _out: *mut f32,
        _n: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn vector_add_f32_sync(
        _a: *const f32,
        _b: *const f32,
        _out: *mut f32,
        _n: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn block_copy(
        _src: *const u8,
        _dst: *mut u8,
        _nbytes: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn block_copy_sync(
        _src: *const u8,
        _dst: *mut u8,
        _nbytes: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gptq_gemm_f16(
        _x: *const u16,
        _qweight: *const i32,
        _qzeros: *const i32,
        _scales: *const u16,
        _g_idx: *const u32,
        _out: *mut u16,
        _m: i64,
        _in_features: i64,
        _out_features: i64,
        _num_groups: i64,
        _group_size: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gptq_gemm_f16_sync(
        _x: *const u16,
        _qweight: *const i32,
        _qzeros: *const i32,
        _scales: *const u16,
        _g_idx: *const u32,
        _out: *mut u16,
        _m: i64,
        _in_features: i64,
        _out_features: i64,
        _num_groups: i64,
        _group_size: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn awq_gemm_f16(
        _x: *const u16,
        _qweight: *const i32,
        _qzeros: *const i32,
        _scales: *const u16,
        _out: *mut u16,
        _m: i64,
        _in_features: i64,
        _out_features: i64,
        _num_groups: i64,
        _group_size: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn awq_gemm_f16_sync(
        _x: *const u16,
        _qweight: *const i32,
        _qzeros: *const i32,
        _scales: *const u16,
        _out: *mut u16,
        _m: i64,
        _in_features: i64,
        _out_features: i64,
        _num_groups: i64,
        _group_size: i64,
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
        fn vector_add_returns_not_available() {
            let mut buf = [0.0f32; 4];
            let result = vector_add_f32(buf.as_ptr(), buf.as_ptr(), buf.as_mut_ptr(), 4, 0);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                matches!(err, CudaKernelError::NotAvailable),
                "expected NotAvailable, got {err:?}"
            );
        }

        #[test]
        fn vector_add_sync_returns_not_available() {
            let mut buf = [0.0f32; 4];
            let result = vector_add_f32_sync(buf.as_ptr(), buf.as_ptr(), buf.as_mut_ptr(), 4);
            assert!(result.is_err());
        }

        #[test]
        fn block_copy_returns_not_available() {
            let mut buf = [0u8; 16];
            let result = block_copy(buf.as_ptr(), buf.as_mut_ptr(), 16, 0);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), CudaKernelError::NotAvailable));
        }

        #[test]
        fn block_copy_sync_returns_not_available() {
            let mut buf = [0u8; 16];
            let result = block_copy_sync(buf.as_ptr(), buf.as_mut_ptr(), 16);
            assert!(result.is_err());
        }

        #[test]
        fn gptq_gemm_returns_not_available() {
            let mut out = [0u16; 8];
            let result = gptq_gemm_f16(
                out.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                out.as_ptr(),
                std::ptr::null(),
                out.as_mut_ptr(),
                1,
                8,
                8,
                1,
                8,
                0,
            );
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), CudaKernelError::NotAvailable));
        }

        #[test]
        fn error_display_message() {
            let err = CudaKernelError::NotAvailable;
            assert!(err.to_string().contains("not available"));

            let err = CudaKernelError::KernelError { code: 42 };
            assert!(err.to_string().contains("42"));
        }
    }

    #[cfg(has_cuda)]
    mod with_cuda {
        use super::*;
        use crate::cache_ops::{gpu_alloc, gpu_free, gpu_memcpy_d2h, gpu_memcpy_h2d};

        fn f32_to_f16_bits(f: f32) -> u16 {
            let x = f.to_bits();
            let sign = (x >> 31) & 0x1;
            let exp = ((x >> 23) & 0xFF) as i32;
            let mantissa = x & 0x7FFFFF;
            if exp == 0 {
                return (sign << 15) as u16;
            }
            if exp == 255 {
                return ((sign << 15) | 0x7C00) as u16;
            }
            let new_exp = exp - 127 + 15;
            if new_exp <= 0 {
                return (sign << 15) as u16;
            }
            if new_exp >= 31 {
                return ((sign << 15) | 0x7C00) as u16;
            }
            ((sign << 15) | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
        }

        fn f16_bits_to_f32(h: u16) -> f32 {
            let sign = (h >> 15) & 0x1;
            let exponent = (h >> 10) & 0x1F;
            let mantissa = h & 0x3FF;
            if exponent == 0 {
                if mantissa == 0 {
                    return f32::from_bits((sign as u32) << 31);
                }
                let mut e = 0u32;
                let mut m = mantissa;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                m &= 0x3FF;
                return f32::from_bits(
                    ((sign as u32) << 31) | ((127 - 15 - e) << 23) | ((m as u32) << 13),
                );
            }
            if exponent == 31 {
                return f32::from_bits(
                    ((sign as u32) << 31) | 0x7F800000 | ((mantissa as u32) << 13),
                );
            }
            f32::from_bits(
                ((sign as u32) << 31)
                    | (((exponent as u32) + 112) << 23)
                    | ((mantissa as u32) << 13),
            )
        }

        unsafe fn upload_u16(data: &[u16]) -> *mut u16 {
            let nbytes = std::mem::size_of_val(data);
            let ptr = unsafe { gpu_alloc(nbytes).expect("gpu_alloc failed") as *mut u16 };
            unsafe {
                gpu_memcpy_h2d(ptr as *mut u8, data.as_ptr() as *const u8, nbytes).unwrap();
            }
            ptr
        }

        unsafe fn upload_i32(data: &[i32]) -> *mut i32 {
            let nbytes = std::mem::size_of_val(data);
            let ptr = unsafe { gpu_alloc(nbytes).expect("gpu_alloc failed") as *mut i32 };
            unsafe {
                gpu_memcpy_h2d(ptr as *mut u8, data.as_ptr() as *const u8, nbytes).unwrap();
            }
            ptr
        }

        unsafe fn upload_u32(data: &[u32]) -> *mut u32 {
            let nbytes = std::mem::size_of_val(data);
            let ptr = unsafe { gpu_alloc(nbytes).expect("gpu_alloc failed") as *mut u32 };
            unsafe {
                gpu_memcpy_h2d(ptr as *mut u8, data.as_ptr() as *const u8, nbytes).unwrap();
            }
            ptr
        }

        unsafe fn download_u16(ptr: *mut u16, len: usize) -> Vec<u16> {
            let mut host = vec![0u16; len];
            let nbytes = len * std::mem::size_of::<u16>();
            unsafe {
                gpu_memcpy_d2h(host.as_mut_ptr() as *mut u8, ptr as *const u8, nbytes).unwrap();
            }
            host
        }

        #[test]
        fn vector_add_correctness() {
            let a = [1.0f32, 2.0, 3.0, 4.0];
            let b = [10.0f32, 20.0, 30.0, 40.0];
            let mut out = [0.0f32; 4];
            let nbytes = 4 * std::mem::size_of::<f32>();
            unsafe {
                let d_a = gpu_alloc(nbytes).unwrap() as *mut f32;
                let d_b = gpu_alloc(nbytes).unwrap() as *mut f32;
                let d_out = gpu_alloc(nbytes).unwrap() as *mut f32;

                gpu_memcpy_h2d(d_a as *mut u8, a.as_ptr() as *const u8, nbytes).unwrap();
                gpu_memcpy_h2d(d_b as *mut u8, b.as_ptr() as *const u8, nbytes).unwrap();

                vector_add_f32_sync(d_a, d_b, d_out, 4).expect("vector_add_f32_sync failed");

                gpu_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_out as *const u8, nbytes).unwrap();

                gpu_free(d_a as *mut u8).unwrap();
                gpu_free(d_b as *mut u8).unwrap();
                gpu_free(d_out as *mut u8).unwrap();
            }
            assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
        }

        #[test]
        fn block_copy_correctness() {
            let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
            let mut dst = [0u8; 8];
            unsafe {
                let d_src = gpu_alloc(8).unwrap();
                let d_dst = gpu_alloc(8).unwrap();

                gpu_memcpy_h2d(d_src, src.as_ptr(), 8).unwrap();

                block_copy_sync(d_src, d_dst, 8).expect("block_copy_sync failed");

                gpu_memcpy_d2h(dst.as_mut_ptr(), d_dst, 8).unwrap();

                gpu_free(d_src).unwrap();
                gpu_free(d_dst).unwrap();
            }
            assert_eq!(src, dst);
        }

        #[test]
        fn gptq_gemm_correctness() {
            let x_f = [1.0f32, -1.0, 0.5, 2.0, 0.25, -0.5, 1.5, -2.0];
            let x_h = x_f.iter().map(|&v| f32_to_f16_bits(v)).collect::<Vec<_>>();
            let qweight = [0x76543210i32; 8];
            let qzeros = [0x11111111i32];
            let scales = [f32_to_f16_bits(0.5f32); 8];
            let g_idx = [0u32; 8];

            let d_x = unsafe { upload_u16(&x_h) };
            let d_qweight = unsafe { upload_i32(&qweight) };
            let d_qzeros = unsafe { upload_i32(&qzeros) };
            let d_scales = unsafe { upload_u16(&scales) };
            let d_gidx = unsafe { upload_u32(&g_idx) };
            let d_out = unsafe {
                gpu_alloc(8 * std::mem::size_of::<u16>()).expect("gpu_alloc failed") as *mut u16
            };

            unsafe {
                gptq_gemm_f16_sync(
                    d_x, d_qweight, d_qzeros, d_scales, d_gidx, d_out, 1, 8, 8, 1, 8,
                )
                .expect("gptq_gemm_f16_sync failed");
            }

            let out = unsafe { download_u16(d_out, 8) };
            unsafe {
                gpu_free(d_x as *mut u8).expect("gpu_free failed");
                gpu_free(d_qweight as *mut u8).expect("gpu_free failed");
                gpu_free(d_qzeros as *mut u8).expect("gpu_free failed");
                gpu_free(d_scales as *mut u8).expect("gpu_free failed");
                gpu_free(d_gidx as *mut u8).expect("gpu_free failed");
                gpu_free(d_out as *mut u8).expect("gpu_free failed");
            }

            for col in 0..8 {
                let mut expected = 0.0f32;
                for k in 0..8 {
                    let q = k as f32;
                    let zero = 2.0f32;
                    let w = (q - zero) * 0.5f32;
                    expected += x_f[k] * w;
                }
                let actual = f16_bits_to_f32(out[col]);
                assert!(
                    (actual - expected).abs() < 0.2,
                    "col {col}: expected {expected:.4}, got {actual:.4}"
                );
            }
        }
    }
}
