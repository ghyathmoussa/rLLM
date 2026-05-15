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

    extern "C" {
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
        pub fn rllm_block_copy(
            src: *const u8,
            dst: *mut u8,
            nbytes: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_block_copy_sync(
            src: *const u8,
            dst: *mut u8,
            nbytes: i64,
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
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaKernelError::KernelError { code: rc })
    }
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
            let result =
                vector_add_f32_sync(buf.as_ptr(), buf.as_ptr(), buf.as_mut_ptr(), 4);
            assert!(result.is_err());
        }

        #[test]
        fn block_copy_returns_not_available() {
            let mut buf = [0u8; 16];
            let result = block_copy(buf.as_ptr(), buf.as_mut_ptr(), 16, 0);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                CudaKernelError::NotAvailable
            ));
        }

        #[test]
        fn block_copy_sync_returns_not_available() {
            let mut buf = [0u8; 16];
            let result = block_copy_sync(buf.as_ptr(), buf.as_mut_ptr(), 16);
            assert!(result.is_err());
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

        #[test]
        fn vector_add_correctness() {
            let a = [1.0f32, 2.0, 3.0, 4.0];
            let b = [10.0f32, 20.0, 30.0, 40.0];
            let mut out = [0.0f32; 4];
            unsafe {
                vector_add_f32_sync(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4)
                    .expect("vector_add_f32_sync failed");
            }
            assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
        }

        #[test]
        fn block_copy_correctness() {
            let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
            let mut dst = [0u8; 8];
            unsafe {
                block_copy_sync(src.as_ptr(), dst.as_mut_ptr(), 8)
                    .expect("block_copy_sync failed");
            }
            assert_eq!(src, dst);
        }
    }
}
