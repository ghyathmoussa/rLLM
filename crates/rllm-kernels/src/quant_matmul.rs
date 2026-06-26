//! INT8 quantized matmul kernels.
//!
//! The first CUDA path is a simple W8A8 kernel for weight-only INT8 checkpoints:
//! activations are dynamically quantized per row, int8 weights are accumulated
//! into int32, and the result is dequantized to FP16.

use crate::cuda::CudaKernelError;

#[cfg(has_cuda)]
mod ffi {
    use std::os::raw::c_int;

    unsafe extern "C" {
        pub fn rllm_int8_matmul_w8a8_f16(
            x: *const u16,
            qweight: *const i8,
            weight_scale: *const f32,
            output: *mut u16,
            rows: i64,
            out_features: i64,
            in_features: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_int8_matmul_w8a8_f16_sync(
            x: *const u16,
            qweight: *const i8,
            weight_scale: *const f32,
            output: *mut u16,
            rows: i64,
            out_features: i64,
            in_features: i64,
        ) -> c_int;

        pub fn rllm_mxfp8_matmul_w8a16_f16(
            x: *const u16,
            qweight: *const i8,
            scales: *const u16,
            output: *mut u16,
            rows: i64,
            out_features: i64,
            in_features: i64,
            group_size: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_mxfp4_matmul_w4a16_f16(
            x: *const u16,
            qweight: *const u8,
            scales: *const u16,
            output: *mut u16,
            rows: i64,
            out_features: i64,
            in_features: i64,
            group_size: i64,
            stream: usize,
        ) -> c_int;
    }
}

#[cfg(has_cuda)]
fn check(rc: i32) -> Result<(), CudaKernelError> {
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
}

/// Launch async W8A8 matmul.
///
/// Computes `output = x @ dequant(qweight).T`, where `x` and `output` are FP16,
/// `qweight` is row-major `[out_features, in_features]`, and `weight_scale` is
/// per-output-channel `[out_features]`.
///
/// # Safety
/// - All pointers must be valid CUDA device pointers.
/// - `x` must have `rows * in_features` FP16 values.
/// - `qweight` must have `out_features * in_features` i8 values.
/// - `weight_scale` must have `out_features` f32 values.
/// - `output` must have `rows * out_features` FP16 values.
#[cfg(has_cuda)]
pub unsafe fn int8_matmul_w8a8_f16(
    x: *const u16,
    qweight: *const i8,
    weight_scale: *const f32,
    output: *mut u16,
    rows: i64,
    out_features: i64,
    in_features: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_int8_matmul_w8a8_f16(
            x,
            qweight,
            weight_scale,
            output,
            rows,
            out_features,
            in_features,
            stream,
        )
    };
    check(rc)
}

/// Synchronous W8A8 matmul for tests and debugging.
///
/// # Safety
/// Same as [`int8_matmul_w8a8_f16`].
#[cfg(has_cuda)]
pub unsafe fn int8_matmul_w8a8_f16_sync(
    x: *const u16,
    qweight: *const i8,
    weight_scale: *const f32,
    output: *mut u16,
    rows: i64,
    out_features: i64,
    in_features: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_int8_matmul_w8a8_f16_sync(
            x,
            qweight,
            weight_scale,
            output,
            rows,
            out_features,
            in_features,
        )
    };
    check(rc)
}

/// Launch async MXFP8 matmul.
///
/// # Safety
/// - All pointers must be valid CUDA device pointers.
#[cfg(has_cuda)]
pub unsafe fn mxfp8_matmul_w8a16_f16(
    x: *const u16,
    qweight: *const i8,
    scales: *const u16,
    output: *mut u16,
    rows: i64,
    out_features: i64,
    in_features: i64,
    group_size: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_mxfp8_matmul_w8a16_f16(
            x,
            qweight,
            scales,
            output,
            rows,
            out_features,
            in_features,
            group_size,
            stream,
        )
    };
    check(rc)
}

/// Launch async MXFP4 matmul.
///
/// # Safety
/// - All pointers must be valid CUDA device pointers.
#[cfg(has_cuda)]
pub unsafe fn mxfp4_matmul_w4a16_f16(
    x: *const u16,
    qweight: *const u8,
    scales: *const u16,
    output: *mut u16,
    rows: i64,
    out_features: i64,
    in_features: i64,
    group_size: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_mxfp4_matmul_w4a16_f16(
            x,
            qweight,
            scales,
            output,
            rows,
            out_features,
            in_features,
            group_size,
            stream,
        )
    };
    check(rc)
}

#[cfg(not(has_cuda))]
pub use stubs::*;

#[cfg(not(has_cuda))]
mod stubs {
    use super::CudaKernelError;

    /// # Safety
    /// Stub for non-CUDA builds: returns `NotAvailable` and never dereferences
    /// any pointer. Signature mirrors the CUDA build for source compatibility.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn int8_matmul_w8a16(
        _x: *const u16,
        _qweight: *const i8,
        _weight_scale: *const f32,
        _output: *mut u16,
        _rows: i64,
        _out_features: i64,
        _in_features: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    /// # Safety
    /// Stub for non-CUDA builds: returns `NotAvailable` and never dereferences
    /// any pointer. Signature mirrors the CUDA build for source compatibility.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn int8_matmul_w8a8_f16(
        _x: *const u16,
        _qweight: *const i8,
        _weight_scale: *const f32,
        _output: *mut u16,
        _rows: i64,
        _out_features: i64,
        _in_features: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    /// # Safety
    /// Stub for non-CUDA builds: returns `NotAvailable` and never dereferences
    /// any pointer. Signature mirrors the CUDA build for source compatibility.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn int8_matmul_w8a8_f16_sync(
        _x: *const u16,
        _qweight: *const i8,
        _weight_scale: *const f32,
        _output: *mut u16,
        _rows: i64,
        _out_features: i64,
        _in_features: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mxfp8_matmul_w8a16_f16(
        _x: *const u16,
        _qweight: *const i8,
        _scales: *const u16,
        _output: *mut u16,
        _rows: i64,
        _out_features: i64,
        _in_features: i64,
        _group_size: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mxfp4_matmul_w4a16_f16(
        _x: *const u16,
        _qweight: *const u8,
        _scales: *const u16,
        _output: *mut u16,
        _rows: i64,
        _out_features: i64,
        _in_features: i64,
        _group_size: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(has_cuda))]
    #[test]
    fn int8_matmul_returns_not_available_without_cuda() {
        let mut output = [0u16; 4];
        let result = unsafe {
            int8_matmul_w8a8_f16(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                output.as_mut_ptr(),
                0,
                0,
                0,
                0,
            )
        };
        assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
    }
}
