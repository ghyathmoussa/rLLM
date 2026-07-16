use crate::cuda::CudaKernelError;

#[cfg(has_cuda)]
mod ffi {
    unsafe extern "C" {
        pub fn rllm_deepseek_fp8_block_matmul_f16(
            x: *const u16,
            weight: *const u8,
            scales: *const f32,
            output: *mut u16,
            rows: i64,
            out_features: i64,
            in_features: i64,
            block_size: i64,
            stream: usize,
        ) -> i32;
        pub fn rllm_deepseek_fp8_selected_expert_matmul_f16(
            x: *const u16,
            expert_ids: *const u32,
            weights: *const u8,
            scales: *const f32,
            output: *mut u16,
            tokens: i64,
            top_k: i64,
            num_experts: i64,
            out_features: i64,
            in_features: i64,
            block_size: i64,
            shared_input: i32,
            stream: usize,
        ) -> i32;
    }
}

#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
/// Launch selected-expert FP8 GEMM on a CUDA stream.
///
/// # Safety
///
/// All pointers must reference device allocations large enough for the shapes
/// described by the dimension arguments and remain valid until the stream has
/// completed the asynchronous kernel.
pub unsafe fn fp8_selected_expert_matmul_f16(
    x: *const u16,
    expert_ids: *const u32,
    weights: *const u8,
    scales: *const f32,
    output: *mut u16,
    tokens: i64,
    top_k: i64,
    num_experts: i64,
    out_features: i64,
    in_features: i64,
    block_size: i64,
    shared_input: bool,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_deepseek_fp8_selected_expert_matmul_f16(
            x,
            expert_ids,
            weights,
            scales,
            output,
            tokens,
            top_k,
            num_experts,
            out_features,
            in_features,
            block_size,
            i32::from(shared_input),
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
}

#[cfg(not(has_cuda))]
#[allow(clippy::too_many_arguments)]
/// Non-CUDA stub for selected-expert FP8 GEMM.
///
/// # Safety
///
/// The pointers are not dereferenced by this stub.
pub unsafe fn fp8_selected_expert_matmul_f16(
    _x: *const u16,
    _expert_ids: *const u32,
    _weights: *const u8,
    _scales: *const f32,
    _output: *mut u16,
    _tokens: i64,
    _top_k: i64,
    _num_experts: i64,
    _out_features: i64,
    _in_features: i64,
    _block_size: i64,
    _shared_input: bool,
    _stream: usize,
) -> Result<(), CudaKernelError> {
    Err(CudaKernelError::NotAvailable)
}

#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
/// Launch two-dimensional block-scaled FP8 GEMM on a CUDA stream.
///
/// # Safety
///
/// All pointers must reference device allocations large enough for the shapes
/// described by the dimension arguments and remain valid until the stream has
/// completed the asynchronous kernel.
pub unsafe fn fp8_block_matmul_f16(
    x: *const u16,
    weight: *const u8,
    scales: *const f32,
    output: *mut u16,
    rows: i64,
    out_features: i64,
    in_features: i64,
    block_size: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_deepseek_fp8_block_matmul_f16(
            x,
            weight,
            scales,
            output,
            rows,
            out_features,
            in_features,
            block_size,
            stream,
        )
    };
    if rc == 0 { Ok(()) } else { Err(CudaKernelError::KernelError { code: rc }) }
}

#[cfg(not(has_cuda))]
#[allow(clippy::too_many_arguments)]
/// Non-CUDA stub for block-scaled FP8 GEMM.
///
/// # Safety
///
/// The pointers are not dereferenced by this stub.
pub unsafe fn fp8_block_matmul_f16(
    _x: *const u16,
    _weight: *const u8,
    _scales: *const f32,
    _output: *mut u16,
    _rows: i64,
    _out_features: i64,
    _in_features: i64,
    _block_size: i64,
    _stream: usize,
) -> Result<(), CudaKernelError> {
    Err(CudaKernelError::NotAvailable)
}
