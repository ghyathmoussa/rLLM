// rLLM CUDA kernels — minimal smoke test kernels for Phase 5.
// All kernel entry points use C linkage for Rust FFI.

#include <cstdint>

// ── Error codes ──────────────────────────────────────────────────────────
// 0 = success, non-zero = error.

extern "C" {

// ── Vector Add ───────────────────────────────────────────────────────────
// Element-wise: out[i] = a[i] + b[i] for float32 arrays.

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int32_t rllm_vector_add_f32(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    if (n <= 0) return 0;
    int64_t threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// Synchronous version for testing/debugging.
int32_t rllm_vector_add_f32_sync(const float* a, const float* b, float* out, int64_t n) {
    int32_t rc = rllm_vector_add_f32(a, b, out, n, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── Block Copy ───────────────────────────────────────────────────────────
// Copy `n` bytes from src to dst on the device.

__global__ void block_copy_kernel(const uint8_t* src, uint8_t* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int32_t rllm_block_copy(const uint8_t* src, uint8_t* dst, int64_t nbytes, cudaStream_t stream) {
    if (nbytes <= 0) return 0;
    int64_t threads = 256;
    int64_t blocks = (nbytes + threads - 1) / threads;
    block_copy_kernel<<<blocks, threads, 0, stream>>>(src, dst, nbytes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_block_copy_sync(const uint8_t* src, uint8_t* dst, int64_t nbytes) {
    int32_t rc = rllm_block_copy(src, dst, nbytes, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
