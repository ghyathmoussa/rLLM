// rLLM CUDA kernels — GPTQ packed INT4 GEMM.
// This is a correctness-first implementation that directly consumes the loader's
// packed GPTQ layout. It can be optimized later without changing the contract.

#include <cstdint>
#include <cuda_fp16.h>

extern "C" {

__device__ __forceinline__ int unpack_qweight(const int32_t word, const int nibble) {
    return (static_cast<uint32_t>(word) >> (4 * nibble)) & 0xF;
}

__device__ __forceinline__ int unpack_qzero(const int32_t word, const int nibble) {
    return ((static_cast<uint32_t>(word) >> (4 * nibble)) & 0xF) + 1;
}

__global__ void gptq_gemm_f16_kernel(
    const __half* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const __half* scales,
    const uint32_t* g_idx,
    __half* out,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t group_size) {

    (void)group_size;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= out_features) return;

    int64_t packed_out_cols = out_features / 8;
    float acc = 0.0f;
    for (int64_t k = 0; k < in_features; ++k) {
        int64_t packed_k = k / 8;
        int64_t nibble_k = k % 8;
        int32_t qweight_word = qweight[packed_k * out_features + col];
        int q = unpack_qweight(qweight_word, static_cast<int>(nibble_k));

        uint32_t group = g_idx[k];
        if (group >= static_cast<uint32_t>(num_groups)) return;
        int32_t qzero_word = qzeros[group * packed_out_cols + (col / 8)];
        int zero = unpack_qzero(qzero_word, static_cast<int>(col % 8));
        float scale = __half2float(scales[group * out_features + col]);
        float weight = static_cast<float>(q - zero) * scale;
        float x_val = __half2float(x[row * in_features + k]);
        acc += x_val * weight;
    }

    out[row * out_features + col] = __float2half(acc);
}

int32_t rllm_gptq_gemm_f16(
    const __half* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const __half* scales,
    const uint32_t* g_idx,
    __half* out,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t group_size,
    cudaStream_t stream) {

    if (m <= 0 || in_features <= 0 || out_features <= 0) return 0;
    if ((in_features % 8) != 0 || (out_features % 8) != 0) return 1;

    dim3 threads(16, 16);
    dim3 blocks(
        static_cast<unsigned int>((out_features + threads.x - 1) / threads.x),
        static_cast<unsigned int>((m + threads.y - 1) / threads.y));
    gptq_gemm_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, qweight, qzeros, scales, g_idx, out, m, in_features, out_features, num_groups, group_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_gptq_gemm_f16_sync(
    const __half* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const __half* scales,
    const uint32_t* g_idx,
    __half* out,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t group_size) {

    int32_t rc = rllm_gptq_gemm_f16(
        x, qweight, qzeros, scales, g_idx, out, m, in_features, out_features, num_groups, group_size, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
