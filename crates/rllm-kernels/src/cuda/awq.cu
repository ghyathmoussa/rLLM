// rLLM CUDA kernels — AWQ packed INT4 GEMM.
// Correctness-first implementation matching the standard AWQ packing format.

#include <cstdint>
#include <cuda_fp16.h>

extern "C" {

__device__ __forceinline__ int unpack_awq_qweight(const int32_t word, const int nibble) {
    return (static_cast<uint32_t>(word) >> (4 * nibble)) & 0xF;
}

__device__ __forceinline__ int unpack_awq_qzero(const int32_t word, const int nibble) {
    return (static_cast<uint32_t>(word) >> (4 * nibble)) & 0xF;
}

__global__ void awq_gemm_f16_kernel(
    const __half* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const __half* scales,
    __half* out,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t group_size) {

    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= out_features) return;

    int64_t packed_out_cols = out_features / 8;
    int64_t col_idx_8 = col / 8;
    int64_t j = col % 8;

    // AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    const int awq_reverse_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    int nibble = awq_reverse_order[j];

    float acc = 0.0f;
    for (int64_t k = 0; k < in_features; ++k) {
        int32_t qweight_word = qweight[k * packed_out_cols + col_idx_8];
        int q = unpack_awq_qweight(qweight_word, nibble);

        int64_t group = k / group_size;
        int32_t qzero_word = qzeros[group * packed_out_cols + col_idx_8];
        int zero = unpack_awq_qzero(qzero_word, nibble);

        float scale = __half2float(scales[group * out_features + col]);
        float weight = static_cast<float>(q - zero) * scale;
        float x_val = __half2float(x[row * in_features + k]);
        acc += x_val * weight;
    }

    out[row * out_features + col] = __float2half(acc);
}

int32_t rllm_awq_gemm_f16(
    const __half* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const __half* scales,
    __half* out,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t group_size,
    cudaStream_t stream) {

    if (m <= 0 || in_features <= 0 || out_features <= 0) return 0;
    if ((out_features % 8) != 0) return 1;

    dim3 threads(16, 16);
    dim3 blocks(
        static_cast<unsigned int>((out_features + threads.x - 1) / threads.x),
        static_cast<unsigned int>((m + threads.y - 1) / threads.y));
    awq_gemm_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, qweight, qzeros, scales, out, m, in_features, out_features, num_groups, group_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_awq_gemm_f16_sync(
    const __half* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const __half* scales,
    __half* out,
    int64_t m,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups,
    int64_t group_size) {

    int32_t rc = rllm_awq_gemm_f16(
        x, qweight, qzeros, scales, out, m, in_features, out_features, num_groups, group_size, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
