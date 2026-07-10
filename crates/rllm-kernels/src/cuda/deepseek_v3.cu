#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ inline float deepseek_fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;
    if (exp == 0) {
        float value = 0.001953125f * static_cast<float>(mant);
        return sign ? -value : value;
    }
    uint32_t fp32_bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
    return __uint_as_float(fp32_bits);
}

__global__ void deepseek_fp8_block_matmul_f16_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    __half* __restrict__ output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t block_size,
    int64_t in_blocks) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * out_features;
    if (index >= total) return;

    int64_t row = index / out_features;
    int64_t out = index % out_features;
    int64_t out_block = out / block_size;
    float acc = 0.0f;
    for (int64_t col = 0; col < in_features; ++col) {
        int64_t in_block = col / block_size;
        float scale = scales[out_block * in_blocks + in_block];
        float w = deepseek_fp8_e4m3_to_float(weight[out * in_features + col]) * scale;
        acc += __half2float(x[row * in_features + col]) * w;
    }
    output[index] = __float2half_rn(acc);
}

extern "C" int32_t rllm_deepseek_fp8_block_matmul_f16(
    const __half* x,
    const uint8_t* weight,
    const float* scales,
    __half* output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t block_size,
    cudaStream_t stream) {
    if (rows <= 0 || out_features <= 0 || in_features <= 0 || block_size <= 0) return 0;
    int64_t in_blocks = (in_features + block_size - 1) / block_size;
    int64_t total = rows * out_features;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    deepseek_fp8_block_matmul_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, scales, output, rows, out_features, in_features, block_size, in_blocks);
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : static_cast<int32_t>(error);
}

__global__ void deepseek_fp8_selected_expert_matmul_f16_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ weights,
    const float* __restrict__ scales,
    __half* __restrict__ output,
    int64_t tokens,
    int64_t top_k,
    int64_t num_experts,
    int64_t out_features,
    int64_t in_features,
    int64_t block_size,
    int64_t in_blocks,
    int32_t shared_input) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = tokens * top_k * out_features;
    if (index >= total) return;

    int64_t out = index % out_features;
    int64_t route = (index / out_features) % top_k;
    int64_t token = index / (out_features * top_k);
    uint32_t expert = expert_ids[token * top_k + route];
    if (expert >= static_cast<uint32_t>(num_experts)) {
        output[index] = __float2half(0.0f);
        return;
    }

    int64_t input_row = shared_input ? token : token * top_k + route;
    int64_t out_block = out / block_size;
    int64_t expert_weight_offset = static_cast<int64_t>(expert) * out_features * in_features;
    int64_t expert_scale_offset = static_cast<int64_t>(expert) *
        ((out_features + block_size - 1) / block_size) * in_blocks;
    float acc = 0.0f;
    for (int64_t col = 0; col < in_features; ++col) {
        int64_t in_block = col / block_size;
        float scale = scales[expert_scale_offset + out_block * in_blocks + in_block];
        uint8_t bits = weights[expert_weight_offset + out * in_features + col];
        float w = deepseek_fp8_e4m3_to_float(bits) * scale;
        acc += __half2float(x[input_row * in_features + col]) * w;
    }
    output[index] = __float2half_rn(acc);
}

extern "C" int32_t rllm_deepseek_fp8_selected_expert_matmul_f16(
    const __half* x,
    const uint32_t* expert_ids,
    const uint8_t* weights,
    const float* scales,
    __half* output,
    int64_t tokens,
    int64_t top_k,
    int64_t num_experts,
    int64_t out_features,
    int64_t in_features,
    int64_t block_size,
    int32_t shared_input,
    cudaStream_t stream) {
    if (tokens <= 0 || top_k <= 0 || out_features <= 0 || in_features <= 0) return 0;
    int64_t in_blocks = (in_features + block_size - 1) / block_size;
    int64_t total = tokens * top_k * out_features;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    deepseek_fp8_selected_expert_matmul_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, expert_ids, weights, scales, output, tokens, top_k, num_experts,
        out_features, in_features, block_size, in_blocks, shared_input);
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : static_cast<int32_t>(error);
}
