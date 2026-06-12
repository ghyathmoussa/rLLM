// rLLM CUDA INT8 quantized matmul kernels.

#include <cstdint>
#include <cuda_fp16.h>
#include <cmath>

extern "C" {

__device__ inline int8_t quantize_i8(float value, float inv_scale) {
    float q = nearbyintf(value * inv_scale);
    q = fminf(127.0f, fmaxf(-127.0f, q));
    return static_cast<int8_t>(q);
}

__global__ void int8_matmul_w8a8_f16_kernel(
    const __half* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ weight_scale,
    __half* __restrict__ output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * out_features;
    if (idx >= total) return;

    int64_t row = idx / out_features;
    int64_t out = idx % out_features;
    const __half* x_row = x + row * in_features;
    const int8_t* w_row = qweight + out * in_features;

    float absmax = 0.0f;
    for (int64_t k = 0; k < in_features; ++k) {
        absmax = fmaxf(absmax, fabsf(__half2float(x_row[k])));
    }
    float act_scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    float inv_act_scale = 1.0f / act_scale;

    int32_t acc = 0;
    for (int64_t k = 0; k < in_features; ++k) {
        int8_t qx = quantize_i8(__half2float(x_row[k]), inv_act_scale);
        acc += static_cast<int32_t>(qx) * static_cast<int32_t>(w_row[k]);
    }

    float deq = static_cast<float>(acc) * act_scale * weight_scale[out];
    output[idx] = __float2half_rn(deq);
}

int32_t rllm_int8_matmul_w8a8_f16(
    const __half* x,
    const int8_t* qweight,
    const float* weight_scale,
    __half* output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    cudaStream_t stream) {

    if (rows <= 0 || out_features <= 0 || in_features <= 0) return 0;
    int64_t total = rows * out_features;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    int8_matmul_w8a8_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, qweight, weight_scale, output, rows, out_features, in_features);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_int8_matmul_w8a8_f16_sync(
    const __half* x,
    const int8_t* qweight,
    const float* weight_scale,
    __half* output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {

    int32_t rc = rllm_int8_matmul_w8a8_f16(
        x, qweight, weight_scale, output, rows, out_features, in_features, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
