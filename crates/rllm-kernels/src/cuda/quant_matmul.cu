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

__device__ inline float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;
    
    if (exp == 0) {
        float val = 0.001953125f * static_cast<float>(mant);
        return sign ? -val : val;
    } else {
        uint32_t float_val_bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
        return __uint_as_float(float_val_bits);
    }
}

__device__ inline float fp4_e2m1_to_float(uint8_t code) {
    const float lut[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[code & 0x0F];
}

__global__ void mxfp8_matmul_w8a16_f16_kernel(
    const __half* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * out_features;
    if (idx >= total) return;

    int64_t row = idx / out_features;
    int64_t out = idx % out_features;
    const __half* x_row = x + row * in_features;
    const int8_t* w_row = qweight + out * in_features;
    int64_t num_groups = in_features / group_size;
    const __half* scale_row = scales + out * num_groups;

    float acc = 0.0f;
    for (int64_t g = 0; g < num_groups; ++g) {
        float scale = __half2float(scale_row[g]);
        for (int64_t k = 0; k < group_size; ++k) {
            int64_t col = g * group_size + k;
            uint8_t qw = static_cast<uint8_t>(w_row[col]);
            float w_val = fp8_e4m3_to_float(qw) * scale;
            acc += __half2float(x_row[col]) * w_val;
        }
    }

    output[idx] = __float2half_rn(acc);
}

__global__ void mxfp4_matmul_w4a16_f16_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * out_features;
    if (idx >= total) return;

    int64_t row = idx / out_features;
    int64_t out = idx % out_features;
    const __half* x_row = x + row * in_features;
    const uint8_t* w_row = qweight + out * (in_features / 2);
    int64_t num_groups = in_features / group_size;
    const __half* scale_row = scales + out * num_groups;

    float acc = 0.0f;
    for (int64_t g = 0; g < num_groups; ++g) {
        float scale = __half2float(scale_row[g]);
        for (int64_t k = 0; k < group_size; ++k) {
            int64_t col = g * group_size + k;
            uint8_t byte = w_row[col / 2];
            uint8_t code = (col % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float w_val = fp4_e2m1_to_float(code) * scale;
            acc += __half2float(x_row[col]) * w_val;
        }
    }

    output[idx] = __float2half_rn(acc);
}

int32_t rllm_mxfp8_matmul_w8a16_f16(
    const __half* x,
    const int8_t* qweight,
    const __half* scales,
    __half* output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size,
    cudaStream_t stream) {

    if (rows <= 0 || out_features <= 0 || in_features <= 0) return 0;
    int64_t total = rows * out_features;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    mxfp8_matmul_w8a16_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, qweight, scales, output, rows, out_features, in_features, group_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_mxfp4_matmul_w4a16_f16(
    const __half* x,
    const uint8_t* qweight,
    const __half* scales,
    __half* output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size,
    cudaStream_t stream) {

    if (rows <= 0 || out_features <= 0 || in_features <= 0) return 0;
    int64_t total = rows * out_features;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    mxfp4_matmul_w4a16_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, qweight, scales, output, rows, out_features, in_features, group_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

__global__ void nvfp4_matmul_w4a16_f16_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const __half* __restrict__ scales,
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
    const uint8_t* w_row = qweight + out * (in_features / 2);
    int64_t num_groups = in_features / 16;
    const __half* scale_row = scales + out * num_groups;

    float acc = 0.0f;
    for (int64_t g = 0; g < num_groups; ++g) {
        float scale = __half2float(scale_row[g]);
        #pragma unroll
        for (int64_t k = 0; k < 16; ++k) {
            int64_t col = g * 16 + k;
            uint8_t byte = w_row[col / 2];
            uint8_t code = (col % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float w_val = fp4_e2m1_to_float(code) * scale;
            acc += __half2float(x_row[col]) * w_val;
        }
    }

    output[idx] = __float2half_rn(acc);
}

int32_t rllm_nvfp4_matmul_w4a16_f16(
    const __half* x,
    const uint8_t* qweight,
    const __half* scales,
    __half* output,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    cudaStream_t stream) {

    if (rows <= 0 || out_features <= 0 || in_features <= 0) return 0;
    int64_t total = rows * out_features;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    nvfp4_matmul_w4a16_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, qweight, scales, output, rows, out_features, in_features);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
