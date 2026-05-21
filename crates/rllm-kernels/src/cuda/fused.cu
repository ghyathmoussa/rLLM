// rLLM fused CUDA kernels — RMSNorm, RoPE, SiLU-mul for Phase 15.
// All kernel entry points use C linkage for Rust FFI.

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>

extern "C" {

// ── Fused RMSNorm (FP16) ──────────────────────────────────────────────────
// Applies RMS normalization: output = input / sqrt(mean(input^2) + eps) * weight
//
// input:    [n_elements] (__half)
// weight:   [hidden_size] (__half) — broadcast across rows
// output:   [n_elements] (__half)
// hidden_size: size of the last dimension
// n_elements: total number of elements (must be multiple of hidden_size)
// eps:      small constant for numerical stability (e.g. 1e-6)

__global__ void fused_rmsnorm_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    int64_t hidden_size,
    int64_t n_elements,
    float eps) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t num_rows = n_elements / hidden_size;
    int64_t row = idx / hidden_size;
    int64_t col = idx % hidden_size;

    if (row >= num_rows) return;

    // Compute sum of squares for this row using a single-thread approach.
    // Each thread computes its own element's contribution, then we reduce.
    float val = __half2float(input[idx]);
    float sq = val * val;

    // Shared memory for block-level reduction
    extern __shared__ float s_sum[];
    s_sum[threadIdx.x] = sq;
    __syncthreads();

    // Only threads within hidden_size contribute to the same row.
    // We reduce across the block.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < hidden_size) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = s_sum[0] / static_cast<float>(hidden_size);
    float rms = rsqrtf(variance + eps);

    float w = __half2float(weight[col]);
    output[idx] = __float2half(val * rms * w);
}

int32_t rllm_fused_rmsnorm_f16(
    __half* output,
    const __half* input,
    const __half* weight,
    int64_t hidden_size,
    int64_t n_elements,
    float eps,
    cudaStream_t stream) {

    if (n_elements <= 0 || hidden_size <= 0) return 0;
    int64_t threads = 256;
    // Round threads to at least hidden_size for correct reduction
    if (threads < hidden_size) threads = ((hidden_size + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;
    int64_t blocks = (n_elements + threads - 1) / threads;
    int64_t shared_mem = threads * sizeof(float);

    fused_rmsnorm_kernel<<<blocks, threads, shared_mem, stream>>>(
        output, input, weight, hidden_size, n_elements, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_fused_rmsnorm_f16_sync(
    __half* output,
    const __half* input,
    const __half* weight,
    int64_t hidden_size,
    int64_t n_elements,
    float eps) {

    int32_t rc = rllm_fused_rmsnorm_f16(output, input, weight, hidden_size, n_elements, eps, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── Fused SiLU-Mul (FP16) ─────────────────────────────────────────────────
// Computes output[i] = silu(gate[i]) * up[i] where silu(x) = x * sigmoid(x).
// Used in SwiGLU MLP: gate_proj(x).silu() * up_proj(x).
//
// output:  [n_elements] (__half)
// gate:    [n_elements] (__half)
// up:      [n_elements] (__half)

__global__ void fused_silu_mul_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    int64_t n_elements) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_val = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu_val * u);
}

int32_t rllm_fused_silu_mul_f16(
    __half* output,
    const __half* gate,
    const __half* up,
    int64_t n_elements,
    cudaStream_t stream) {

    if (n_elements <= 0) return 0;
    int64_t threads = 256;
    int64_t blocks = (n_elements + threads - 1) / threads;
    fused_silu_mul_kernel<<<blocks, threads, 0, stream>>>(output, gate, up, n_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_fused_silu_mul_f16_sync(
    __half* output,
    const __half* gate,
    const __half* up,
    int64_t n_elements) {

    int32_t rc = rllm_fused_silu_mul_f16(output, gate, up, n_elements, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── Fused RoPE (FP16) ─────────────────────────────────────────────────────
// Applies rotary position embeddings to query and key tensors.
//
// For each token t and head dimension pair (2i, 2i+1):
//   freq = 1.0 / (theta ^ (2i / head_dim))
//   cos_val = cos(position * freq)
//   sin_val = sin(position * freq)
//   out[2i]   = q[2i]   * cos_val - q[2i+1] * sin_val
//   out[2i+1] = q[2i+1] * cos_val + q[2i]   * sin_val
//
// query/key layout: [num_tokens, num_heads, head_dim]
// positions:        [num_tokens] (int32)
// out_q/out_k:      same layout as query/key

__global__ void fused_rope_kernel(
    __half* __restrict__ out_q,
    __half* __restrict__ out_k,
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const int32_t* __restrict__ positions,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float rope_theta) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_q = num_tokens * num_q_heads * head_dim;
    if (idx >= total_q) return;

    int64_t d = idx % head_dim;
    if (d % 2 != 0) return; // only process even indices; odd handled together

    int64_t remainder = idx / head_dim;
    int64_t token = remainder / num_q_heads;
    int64_t q_head = remainder % num_q_heads;

    float pos = static_cast<float>(positions[token]);
    float dim = static_cast<float>(d);
    float freq = 1.0f / powf(rope_theta, dim / static_cast<float>(head_dim));
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Apply to query
    int64_t q_idx_even = idx;
    int64_t q_idx_odd = idx + 1;
    float q_even = __half2float(query[q_idx_even]);
    float q_odd = __half2float(query[q_idx_odd]);
    out_q[q_idx_even] = __float2half(q_even * cos_val - q_odd * sin_val);
    out_q[q_idx_odd]  = __float2half(q_odd * cos_val + q_even * sin_val);

    // Apply to key (GQA: map q_head to kv_head)
    int64_t kv_head = q_head * num_kv_heads / num_q_heads;
    int64_t k_base = (token * num_kv_heads + kv_head) * head_dim;
    int64_t k_idx_even = k_base + d;
    int64_t k_idx_odd = k_base + d + 1;

    if (k_idx_odd < (token + 1) * num_kv_heads * head_dim) {
        float k_even = __half2float(key[k_idx_even]);
        float k_odd = __half2float(key[k_idx_odd]);
        out_k[k_idx_even] = __float2half(k_even * cos_val - k_odd * sin_val);
        out_k[k_idx_odd]  = __float2half(k_odd * cos_val + k_even * sin_val);
    }
}

int32_t rllm_fused_rope_f16(
    __half* out_q,
    __half* out_k,
    const __half* query,
    const __half* key,
    const int32_t* positions,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float rope_theta,
    cudaStream_t stream) {

    if (num_tokens <= 0) return 0;
    int64_t total_q = num_tokens * num_q_heads * head_dim;
    int64_t threads = 256;
    int64_t blocks = (total_q + threads - 1) / threads;
    fused_rope_kernel<<<blocks, threads, 0, stream>>>(
        out_q, out_k, query, key, positions,
        num_tokens, num_q_heads, num_kv_heads, head_dim, rope_theta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_fused_rope_f16_sync(
    __half* out_q,
    __half* out_k,
    const __half* query,
    const __half* key,
    const int32_t* positions,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float rope_theta) {

    int32_t rc = rllm_fused_rope_f16(
        out_q, out_k, query, key, positions,
        num_tokens, num_q_heads, num_kv_heads, head_dim, rope_theta, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
