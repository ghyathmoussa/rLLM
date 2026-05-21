// rLLM PagedAttention CUDA kernels — decode and prefill paths for Phase 7.
// All kernel entry points use C linkage for Rust FFI.

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>

extern "C" {

// ── Decode PagedAttention (FP16) ──────────────────────────────────────────
//
// One query token per scheduled sequence. Gathers scattered KV cache blocks
// via block tables, computes scaled dot-product attention with online softmax.
//
// Thread mapping: one thread block per (sequence, Q head).
// Each thread (tid < head_dim) holds one element of the Q vector and
// cooperatively computes the QK dot product via shared memory reduction.
//
// Q layout:      [num_seqs, num_q_heads, head_dim]
// K/V cache:     [num_blocks, num_kv_heads, head_dim, block_size] (NHD)
// block_tables:  [num_seqs, max_num_blocks_per_seq] (int32, -1 = unused)
// seq_lens:      [num_seqs] (int32)
// output:        [num_seqs, num_q_heads, head_dim]

__global__ void paged_attention_decode_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    int64_t num_seqs,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks_per_seq,
    float scale) {

    int64_t seq_idx = blockIdx.x / num_q_heads;
    int64_t q_head  = blockIdx.x % num_q_heads;
    int64_t tid     = threadIdx.x;

    if (seq_idx >= num_seqs) return;

    int64_t seq_len = seq_lens[seq_idx];
    if (seq_len == 0 || tid >= head_dim) return;

    // GQA: map Q head to KV head
    int64_t kv_head = q_head * num_kv_heads / num_q_heads;

    // Each thread loads its Q element once
    int64_t q_idx = (seq_idx * num_q_heads + q_head) * head_dim + tid;
    float q_val = __half2float(query[q_idx]);

    // Shared memory for block reduction of dot products
    extern __shared__ float s_data[];

    // Online softmax state
    float max_logits = -INFINITY;
    float exp_sum = 0.0f;
    float out_val = 0.0f;

    int64_t num_blocks_needed = (seq_len + block_size - 1) / block_size;

    for (int64_t b = 0; b < num_blocks_needed; b++) {
        int64_t physical_block = block_tables[seq_idx * max_num_blocks_per_seq + b];
        if (physical_block < 0) continue;

        int64_t block_start = b * block_size;
        int64_t block_end = block_start + block_size;
        if (block_end > seq_len) block_end = seq_len;

        for (int64_t offset = block_start; offset < block_end; offset++) {
            int64_t blk_off = offset - block_start;

            // Load K for this (kv_head, position) — each thread gets one element
            int64_t k_idx = ((physical_block * num_kv_heads + kv_head) * head_dim + tid)
                            * block_size + blk_off;
            float k_val = __half2float(key_cache[k_idx]);

            // Partial QK dot product
            s_data[tid] = q_val * k_val;
            __syncthreads();

            // Parallel reduction in shared memory
            for (int64_t stride = (head_dim + 1) / 2; stride > 0; stride >>= 1) {
                if (tid < stride && (tid + stride) < head_dim) {
                    s_data[tid] += s_data[tid + stride];
                }
                __syncthreads();
            }

            float logit = s_data[0] * scale;

            // Online softmax update
            float old_max = max_logits;
            max_logits = fmaxf(max_logits, logit);
            float correction = expf(old_max - max_logits);
            exp_sum = exp_sum * correction + expf(logit - max_logits);

            // Weighted V accumulation
            int64_t v_idx = ((physical_block * num_kv_heads + kv_head) * head_dim + tid)
                            * block_size + blk_off;
            float v_val = __half2float(value_cache[v_idx]);
            out_val = out_val * correction + v_val * expf(logit - max_logits);
        }
    }

    // Normalize and write output
    if (exp_sum > 0.0f) {
        out_val /= exp_sum;
    }
    int64_t out_idx = (seq_idx * num_q_heads + q_head) * head_dim + tid;
    output[out_idx] = __float2half(out_val);
}

int32_t rllm_paged_attention_decode_f16(
    __half* output,
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int64_t num_seqs,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks_per_seq,
    float scale,
    cudaStream_t stream) {

    if (num_seqs <= 0) return 0;
    int64_t grid = num_seqs * num_q_heads;
    int64_t threads = ((head_dim + 31) / 32) * 32; // round up to nearest warp
    if (threads > 1024) threads = 1024;
    int64_t shared_mem = threads * sizeof(float);

    paged_attention_decode_kernel<<<grid, threads, shared_mem, stream>>>(
        output, query, key_cache, value_cache, block_tables, seq_lens,
        num_seqs, num_q_heads, num_kv_heads, head_dim, block_size,
        max_num_blocks_per_seq, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_paged_attention_decode_f16_sync(
    __half* output,
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int64_t num_seqs,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks_per_seq,
    float scale) {

    int32_t rc = rllm_paged_attention_decode_f16(
        output, query, key_cache, value_cache, block_tables, seq_lens,
        num_seqs, num_q_heads, num_kv_heads, head_dim, block_size,
        max_num_blocks_per_seq, scale, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── Prefill PagedAttention (FP16) ─────────────────────────────────────────
//
// Multiple query tokens per sequence with paged KV access and causal masking.
//
// Thread mapping: one thread block per (token, Q head).
// Each thread (tid < head_dim) holds one Q element and iterates over all
// KV positions up to the causal limit (position within sequence).
//
// Q layout:      [num_tokens, num_q_heads, head_dim]
// K/V cache:     [num_blocks, num_kv_heads, head_dim, block_size] (NHD)
// block_tables:  [num_seqs, max_num_blocks_per_seq] (int32)
// seq_lens:      [num_seqs] (int32) — total KV length per sequence
// query_start_loc: [num_seqs + 1] (int32) — prefix sum of token counts
// output:        [num_tokens, num_q_heads, head_dim]

__global__ void paged_attention_prefill_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_loc,
    int64_t num_seqs,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks_per_seq,
    float scale) {

    int64_t token_idx = blockIdx.x / num_q_heads;
    int64_t q_head   = blockIdx.x % num_q_heads;
    int64_t tid      = threadIdx.x;

    if (token_idx >= num_tokens || tid >= head_dim) return;

    // Find which sequence this token belongs to (binary search on query_start_loc)
    int64_t seq_idx = 0;
    for (int64_t s = num_seqs - 1; s >= 0; s--) {
        if (token_idx >= query_start_loc[s]) {
            seq_idx = s;
            break;
        }
    }

    // Position within the sequence (0-indexed, for causal masking)
    int64_t position = token_idx - query_start_loc[seq_idx];
    int64_t causal_len = position + 1; // attend to positions 0..position

    // GQA
    int64_t kv_head = q_head * num_kv_heads / num_q_heads;

    // Load Q element
    int64_t q_idx = (token_idx * num_q_heads + q_head) * head_dim + tid;
    float q_val = __half2float(query[q_idx]);

    extern __shared__ float s_data[];

    float max_logits = -INFINITY;
    float exp_sum = 0.0f;
    float out_val = 0.0f;

    int64_t num_blocks_needed = (causal_len + block_size - 1) / block_size;

    for (int64_t b = 0; b < num_blocks_needed; b++) {
        int64_t physical_block = block_tables[seq_idx * max_num_blocks_per_seq + b];
        if (physical_block < 0) continue;

        int64_t block_start = b * block_size;
        int64_t block_end = block_start + block_size;
        if (block_end > causal_len) block_end = causal_len;

        for (int64_t offset = block_start; offset < block_end; offset++) {
            int64_t blk_off = offset - block_start;

            int64_t k_idx = ((physical_block * num_kv_heads + kv_head) * head_dim + tid)
                            * block_size + blk_off;
            float k_val = __half2float(key_cache[k_idx]);

            s_data[tid] = q_val * k_val;
            __syncthreads();

            for (int64_t stride = (head_dim + 1) / 2; stride > 0; stride >>= 1) {
                if (tid < stride && (tid + stride) < head_dim) {
                    s_data[tid] += s_data[tid + stride];
                }
                __syncthreads();
            }

            float logit = s_data[0] * scale;

            float old_max = max_logits;
            max_logits = fmaxf(max_logits, logit);
            float correction = expf(old_max - max_logits);
            exp_sum = exp_sum * correction + expf(logit - max_logits);

            int64_t v_idx = ((physical_block * num_kv_heads + kv_head) * head_dim + tid)
                            * block_size + blk_off;
            float v_val = __half2float(value_cache[v_idx]);
            out_val = out_val * correction + v_val * expf(logit - max_logits);
        }
    }

    if (exp_sum > 0.0f) {
        out_val /= exp_sum;
    }
    int64_t out_idx = (token_idx * num_q_heads + q_head) * head_dim + tid;
    output[out_idx] = __float2half(out_val);
}

int32_t rllm_paged_attention_prefill_f16(
    __half* output,
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    const int32_t* query_start_loc,
    int64_t num_seqs,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks_per_seq,
    float scale,
    cudaStream_t stream) {

    if (num_tokens <= 0) return 0;
    int64_t grid = num_tokens * num_q_heads;
    int64_t threads = ((head_dim + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;
    int64_t shared_mem = threads * sizeof(float);

    paged_attention_prefill_kernel<<<grid, threads, shared_mem, stream>>>(
        output, query, key_cache, value_cache, block_tables, seq_lens,
        query_start_loc, num_seqs, num_tokens, num_q_heads, num_kv_heads,
        head_dim, block_size, max_num_blocks_per_seq, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_paged_attention_prefill_f16_sync(
    __half* output,
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    const int32_t* query_start_loc,
    int64_t num_seqs,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks_per_seq,
    float scale) {

    int32_t rc = rllm_paged_attention_prefill_f16(
        output, query, key_cache, value_cache, block_tables, seq_lens,
        query_start_loc, num_seqs, num_tokens, num_q_heads, num_kv_heads,
        head_dim, block_size, max_num_blocks_per_seq, scale, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
