// rLLM CUDA cache kernels — cache write, block copy, and zeroing for Phase 6.
// All kernel entry points use C linkage for Rust FFI.

#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

extern "C" {

// ── Cache Write ──────────────────────────────────────────────────────────
// Writes new key/value data into the physical KV cache at positions given by
// a slot mapping. Each slot maps a token index to a (block, offset) pair.
//
// The slot is encoded as: slot = block_id * block_size + offset_in_block
//
// key_cache:   [num_blocks, num_kv_heads, head_dim, block_size] (NHD layout)
// value_cache: [num_blocks, num_kv_heads, head_dim, block_size] (NHD layout)
// new_key:     [num_tokens, num_kv_heads, head_dim]
// new_value:   [num_tokens, num_kv_heads, head_dim]
// slot_mapping: [num_tokens] — slot index for each token (-1 = skip)

__global__ void cache_write_kernel(
    __half* key_cache,
    __half* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_tokens * num_kv_heads * head_dim;
    if (idx >= total) return;

    int64_t token_idx = idx / (num_kv_heads * head_dim);
    int64_t remainder = idx % (num_kv_heads * head_dim);
    int64_t kv_head = remainder / head_dim;
    int64_t d = remainder % head_dim;

    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return; // skip padding tokens

    int64_t block_id = slot / block_size;
    int64_t block_offset = slot % block_size;

    if (block_id >= num_blocks) return;

    // NHD layout: [num_blocks, num_kv_heads, head_dim, block_size]
    int64_t cache_idx = ((block_id * num_kv_heads + kv_head) * head_dim + d) * block_size + block_offset;

    // Key
    int64_t key_src = (token_idx * num_kv_heads + kv_head) * head_dim + d;
    key_cache[cache_idx] = new_key[key_src];

    // Value
    int64_t val_src = (token_idx * num_kv_heads + kv_head) * head_dim + d;
    value_cache[cache_idx] = new_value[val_src];
}

int32_t rllm_cache_write_f16(
    __half* key_cache,
    __half* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks,
    cudaStream_t stream) {

    if (num_tokens <= 0) return 0;
    int64_t total = num_tokens * num_kv_heads * head_dim;
    int64_t threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    cache_write_kernel<<<blocks, threads, 0, stream>>>(
        key_cache, value_cache, new_key, new_value, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_cache_write_f16_sync(
    __half* key_cache,
    __half* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks) {

    int32_t rc = rllm_cache_write_f16(
        key_cache, value_cache, new_key, new_value, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── Cache Block Copy ─────────────────────────────────────────────────────
// Copies N full blocks from src to dst for prefix sharing/forking.
// block_nbytes is the size of one block in bytes.

__global__ void cache_block_copy_kernel(
    const uint8_t* src,
    uint8_t* dst,
    int64_t block_nbytes,
    int64_t num_blocks) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = block_nbytes * num_blocks;
    if (idx >= total) return;
    dst[idx] = src[idx];
}

int32_t rllm_cache_block_copy(
    const uint8_t* src,
    uint8_t* dst,
    int64_t block_nbytes,
    int64_t num_blocks,
    cudaStream_t stream) {

    if (block_nbytes <= 0 || num_blocks <= 0) return 0;
    int64_t total = block_nbytes * num_blocks;
    int64_t threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    cache_block_copy_kernel<<<blocks, threads, 0, stream>>>(src, dst, block_nbytes, num_blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_cache_block_copy_sync(
    const uint8_t* src,
    uint8_t* dst,
    int64_t block_nbytes,
    int64_t num_blocks) {

    int32_t rc = rllm_cache_block_copy(src, dst, block_nbytes, num_blocks, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── Cache Zero ───────────────────────────────────────────────────────────
// Zero out nbytes of cache memory.

__global__ void cache_zero_kernel(uint8_t* ptr, int64_t nbytes) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbytes) {
        ptr[idx] = 0;
    }
}

int32_t rllm_cache_zero(uint8_t* ptr, int64_t nbytes, cudaStream_t stream) {
    if (nbytes <= 0) return 0;
    int64_t threads = 256;
    int64_t blocks = (nbytes + threads - 1) / threads;
    cache_zero_kernel<<<blocks, threads, 0, stream>>>(ptr, nbytes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_cache_zero_sync(uint8_t* ptr, int64_t nbytes) {
    int32_t rc = rllm_cache_zero(ptr, nbytes, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

// ── GPU Memory Alloc/Free ────────────────────────────────────────────────

int32_t rllm_gpu_alloc(void** ptr, int64_t nbytes) {
    cudaError_t err = cudaMalloc(ptr, static_cast<size_t>(nbytes));
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_gpu_free(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_gpu_alloc_host(void** ptr, int64_t nbytes) {
    cudaError_t err = cudaMallocHost(ptr, static_cast<size_t>(nbytes));
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_gpu_free_host(void* ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
