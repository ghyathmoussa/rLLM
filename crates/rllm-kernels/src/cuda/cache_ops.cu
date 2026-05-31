// rLLM CUDA cache kernels — cache write, block copy, and zeroing for Phase 6.
// All kernel entry points use C linkage for Rust FFI.

#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cmath>

// FP8 conversion helpers (software emulated)
// E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7)
// E5M2 (1 sign bit, 5 exponent bits, 2 mantissa bits, bias = 15)

__device__ inline uint8_t float_to_fp8_e4m3(float val) {
    if (isnan(val)) {
        return 0x7F; // NaN representation
    }
    union {
        float f;
        uint32_t i;
    } u;
    u.f = val;
    uint32_t sign = (u.i >> 31) & 0x01;
    uint32_t exp = (u.i >> 23) & 0xFF;
    uint32_t mant = u.i & 0x7FFFFF;

    if (exp == 0) {
        // zero or subnormal
        return sign << 7;
    } else if (exp == 255) {
        // infinity or NaN
        // Clip to max e4m3 value (448)
        return (sign << 7) | 0x7E;
    }

    int new_exp = (int)exp - 127 + 7;
    if (new_exp <= 0) {
        // underflow to subnormal/zero
        return sign << 7;
    } else if (new_exp >= 16) {
        // overflow, clip to max
        return (sign << 7) | 0x7E; // max magnitude is 448
    }

    uint8_t res = (sign << 7) | (new_exp << 3) | (mant >> 20);
    return res;
}

__device__ inline float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 0x01;
    uint32_t exp = (val >> 3) & 0x0F;
    uint32_t mant = val & 0x07;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal: (-1)^sign * 2^(-6) * (0.mant)
        float f = (float)mant * 0.0078125f; // mant / 128
        return sign ? -f : f;
    } else if (exp == 15 && mant == 7) {
        return nanf(""); // NaN
    }

    // Normal: (-1)^sign * 2^(exp - 7) * (1.mant)
    int shift = (int)exp - 7;
    float f = 1.0f + (float)mant * 0.125f;
    if (shift >= 0) {
        f *= (float)(1 << shift);
    } else {
        f /= (float)(1 << (-shift));
    }
    return sign ? -f : f;
}

__device__ inline uint8_t float_to_fp8_e5m2(float val) {
    if (isnan(val)) {
        return 0x7F;
    }
    union {
        float f;
        uint32_t i;
    } u;
    u.f = val;
    uint32_t sign = (u.i >> 31) & 0x01;
    uint32_t exp = (u.i >> 23) & 0xFF;
    uint32_t mant = u.i & 0x7FFFFF;

    if (exp == 0) {
        return sign << 7;
    } else if (exp == 255) {
        return (sign << 7) | 0x7B; // max value
    }

    int new_exp = (int)exp - 127 + 15;
    if (new_exp <= 0) {
        return sign << 7;
    } else if (new_exp >= 31) {
        return (sign << 7) | 0x7B; // overflow, clip to max E5M2 (57344)
    }

    uint8_t res = (sign << 7) | (new_exp << 2) | (mant >> 21);
    return res;
}

__device__ inline float fp8_e5m2_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 0x01;
    uint32_t exp = (val >> 2) & 0x1F;
    uint32_t mant = val & 0x03;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        float f = (float)mant * 0.000015258789f; // mant * 2^(-16)
        return sign ? -f : f;
    } else if (exp == 31) {
        return nanf("");
    }

    int shift = (int)exp - 15;
    float f = 1.0f + (float)mant * 0.25f;
    if (shift >= 0) {
        f *= (float)(1 << shift);
    } else {
        f /= (float)(1 << (-shift));
    }
    return sign ? -f : f;
}

__device__ inline uint8_t float_to_fp8(float val, bool is_e5m2) {
    if (is_e5m2) {
        return float_to_fp8_e5m2(val);
    } else {
        return float_to_fp8_e4m3(val);
    }
}

__device__ inline float fp8_to_float(uint8_t val, bool is_e5m2) {
    if (is_e5m2) {
        return fp8_e5m2_to_float(val);
    } else {
        return fp8_e4m3_to_float(val);
    }
}

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

__global__ void cache_write_fp8_kernel(
    uint8_t* key_cache,
    uint8_t* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks,
    int32_t is_e5m2) {

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
    float k_val = __half2float(new_key[key_src]);
    key_cache[cache_idx] = float_to_fp8(k_val, is_e5m2 != 0);

    // Value
    int64_t val_src = (token_idx * num_kv_heads + kv_head) * head_dim + d;
    float v_val = __half2float(new_value[val_src]);
    value_cache[cache_idx] = float_to_fp8(v_val, is_e5m2 != 0);
}

int32_t rllm_cache_write_fp8(
    uint8_t* key_cache,
    uint8_t* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks,
    int32_t is_e5m2,
    cudaStream_t stream) {

    if (num_tokens <= 0) return 0;
    int64_t total = num_tokens * num_kv_heads * head_dim;
    int64_t threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    cache_write_fp8_kernel<<<blocks, threads, 0, stream>>>(
        key_cache, value_cache, new_key, new_value, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks, is_e5m2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_cache_write_fp8_sync(
    uint8_t* key_cache,
    uint8_t* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks,
    int32_t is_e5m2) {

    int32_t rc = rllm_cache_write_fp8(
        key_cache, value_cache, new_key, new_value, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks, is_e5m2, 0);
    if (rc != 0) return rc;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

__device__ inline int8_t float_to_i8_fixed(float val) {
    float clipped = fminf(1.0f, fmaxf(-1.0f, val));
    return static_cast<int8_t>(nearbyintf(clipped * 127.0f));
}

__global__ void cache_write_i8_kernel(
    int8_t* key_cache,
    int8_t* value_cache,
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

    int64_t key_src = (token_idx * num_kv_heads + kv_head) * head_dim + d;
    key_cache[cache_idx] = float_to_i8_fixed(__half2float(new_key[key_src]));

    int64_t val_src = (token_idx * num_kv_heads + kv_head) * head_dim + d;
    value_cache[cache_idx] = float_to_i8_fixed(__half2float(new_value[val_src]));
}

int32_t rllm_cache_write_i8(
    int8_t* key_cache,
    int8_t* value_cache,
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
    cache_write_i8_kernel<<<blocks, threads, 0, stream>>>(
        key_cache, value_cache, new_key, new_value, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_cache_write_i8_sync(
    int8_t* key_cache,
    int8_t* value_cache,
    const __half* new_key,
    const __half* new_value,
    const int64_t* slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks) {

    int32_t rc = rllm_cache_write_i8(
        key_cache, value_cache, new_key, new_value, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks, 0);
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

int32_t rllm_gpu_memcpy_to_device(void* dst, const void* src, int64_t nbytes) {
    cudaError_t err = cudaMemcpy(dst, src, static_cast<size_t>(nbytes), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

int32_t rllm_gpu_memcpy_to_host(void* dst, const void* src, int64_t nbytes) {
    cudaError_t err = cudaMemcpy(dst, src, static_cast<size_t>(nbytes), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

} // extern "C"
