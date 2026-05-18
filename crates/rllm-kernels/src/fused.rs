//! Fused CUDA kernels for RMSNorm, RoPE, and SiLU-mul.
//!
//! Provides:
//! - Fused RMSNorm: normalization + scaling in one kernel
//! - Fused RoPE: rotary position embedding application
//! - Fused SiLU-Mul: SiLU activation + element-wise multiply (SwiGLU MLP)

use crate::cuda::CudaKernelError;

// ── FFI declarations ──────────────────────────────────────────────────────

#[cfg(has_cuda)]
mod ffi {
    use std::os::raw::c_int;

    extern "C" {
        // Fused RMSNorm (FP16)
        pub fn rllm_fused_rmsnorm_f16(
            output: *mut u16,
            input: *const u16,
            weight: *const u16,
            hidden_size: i64,
            n_elements: i64,
            eps: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_fused_rmsnorm_f16_sync(
            output: *mut u16,
            input: *const u16,
            weight: *const u16,
            hidden_size: i64,
            n_elements: i64,
            eps: f32,
        ) -> c_int;

        // Fused SiLU-Mul (FP16)
        pub fn rllm_fused_silu_mul_f16(
            output: *mut u16,
            gate: *const u16,
            up: *const u16,
            n_elements: i64,
            stream: usize,
        ) -> c_int;

        pub fn rllm_fused_silu_mul_f16_sync(
            output: *mut u16,
            gate: *const u16,
            up: *const u16,
            n_elements: i64,
        ) -> c_int;

        // Fused RoPE (FP16)
        pub fn rllm_fused_rope_f16(
            out_q: *mut u16,
            out_k: *mut u16,
            query: *const u16,
            key: *const u16,
            positions: *const i32,
            num_tokens: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            rope_theta: f32,
            stream: usize,
        ) -> c_int;

        pub fn rllm_fused_rope_f16_sync(
            out_q: *mut u16,
            out_k: *mut u16,
            query: *const u16,
            key: *const u16,
            positions: *const i32,
            num_tokens: i64,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            rope_theta: f32,
        ) -> c_int;
    }
}

#[cfg(has_cuda)]
fn check(rc: i32) -> Result<(), CudaKernelError> {
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaKernelError::KernelError { code: rc })
    }
}

// ── Fused RMSNorm ─────────────────────────────────────────────────────────

/// Launch async fused RMSNorm: output = input / sqrt(mean(input^2) + eps) * weight.
///
/// # Safety
/// - All pointers must be valid device pointers.
/// - `weight` must have `hidden_size` elements.
/// - `input` and `output` must have `n_elements` elements.
/// - `n_elements` must be a multiple of `hidden_size`.
#[cfg(has_cuda)]
pub unsafe fn fused_rmsnorm_f16(
    output: *mut u16,
    input: *const u16,
    weight: *const u16,
    hidden_size: i64,
    n_elements: i64,
    eps: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_fused_rmsnorm_f16(output, input, weight, hidden_size, n_elements, eps, stream)
    };
    check(rc)
}

/// Synchronous fused RMSNorm for testing.
#[cfg(has_cuda)]
pub unsafe fn fused_rmsnorm_f16_sync(
    output: *mut u16,
    input: *const u16,
    weight: *const u16,
    hidden_size: i64,
    n_elements: i64,
    eps: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_fused_rmsnorm_f16_sync(output, input, weight, hidden_size, n_elements, eps)
    };
    check(rc)
}

// ── Fused SiLU-Mul ────────────────────────────────────────────────────────

/// Launch async fused SiLU-Mul: output[i] = silu(gate[i]) * up[i].
///
/// # Safety
/// - All pointers must be valid device pointers with `n_elements` elements.
#[cfg(has_cuda)]
pub unsafe fn fused_silu_mul_f16(
    output: *mut u16,
    gate: *const u16,
    up: *const u16,
    n_elements: i64,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_fused_silu_mul_f16(output, gate, up, n_elements, stream) };
    check(rc)
}

/// Synchronous fused SiLU-Mul for testing.
#[cfg(has_cuda)]
pub unsafe fn fused_silu_mul_f16_sync(
    output: *mut u16,
    gate: *const u16,
    up: *const u16,
    n_elements: i64,
) -> Result<(), CudaKernelError> {
    let rc = unsafe { ffi::rllm_fused_silu_mul_f16_sync(output, gate, up, n_elements) };
    check(rc)
}

// ── Fused RoPE ─────────────────────────────────────────────────────────────

/// Launch async fused RoPE: applies rotary position embeddings to query and key.
///
/// # Safety
/// - All pointers must be valid device pointers.
/// - `query`/`key`/`out_q`/`out_k` must have `num_tokens * num_heads * head_dim` elements.
/// - `positions` must have `num_tokens` elements.
/// - `head_dim` must be even.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_rope_f16(
    out_q: *mut u16,
    out_k: *mut u16,
    query: *const u16,
    key: *const u16,
    positions: *const i32,
    num_tokens: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rope_theta: f32,
    stream: usize,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_fused_rope_f16(
            out_q,
            out_k,
            query,
            key,
            positions,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            stream,
        )
    };
    check(rc)
}

/// Synchronous fused RoPE for testing.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_rope_f16_sync(
    out_q: *mut u16,
    out_k: *mut u16,
    query: *const u16,
    key: *const u16,
    positions: *const i32,
    num_tokens: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rope_theta: f32,
) -> Result<(), CudaKernelError> {
    let rc = unsafe {
        ffi::rllm_fused_rope_f16_sync(
            out_q,
            out_k,
            query,
            key,
            positions,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        )
    };
    check(rc)
}

// ── Non-CUDA stubs ────────────────────────────────────────────────────────

#[cfg(not(has_cuda))]
pub use stubs::*;

#[cfg(not(has_cuda))]
mod stubs {
    use super::CudaKernelError;

    pub fn fused_rmsnorm_f16(
        _output: *mut u16,
        _input: *const u16,
        _weight: *const u16,
        _hidden_size: i64,
        _n_elements: i64,
        _eps: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn fused_rmsnorm_f16_sync(
        _output: *mut u16,
        _input: *const u16,
        _weight: *const u16,
        _hidden_size: i64,
        _n_elements: i64,
        _eps: f32,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn fused_silu_mul_f16(
        _output: *mut u16,
        _gate: *const u16,
        _up: *const u16,
        _n_elements: i64,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    pub fn fused_silu_mul_f16_sync(
        _output: *mut u16,
        _gate: *const u16,
        _up: *const u16,
        _n_elements: i64,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_rope_f16(
        _out_q: *mut u16,
        _out_k: *mut u16,
        _query: *const u16,
        _key: *const u16,
        _positions: *const i32,
        _num_tokens: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _rope_theta: f32,
        _stream: usize,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_rope_f16_sync(
        _out_q: *mut u16,
        _out_k: *mut u16,
        _query: *const u16,
        _key: *const u16,
        _positions: *const i32,
        _num_tokens: i64,
        _num_q_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
        _rope_theta: f32,
    ) -> Result<(), CudaKernelError> {
        Err(CudaKernelError::NotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(has_cuda))]
    mod no_cuda {
        use super::*;

        #[test]
        fn fused_rmsnorm_returns_not_available() {
            let result = fused_rmsnorm_f16(
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                0, 0, 1e-6, 0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn fused_rmsnorm_sync_returns_not_available() {
            let result = fused_rmsnorm_f16_sync(
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                0, 0, 1e-6,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn fused_silu_mul_returns_not_available() {
            let result = fused_silu_mul_f16(
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                0, 0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn fused_silu_mul_sync_returns_not_available() {
            let result = fused_silu_mul_f16_sync(
                std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn fused_rope_returns_not_available() {
            let result = fused_rope_f16(
                std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                0, 0, 0, 0, 10000.0, 0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }

        #[test]
        fn fused_rope_sync_returns_not_available() {
            let result = fused_rope_f16_sync(
                std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                0, 0, 0, 0, 10000.0,
            );
            assert!(matches!(result, Err(CudaKernelError::NotAvailable)));
        }
    }

    #[cfg(has_cuda)]
    mod with_cuda {
        use super::*;
        use crate::cache_ops::{gpu_alloc, gpu_free};

        /// Convert f32 to IEEE 754 FP16 bit pattern.
        fn f32_to_f16_bits(f: f32) -> u16 {
            let bits = f.to_bits();
            let sign = (bits >> 31) & 0x1;
            let exponent = (bits >> 23) & 0xFF;
            let mantissa = bits & 0x7FFFFF;
            if exponent == 0 {
                return (sign << 15) as u16;
            }
            if exponent == 255 {
                return ((sign << 15) | 0x7C00 | (if mantissa != 0 { 1 } else { 0 })) as u16;
            }
            let new_exp = exponent as i32 - 127 + 15;
            if new_exp <= 0 {
                return (sign << 15) as u16;
            }
            if new_exp >= 31 {
                return ((sign << 15) | 0x7C00) as u16;
            }
            ((sign << 15) | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
        }

        /// Convert IEEE 754 FP16 bit pattern to f32.
        fn f16_bits_to_f32(h: u16) -> f32 {
            let sign = (h >> 15) & 0x1;
            let exponent = (h >> 10) & 0x1F;
            let mantissa = h & 0x3FF;
            if exponent == 0 {
                if mantissa == 0 {
                    return f32::from_bits(sign << 31);
                }
                let mut e = 0u32;
                let mut m = mantissa;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                m &= 0x3FF;
                return f32::from_bits((sign << 31) | ((127 - 15 - e) << 23) | (m << 13));
            }
            if exponent == 31 {
                return f32::from_bits((sign << 31) | 0x7F800000 | (mantissa << 13));
            }
            f32::from_bits((sign << 31) | ((exponent + 112) << 23) | (mantissa << 13))
        }

        unsafe fn upload(data: &[u16]) -> *mut u16 {
            let nbytes = data.len() * 2;
            let ptr = gpu_alloc(nbytes).expect("gpu_alloc failed") as *mut u16;
            libc::memcpy(ptr as *mut libc::c_void, data.as_ptr() as *const libc::c_void, nbytes);
            ptr
        }

        unsafe fn download(ptr: *mut u16, len: usize) -> Vec<u16> {
            let mut host = vec![0u16; len];
            libc::memcpy(host.as_mut_ptr() as *mut libc::c_void, ptr as *const libc::c_void, len * 2);
            host
        }

        #[test]
        fn fused_silu_mul_correctness() {
            let gate_f: Vec<f32> = vec![1.0, 2.0, 0.0, -1.0, 0.5, 3.0, -0.5, 4.0];
            let up_f: Vec<f32> = vec![2.0, 0.5, 1.0, 1.0, 3.0, 1.0, 2.0, 0.5];

            let gate_h: Vec<u16> = gate_f.iter().map(|&f| f32_to_f16_bits(f)).collect();
            let up_h: Vec<u16> = up_f.iter().map(|&f| f32_to_f16_bits(f)).collect();
            let n = gate_h.len() as i64;

            let d_gate = unsafe { upload(&gate_h) };
            let d_up = unsafe { upload(&up_h) };
            let d_out = unsafe { gpu_alloc(n as usize * 2).expect("gpu_alloc failed") as *mut u16 };

            unsafe { fused_silu_mul_f16_sync(d_out, d_gate, d_up, n).expect("silu_mul failed") };

            let result = unsafe { download(d_out, n as usize) };
            unsafe { gpu_free(d_gate as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_up as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_out as *mut u8).expect("gpu_free failed") };

            for i in 0..n as usize {
                let g = gate_f[i];
                let u = up_f[i];
                let silu_val = g / (1.0 + (-g).exp());
                let expected = silu_val * u;
                let actual = f16_bits_to_f32(result[i]);
                assert!(
                    (actual - expected).abs() < 0.1,
                    "silu_mul[{}]: expected {:.4}, got {:.4}",
                    i, expected, actual
                );
            }
        }

        #[test]
        fn fused_rmsnorm_correctness() {
            let hidden_size = 4i64;
            let n_elements = 8i64;
            let eps = 1e-6f32;

            let input_f: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
            let weight_f: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

            let input_h: Vec<u16> = input_f.iter().map(|&f| f32_to_f16_bits(f)).collect();
            let weight_h: Vec<u16> = weight_f.iter().map(|&f| f32_to_f16_bits(f)).collect();

            let d_input = unsafe { upload(&input_h) };
            let d_weight = unsafe { upload(&weight_h) };
            let d_output = unsafe { gpu_alloc(n_elements as usize * 2).expect("gpu_alloc failed") as *mut u16 };

            unsafe {
                fused_rmsnorm_f16_sync(d_output, d_input, d_weight, hidden_size, n_elements, eps)
                    .expect("rmsnorm failed");
            }

            let result = unsafe { download(d_output, n_elements as usize) };
            unsafe { gpu_free(d_input as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_weight as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_output as *mut u8).expect("gpu_free failed") };

            // Row 0: [1,2,3,4], variance = (1+4+9+16)/4 = 7.5
            let rms0 = 1.0 / (7.5f32 + eps).sqrt();
            for i in 0..4 {
                let expected = input_f[i] * rms0;
                let actual = f16_bits_to_f32(result[i]);
                assert!(
                    (actual - expected).abs() < 0.1,
                    "rmsnorm row0[{}]: expected {:.4}, got {:.4}",
                    i, expected, actual
                );
            }

            // Row 1: [2,4,6,8], variance = (4+16+36+64)/4 = 30
            let rms1 = 1.0 / (30.0f32 + eps).sqrt();
            for i in 0..4 {
                let expected = input_f[4 + i] * rms1;
                let actual = f16_bits_to_f32(result[4 + i]);
                assert!(
                    (actual - expected).abs() < 0.1,
                    "rmsnorm row1[{}]: expected {:.4}, got {:.4}",
                    i, expected, actual
                );
            }
        }

        #[test]
        fn fused_rope_correctness() {
            let num_tokens = 2i64;
            let num_q_heads = 1i64;
            let num_kv_heads = 1i64;
            let head_dim = 4i64;
            let rope_theta = 10000.0f32;

            // query/key: [2 tokens, 1 head, 4 dims] = 8 elements each
            let query_f: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
            let key_f: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
            let positions: Vec<i32> = vec![0, 1];

            let query_h: Vec<u16> = query_f.iter().map(|&f| f32_to_f16_bits(f)).collect();
            let key_h: Vec<u16> = key_f.iter().map(|&f| f32_to_f16_bits(f)).collect();
            let n = query_h.len() as i64;

            let d_query = unsafe { upload(&query_h) };
            let d_key = unsafe { upload(&key_h) };
            let d_out_q = unsafe { gpu_alloc(n as usize * 2).expect("gpu_alloc failed") as *mut u16 };
            let d_out_k = unsafe { gpu_alloc(n as usize * 2).expect("gpu_alloc failed") as *mut u16 };
            let mut d_positions = unsafe { gpu_alloc(num_tokens as usize * 4).expect("gpu_alloc failed") as *mut i32 };
            unsafe {
                libc::memcpy(
                    d_positions as *mut libc::c_void,
                    positions.as_ptr() as *const libc::c_void,
                    num_tokens as usize * 4,
                );
            }

            unsafe {
                fused_rope_f16_sync(
                    d_out_q, d_out_k, d_query, d_key, d_positions,
                    num_tokens, num_q_heads, num_kv_heads, head_dim, rope_theta,
                ).expect("rope failed");
            }

            let out_q = unsafe { download(d_out_q, n as usize) };

            unsafe { gpu_free(d_query as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_key as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_out_q as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_out_k as *mut u8).expect("gpu_free failed") };
            unsafe { gpu_free(d_positions as *mut u8).expect("gpu_free failed") };

            // Position 0: angle = 0 for all dims → cos=1, sin=0 → no rotation
            for i in 0..4 {
                let actual = f16_bits_to_f32(out_q[i]);
                let expected = query_f[i]; // unchanged at position 0
                assert!(
                    (actual - expected).abs() < 0.1,
                    "rope pos0 [{}]: expected {:.4}, got {:.4}",
                    i, expected, actual
                );
            }

            // Position 1, dim pair 0-1: freq = 1/theta^(0/4) = 1.0, angle = 1.0
            let cos1 = 1.0_f32.cos();
            let sin1 = 1.0_f32.sin();
            let expected_0 = 1.0 * cos1 - 0.0 * sin1; // = cos(1)
            let actual_0 = f16_bits_to_f32(out_q[4]);
            assert!(
                (actual_0 - expected_0).abs() < 0.1,
                "rope pos1 dim0: expected {:.4}, got {:.4}",
                expected_0, actual_0
            );
        }
    }
}
