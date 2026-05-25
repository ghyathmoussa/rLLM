#[cfg(feature = "candle-backend")]
use candle_core::{D, DType, Device, Result, Tensor};

#[cfg(feature = "candle-backend")]
use crate::rope::RotaryEmbedding;

// ── RMSNorm ──────────────────────────────────────────────────────────────

#[cfg(feature = "candle-backend")]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

#[cfg(feature = "candle-backend")]
impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?;
        let x_norm = x.broadcast_div(&variance.sqrt()?)?;
        let out = x_norm.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        out.to_dtype(dtype)
    }
}

// ── Linear (no bias, as in Llama) ────────────────────────────────────────

#[cfg(feature = "candle-backend")]
pub struct Linear {
    weight: Tensor,
}

#[cfg(feature = "candle-backend")]
impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // weight shape: [out_features, in_features]
        // x shape: [..., in_features]
        let in_features = self.weight.dim(D::Minus1)?;
        let out_features = self.weight.dim(D::Minus2)?;
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let x_2d = x.reshape((batch, in_features))?;
        let out = x_2d.matmul(&self.weight.t()?)?;
        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(out_features);
        out.reshape(out_shape)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

// ── LlamaMLP (SwiGLU) ───────────────────────────────────────────────────

#[cfg(feature = "candle-backend")]
pub struct LlamaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

#[cfg(feature = "candle-backend")]
impl LlamaMLP {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        Self {
            gate_proj: Linear::new(gate_proj),
            up_proj: Linear::new(up_proj),
            down_proj: Linear::new(down_proj),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = gate.silu()?;
        self.down_proj.forward(&gate.broadcast_mul(&up)?)
    }
}

// ── LlamaAttention (GQA) ────────────────────────────────────────────────

#[cfg(feature = "candle-backend")]
pub struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

#[cfg(feature = "candle-backend")]
impl LlamaAttention {
    pub fn new(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj: Linear::new(q_proj),
            k_proj: Linear::new(k_proj),
            v_proj: Linear::new(v_proj),
            o_proj: Linear::new(o_proj),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq_len, num_heads, head_dim] then transpose
        let q = q.reshape((bsz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?; // [batch, num_heads, seq_len, head_dim]

        let k = k.reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?; // [batch, num_kv_heads, seq_len, head_dim]

        let v = v.reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?; // [batch, num_kv_heads, seq_len, head_dim]

        // Apply RoPE
        let (q, k) = rope.apply(&q, &k, positions)?;

        // Update KV cache
        let (k, v) = match kv_cache {
            Some((cached_k, cached_v)) => {
                let k = Tensor::cat(&[cached_k.clone(), k.clone()], 2)?;
                let v = Tensor::cat(&[cached_v.clone(), v.clone()], 2)?;
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
            None => {
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };

        // GQA: repeat K, V to match num_heads if needed
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_weights = q
            .matmul(&k.t()?)?
            .broadcast_mul(&Tensor::new(scale, q.device())?.to_dtype(q.dtype())?)?;

        // Apply causal mask for prefill (seq_len > 1)
        let attn_weights = if seq_len > 1 {
            let mask = causal_mask(seq_len, q.device())?.to_dtype(q.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        let attn_output =
            attn_output.transpose(1, 2)?.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    /// Paged attention forward pass.
    ///
    /// Computes Q/K/V projections and RoPE, then writes K/V into the global
    /// GPU KV cache and computes attention using PagedAttention kernels.
    ///
    /// This replaces the native Candle matmul-based attention with block-addressed
    /// attention for efficient KV cache reuse across requests.
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states [batch, seq_len, hidden_size]
    /// * `positions` - Token positions for RoPE
    /// * `gpu_kv_cache` - Global GPU KV cache with block-addressed storage
    /// * `attn_meta` - Attention metadata (block tables, slot mappings, seq lens)
    /// * `layer_idx` - Layer index for KV cache addressing
    pub fn forward_paged(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
        layer_idx: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq_len, num_heads, head_dim] then transpose
        let q = q.reshape((bsz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = rope.apply(&q, &k, positions)?;

        // Write K/V into the global GPU cache at slot-mapped positions.
        //
        // When CUDA is available, we call cache_write_f16 to scatter-write
        // the new K/V data into the physical cache blocks. Without CUDA,
        // we fall back to the native Candle attention path (this branch
        // should not be reached in production paged mode).
        #[cfg(has_cuda)]
        {
            use rllm_kernels::cache_ops;

            // Flatten K/V for cache write: [num_tokens, num_kv_heads, head_dim]
            let k_flat = k.transpose(1, 2)?.reshape((bsz * seq_len, self.num_kv_heads, self.head_dim))?;
            let v_flat = v.transpose(1, 2)?.reshape((bsz * seq_len, self.num_kv_heads, self.head_dim))?;

            // Convert to F16 for cache write
            let k_f16 = k_flat.to_dtype(candle_core::DType::F16)?.contiguous()?;
            let v_f16 = v_flat.to_dtype(candle_core::DType::F16)?.contiguous()?;

            let k_ptr = k_f16.as_ptr() as *const u16;
            let v_ptr = v_f16.as_ptr() as *const u16;

            let key_cache_ptr = gpu_kv_cache.key_ptr(layer_idx) as *mut u16;
            let value_cache_ptr = gpu_kv_cache.value_ptr(layer_idx) as *mut u16;

            // Build slot mapping tensor on device
            let slot_mapping = &attn_meta.slot_mapping;

            if !slot_mapping.is_empty() {
                unsafe {
                    cache_ops::cache_write_f16(
                        key_cache_ptr,
                        value_cache_ptr,
                        k_ptr,
                        v_ptr,
                        slot_mapping.as_ptr(),
                        (bsz * seq_len) as i64,
                        self.num_kv_heads as i64,
                        self.head_dim as i64,
                        gpu_kv_cache.block_size() as i64,
                        gpu_kv_cache.num_blocks() as i64,
                        0, // default stream
                    ).map_err(|e| candle_core::Error::Msg(format!("cache_write_f16 failed: {e}")))?;
                }
            }

            // Run PagedAttention kernel for attention computation.
            // Flatten query for kernel: [num_tokens, num_q_heads * head_dim]
            let q_flat = q.transpose(1, 2)?.reshape((bsz * seq_len, self.num_heads * self.head_dim))?;
            let q_f16 = q_flat.to_dtype(candle_core::DType::F16)?.contiguous()?;
            let q_ptr = q_f16.as_ptr() as *const u16;

            // Allocate output buffer
            let out_elems = bsz * seq_len * self.num_heads * self.head_dim;
            let out_bytes = out_elems * 2; // FP16
            let out_ptr = unsafe { cache_ops::gpu_alloc(out_bytes)? } as *mut u16;

            let flat_block_tables = attn_meta.flatten_block_tables();
            let seq_lens_i32: Vec<i32> = attn_meta.seq_lens.iter().map(|&s| s as i32).collect();
            let scale = 1.0f32 / (self.head_dim as f32).sqrt();

            let is_prefill = attn_meta.num_prefill_tokens > 0 && attn_meta.num_decode_tokens == 0;

            let kernel_result = if is_prefill {
                let query_start_loc_i32: Vec<i32> = attn_meta.query_start_loc.iter().map(|&s| s as i32).collect();
                unsafe {
                    rllm_kernels::paged_attention_prefill_f16(
                        out_ptr,
                        q_ptr,
                        gpu_kv_cache.key_ptr(layer_idx) as *const u16,
                        gpu_kv_cache.value_ptr(layer_idx) as *const u16,
                        flat_block_tables.as_ptr(),
                        seq_lens_i32.as_ptr(),
                        query_start_loc_i32.as_ptr(),
                        attn_meta.num_seqs() as i64,
                        (bsz * seq_len) as i64,
                        self.num_heads as i64,
                        self.num_kv_heads as i64,
                        self.head_dim as i64,
                        gpu_kv_cache.block_size() as i64,
                        attn_meta.max_num_blocks_per_seq as i64,
                        scale,
                        0, // default stream
                    )
                }
            } else {
                unsafe {
                    rllm_kernels::paged_attention_decode_f16(
                        out_ptr,
                        q_ptr,
                        gpu_kv_cache.key_ptr(layer_idx) as *const u16,
                        gpu_kv_cache.value_ptr(layer_idx) as *const u16,
                        flat_block_tables.as_ptr(),
                        seq_lens_i32.as_ptr(),
                        attn_meta.num_seqs() as i64,
                        self.num_heads as i64,
                        self.num_kv_heads as i64,
                        self.head_dim as i64,
                        gpu_kv_cache.block_size() as i64,
                        attn_meta.max_num_blocks_per_seq as i64,
                        scale,
                        0, // default stream
                    )
                }
            };

            kernel_result.map_err(|e| candle_core::Error::Msg(format!("paged_attention failed: {e}")))?;

            // Convert output back to a Candle tensor
            // Read from GPU pointer into a Vec, then create tensor
            let mut out_host = vec![0u16; out_elems];
            unsafe {
                std::ptr::copy_nonoverlapping(out_ptr, out_host.as_mut_ptr(), out_elems);
                cache_ops::gpu_free(out_ptr as *mut u8)
                    .map_err(|e| candle_core::Error::Msg(format!("gpu_free failed: {e}")))?;
            }

            // Build Candle tensor from the F16 output
            let attn_output = Tensor::from_raw_buffer(
                &out_host.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>(),
                candle_core::DType::F16,
                &[bsz, seq_len, self.num_heads * self.head_dim],
                hidden_states.device(),
            )?.to_dtype(hidden_states.dtype())?;

            return self.o_proj.forward(&attn_output).map_err(|e| anyhow::anyhow!("{e}"));
        }

        // Non-CUDA fallback: use native attention
        #[cfg(not(has_cuda))]
        {
            let _ = (gpu_kv_cache, attn_meta, layer_idx);

            // GQA: repeat K, V to match num_heads if needed
            let (k, v) = if self.num_kv_heads < self.num_heads {
                let n_rep = self.num_heads / self.num_kv_heads;
                (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?)
            } else {
                (k, v)
            };

            let scale = 1.0f32 / (self.head_dim as f32).sqrt();
            let attn_weights = q
                .matmul(&k.t()?)?
                .broadcast_mul(&Tensor::new(scale, q.device())?.to_dtype(q.dtype())?)?;

            let attn_weights = if seq_len > 1 {
                let mask = causal_mask(seq_len, q.device())?.to_dtype(q.dtype())?;
                attn_weights.broadcast_add(&mask)?
            } else {
                attn_weights
            };

            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
            let attn_output = attn_weights.matmul(&v)?;
            let attn_output =
                attn_output.transpose(1, 2)?.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

            self.o_proj.forward(&attn_output)
        }
    }
}

#[cfg(feature = "candle-backend")]
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    // x: [batch, num_kv_heads, seq_len, head_dim]
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x.unsqueeze(2)?.expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?.reshape((
        batch,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    ))?;
    Ok(x)
}

#[cfg(feature = "candle-backend")]
fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    // Upper triangular mask with -inf for positions that should be masked
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::from_vec(mask, (seq_len, seq_len), device)?;
    // Broadcast to [1, 1, seq_len, seq_len]
    mask.reshape((1, 1, seq_len, seq_len))
}

// ── LlamaDecoderLayer ────────────────────────────────────────────────────

#[cfg(feature = "candle-backend")]
pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

#[cfg(feature = "candle-backend")]
impl LlamaDecoderLayer {
    pub fn new(
        self_attn: LlamaAttention,
        mlp: LlamaMLP,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
    ) -> Self {
        Self { self_attn, mlp, input_layernorm, post_attention_layernorm }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        // Self attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, positions, kv_cache, rope)?;
        let hidden_states = (residual + hidden_states)?;

        // MLP with residual
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }

    /// Paged forward pass using PagedAttention kernels.
    pub fn forward_paged(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
        layer_idx: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        // Self attention with residual (paged)
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward_paged(
            &hidden_states, positions, gpu_kv_cache, attn_meta, layer_idx, rope,
        )?;
        let hidden_states = (residual + hidden_states)?;

        // MLP with residual (unchanged)
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }
}

#[cfg(all(test, feature = "candle-backend"))]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones(64, DType::F32, &device)?;
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device)?;
        let out = norm.forward(&x)?;
        assert_eq!(out.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn linear_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0f32, (128, 64), &device)?;
        let linear = Linear::new(weight);

        let x = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device)?;
        let out = linear.forward(&x)?;
        assert_eq!(out.dims(), &[2, 10, 128]);
        Ok(())
    }

    #[test]
    fn swiglu_mlp_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 64;
        let intermediate = 128;

        let mlp = LlamaMLP::new(
            Tensor::randn(0.0f32, 1.0f32, (intermediate, hidden), &device)?,
            Tensor::randn(0.0f32, 1.0f32, (intermediate, hidden), &device)?,
            Tensor::randn(0.0f32, 1.0f32, (hidden, intermediate), &device)?,
        );

        let x = Tensor::randn(0.0f32, 1.0f32, (1, 5, hidden), &device)?;
        let out = mlp.forward(&x)?;
        assert_eq!(out.dims(), &[1, 5, hidden]);
        Ok(())
    }

    #[test]
    fn attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden / num_heads;

        let attn = LlamaAttention::new(
            Tensor::randn(0.0f32, 1.0f32, (num_heads * head_dim, hidden), &device)?,
            Tensor::randn(0.0f32, 1.0f32, (num_kv_heads * head_dim, hidden), &device)?,
            Tensor::randn(0.0f32, 1.0f32, (num_kv_heads * head_dim, hidden), &device)?,
            Tensor::randn(0.0f32, 1.0f32, (hidden, num_heads * head_dim), &device)?,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        let rope = RotaryEmbedding::new(head_dim, 512, 10000.0, &device)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 5, hidden), &device)?;
        let mut kv_cache = None;

        let out = attn.forward(&x, &[0, 1, 2, 3, 4], &mut kv_cache, &rope)?;
        assert_eq!(out.dims(), &[1, 5, hidden]);
        assert!(kv_cache.is_some());
        Ok(())
    }

    #[test]
    fn causal_mask_correctness() -> Result<()> {
        let device = Device::Cpu;
        let mask = causal_mask(4, &device)?;
        // mask shape: [1, 1, 4, 4]
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        let vals = mask.reshape((4, 4))?.to_vec2::<f32>()?;

        // Position 0 can only see position 0
        assert!(vals[0][0].is_finite());
        assert!(vals[0][1].is_infinite());
        assert!(vals[0][2].is_infinite());
        assert!(vals[0][3].is_infinite());

        // Position 2 can see 0, 1, 2 but not 3
        assert!(vals[2][0].is_finite());
        assert!(vals[2][1].is_finite());
        assert!(vals[2][2].is_finite());
        assert!(vals[2][3].is_infinite());
        Ok(())
    }
}
