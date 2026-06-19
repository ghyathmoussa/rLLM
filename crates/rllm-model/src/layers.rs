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
pub enum LinearWeight {
    Fp(Tensor),
    Gptq {
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        g_idx: Tensor,
        bits: usize,
        group_size: usize,
        dequantized: std::sync::OnceLock<Tensor>,
    },
}

#[cfg(feature = "candle-backend")]
pub fn dequantize_gptq(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: &Tensor,
    bits: usize,
    _group_size: usize,
) -> Result<Tensor> {
    if bits != 4 {
        return Err(candle_core::Error::Msg(format!(
            "Only 4-bit GPTQ is supported, got {bits}"
        )));
    }

    let device = qweight.device();
    
    // We expect qweight to be (in_features / 8, out_features)
    let (packed_in_features, out_features) = qweight.dims2()?;
    let in_features = packed_in_features * 8;

    // Convert qweight from signed I32 to unsigned I64 to avoid sign-extension bugs.
    // If it's negative, add 2^32.
    let qweight_i64 = qweight.to_dtype(DType::I64)?;
    let zero_w = Tensor::new(0i64, device)?.broadcast_as(qweight_i64.shape())?;
    let is_neg = qweight_i64.lt(&zero_w)?;
    let offset_w = Tensor::new(4294967296i64, device)?.broadcast_as(qweight_i64.shape())?;
    let qweight_u64 = qweight_i64.add(&is_neg.to_dtype(DType::I64)?.mul(&offset_w)?)?;

    // Unpack qweight (bits = 4, so 8 values per i32/i64)
    let mut w_unpacked = Vec::with_capacity(8);
    let c16_w = Tensor::new(16i64, device)?.broadcast_as(qweight_u64.shape())?;
    for i in 0..8 {
        let divisor = 1i64 << (4 * i);
        let divisor_t = Tensor::new(divisor, device)?.broadcast_as(qweight_u64.shape())?;
        let shifted = qweight_u64.div(&divisor_t)?;
        let temp = shifted.div(&c16_w)?.mul(&c16_w)?;
        let masked = shifted.sub(&temp)?;
        w_unpacked.push(masked);
    }
    // Stack along dimension 1 (shape: [packed_in_features, 8, out_features])
    let w_stacked = Tensor::stack(&w_unpacked, 1)?;
    // Reshape to [in_features, out_features]
    let w_raw = w_stacked.reshape((in_features, out_features))?.to_dtype(DType::F32)?;

    // Convert qzeros from signed I32 to unsigned I64.
    let qzeros_i64 = qzeros.to_dtype(DType::I64)?;
    let zero_z = Tensor::new(0i64, device)?.broadcast_as(qzeros_i64.shape())?;
    let is_neg_z = qzeros_i64.lt(&zero_z)?;
    let offset_z = Tensor::new(4294967296i64, device)?.broadcast_as(qzeros_i64.shape())?;
    let qzeros_u64 = qzeros_i64.add(&is_neg_z.to_dtype(DType::I64)?.mul(&offset_z)?)?;

    // Unpack qzeros
    let mut z_unpacked = Vec::with_capacity(8);
    let c16_z = Tensor::new(16i64, device)?.broadcast_as(qzeros_u64.shape())?;
    let one_z = Tensor::new(1.0f32, device)?.broadcast_as(qzeros_u64.shape())?;
    for i in 0..8 {
        let divisor = 1i64 << (4 * i);
        let divisor_t = Tensor::new(divisor, device)?.broadcast_as(qzeros_u64.shape())?;
        let shifted = qzeros_u64.div(&divisor_t)?;
        let temp = shifted.div(&c16_z)?.mul(&c16_z)?;
        let masked = shifted.sub(&temp)?;
        let adjusted = masked.to_dtype(DType::F32)?.add(&one_z)?;
        z_unpacked.push(adjusted);
    }
    // Stack along dimension 2 (shape: [num_groups, out_features / 8, 8])
    let z_stacked = Tensor::stack(&z_unpacked, 2)?;
    // Reshape to [num_groups, out_features]
    let z_raw = z_stacked.reshape((qzeros.dim(0)?, out_features))?;

    // Prepare g_idx (cast to U32 for index_select)
    let g_idx_u32 = g_idx.to_dtype(DType::U32)?;

    // Select scales and zero-points for each input feature
    let select_scales = scales.index_select(&g_idx_u32, 0)?; // [in_features, out_features]
    let select_zeros = z_raw.index_select(&g_idx_u32, 0)?;   // [in_features, out_features]

    // Apply dequantization formula: (W_q - ZP) * Scale
    let target_dtype = scales.dtype();
    let w_dequant = w_raw.sub(&select_zeros)?.mul(&select_scales.to_dtype(DType::F32)?)?;
    let w_dequant = w_dequant.to_dtype(target_dtype)?;

    // Transpose back to match [out_features, in_features] shape
    w_dequant.t()
}

#[cfg(feature = "candle-backend")]
pub struct Linear {
    weight: LinearWeight,
}

#[cfg(feature = "candle-backend")]
impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self {
            weight: LinearWeight::Fp(weight),
        }
    }

    pub fn new_gptq(
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        g_idx: Tensor,
        bits: usize,
        group_size: usize,
    ) -> Self {
        Self {
            weight: LinearWeight::Gptq {
                qweight,
                qzeros,
                scales,
                g_idx,
                bits,
                group_size,
                dequantized: std::sync::OnceLock::new(),
            },
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = match &self.weight {
            LinearWeight::Fp(w) => w,
            LinearWeight::Gptq {
                qweight,
                qzeros,
                scales,
                g_idx,
                bits,
                group_size,
                dequantized,
            } => {
                if let Some(w) = dequantized.get() {
                    w
                } else {
                    let w = dequantize_gptq(qweight, qzeros, scales, g_idx, *bits, *group_size)?;
                    let _ = dequantized.set(w);
                    dequantized.get().unwrap()
                }
            }
        };

        // weight shape: [out_features, in_features]
        // x shape: [..., in_features]
        let in_features = weight.dim(D::Minus1)?;
        let out_features = weight.dim(D::Minus2)?;
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let x_2d = x.reshape((batch, in_features))?;
        let out = x_2d.matmul(&weight.t()?)?;
        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(out_features);
        out.reshape(out_shape)
    }

    pub fn weight(&self) -> Result<&Tensor> {
        match &self.weight {
            LinearWeight::Fp(w) => Ok(w),
            LinearWeight::Gptq {
                qweight,
                qzeros,
                scales,
                g_idx,
                bits,
                group_size,
                dequantized,
            } => {
                if let Some(w) = dequantized.get() {
                    Ok(w)
                } else {
                    let w = dequantize_gptq(qweight, qzeros, scales, g_idx, *bits, *group_size)?;
                    let _ = dequantized.set(w);
                    Ok(dequantized.get().unwrap())
                }
            }
        }
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
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
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
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (intermediate, hidden), &device)?),
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (intermediate, hidden), &device)?),
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (hidden, intermediate), &device)?),
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
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (num_heads * head_dim, hidden), &device)?),
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (num_kv_heads * head_dim, hidden), &device)?),
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (num_kv_heads * head_dim, hidden), &device)?),
            Linear::new(Tensor::randn(0.0f32, 1.0f32, (hidden, num_heads * head_dim), &device)?),
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

    #[test]
    fn gptq_dequantization_correctness() -> Result<()> {
        let device = Device::Cpu;

        // Shape: (1, 8)
        let qweight_data = vec![
            1985229328i32,  // Col 0: 0x76543210
            -19088744i32,   // Col 1: 0xFEDCBA98
            -324508640i32,  // Col 2: 0xECA86420
            -1985229329i32, // Col 3: 0x89ABCDEF
            0i32, 0i32, 0i32, 0i32, // Col 4-7
        ];
        let qweight = Tensor::from_vec(qweight_data, (1, 8), &device)?;

        // Shape: (1, 1)
        // Zero points packed: 7 for all 8 cols -> 0x77777777
        let qzeros = Tensor::from_vec(vec![2004318071i32], (1, 1), &device)?;

        // Shape: (1, 8)
        let scales = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], (1, 8), &device)?;

        // Shape: (8,)
        let g_idx = Tensor::from_vec(vec![0u32; 8], (8,), &device)?;

        let w_dequant = dequantize_gptq(&qweight, &qzeros, &scales, &g_idx, 4, 8)?;
        // Expected shape: (8, 8)
        assert_eq!(w_dequant.dims(), &[8, 8]);

        let w_vals = w_dequant.to_vec2::<f32>()?;

        // Verify Col 0: expected [r - 8]
        for r in 0..8 {
            assert_eq!(w_vals[0][r], (r as f32) - 8.0);
        }

        // Verify Col 1: expected [2.0 * r]
        for r in 0..8 {
            assert_eq!(w_vals[1][r], 2.0 * (r as f32));
        }

        // Verify Col 2: expected [(2 * r - 8) * 3.0]
        for r in 0..8 {
            assert_eq!(w_vals[2][r], ((2 * r) as f32 - 8.0) * 3.0);
        }

        // Verify Col 3: expected [(7 - r) * 4.0]
        for r in 0..8 {
            assert_eq!(w_vals[3][r], (7.0 - (r as f32)) * 4.0);
        }

        Ok(())
    }
}
