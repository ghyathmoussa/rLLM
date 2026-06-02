#[cfg(feature = "candle-backend")]
use candle_core::{D, DType, Device, Result, Tensor};
#[cfg(feature = "candle-backend")]
use rllm_core::optimizations::QuantizationPlan;
#[cfg(feature = "candle-backend")]
use rllm_quant::{LinearMethod, UnquantizedLinear};

#[cfg(feature = "candle-backend")]
use crate::rope::RotaryEmbedding;

#[cfg(feature = "candle-backend")]
pub fn simulate_weight_quantization(weight: &Tensor, plan: &QuantizationPlan) -> Result<Tensor> {
    use rllm_core::optimizations::QuantizedWeightFormat;
    let format = plan.format;
    if format == QuantizedWeightFormat::Unquantized {
        return Ok(weight.clone());
    }

    let dtype = weight.dtype();
    let w_f32 = weight.to_dtype(DType::F32)?;
    let q_w = match format {
        QuantizedWeightFormat::Mxfp4 => {
            let group_size = plan.group_size.unwrap_or(32);
            simulate_group_quant(&w_f32, group_size, 4)?
        }
        QuantizedWeightFormat::Mxfp8 => {
            let group_size = plan.group_size.unwrap_or(32);
            simulate_group_quant(&w_f32, group_size, 8)?
        }
        QuantizedWeightFormat::Nvfp4 => simulate_uniform_quant(&w_f32, 4, true)?,
        QuantizedWeightFormat::Int8
        | QuantizedWeightFormat::Gptq
        | QuantizedWeightFormat::Awq
        | QuantizedWeightFormat::Gguf
        | QuantizedWeightFormat::CompressedTensors
        | QuantizedWeightFormat::ModelOpt
        | QuantizedWeightFormat::TorchAo => {
            if let Some(gs) = plan.group_size {
                simulate_group_quant(&w_f32, gs, 8)?
            } else {
                simulate_channel_quant(&w_f32, 8)?
            }
        }
        QuantizedWeightFormat::Int4 => {
            let group_size = plan.group_size.unwrap_or(128);
            simulate_group_quant(&w_f32, group_size, 4)?
        }
        QuantizedWeightFormat::Unquantized => w_f32,
        _ => w_f32,
    };
    q_w.to_dtype(dtype)
}

#[cfg(feature = "candle-backend")]
fn simulate_uniform_quant(weight: &Tensor, bits: u32, symmetric: bool) -> Result<Tensor> {
    let max_val = weight.flatten_all()?.max(0)?.to_scalar::<f32>()? as f64;
    let min_val = weight.flatten_all()?.min(0)?.to_scalar::<f32>()? as f64;
    let levels = (1 << bits) - 1;
    let (scale, zero_point) = if symmetric {
        let abs_max = max_val.max(min_val.abs());
        let scale = if abs_max > 0.0 { abs_max / (levels as f64 / 2.0) } else { 1.0 };
        (scale, 0.0)
    } else {
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / levels as f64 } else { 1.0 };
        (scale, min_val)
    };

    let q = weight
        .broadcast_sub(&Tensor::new(zero_point as f32, weight.device())?)?
        .broadcast_div(&Tensor::new(scale as f32, weight.device())?)?
        .round()?
        .clamp(-(levels as f32 / 2.0), levels as f32 / 2.0)?
        .broadcast_mul(&Tensor::new(scale as f32, weight.device())?)?
        .broadcast_add(&Tensor::new(zero_point as f32, weight.device())?)?;
    Ok(q)
}

#[cfg(feature = "candle-backend")]
fn simulate_channel_quant(weight: &Tensor, bits: usize) -> Result<Tensor> {
    let _out_features = weight.dim(0)?;
    let _in_features = weight.dim(1)?;
    let abs_w = weight.abs()?;
    let max_abs = abs_w.max_keepdim(1)?;
    let q_max = (1 << (bits - 1)) - 1;
    let scale = max_abs.broadcast_div(&Tensor::new(q_max as f32, weight.device())?)?;
    let eps = Tensor::new(1e-8f32, weight.device())?;
    let scale_safe = scale.broadcast_add(&eps)?;
    let w_quant = weight.broadcast_div(&scale_safe)?.round()?;
    let w_clamp = w_quant.clamp(-(q_max as f32), q_max as f32)?;
    let w_dequant = w_clamp.broadcast_mul(&scale_safe)?;
    Ok(w_dequant)
}

#[cfg(feature = "candle-backend")]
fn simulate_group_quant(weight: &Tensor, group_size: usize, bits: usize) -> Result<Tensor> {
    let out_features = weight.dim(0)?;
    let in_features = weight.dim(1)?;
    if in_features % group_size != 0 {
        return simulate_channel_quant(weight, bits);
    }
    let num_groups = in_features / group_size;
    let w_reshaped = weight.reshape((out_features * num_groups, group_size))?;
    let abs_w = w_reshaped.abs()?;
    let max_abs = abs_w.max_keepdim(1)?;
    let q_max = (1 << (bits - 1)) - 1;
    let scale = max_abs.broadcast_div(&Tensor::new(q_max as f32, weight.device())?)?;
    let eps = Tensor::new(1e-8f32, weight.device())?;
    let scale_safe = scale.broadcast_add(&eps)?;
    let w_quant = w_reshaped.broadcast_div(&scale_safe)?.round()?;
    let w_clamp = w_quant.clamp(-(q_max as f32), q_max as f32)?;
    let w_dequant = w_clamp.broadcast_mul(&scale_safe)?;
    w_dequant.reshape((out_features, in_features))
}

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
    method: Box<dyn LinearMethod>,
}

#[cfg(feature = "candle-backend")]
impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { method: Box::new(UnquantizedLinear::new(weight)) }
    }

    pub fn from_method(method: Box<dyn LinearMethod>) -> Self {
        Self { method }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.method.apply(x)
    }

    pub fn weight(&self) -> &Tensor {
        self.method
            .weight()
            .expect("Linear::weight() is only available for unquantized linear layers")
    }

    pub fn is_quantized(&self) -> bool {
        self.method.is_quantized()
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
    pub fn from_linears(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

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
    pub fn from_linears(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self { q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim }
    }

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
            let _ = (rope, positions);

            // Opt-in gate. The paged path must be validated against the eager
            // forward on the GPU box before it becomes the default; until then,
            // without the env flag we signal the caller to use the proven legacy
            // per-request forward (`execute_model_step` in rllm-executor), which is
            // numerically correct. Set `RLLM_PAGED_ATTENTION=1` to exercise this
            // path. See docs/quantization-int8-kvcache-plan.md (Layer 0).
            if !paged_attention_enabled() {
                let _ = (gpu_kv_cache, attn_meta, layer_idx, &q, &k, &v);
                return Err(candle_core::Error::Msg(
                    "paged-attention disabled (set RLLM_PAGED_ATTENTION=1 to enable); \
                     using legacy forward"
                        .to_string(),
                ));
            }

            let num_tokens = bsz * seq_len;

            // The kernels expect token-major, contiguous f16 tensors:
            // q [num_tokens, num_heads, head_dim], k/v [num_tokens, num_kv_heads, head_dim].
            let q_tok = q
                .transpose(1, 2)?
                .reshape((num_tokens, self.num_heads, self.head_dim))?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let k_tok = k
                .transpose(1, 2)?
                .reshape((num_tokens, self.num_kv_heads, self.head_dim))?
                .contiguous()?
                .to_dtype(DType::F16)?;
            let v_tok = v
                .transpose(1, 2)?
                .reshape((num_tokens, self.num_kv_heads, self.head_dim))?
                .contiguous()?
                .to_dtype(DType::F16)?;

            let op = PagedAttentionOp {
                key_cache: gpu_kv_cache.key_ptr(layer_idx) as usize,
                value_cache: gpu_kv_cache.value_ptr(layer_idx) as usize,
                cache_dtype: gpu_kv_cache.dtype(),
                k_scale: gpu_kv_cache.k_scale(),
                v_scale: gpu_kv_cache.v_scale(),
                num_blocks: gpu_kv_cache.num_blocks() as i64,
                block_size: gpu_kv_cache.block_size() as i64,
                num_q_heads: self.num_heads as i64,
                num_kv_heads: self.num_kv_heads as i64,
                head_dim: self.head_dim as i64,
                num_tokens: num_tokens as i64,
                num_seqs: attn_meta.num_seqs() as i64,
                max_num_blocks_per_seq: attn_meta.max_num_blocks_per_seq as i64,
                scale: 1.0f32 / (self.head_dim as f32).sqrt(),
                is_prefill: seq_len > 1,
                slot_mapping: attn_meta.slot_mapping.clone(),
                block_tables_flat: attn_meta.flatten_block_tables(),
                seq_lens: attn_meta.seq_lens.iter().map(|&s| s as i32).collect(),
                query_start_loc: attn_meta.query_start_loc.iter().map(|&s| s as i32).collect(),
            };

            // Custom op writes K/V into the paged cache then runs PagedAttention,
            // returning [num_tokens, num_heads, head_dim].
            let attn_output = q_tok.apply_op3_no_bwd(&k_tok, &v_tok, &op)?;
            let attn_output = attn_output.to_dtype(hidden_states.dtype())?.reshape((
                bsz,
                seq_len,
                self.num_heads * self.head_dim,
            ))?;
            return self.o_proj.forward(&attn_output);
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
            let attn_output = attn_output.transpose(1, 2)?.reshape((
                bsz,
                seq_len,
                self.num_heads * self.head_dim,
            ))?;

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

// ── Paged-attention CUDA op (Layers 0 + 3) ───────────────────────────────
//
// Wraps one decoder layer's paged-attention step as a candle `CustomOp3` over
// (q, k, v): it scatter-writes the new K/V into the block-addressed `GpuKVCache`,
// then runs the paged-attention read, returning the attention output as a candle
// tensor. The device-pointer extraction mirrors the validated pattern in
// `rllm_quant::int8::Int8MatmulOp::cuda_fwd`.
//
// Layer 3 makes the write+read kernels dispatch on the cache dtype:
//   - F16 / BF16 → `cache_write_f16` + `paged_attention_*_f16`
//   - FP8E4M3 / FP8E5M2 → `cache_write_fp8` + `paged_attention_*_fp8` (+ `is_e5m2`)
//   - INT8 → `cache_write_i8` + `paged_attention_*_i8` (+ `k_scale`/`v_scale`)
// Q/K/V are always f16 (the projections are cast to f16 before the op); only the
// cache element type and the selected kernels change.
//
// This code is only compiled with `--features cuda` (sets `has_cuda`); it has
// not been compiled on this dev host (no nvcc) and must be built/validated on
// the GPU box. See docs/quantization-int8-kvcache-plan.md (Layers 0–3).

/// Whether the opt-in paged-attention path is enabled (`RLLM_PAGED_ATTENTION=1`).
#[cfg(has_cuda)]
fn paged_attention_enabled() -> bool {
    std::env::var("RLLM_PAGED_ATTENTION").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Inputs: q `[num_tokens, num_q_heads, head_dim]`, k/v `[num_tokens, num_kv_heads, head_dim]`
/// (token-major, contiguous, f16). Output: `[num_tokens, num_q_heads, head_dim]`.
/// The write+read kernels are selected at runtime by `cache_dtype`.
#[cfg(has_cuda)]
struct PagedAttentionOp {
    /// Per-layer cache device pointers (raw addresses; cast back in `cuda_fwd`).
    key_cache: usize,
    value_cache: usize,
    /// KV cache element dtype; selects the write+read kernel family.
    cache_dtype: rllm_core::dtype::DType,
    /// INT8 dequant scales (`x ~= q * scale`); ignored for non-INT8 dtypes.
    k_scale: f32,
    v_scale: f32,
    num_blocks: i64,
    block_size: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    num_tokens: i64,
    num_seqs: i64,
    max_num_blocks_per_seq: i64,
    scale: f32,
    is_prefill: bool,
    slot_mapping: Vec<i64>,
    block_tables_flat: Vec<i32>,
    seq_lens: Vec<i32>,
    query_start_loc: Vec<i32>,
}

#[cfg(has_cuda)]
impl candle_core::CustomOp3 for PagedAttentionOp {
    fn name(&self) -> &'static str {
        "rllm-paged-attention"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
        _s3: &candle_core::CpuStorage,
        _l3: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Err(candle_core::Error::Msg("rllm-paged-attention runs only on CUDA".to_string()))
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        ql: &candle_core::Layout,
        k: &candle_core::CudaStorage,
        kl: &candle_core::Layout,
        v: &candle_core::CudaStorage,
        vl: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        if !ql.is_contiguous() || !kl.is_contiguous() || !vl.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "paged-attention inputs must be contiguous".to_string(),
            ));
        }

        let device = q.device.clone();
        let stream = device.cuda_stream();

        let q_slice = q.as_cuda_slice::<half::f16>()?;
        let q_slice = q_slice.slice(ql.start_offset()..ql.start_offset() + ql.shape().elem_count());
        let k_slice = k.as_cuda_slice::<half::f16>()?;
        let k_slice = k_slice.slice(kl.start_offset()..kl.start_offset() + kl.shape().elem_count());
        let v_slice = v.as_cuda_slice::<half::f16>()?;
        let v_slice = v_slice.slice(vl.start_offset()..vl.start_offset() + vl.shape().elem_count());

        // Per-step metadata uploads (these change every step, unlike weights).
        let slot_dev = device.clone_htod(self.slot_mapping.as_slice())?;
        let bt_dev = device.clone_htod(self.block_tables_flat.as_slice())?;
        let seq_dev = device.clone_htod(self.seq_lens.as_slice())?;
        let qsl_dev = device.clone_htod(self.query_start_loc.as_slice())?;

        let out_len = (self.num_tokens * self.num_q_heads * self.head_dim) as usize;
        let mut output = unsafe { device.alloc::<half::f16>(out_len)? };

        {
            let (q_ptr, _gq) = q_slice.device_ptr(&stream);
            let (k_ptr, _gk) = k_slice.device_ptr(&stream);
            let (v_ptr, _gv) = v_slice.device_ptr(&stream);
            let (slot_ptr, _gs) = slot_dev.device_ptr(&stream);
            let (bt_ptr, _gb) = bt_dev.device_ptr(&stream);
            let (seq_ptr, _gl) = seq_dev.device_ptr(&stream);
            let (qsl_ptr, _gqsl) = qsl_dev.device_ptr(&stream);
            let (out_ptr, _go) = output.device_ptr_mut(&stream);
            let cu = stream.cu_stream() as usize;

            // Dispatch the write + read kernels on the cache element dtype.
            // Q/K/V are f16; only the cache element type and kernels differ.
            // `fp8_e5m2` distinguishes the two fp8 encodings.
            use rllm_core::dtype::DType;
            let (k_cache_u8, v_cache_u8) = (self.key_cache as *mut u8, self.value_cache as *mut u8);
            let fp8_e5m2 = matches!(self.cache_dtype, DType::FP8E5M2);

            match self.cache_dtype {
                DType::F16 | DType::BF16 => unsafe {
                    let key_cache = self.key_cache as *mut u16;
                    let value_cache = self.value_cache as *mut u16;
                    rllm_kernels::cache_ops::cache_write_f16(
                        key_cache,
                        value_cache,
                        k_ptr as *const u16,
                        v_ptr as *const u16,
                        slot_ptr as *const i64,
                        self.num_tokens,
                        self.num_kv_heads,
                        self.head_dim,
                        self.block_size,
                        self.num_blocks,
                        cu,
                    )
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                    if self.is_prefill {
                        rllm_kernels::attention::paged_attention_prefill_f16(
                            out_ptr as *mut u16,
                            q_ptr as *const u16,
                            key_cache as *const u16,
                            value_cache as *const u16,
                            bt_ptr as *const i32,
                            seq_ptr as *const i32,
                            qsl_ptr as *const i32,
                            self.num_seqs,
                            self.num_tokens,
                            self.num_q_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                            self.max_num_blocks_per_seq,
                            self.scale,
                            cu,
                        )
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    } else {
                        rllm_kernels::attention::paged_attention_decode_f16(
                            out_ptr as *mut u16,
                            q_ptr as *const u16,
                            key_cache as *const u16,
                            value_cache as *const u16,
                            bt_ptr as *const i32,
                            seq_ptr as *const i32,
                            self.num_seqs,
                            self.num_q_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                            self.max_num_blocks_per_seq,
                            self.scale,
                            cu,
                        )
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    }
                },
                DType::FP8E4M3 | DType::FP8E5M2 => unsafe {
                    rllm_kernels::cache_ops::cache_write_fp8(
                        k_cache_u8,
                        v_cache_u8,
                        k_ptr as *const u16,
                        v_ptr as *const u16,
                        slot_ptr as *const i64,
                        self.num_tokens,
                        self.num_kv_heads,
                        self.head_dim,
                        self.block_size,
                        self.num_blocks,
                        fp8_e5m2,
                        cu,
                    )
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                    if self.is_prefill {
                        rllm_kernels::attention::paged_attention_prefill_fp8(
                            out_ptr as *mut u16,
                            q_ptr as *const u16,
                            k_cache_u8 as *const u8,
                            v_cache_u8 as *const u8,
                            bt_ptr as *const i32,
                            seq_ptr as *const i32,
                            qsl_ptr as *const i32,
                            self.num_seqs,
                            self.num_tokens,
                            self.num_q_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                            self.max_num_blocks_per_seq,
                            self.scale,
                            fp8_e5m2,
                            cu,
                        )
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    } else {
                        rllm_kernels::attention::paged_attention_decode_fp8(
                            out_ptr as *mut u16,
                            q_ptr as *const u16,
                            k_cache_u8 as *const u8,
                            v_cache_u8 as *const u8,
                            bt_ptr as *const i32,
                            seq_ptr as *const i32,
                            self.num_seqs,
                            self.num_q_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                            self.max_num_blocks_per_seq,
                            self.scale,
                            fp8_e5m2,
                            cu,
                        )
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    }
                },
                DType::INT8 => unsafe {
                    let key_cache = self.key_cache as *mut i8;
                    let value_cache = self.value_cache as *mut i8;
                    rllm_kernels::cache_ops::cache_write_i8(
                        key_cache,
                        value_cache,
                        k_ptr as *const u16,
                        v_ptr as *const u16,
                        slot_ptr as *const i64,
                        self.num_tokens,
                        self.num_kv_heads,
                        self.head_dim,
                        self.block_size,
                        self.num_blocks,
                        self.k_scale,
                        self.v_scale,
                        cu,
                    )
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                    if self.is_prefill {
                        rllm_kernels::attention::paged_attention_prefill_i8(
                            out_ptr as *mut u16,
                            q_ptr as *const u16,
                            key_cache as *const i8,
                            value_cache as *const i8,
                            bt_ptr as *const i32,
                            seq_ptr as *const i32,
                            qsl_ptr as *const i32,
                            self.num_seqs,
                            self.num_tokens,
                            self.num_q_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                            self.max_num_blocks_per_seq,
                            self.scale,
                            self.k_scale,
                            self.v_scale,
                            cu,
                        )
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    } else {
                        rllm_kernels::attention::paged_attention_decode_i8(
                            out_ptr as *mut u16,
                            q_ptr as *const u16,
                            key_cache as *const i8,
                            value_cache as *const i8,
                            bt_ptr as *const i32,
                            seq_ptr as *const i32,
                            self.num_seqs,
                            self.num_q_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                            self.max_num_blocks_per_seq,
                            self.scale,
                            self.k_scale,
                            self.v_scale,
                            cu,
                        )
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    }
                },
                other => {
                    return Err(candle_core::Error::Msg(format!(
                        "paged attention: unsupported KV cache dtype {other:?}"
                    )));
                }
            }
        }

        let storage = candle_core::CudaStorage::wrap_cuda_slice(output, device);
        let shape = (self.num_tokens as usize, self.num_q_heads as usize, self.head_dim as usize);
        Ok((storage, shape.into()))
    }
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
            &hidden_states,
            positions,
            gpu_kv_cache,
            attn_meta,
            layer_idx,
            rope,
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

    #[test]
    fn weight_quantization_simulation() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0f32, (128, 64), &device)?;

        // MXFP8 test
        let plan_mxfp8 = QuantizationPlan::mxfp8();
        let q_mxfp8 = simulate_weight_quantization(&weight, &plan_mxfp8)?;
        assert_eq!(q_mxfp8.dims(), weight.dims());

        // INT4 test
        let plan_int4 = QuantizationPlan::int4();
        let q_int4 = simulate_weight_quantization(&weight, &plan_int4)?;
        assert_eq!(q_int4.dims(), weight.dims());

        // NVFP4 test
        let plan_nvfp4 = QuantizationPlan::nvfp4();
        let q_nvfp4 = simulate_weight_quantization(&weight, &plan_nvfp4)?;
        assert_eq!(q_nvfp4.dims(), weight.dims());

        Ok(())
    }
}
