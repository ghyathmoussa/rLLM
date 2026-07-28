#[cfg(feature = "candle-backend")]
use candle_core::{D, DType, Device, Result, Tensor};
#[cfg(feature = "candle-backend")]
use rllm_core::optimizations::QuantizationPlan;
#[cfg(feature = "candle-backend")]
use rllm_quant::LinearMethod;

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
        QuantizedWeightFormat::Nvfp4 => {
            let group_size = plan.group_size.unwrap_or(16);
            simulate_group_quant(&w_f32, group_size, 4)?
        }
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
#[allow(dead_code)]
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

    pub(crate) fn hidden_size(&self) -> Result<usize> {
        self.weight.dim(D::Minus1)
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
    Awq {
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        bits: usize,
        group_size: usize,
        dequantized: std::sync::OnceLock<Tensor>,
    },
    Method(Box<dyn LinearMethod>),
}

#[cfg(feature = "candle-backend")]
pub fn dequantize_gptq(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<Tensor> {
    let device = qweight.device();
    if let Device::Cuda(_) = device {
        // Dequantize on CPU. The packed-INT4 unpacking relies on I64/integer
        // ops whose CUDA kernels are incomplete in this Candle/cudarc build
        // (they raise "named symbol not found"). The result is cached per layer,
        // so the host transfer is a one-time cost; the dequantized FP weights
        // then live on the GPU for the matmul.
        let cpu = Device::Cpu;
        let out = dequantize_gptq_impl(
            &qweight.to_device(&cpu)?,
            &qzeros.to_device(&cpu)?,
            &scales.to_device(&cpu)?,
            &g_idx.to_device(&cpu)?,
            bits,
            group_size,
            &cpu,
        )?;
        return out.to_device(device);
    }
    dequantize_gptq_impl(qweight, qzeros, scales, g_idx, bits, group_size, device)
}

#[cfg(feature = "candle-backend")]
fn dequantize_gptq_impl(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: &Tensor,
    bits: usize,
    _group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    if bits != 4 {
        return Err(candle_core::Error::Msg(format!("Only 4-bit GPTQ is supported, got {bits}")));
    }

    // We expect qweight to be (in_features / 8, out_features)
    let (packed_in_features, out_features) = qweight.dims2()?;
    let in_features = packed_in_features * 8;

    // Convert qweight from signed I32 to unsigned F64 to avoid sign-extension bugs.
    // If it's negative, add 2^32.
    let qweight_f64 = qweight.to_dtype(DType::F64)?;
    let zero_w = Tensor::new(0.0f64, device)?.broadcast_as(qweight_f64.shape())?;
    let is_neg = qweight_f64.lt(&zero_w)?;
    let offset_w = Tensor::new(4294967296.0f64, device)?.broadcast_as(qweight_f64.shape())?;
    let qweight_u64 = qweight_f64.add(&is_neg.to_dtype(DType::F64)?.mul(&offset_w)?)?;

    // Unpack qweight (bits = 4, so 8 values per i32/i64)
    let mut w_unpacked = Vec::with_capacity(8);
    let c16_w = Tensor::new(16.0f64, device)?.broadcast_as(qweight_u64.shape())?;
    for i in 0..8 {
        let divisor = (1i64 << (4 * i)) as f64;
        let divisor_t = Tensor::new(divisor, device)?.broadcast_as(qweight_u64.shape())?;
        let shifted = qweight_u64.div(&divisor_t)?.floor()?;
        let temp = shifted.div(&c16_w)?.floor()?.mul(&c16_w)?;
        let masked = shifted.sub(&temp)?;
        w_unpacked.push(masked);
    }
    // Stack along dimension 1 (shape: [packed_in_features, 8, out_features])
    let w_stacked = Tensor::stack(&w_unpacked, 1)?;
    // Reshape to [in_features, out_features]
    let w_raw = w_stacked.reshape((in_features, out_features))?.to_dtype(DType::F32)?;

    // Convert qzeros from signed I32 to unsigned F64.
    let qzeros_f64 = qzeros.to_dtype(DType::F64)?;
    let zero_z = Tensor::new(0.0f64, device)?.broadcast_as(qzeros_f64.shape())?;
    let is_neg_z = qzeros_f64.lt(&zero_z)?;
    let offset_z = Tensor::new(4294967296.0f64, device)?.broadcast_as(qzeros_f64.shape())?;
    let qzeros_u64 = qzeros_f64.add(&is_neg_z.to_dtype(DType::F64)?.mul(&offset_z)?)?;

    // Unpack qzeros
    let mut z_unpacked = Vec::with_capacity(8);
    let c16_z = Tensor::new(16.0f64, device)?.broadcast_as(qzeros_u64.shape())?;
    let one_z = Tensor::new(1.0f32, device)?.broadcast_as(qzeros_u64.shape())?;
    for i in 0..8 {
        let divisor = (1i64 << (4 * i)) as f64;
        let divisor_t = Tensor::new(divisor, device)?.broadcast_as(qzeros_u64.shape())?;
        let shifted = qzeros_u64.div(&divisor_t)?.floor()?;
        let temp = shifted.div(&c16_z)?.floor()?.mul(&c16_z)?;
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
    let select_zeros = z_raw.index_select(&g_idx_u32, 0)?; // [in_features, out_features]

    // Apply dequantization formula: (W_q - ZP) * Scale
    let target_dtype = scales.dtype();
    let w_dequant = w_raw.sub(&select_zeros)?.mul(&select_scales.to_dtype(DType::F32)?)?;
    let w_dequant = w_dequant.to_dtype(target_dtype)?;

    // Transpose back to match [out_features, in_features] shape
    w_dequant.t()
}

#[cfg(feature = "candle-backend")]
pub fn dequantize_awq(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<Tensor> {
    let device = qweight.device();
    if let Device::Cuda(_) = device {
        let cpu = Device::Cpu;
        let out = dequantize_awq_impl(
            &qweight.to_device(&cpu)?,
            &qzeros.to_device(&cpu)?,
            &scales.to_device(&cpu)?,
            bits,
            group_size,
            &cpu,
        )?;
        return out.to_device(device);
    }
    dequantize_awq_impl(qweight, qzeros, scales, bits, group_size, device)
}

#[cfg(feature = "candle-backend")]
fn dequantize_awq_impl(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: usize,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    if bits != 4 {
        return Err(candle_core::Error::Msg(format!("Only 4-bit AWQ is supported, got {bits}")));
    }

    let (in_features, packed_out_features) = qweight.dims2()?;
    let out_features = packed_out_features * 8;
    let num_groups = in_features / group_size;

    // Convert qweight from signed I32 to unsigned F64 to avoid sign-extension bugs.
    // If it's negative, add 2^32.
    let qweight_f64 = qweight.to_dtype(DType::F64)?;
    let zero_w = Tensor::new(0.0f64, device)?.broadcast_as(qweight_f64.shape())?;
    let is_neg = qweight_f64.lt(&zero_w)?;
    let offset_w = Tensor::new(4294967296.0f64, device)?.broadcast_as(qweight_f64.shape())?;
    let qweight_u64 = qweight_f64.add(&is_neg.to_dtype(DType::F64)?.mul(&offset_w)?)?;

    // Unpack qweight (bits = 4, 8 values per i32/f64 word)
    let mut w_unpacked = Vec::with_capacity(8);
    let c16_w = Tensor::new(16.0f64, device)?.broadcast_as(qweight_u64.shape())?;
    for i in 0..8 {
        let divisor = (1i64 << (4 * i)) as f64;
        let divisor_t = Tensor::new(divisor, device)?.broadcast_as(qweight_u64.shape())?;
        let shifted = qweight_u64.div(&divisor_t)?.floor()?;
        let temp = shifted.div(&c16_w)?.floor()?.mul(&c16_w)?;
        let masked = shifted.sub(&temp)?;
        w_unpacked.push(masked);
    }

    let awq_reverse_order = [0, 4, 1, 5, 2, 6, 3, 7];
    let mut w_unpacked_reordered = Vec::with_capacity(8);
    for &idx in &awq_reverse_order {
        w_unpacked_reordered.push(w_unpacked[idx].clone());
    }
    // Stack along dimension 2 (shape: [in_features, packed_out_features, 8])
    let w_stacked = Tensor::stack(&w_unpacked_reordered, 2)?;
    // Reshape to [in_features, out_features]
    let w_raw = w_stacked.reshape((in_features, out_features))?.to_dtype(scales.dtype())?;

    // Convert qzeros from signed I32 to unsigned F64.
    let qzeros_f64 = qzeros.to_dtype(DType::F64)?;
    let zero_z = Tensor::new(0.0f64, device)?.broadcast_as(qzeros_f64.shape())?;
    let is_neg_z = qzeros_f64.lt(&zero_z)?;
    let offset_z = Tensor::new(4294967296.0f64, device)?.broadcast_as(qzeros_f64.shape())?;
    let qzeros_u64 = qzeros_f64.add(&is_neg_z.to_dtype(DType::F64)?.mul(&offset_z)?)?;

    // Unpack qzeros
    let mut z_unpacked = Vec::with_capacity(8);
    let c16_z = Tensor::new(16.0f64, device)?.broadcast_as(qzeros_u64.shape())?;
    for i in 0..8 {
        let divisor = (1i64 << (4 * i)) as f64;
        let divisor_t = Tensor::new(divisor, device)?.broadcast_as(qzeros_u64.shape())?;
        let shifted = qzeros_u64.div(&divisor_t)?.floor()?;
        let temp = shifted.div(&c16_z)?.floor()?.mul(&c16_z)?;
        let masked = shifted.sub(&temp)?;
        z_unpacked.push(masked);
    }

    let mut z_unpacked_reordered = Vec::with_capacity(8);
    for &idx in &awq_reverse_order {
        z_unpacked_reordered.push(z_unpacked[idx].clone());
    }
    // Stack along dimension 2 (shape: [num_groups, packed_out_features, 8])
    let z_stacked = Tensor::stack(&z_unpacked_reordered, 2)?;
    // Reshape to [num_groups, out_features]
    let z_raw = z_stacked.reshape((num_groups, out_features))?.to_dtype(scales.dtype())?;

    // Expand scales and zeros to shape [in_features, out_features]
    let scales_expanded = scales
        .unsqueeze(1)?
        .expand((num_groups, group_size, out_features))?
        .reshape((in_features, out_features))?;
    let z_expanded = z_raw
        .unsqueeze(1)?
        .expand((num_groups, group_size, out_features))?
        .reshape((in_features, out_features))?;

    // Apply dequantization formula: (W_q - ZP) * Scale
    let w_dequant = w_raw.sub(&z_expanded)?.mul(&scales_expanded)?;

    // Transpose back to match [out_features, in_features] shape
    w_dequant.t()
}

#[cfg(feature = "candle-backend")]
pub struct Linear {
    weight: LinearWeight,
    bias: Option<Tensor>,
}

#[cfg(feature = "candle-backend")]
impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { weight: LinearWeight::Fp(weight), bias: None }
    }

    pub fn from_method(method: Box<dyn LinearMethod>) -> Self {
        Self { weight: LinearWeight::Method(method), bias: None }
    }

    /// Attach an output bias while preserving the underlying dense or
    /// quantized weight implementation.
    pub fn with_bias(mut self, bias: Tensor) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn new_awq(
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        bits: usize,
        group_size: usize,
    ) -> Self {
        Self {
            weight: LinearWeight::Awq {
                qweight,
                qzeros,
                scales,
                bits,
                group_size,
                dequantized: std::sync::OnceLock::new(),
            },
            bias: None,
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
            bias: None,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = match &self.weight {
            LinearWeight::Fp(weight) => self.forward_fp(x, weight),
            LinearWeight::Method(method) => method.apply(x),
            LinearWeight::Gptq {
                qweight,
                qzeros,
                scales,
                g_idx,
                bits,
                group_size,
                dequantized,
            } => {
                #[cfg(feature = "cuda")]
                if let Some(out) = self.try_forward_gptq_cuda(
                    x,
                    qweight,
                    qzeros,
                    scales,
                    g_idx,
                    *bits,
                    *group_size,
                )? {
                    return self.add_bias(out);
                }

                let weight = if let Some(w) = dequantized.get() {
                    w
                } else {
                    let w = dequantize_gptq(qweight, qzeros, scales, g_idx, *bits, *group_size)?;
                    let _ = dequantized.set(w);
                    dequantized.get().unwrap()
                };
                self.forward_fp(x, weight)
            }
            LinearWeight::Awq { qweight, qzeros, scales, bits, group_size, dequantized } => {
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    self.try_forward_awq_cuda(x, qweight, qzeros, scales, *bits, *group_size)?
                {
                    return self.add_bias(out);
                }

                let weight = if let Some(w) = dequantized.get() {
                    w
                } else {
                    let w = dequantize_awq(qweight, qzeros, scales, *bits, *group_size)?;
                    let _ = dequantized.set(w);
                    dequantized.get().unwrap()
                };
                self.forward_fp(x, weight)
            }
        }?;
        self.add_bias(output)
    }

    fn add_bias(&self, output: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }

    pub fn weight(&self) -> Result<&Tensor> {
        match &self.weight {
            LinearWeight::Fp(w) => Ok(w),
            LinearWeight::Method(method) => method
                .weight()
                .ok_or_else(|| candle_core::Error::Msg("method weight not available".to_string())),
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
            LinearWeight::Awq { qweight, qzeros, scales, bits, group_size, dequantized } => {
                if let Some(w) = dequantized.get() {
                    Ok(w)
                } else {
                    let w = dequantize_awq(qweight, qzeros, scales, *bits, *group_size)?;
                    let _ = dequantized.set(w);
                    Ok(dequantized.get().unwrap())
                }
            }
        }
    }

    pub fn is_quantized(&self) -> bool {
        match &self.weight {
            LinearWeight::Gptq { .. } => true,
            LinearWeight::Awq { .. } => true,
            LinearWeight::Method(method) => method.is_quantized(),
            LinearWeight::Fp(_) => false,
        }
    }

    fn forward_fp(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_forward_gptq_cuda(
        &self,
        x: &Tensor,
        qweight: &Tensor,
        qzeros: &Tensor,
        scales: &Tensor,
        g_idx: &Tensor,
        bits: usize,
        group_size: usize,
    ) -> Result<Option<Tensor>> {
        if bits != 4
            || x.dtype() != DType::F16
            || scales.dtype() != DType::F16
            || !matches!(x.device(), Device::Cuda(_))
            || !matches!(qweight.device(), Device::Cuda(_))
            || !matches!(qzeros.device(), Device::Cuda(_))
            || !matches!(scales.device(), Device::Cuda(_))
            || !matches!(g_idx.device(), Device::Cuda(_))
        {
            return Ok(None);
        }

        let (_packed_in_features, out_features) = qweight.dims2()?;
        if out_features % 8 != 0 {
            return Ok(None);
        }

        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let in_features = x.dim(D::Minus1)?;
        if in_features % 8 != 0 || g_idx.dim(0)? != in_features {
            return Ok(None);
        }

        // Ensure all tensors are contiguous to pass safely to raw CUDA kernel
        let x_contig = x.contiguous()?;
        let qweight_contig = qweight.contiguous()?;
        let qzeros_contig = qzeros.contiguous()?;
        let scales_contig = scales.contiguous()?;
        let g_idx_contig = g_idx.contiguous()?;

        let out = Tensor::zeros((batch, out_features), DType::F16, x.device())?;

        let p_x = get_cuda_ptr::<half::f16>(&x_contig)?;
        let p_qweight = get_cuda_ptr::<i32>(&qweight_contig)?;
        let p_qzeros = get_cuda_ptr::<i32>(&qzeros_contig)?;
        let p_scales = get_cuda_ptr::<half::f16>(&scales_contig)?;
        let p_gidx = get_cuda_ptr::<u32>(&g_idx_contig)?;
        let p_out = get_cuda_ptr::<half::f16>(&out)?;

        let num_groups = (in_features / group_size) as i64;

        unsafe {
            rllm_kernels::cuda::gptq_gemm_f16_sync(
                p_x as *const u16,
                p_qweight,
                p_qzeros,
                p_scales as *const u16,
                p_gidx,
                p_out as *mut u16,
                batch as i64,
                in_features as i64,
                out_features as i64,
                num_groups,
                group_size as i64,
            )
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }

        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(out_features);
        out.reshape(out_shape).map(Some)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_forward_awq_cuda(
        &self,
        x: &Tensor,
        qweight: &Tensor,
        qzeros: &Tensor,
        scales: &Tensor,
        bits: usize,
        group_size: usize,
    ) -> Result<Option<Tensor>> {
        if bits != 4
            || x.dtype() != DType::F16
            || scales.dtype() != DType::F16
            || !matches!(x.device(), Device::Cuda(_))
            || !matches!(qweight.device(), Device::Cuda(_))
            || !matches!(qzeros.device(), Device::Cuda(_))
            || !matches!(scales.device(), Device::Cuda(_))
        {
            return Ok(None);
        }

        let (in_features, packed_out_features) = qweight.dims2()?;
        let out_features = packed_out_features * 8;
        if out_features % 8 != 0 {
            return Ok(None);
        }

        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let x_in_features = x.dim(D::Minus1)?;
        if x_in_features != in_features || in_features % 8 != 0 {
            return Ok(None);
        }

        // Ensure all tensors are contiguous to pass safely to raw CUDA kernel
        let x_contig = x.contiguous()?;
        let qweight_contig = qweight.contiguous()?;
        let qzeros_contig = qzeros.contiguous()?;
        let scales_contig = scales.contiguous()?;

        let out = Tensor::zeros((batch, out_features), DType::F16, x.device())?;

        let p_x = get_cuda_ptr::<half::f16>(&x_contig)?;
        let p_qweight = get_cuda_ptr::<i32>(&qweight_contig)?;
        let p_qzeros = get_cuda_ptr::<i32>(&qzeros_contig)?;
        let p_scales = get_cuda_ptr::<half::f16>(&scales_contig)?;
        let p_out = get_cuda_ptr::<half::f16>(&out)?;

        let num_groups = (in_features / group_size) as i64;

        unsafe {
            rllm_kernels::cuda::awq_gemm_f16_sync(
                p_x as *const u16,
                p_qweight,
                p_qzeros,
                p_scales as *const u16,
                p_out as *mut u16,
                batch as i64,
                in_features as i64,
                out_features as i64,
                num_groups,
                group_size as i64,
            )
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }

        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(out_features);
        out.reshape(out_shape).map(Some)
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
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = gate.silu()?;
        self.down_proj.forward(&gate.broadcast_mul(&up)?)
    }

    pub fn gate_proj(&self) -> &Linear {
        &self.gate_proj
    }

    pub fn up_proj(&self) -> &Linear {
        &self.up_proj
    }

    pub fn down_proj(&self) -> &Linear {
        &self.down_proj
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
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
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
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            q_norm: None,
            k_norm: None,
        }
    }

    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::from_linears(q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim)
    }

    /// Apply per-head Q/K normalization before RoPE, as required by Qwen3.
    pub fn with_qk_norm(mut self, q_norm: RmsNorm, k_norm: RmsNorm) -> Self {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
    }

    fn project_qkv(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (bsz, seq_len, _) = hidden_states.dims3()?;
        let mut q = self.q_proj.forward(hidden_states)?.reshape((
            bsz,
            seq_len,
            self.num_heads,
            self.head_dim,
        ))?;
        let mut k = self.k_proj.forward(hidden_states)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = self.v_proj.forward(hidden_states)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        Ok((q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?))
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = hidden_states.dims3()?;

        let (q, k, v) = self.project_qkv(hidden_states)?;

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

        // Scaled dot-product attention. Candle's CUDA matmul requires contiguous
        // operands; `q`, `k`/`v` are non-contiguous views after transpose (and RoPE/cat),
        // so materialize them before the batched matmul.
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let q_contig = q.contiguous()?;
        let k_t = k.t()?.contiguous()?;
        let attn_weights = q_contig
            .matmul(&k_t)?
            .broadcast_mul(&Tensor::new(scale, q.device())?.to_dtype(q.dtype())?)?;

        // Apply causal mask for prefill (seq_len > 1)
        let attn_weights = if seq_len > 1 {
            let mask = causal_mask(seq_len, q.device())?.to_dtype(q.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let v_contig = v.contiguous()?;
        let attn_output = attn_weights.matmul(&v_contig)?;

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        let attn_output =
            attn_output.transpose(1, 2)?.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

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

        let (q, k, v) = self.project_qkv(hidden_states)?;

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
                k_scale: gpu_kv_cache.k_scale(layer_idx),
                v_scale: gpu_kv_cache.v_scale(layer_idx),
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
            self.o_proj.forward(&attn_output)
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

    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }

    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }

    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }

    pub fn o_proj(&self) -> &Linear {
        &self.o_proj
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(feature = "candle-backend")]
pub(crate) fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
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
pub(crate) fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
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
pub(crate) fn paged_attention_enabled() -> bool {
    std::env::var("RLLM_PAGED_ATTENTION").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Inputs: q `[num_tokens, num_q_heads, head_dim]`, k/v `[num_tokens, num_kv_heads, head_dim]`
/// (token-major, contiguous, f16). Output: `[num_tokens, num_q_heads, head_dim]`.
/// The write+read kernels are selected at runtime by `cache_dtype`.
#[cfg(has_cuda)]
pub(crate) struct PagedAttentionOp {
    /// Per-layer cache device pointers (raw addresses; cast back in `cuda_fwd`).
    pub(crate) key_cache: usize,
    pub(crate) value_cache: usize,
    /// KV cache element dtype; selects the write+read kernel family.
    pub(crate) cache_dtype: rllm_core::dtype::DType,
    /// INT8 dequant scales (`x ~= q * scale`); ignored for non-INT8 dtypes.
    pub(crate) k_scale: f32,
    pub(crate) v_scale: f32,
    pub(crate) num_blocks: i64,
    pub(crate) block_size: i64,
    pub(crate) num_q_heads: i64,
    pub(crate) num_kv_heads: i64,
    pub(crate) head_dim: i64,
    pub(crate) num_tokens: i64,
    pub(crate) num_seqs: i64,
    pub(crate) max_num_blocks_per_seq: i64,
    pub(crate) scale: f32,
    pub(crate) is_prefill: bool,
    pub(crate) slot_mapping: Vec<i64>,
    pub(crate) block_tables_flat: Vec<i32>,
    pub(crate) seq_lens: Vec<i32>,
    pub(crate) query_start_loc: Vec<i32>,
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

    pub fn self_attn(&self) -> &LlamaAttention {
        &self.self_attn
    }

    pub fn mlp(&self) -> &LlamaMLP {
        &self.mlp
    }

    pub fn input_layernorm(&self) -> &RmsNorm {
        &self.input_layernorm
    }

    pub fn post_attention_layernorm(&self) -> &RmsNorm {
        &self.post_attention_layernorm
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
    #[allow(clippy::needless_range_loop)]
    fn gptq_dequantization_correctness() -> Result<()> {
        let device = Device::Cpu;

        // Shape: (1, 8)
        let qweight_data = vec![
            1985229328i32,  // Col 0: 0x76543210
            -19088744i32,   // Col 1: 0xFEDCBA98
            -324508640i32,  // Col 2: 0xECA86420
            -1985229329i32, // Col 3: 0x89ABCDEF
            0i32,
            0i32,
            0i32,
            0i32, // Col 4-7
        ];
        let qweight = Tensor::from_vec(qweight_data, (1, 8), &device)?;

        // Shape: (1, 1)
        // Zero points packed: 7 for all 8 cols -> 0x77777777
        let qzeros = Tensor::from_vec(vec![2004318071i32], (1, 1), &device)?;

        // Shape: (1, 8)
        let scales = Tensor::from_vec(
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32],
            (1, 8),
            &device,
        )?;

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

    #[cfg(feature = "cuda")]
    #[test]
    fn gptq_cuda_linear_forward_correctness() -> Result<()> {
        let device = Device::new_cuda(0)?;

        // Setup same weights as CPU test but on CUDA device
        let qweight_data = vec![
            1985229328i32,  // Col 0: 0x76543210
            -19088744i32,   // Col 1: 0xFEDCBA98
            -324508640i32,  // Col 2: 0xECA86420
            -1985229329i32, // Col 3: 0x89ABCDEF
            0i32,
            0i32,
            0i32,
            0i32, // Col 4-7
        ];
        let qweight = Tensor::from_vec(qweight_data, (1, 8), &device)?;

        // Shape: (1, 1)
        let qzeros = Tensor::from_vec(vec![2004318071i32], (1, 1), &device)?;

        // Shape: (1, 8)
        let scales_f32 = Tensor::from_vec(
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32],
            (1, 8),
            &device,
        )?;
        let scales = scales_f32.to_dtype(DType::F16)?;

        // Shape: (8,)
        let g_idx = Tensor::from_vec(vec![0u32; 8], (8,), &device)?;

        let linear = Linear::new_gptq(qweight, qzeros, scales, g_idx, 4, 8);

        // Input x (shape [1, 8], dtype F16)
        let x_data = vec![1.0f32, -1.0, 0.5, 2.0, 0.25, -0.5, 1.5, -2.0];
        let x = Tensor::from_vec(x_data, (1, 8), &device)?.to_dtype(DType::F16)?;

        let out = linear.forward(&x)?;
        assert_eq!(out.dims(), &[1, 8]);

        let out_f32 = out.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // Let's compute the expected outputs using the CPU baseline
        let expected_out = x
            .to_device(&Device::Cpu)?
            .reshape((1, 8))?
            .matmul(&linear.weight()?.to_device(&Device::Cpu)?.t()?)?;
        let expected_vals = expected_out.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();

        for col in 0..8 {
            let expected = expected_vals[col];
            let actual = out_f32[0][col];
            assert!(
                (actual - expected).abs() < 0.2,
                "col {col}: expected {expected:.4}, got {actual:.4}"
            );
        }
        Ok(())
    }

    #[test]
    fn awq_dequantization_correctness() -> Result<()> {
        let device = Device::Cpu;

        // Shape: (8, 1) — 8 rows (in_features = 8), 1 packed col (out_features = 8)
        let qweight_data = vec![
            1985229328i32, // Row 0: 0x76543210 -> unpacked: [0, 4, 1, 5, 2, 6, 3, 7]
            1985229328i32, // Row 1
            1985229328i32, // Row 2
            1985229328i32, // Row 3
            1985229328i32, // Row 4
            1985229328i32, // Row 5
            1985229328i32, // Row 6
            1985229328i32, // Row 7
        ];
        let qweight = Tensor::from_vec(qweight_data, (8, 1), &device)?;

        // Shape: (1, 1) — 1 group, 1 packed col. Zeros = 3 -> 0x33333333 = 858993459
        let qzeros = Tensor::from_vec(vec![858993459i32], (1, 1), &device)?;

        // Shape: (1, 8) — 1 group, 8 cols. Scales = 1.0
        let scales = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &device)?;

        let w_dequant = dequantize_awq(&qweight, &qzeros, &scales, 4, 8)?;
        // Expected shape: (8, 8) — out_features, in_features
        assert_eq!(w_dequant.dims(), &[8, 8]);

        let w_vals = w_dequant.to_vec2::<f32>()?;
        let expected_row = [-3.0, 1.0, -2.0, 2.0, -1.0, 3.0, 0.0, 4.0];

        // w_dequant has shape [out_features, in_features]
        // so w_vals[col][row] should be expected_row[col] (since all rows are identical)
        for col in 0..8 {
            for &val in &w_vals[col] {
                assert_eq!(val, expected_row[col]);
            }
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn awq_cuda_linear_forward_correctness() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let qweight_data = vec![
            1985229328i32,
            1985229328i32,
            1985229328i32,
            1985229328i32,
            1985229328i32,
            1985229328i32,
            1985229328i32,
            1985229328i32,
        ];
        let qweight = Tensor::from_vec(qweight_data, (8, 1), &device)?;
        let qzeros = Tensor::from_vec(vec![858993459i32], (1, 1), &device)?;
        let scales = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &device)?.to_dtype(DType::F16)?;

        let linear = Linear::new_awq(qweight, qzeros, scales, 4, 8);

        // Input x (shape [1, 8], dtype F16)
        let x_data = vec![1.0f32, -1.0, 0.5, 2.0, 0.25, -0.5, 1.5, -2.0];
        let x = Tensor::from_vec(x_data, (1, 8), &device)?.to_dtype(DType::F16)?;

        let out = linear.forward(&x)?;
        assert_eq!(out.dims(), &[1, 8]);

        let out_f32 = out.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // Compute expected outputs using CPU baseline
        let expected_out = x
            .to_device(&Device::Cpu)?
            .reshape((1, 8))?
            .matmul(&linear.weight()?.to_device(&Device::Cpu)?.t()?)?;
        let expected_vals = expected_out.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();

        for col in 0..8 {
            let expected = expected_vals[col];
            let actual = out_f32[0][col];
            assert!(
                (actual - expected).abs() < 0.2,
                "col {col}: expected {expected:.4}, got {actual:.4}"
            );
        }
        Ok(())
    }
}

#[cfg(all(feature = "candle-backend", feature = "cuda"))]
fn get_cuda_ptr<T: candle_core::cuda_backend::CudaDType + 'static>(t: &Tensor) -> Result<*const T> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<T>()?;
            let stream = cuda_storage.device.cuda_stream();
            let (raw_ptr_u64, _guard) = slice.device_ptr(&stream);
            let offset = layout.start_offset();
            let raw_device_ptr = raw_ptr_u64 as *const T;
            unsafe { Ok(raw_device_ptr.add(offset)) }
        }
        _ => Err(candle_core::Error::Msg("Not a CUDA tensor".to_string())),
    }
}
