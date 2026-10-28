use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::method::{LinearMethod, QuantMethodFactory, WeightSource};

const FP4_VALUES: [f32; 16] =
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

pub struct MxfpWeightOnlyFactory {
    bits: u8,
    group_size: usize,
}

impl MxfpWeightOnlyFactory {
    pub fn new(bits: u8, group_size: usize) -> Self {
        Self { bits, group_size }
    }
}

impl QuantMethodFactory for MxfpWeightOnlyFactory {
    fn build_linear(
        &self,
        prefix: &str,
        source: &mut WeightSource<'_>,
    ) -> Result<Box<dyn LinearMethod>> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");

        // Try to load pre-quantized weights first
        if source.has_tensor(&weight_name) {
            let weight = source.remove_tensor(&weight_name)?;
            if source.has_tensor(&scale_name) {
                let scales = decode_e8m0_scales(source.remove_tensor(&scale_name)?)?;
                let packed = if self.bits == 8 && weight.dtype() == DType::F8E4M3 {
                    let device = weight.device().clone();
                    let shape = weight.dims().to_vec();
                    let values = weight
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<float8::F8E4M3>()?;
                    let bytes = values.iter().map(float8::F8E4M3::to_bits).collect();
                    Tensor::from_vec(bytes, shape, &device)?
                } else {
                    weight
                };
                return Ok(Box::new(MxfpLinear::new(packed, scales, self.bits, self.group_size)?));
            } else {
                // Perform online quantization if no scale is found
                let (qweight, scales) = if self.bits == 8 {
                    quantize_mxfp8_cpu(&weight, self.group_size)?
                } else {
                    quantize_mxfp4_cpu(&weight, self.group_size)?
                };
                return Ok(Box::new(MxfpLinear::new(qweight, scales, self.bits, self.group_size)?));
            }
        }

        // Check in quantized map (from safetensors load)
        if source.has_quant_tensor(&weight_name) {
            let qweight = source.remove_quant_tensor(&weight_name)?;
            let scales = decode_e8m0_scales(source.remove_tensor(&scale_name)?)?;
            // Convert QuantTensor inner data to candle Tensor
            let device = qweight.device().clone();
            let data = qweight.data().iter().map(|&x| x as u8).collect::<Vec<u8>>();
            let qweight_tensor = Tensor::from_vec(data, qweight.shape().to_vec(), &device)?;
            return Ok(Box::new(MxfpLinear::new(
                qweight_tensor,
                scales,
                self.bits,
                self.group_size,
            )?));
        }

        anyhow::bail!("missing weight tensor for {prefix}")
    }
}

/// Convert OCP E8M0 exponent bytes to ordinary scale values for the existing
/// CPU and CUDA matmul paths, which both expect numeric scales.
fn decode_e8m0_scales(scales: Tensor) -> Result<Tensor> {
    if scales.dtype() != DType::U8 {
        return Ok(scales);
    }
    let device = scales.device().clone();
    let shape = scales.dims().to_vec();
    let bytes = scales.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let values = bytes
        .into_iter()
        .map(|exponent| {
            if exponent == u8::MAX {
                anyhow::bail!("MXFP E8M0 scale contains the reserved NaN exponent 255");
            }
            Ok(2.0f32.powi(exponent as i32 - 127))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_vec(values, shape, &device)?)
}

fn power_of_two_scale(absmax: f32, max_value: f32) -> f32 {
    if absmax == 0.0 { 1.0 } else { (absmax / max_value).log2().ceil().exp2() }
}

pub struct MxfpLinear {
    qweight: Tensor,
    scales: Tensor,
    bits: u8,
    group_size: usize,
    in_features: usize,
    out_features: usize,
    dequantized: std::sync::OnceLock<Tensor>,
}

impl MxfpLinear {
    pub fn new(qweight: Tensor, scales: Tensor, bits: u8, group_size: usize) -> Result<Self> {
        let dims = qweight.dims();
        if dims.len() != 2 || dims.contains(&0) {
            anyhow::bail!("MXFP weight must be a non-empty rank-2 tensor, got {dims:?}");
        }
        if bits != 4 && bits != 8 {
            anyhow::bail!("MXFP supports 4 or 8 bits, got {bits}");
        }
        if qweight.dtype() != DType::U8 {
            anyhow::bail!("MXFP packed weight must have U8 dtype, got {:?}", qweight.dtype());
        }
        let out_features = dims[0];
        let in_features = if bits == 4 { dims[1] * 2 } else { dims[1] };
        if group_size == 0 || in_features % group_size != 0 || (bits == 4 && group_size % 2 != 0) {
            anyhow::bail!(
                "MXFP group size {group_size} must divide {in_features} input features and be even for 4-bit weights"
            );
        }
        if scales.dims() != [out_features, in_features / group_size] {
            anyhow::bail!(
                "MXFP scales must have shape [{out_features}, {}], got {:?}",
                in_features / group_size,
                scales.dims()
            );
        }
        let scale_values = scales
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if scale_values.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) {
            anyhow::bail!("MXFP scales must be finite and positive");
        }

        Ok(Self {
            qweight,
            scales,
            bits,
            group_size,
            in_features,
            out_features,
            dequantized: std::sync::OnceLock::new(),
        })
    }

    fn get_or_dequantize(&self, target_device: &Device, target_dtype: DType) -> Result<Tensor> {
        if self.dequantized.get().is_none() {
            let dequant = if self.bits == 8 {
                self.dequantize_mxfp8_cpu()?
            } else {
                self.dequantize_mxfp4_cpu()?
            };
            let _ = self.dequantized.set(dequant);
        }
        self.dequantized
            .get()
            .expect("dequantized weight stored above")
            .to_device(target_device)?
            .to_dtype(target_dtype)
            .map_err(Into::into)
    }

    fn dequantize_mxfp8_cpu(&self) -> Result<Tensor> {
        let q_u8 = self.qweight.to_device(&Device::Cpu)?;
        let q_data_u8 = q_u8.flatten_all()?.to_vec1::<u8>()?;
        let q_data: Vec<f32> =
            q_data_u8.iter().map(|&x| float8::F8E4M3::from_bits(x).to_f32()).collect();

        let scales_f32 = self.scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let scales_data = scales_f32.flatten_all()?.to_vec1::<f32>()?;

        let mut dequant = vec![0.0f32; self.out_features * self.in_features];
        let num_groups = self.in_features / self.group_size;

        for o in 0..self.out_features {
            for g in 0..num_groups {
                let scale = scales_data[o * num_groups + g];
                for k in 0..self.group_size {
                    let idx = o * self.in_features + g * self.group_size + k;
                    dequant[idx] = q_data[idx] * scale;
                }
            }
        }

        let out = Tensor::from_vec(dequant, (self.out_features, self.in_features), &Device::Cpu)?;
        Ok(out)
    }

    fn dequantize_mxfp4_cpu(&self) -> Result<Tensor> {
        // qweight is stored as packed u8 of shape [out_features, in_features / 2]
        let q_u8 = self.qweight.to_device(&Device::Cpu)?.to_dtype(DType::U8)?;
        let q_data = q_u8.flatten_all()?.to_vec1::<u8>()?;
        let scales_f32 = self.scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let scales_data = scales_f32.flatten_all()?.to_vec1::<f32>()?;

        let mut dequant = vec![0.0f32; self.out_features * self.in_features];
        let num_groups = self.in_features / self.group_size;

        for o in 0..self.out_features {
            for g in 0..num_groups {
                let scale = scales_data[o * num_groups + g];
                for k in 0..self.group_size {
                    let i = g * self.group_size + k;
                    let byte_idx = o * (self.in_features / 2) + i / 2;
                    let byte = q_data[byte_idx];
                    let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let val = FP4_VALUES[code as usize];
                    let idx = o * self.in_features + i;
                    dequant[idx] = val * scale;
                }
            }
        }

        let out = Tensor::from_vec(dequant, (self.out_features, self.in_features), &Device::Cpu)?;
        Ok(out)
    }
}

impl LinearMethod for MxfpLinear {
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), candle_core::Device::Cuda(_)) {
            return self.apply_cuda(x);
        }

        let weight = self
            .get_or_dequantize(x.device(), x.dtype())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let x_2d = x.reshape((batch, self.in_features))?;
        let out = x_2d.matmul(&weight.t()?)?;
        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(self.out_features);
        out.reshape(out_shape)
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn is_quantized(&self) -> bool {
        true
    }
}

// --- Online Quantization Helpers ---

fn quantize_mxfp8_cpu(weight: &Tensor, group_size: usize) -> Result<(Tensor, Tensor)> {
    let dev = weight.device();
    let weight_cpu = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let dims = weight_cpu.dims();
    validate_online_shape(dims, group_size, false)?;
    let out_features = dims[0];
    let in_features = dims[1];

    let w_data = weight_cpu.flatten_all()?.to_vec1::<f32>()?;
    let num_groups = in_features / group_size;

    let mut q_data = vec![0u8; out_features * in_features];
    let mut scales = vec![0.0f32; out_features * num_groups];

    for o in 0..out_features {
        for g in 0..num_groups {
            let mut abs_max = 0.0f32;
            for k in 0..group_size {
                let val = w_data[o * in_features + g * group_size + k];
                abs_max = abs_max.max(val.abs());
            }

            let scale = power_of_two_scale(abs_max, 448.0);
            scales[o * num_groups + g] = scale;

            for k in 0..group_size {
                let idx = o * in_features + g * group_size + k;
                let val = w_data[idx];
                q_data[idx] = float8::F8E4M3::from_f32(val / scale).to_bits();
            }
        }
    }

    let qweight = Tensor::from_vec(q_data, (out_features, in_features), dev)?;
    let scale_tensor = Tensor::from_vec(scales, (out_features, num_groups), dev)?;

    Ok((qweight, scale_tensor))
}

fn quantize_mxfp4_cpu(weight: &Tensor, group_size: usize) -> Result<(Tensor, Tensor)> {
    let dev = weight.device();
    let weight_cpu = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let dims = weight_cpu.dims();
    validate_online_shape(dims, group_size, true)?;
    let out_features = dims[0];
    let in_features = dims[1];

    let w_data = weight_cpu.flatten_all()?.to_vec1::<f32>()?;
    let num_groups = in_features / group_size;

    let mut q_data = vec![0u8; out_features * (in_features / 2)];
    let mut scales = vec![0.0f32; out_features * num_groups];

    for o in 0..out_features {
        for g in 0..num_groups {
            let mut abs_max = 0.0f32;
            for k in 0..group_size {
                let val = w_data[o * in_features + g * group_size + k];
                abs_max = abs_max.max(val.abs());
            }

            let scale = power_of_two_scale(abs_max, 6.0);
            scales[o * num_groups + g] = scale;

            for k in (0..group_size).step_by(2) {
                let i0 = g * group_size + k;
                let i1 = g * group_size + k + 1;

                let val0 = w_data[o * in_features + i0] / scale;
                let val1 = w_data[o * in_features + i1] / scale;

                let code0 = find_closest_fp4(val0);
                let code1 = find_closest_fp4(val1);

                let byte_idx = o * (in_features / 2) + i0 / 2;
                q_data[byte_idx] = code0 | (code1 << 4);
            }
        }
    }

    let qweight = Tensor::from_vec(q_data, (out_features, in_features / 2), dev)?;
    let scale_tensor = Tensor::from_vec(scales, (out_features, num_groups), dev)?;

    Ok((qweight, scale_tensor))
}

fn validate_online_shape(dims: &[usize], group_size: usize, packed: bool) -> Result<()> {
    if dims.len() != 2 || dims.contains(&0) {
        anyhow::bail!("MXFP source weight must be a non-empty rank-2 tensor, got {dims:?}");
    }
    let in_features = dims[1];
    if group_size == 0 || in_features % group_size != 0 || (packed && group_size % 2 != 0) {
        anyhow::bail!(
            "MXFP group size {group_size} must divide {in_features} input features and be even for 4-bit weights"
        );
    }
    Ok(())
}

fn find_closest_fp4(val: f32) -> u8 {
    let mut best_idx = 0;
    let mut min_diff = f32::MAX;
    for (idx, &fp4_val) in FP4_VALUES.iter().enumerate() {
        let diff = (val - fp4_val).abs();
        if diff < min_diff {
            min_diff = diff;
            best_idx = idx;
        }
    }
    best_idx as u8
}

// --- GPU CUDA forward implementation ---

#[cfg(feature = "cuda")]
impl MxfpLinear {
    fn apply_cuda(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let x_contig = x.to_dtype(DType::F16)?.reshape((batch, self.in_features))?.contiguous()?;

        let qweight_contig = self.qweight.contiguous()?;
        let scales_contig = self.scales.to_dtype(DType::F16)?.contiguous()?;

        let out = Tensor::zeros((batch, self.out_features), DType::F16, x.device())?;

        let p_x = get_cuda_ptr::<half::f16>(&x_contig)?;
        let p_qweight = get_cuda_ptr::<u8>(&qweight_contig)?;
        let p_scales = get_cuda_ptr::<half::f16>(&scales_contig)?;
        let p_out = get_cuda_ptr::<half::f16>(&out)?;

        let stream_ptr = match x.device() {
            candle_core::Device::Cuda(c) => c.cuda_stream().cu_stream() as usize,
            _ => 0,
        };

        unsafe {
            if self.bits == 8 {
                rllm_kernels::quant_matmul::mxfp8_matmul_w8a16_f16(
                    p_x as *const u16,
                    p_qweight as *const i8,
                    p_scales as *const u16,
                    p_out as *mut u16,
                    batch as i64,
                    self.out_features as i64,
                    self.in_features as i64,
                    self.group_size as i64,
                    stream_ptr,
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            } else {
                rllm_kernels::quant_matmul::mxfp4_matmul_w4a16_f16(
                    p_x as *const u16,
                    p_qweight,
                    p_scales as *const u16,
                    p_out as *mut u16,
                    batch as i64,
                    self.out_features as i64,
                    self.in_features as i64,
                    self.group_size as i64,
                    stream_ptr,
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }
        }

        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(self.out_features);
        out.reshape(out_shape)
            .map(|t| t.to_dtype(x.dtype()))
            .unwrap_or(Err(candle_core::Error::Msg("reshape failed".to_string())))
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_ptr<T: candle_core::cuda_backend::CudaDType>(
    t: &Tensor,
) -> candle_core::Result<*const T> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let (storage, _) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<T>()?;
            let stream = cuda_storage.device.cuda_stream();
            let (device_ptr, _) = slice.device_ptr(&stream);
            Ok(device_ptr as *const T)
        }
        _ => Err(candle_core::Error::Msg("Tensor is not on CUDA device".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxfp8_quantize_and_apply_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0,
                7.0, -8.0, 1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 1.0f32, -2.0, 3.0, -4.0,
                5.0, -6.0, 7.0, -8.0,
            ],
            (1, 32),
            &device,
        )?;

        let (qweight, scales) = quantize_mxfp8_cpu(&weight, 32)?;
        assert_eq!(qweight.dims(), &[1, 32]);
        assert_eq!(scales.dims(), &[1, 1]);
        let bytes = qweight.flatten_all()?.to_vec1::<u8>()?;
        let scale = scales.flatten_all()?.to_vec1::<f32>()?[0];
        let expected: f32 =
            bytes.iter().map(|&byte| float8::F8E4M3::from_bits(byte).to_f32() * scale).sum();

        let linear = MxfpLinear::new(qweight, scales, 8, 32)?;
        let x = Tensor::ones((1, 32), DType::F32, &device)?;
        let out = linear.apply(&x)?;
        assert_eq!(out.dims(), &[1, 1]);

        let out_val = out.flatten_all()?.to_vec1::<f32>()?[0];
        println!("test_mxfp8_quantize_and_apply_cpu out_val: {}", out_val);
        assert!((out_val - expected).abs() < 1e-4);
        assert!((out_val - (-16.0)).abs() < 1.0);
        Ok(())
    }

    #[test]
    fn mxfp8_online_weights_use_e4m3_bytes() -> Result<()> {
        let device = Device::Cpu;
        let mut values = vec![0.0f32; 32];
        values[0] = 448.0;
        values[1] = 1.0;
        values[2] = -1.0;
        let weight = Tensor::from_vec(values, (1, 32), &device)?;
        let (packed, scales) = quantize_mxfp8_cpu(&weight, 32)?;
        assert_eq!(scales.to_vec2::<f32>()?, vec![vec![1.0]]);
        let bytes = packed.to_vec2::<u8>()?;
        assert_eq!(bytes[0][1], float8::F8E4M3::from_f32(1.0).to_bits());
        assert_eq!(bytes[0][2], float8::F8E4M3::from_f32(-1.0).to_bits());
        Ok(())
    }

    #[test]
    fn cached_mxfp_weight_supports_different_input_dtypes() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((1, 32), DType::F32, &device)?;
        let (packed, scales) = quantize_mxfp8_cpu(&weight, 32)?;
        let linear = MxfpLinear::new(packed, scales, 8, 32)?;
        let x_f32 = Tensor::ones((1, 32), DType::F32, &device)?;
        let x_f16 = x_f32.to_dtype(DType::F16)?;
        let f32_result = linear.apply(&x_f32)?.to_vec2::<f32>()?[0][0];
        let f16_result = linear.apply(&x_f16)?.to_vec2::<half::f16>()?[0][0].to_f32();
        assert!((f32_result - f16_result).abs() < 0.1);
        Ok(())
    }

    #[test]
    fn decodes_e8m0_checkpoint_scales() -> Result<()> {
        let device = Device::Cpu;
        let encoded = Tensor::from_vec(vec![127u8, 128, 126], (1, 3), &device)?;
        let scales = decode_e8m0_scales(encoded)?;
        assert_eq!(scales.to_vec2::<f32>()?, vec![vec![1.0, 2.0, 0.5]]);
        let reserved = Tensor::from_vec(vec![255u8], (1, 1), &device)?;
        assert!(decode_e8m0_scales(reserved).is_err());
        Ok(())
    }

    #[test]
    fn rejects_partial_groups_and_wrong_scale_shape() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((1, 30), DType::F32, &device)?;
        assert!(quantize_mxfp8_cpu(&weight, 32).is_err());
        let packed = Tensor::zeros((1, 16), DType::U8, &device)?;
        let bad_scales = Tensor::ones((1, 2), DType::F32, &device)?;
        assert!(MxfpLinear::new(packed, bad_scales, 4, 32).is_err());
        Ok(())
    }

    #[test]
    fn test_mxfp4_quantize_and_apply_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 1.0, -2.0, 1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0,
                1.0, -2.0, 1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 1.0, -2.0, 1.0f32, -2.0, 3.0, -4.0,
                5.0, -6.0, 1.0, -2.0,
            ],
            (1, 32),
            &device,
        )?;

        let (qweight, scales) = quantize_mxfp4_cpu(&weight, 32)?;
        assert_eq!(qweight.dims(), &[1, 16]); // 32 packed 4-bit values = 16 bytes
        assert_eq!(scales.dims(), &[1, 1]);

        let linear = MxfpLinear::new(qweight, scales, 4, 32)?;
        let x = Tensor::ones((1, 32), DType::F32, &device)?;
        let out = linear.apply(&x)?;
        assert_eq!(out.dims(), &[1, 1]);

        let out_val = out.flatten_all()?.to_vec1::<f32>()?[0];
        println!("test_mxfp4_quantize_and_apply_cpu out_val: {}", out_val);
        assert!((out_val - (-20.0)).abs() < 1e-4);
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;

    #[test]
    fn mxfp_cuda_matches_cpu_for_multiple_rows_and_groups() -> Result<()> {
        let cpu = Device::Cpu;
        let gpu = Device::new_cuda(0)?;
        let weight_values: Vec<f32> =
            (0..3 * 64).map(|i| ((i as f32 * 0.37).sin() * (1 + i / 64) as f32) * 3.0).collect();
        let input_values: Vec<f32> = (0..4 * 64).map(|i| (i as f32 * 0.19).cos()).collect();
        let weight = Tensor::from_vec(weight_values, (3, 64), &cpu)?;
        let input = Tensor::from_vec(input_values, (4, 64), &cpu)?.to_dtype(DType::F16)?;

        for bits in [4, 8] {
            let (packed, scales) = if bits == 4 {
                quantize_mxfp4_cpu(&weight, 32)?
            } else {
                quantize_mxfp8_cpu(&weight, 32)?
            };
            let cpu_linear = MxfpLinear::new(packed.clone(), scales.clone(), bits, 32)?;
            let gpu_linear =
                MxfpLinear::new(packed.to_device(&gpu)?, scales.to_device(&gpu)?, bits, 32)?;
            let expected = cpu_linear.apply(&input)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let actual = gpu_linear
                .apply(&input.to_device(&gpu)?)?
                .to_device(&cpu)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            for (row, (expected_row, actual_row)) in expected.iter().zip(actual.iter()).enumerate()
            {
                for (column, (&expected, &actual)) in
                    expected_row.iter().zip(actual_row.iter()).enumerate()
                {
                    assert!(
                        (expected - actual).abs() <= 0.02 * expected.abs().max(1.0),
                        "MXFP{bits} mismatch at ({row}, {column}): CPU {expected}, CUDA {actual}"
                    );
                }
            }
        }
        Ok(())
    }
}
