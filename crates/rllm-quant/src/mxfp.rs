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
                let scales = source.remove_tensor(&scale_name)?;
                return Ok(Box::new(MxfpLinear::new(weight, scales, self.bits, self.group_size)?));
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
            let scales = source.remove_tensor(&scale_name)?;
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
        let out_features = dims[0];
        let in_features = if bits == 4 { dims[1] * 2 } else { dims[1] };

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

    fn get_or_dequantize(&self, target_device: &Device, target_dtype: DType) -> Result<&Tensor> {
        if let Some(dequant) = self.dequantized.get() {
            return Ok(dequant);
        }

        let dequant = if self.bits == 8 {
            self.dequantize_mxfp8_cpu()?
        } else {
            self.dequantize_mxfp4_cpu()?
        };

        let dequant_dev = dequant.to_device(target_device)?.to_dtype(target_dtype)?;
        let _ = self.dequantized.set(dequant_dev);
        Ok(self.dequantized.get().unwrap())
    }

    fn dequantize_mxfp8_cpu(&self) -> Result<Tensor> {
        let q_u8 = self.qweight.to_device(&Device::Cpu)?;
        let q_data_u8 = q_u8.flatten_all()?.to_vec1::<u8>()?;
        let q_data: Vec<f32> = q_data_u8.iter().map(|&x| (x as i8) as f32).collect();

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
    let out_features = dims[0];
    let in_features = dims[1];

    let w_data = weight_cpu.flatten_all()?.to_vec1::<f32>()?;
    let num_groups = in_features / group_size;

    let mut q_data = vec![0i8; out_features * in_features];
    let mut scales = vec![0.0f32; out_features * num_groups];

    for o in 0..out_features {
        for g in 0..num_groups {
            let mut abs_max = 0.0f32;
            for k in 0..group_size {
                let val = w_data[o * in_features + g * group_size + k];
                abs_max = abs_max.max(val.abs());
            }

            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            scales[o * num_groups + g] = scale;

            for k in 0..group_size {
                let idx = o * in_features + g * group_size + k;
                let val = w_data[idx];
                q_data[idx] = (val / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    let q_u8 = unsafe { std::mem::transmute::<Vec<i8>, Vec<u8>>(q_data) };
    let qweight = Tensor::from_vec(q_u8, (out_features, in_features), dev)?;
    let scale_tensor = Tensor::from_vec(scales, (out_features, num_groups), dev)?;

    Ok((qweight, scale_tensor))
}

fn quantize_mxfp4_cpu(weight: &Tensor, group_size: usize) -> Result<(Tensor, Tensor)> {
    let dev = weight.device();
    let weight_cpu = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let dims = weight_cpu.dims();
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

            let scale = if abs_max > 0.0 { abs_max / 6.0 } else { 1.0 };
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

        let linear = MxfpLinear::new(qweight, scales, 8, 32)?;
        let x = Tensor::ones((1, 32), DType::F32, &device)?;
        let out = linear.apply(&x)?;
        assert_eq!(out.dims(), &[1, 1]);

        let out_val = out.flatten_all()?.to_vec1::<f32>()?[0];
        println!("test_mxfp8_quantize_and_apply_cpu out_val: {}", out_val);
        assert!((out_val - (-16.0)).abs() < 0.2);
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
