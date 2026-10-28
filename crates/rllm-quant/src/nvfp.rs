use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::method::{LinearMethod, QuantMethodFactory, WeightSource};

const FP4_VALUES: [f32; 16] =
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

pub struct NvfpWeightOnlyFactory {
    group_size: usize,
}

impl NvfpWeightOnlyFactory {
    pub fn new(group_size: usize) -> Self {
        Self { group_size }
    }
}

impl QuantMethodFactory for NvfpWeightOnlyFactory {
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
                return Ok(Box::new(NvfpLinear::new(weight, scales, self.group_size)?));
            } else {
                // Perform online quantization if no scale is found
                let (qweight, scales) = quantize_nvfp4_cpu(&weight, self.group_size)?;
                return Ok(Box::new(NvfpLinear::new(qweight, scales, self.group_size)?));
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
            return Ok(Box::new(NvfpLinear::new(qweight_tensor, scales, self.group_size)?));
        }

        anyhow::bail!("missing weight tensor for {prefix}")
    }
}

pub struct NvfpLinear {
    qweight: Tensor,
    scales: Tensor,
    group_size: usize,
    in_features: usize,
    out_features: usize,
    dequantized: std::sync::OnceLock<Tensor>,
}

impl NvfpLinear {
    pub fn new(qweight: Tensor, scales: Tensor, group_size: usize) -> Result<Self> {
        let dims = qweight.dims();
        if dims.len() != 2 || dims.contains(&0) {
            anyhow::bail!("NVFP4 weight must be a non-empty rank-2 tensor, got {dims:?}");
        }
        if qweight.dtype() != DType::U8 {
            anyhow::bail!("NVFP4 packed weight must have U8 dtype, got {:?}", qweight.dtype());
        }
        let out_features = dims[0];
        let in_features = dims[1] * 2;
        validate_group_size(in_features, group_size)?;
        if scales.dims() != [out_features, in_features / group_size] {
            anyhow::bail!(
                "NVFP4 scales must have shape [{out_features}, {}], got {:?}",
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
            anyhow::bail!("NVFP4 scales must be finite and positive");
        }

        Ok(Self {
            qweight,
            scales,
            group_size,
            in_features,
            out_features,
            dequantized: std::sync::OnceLock::new(),
        })
    }

    fn get_or_dequantize(&self, target_device: &Device, target_dtype: DType) -> Result<Tensor> {
        if self.dequantized.get().is_none() {
            let _ = self.dequantized.set(self.dequantize_nvfp4_cpu()?);
        }
        self.dequantized
            .get()
            .expect("dequantized weight stored above")
            .to_device(target_device)?
            .to_dtype(target_dtype)
            .map_err(Into::into)
    }

    fn dequantize_nvfp4_cpu(&self) -> Result<Tensor> {
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

impl LinearMethod for NvfpLinear {
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

    fn weight(&self) -> Option<&Tensor> {
        self.dequantized.get()
    }

    fn is_quantized(&self) -> bool {
        true
    }
}

// --- Online Quantization Helpers ---

fn quantize_nvfp4_cpu(weight: &Tensor, group_size: usize) -> Result<(Tensor, Tensor)> {
    let dev = weight.device();
    let weight_cpu = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let dims = weight_cpu.dims();
    if dims.len() != 2 || dims.contains(&0) {
        anyhow::bail!("NVFP4 source weight must be a non-empty rank-2 tensor, got {dims:?}");
    }
    let out_features = dims[0];
    let in_features = dims[1];
    validate_group_size(in_features, group_size)?;

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

fn validate_group_size(in_features: usize, group_size: usize) -> Result<()> {
    if group_size != 16 || in_features % group_size != 0 {
        anyhow::bail!(
            "NVFP4 requires group size 16, which must divide {in_features} input features; got {group_size}"
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
impl NvfpLinear {
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
            rllm_kernels::quant_matmul::nvfp4_matmul_w4a16_f16(
                p_x as *const u16,
                p_qweight,
                p_scales as *const u16,
                p_out as *mut u16,
                batch as i64,
                self.out_features as i64,
                self.in_features as i64,
                stream_ptr,
            )
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
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
    fn test_nvfp4_quantize_and_apply_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 1.0, -2.0, 1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0,
                1.0, -2.0,
            ],
            (1, 16),
            &device,
        )?;

        let (qweight, scales) = quantize_nvfp4_cpu(&weight, 16)?;
        assert_eq!(qweight.dims(), &[1, 8]); // 16 packed 4-bit values = 8 bytes
        assert_eq!(scales.dims(), &[1, 1]);

        let linear = NvfpLinear::new(qweight, scales, 16)?;
        let x = Tensor::ones((1, 16), DType::F32, &device)?;
        let out = linear.apply(&x)?;
        assert_eq!(out.dims(), &[1, 1]);

        let out_val = out.flatten_all()?.to_vec1::<f32>()?[0];
        println!("test_nvfp4_quantize_and_apply_cpu out_val: {}", out_val);
        assert!((out_val - (-10.0)).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn rejects_partial_groups_and_wrong_scale_shape() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((1, 18), DType::F32, &device)?;
        assert!(quantize_nvfp4_cpu(&weight, 16).is_err());
        let packed = Tensor::zeros((1, 8), DType::U8, &device)?;
        let bad_scales = Tensor::ones((1, 2), DType::F32, &device)?;
        assert!(NvfpLinear::new(packed, bad_scales, 16).is_err());
        Ok(())
    }

    #[test]
    fn cached_nvfp4_weight_supports_different_input_dtypes() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((1, 16), DType::F32, &device)?;
        let (packed, scales) = quantize_nvfp4_cpu(&weight, 16)?;
        let linear = NvfpLinear::new(packed, scales, 16)?;
        let x_f32 = Tensor::ones((1, 16), DType::F32, &device)?;
        let x_f16 = x_f32.to_dtype(DType::F16)?;
        let f32_result = linear.apply(&x_f32)?.to_vec2::<f32>()?[0][0];
        let f16_result = linear.apply(&x_f16)?.to_vec2::<half::f16>()?[0][0].to_f32();
        assert!((f32_result - f16_result).abs() < 0.1);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_nvfp4_quantize_and_apply_gpu() -> Result<()> {
        let cpu_device = Device::Cpu;
        let gpu_device = Device::new_cuda(0)?;

        let weight_data = vec![
            1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 1.0, -2.0, 1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 1.0,
            -2.0,
        ];
        let weight = Tensor::from_vec(weight_data, (1, 16), &cpu_device)?;

        let (qweight, scales) = quantize_nvfp4_cpu(&weight, 16)?;

        // Run on CPU
        let linear_cpu = NvfpLinear::new(qweight.clone(), scales.clone(), 16)?;
        let x_cpu = Tensor::ones((1, 16), DType::F16, &cpu_device)?;
        let out_cpu = linear_cpu.apply(&x_cpu)?;
        let out_cpu_val = out_cpu.flatten_all()?.to_vec1::<half::f16>()?[0].to_f32();

        // Run on GPU
        let qweight_gpu = qweight.to_device(&gpu_device)?;
        let scales_gpu = scales.to_device(&gpu_device)?;
        let linear_gpu = NvfpLinear::new(qweight_gpu, scales_gpu, 16)?;
        let x_gpu = Tensor::ones((1, 16), DType::F16, &gpu_device)?;
        let out_gpu = linear_gpu.apply(&x_gpu)?;
        let out_gpu_val =
            out_gpu.to_device(&cpu_device)?.flatten_all()?.to_vec1::<half::f16>()?[0].to_f32();

        println!("GPU test - CPU out: {}, GPU out: {}", out_cpu_val, out_gpu_val);
        assert!((out_cpu_val - out_gpu_val).abs() < 1e-3);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_cuda_matches_cpu_for_multiple_rows_and_groups() -> Result<()> {
        let cpu = Device::Cpu;
        let gpu = Device::new_cuda(0)?;
        let weight_values: Vec<f32> =
            (0..3 * 64).map(|i| (i as f32 * 0.31).sin() * (1 + i / 64) as f32).collect();
        let input_values: Vec<f32> = (0..4 * 64).map(|i| (i as f32 * 0.13).cos()).collect();
        let weight = Tensor::from_vec(weight_values, (3, 64), &cpu)?;
        let input = Tensor::from_vec(input_values, (4, 64), &cpu)?.to_dtype(DType::F16)?;
        let (packed, scales) = quantize_nvfp4_cpu(&weight, 16)?;
        let cpu_linear = NvfpLinear::new(packed.clone(), scales.clone(), 16)?;
        let gpu_linear = NvfpLinear::new(packed.to_device(&gpu)?, scales.to_device(&gpu)?, 16)?;
        let expected = cpu_linear.apply(&input)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let actual = gpu_linear
            .apply(&input.to_device(&gpu)?)?
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        for (row, (expected_row, actual_row)) in expected.iter().zip(actual.iter()).enumerate() {
            for (column, (&expected, &actual)) in
                expected_row.iter().zip(actual_row.iter()).enumerate()
            {
                assert!(
                    (expected - actual).abs() <= 0.02 * expected.abs().max(1.0),
                    "NVFP4 mismatch at ({row}, {column}): CPU {expected}, CUDA {actual}"
                );
            }
        }
        Ok(())
    }
}
