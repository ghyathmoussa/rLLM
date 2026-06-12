use anyhow::Result;
use candle_core::{DType, Tensor};
use rllm_core::dtype::DType as RllmDType;

use crate::{
    method::{LinearMethod, QuantMethodFactory, WeightSource},
    qtensor::QuantTensor,
    unquant::UnquantizedLinear,
};

pub struct Int8WeightOnlyFactory {
    ignore: Vec<String>,
    strict: bool,
}

impl Int8WeightOnlyFactory {
    pub fn new(ignore: Vec<String>, strict: bool) -> Self {
        Self { ignore, strict }
    }

    fn is_ignored(&self, prefix: &str) -> bool {
        self.ignore.iter().any(|name| prefix == name || prefix.ends_with(&format!(".{name}")))
    }
}

impl QuantMethodFactory for Int8WeightOnlyFactory {
    fn kv_cache_dtype(&self) -> RllmDType {
        RllmDType::INT8
    }

    fn build_linear(
        &self,
        prefix: &str,
        source: &mut WeightSource<'_>,
    ) -> Result<Box<dyn LinearMethod>> {
        if self.is_ignored(prefix) {
            let weight = source.remove_tensor(&format!("{prefix}.weight"))?;
            return Ok(Box::new(UnquantizedLinear::new(weight)));
        }

        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        if !source.has_quant_tensor(&weight_name) {
            if self.strict {
                anyhow::bail!(
                    "checkpoint quantization schema marks {weight_name} as INT8, but no raw I8 tensor was loaded"
                );
            }
            let weight = source.remove_tensor(&weight_name)?;
            let (qweight, scale) = quantize_weight_per_channel(&weight)?;
            return Ok(Box::new(Int8Linear::new(qweight, scale)?));
        }

        let qweight = source.remove_quant_tensor(&weight_name)?;
        let scale = source.remove_tensor(&scale_name)?;
        Ok(Box::new(Int8Linear::new(qweight, scale)?))
    }
}

pub struct Int8Linear {
    qweight: QuantTensor,
    scale: Tensor,
    #[cfg(feature = "cuda")]
    scale_values: Vec<f32>,
    /// Device-resident int8 weights + scale, uploaded once on the first CUDA
    /// forward and reused thereafter (vLLM keeps quantized weights resident as
    /// a device Parameter; only activations are quantized per forward).
    #[cfg(feature = "cuda")]
    device_weights: std::sync::OnceLock<DeviceInt8Weights>,
    in_features: usize,
    out_features: usize,
}

impl Int8Linear {
    pub fn new(qweight: QuantTensor, scale: Tensor) -> Result<Self> {
        let dims = qweight.shape();
        if dims.len() != 2 {
            anyhow::bail!("INT8 linear weight must be rank 2, got shape {dims:?}");
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let scale_values = scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if scale_values.len() != out_features {
            anyhow::bail!(
                "INT8 scale must have one value per output channel, got {} for {out_features} outputs",
                scale_values.len()
            );
        }
        Ok(Self {
            qweight,
            scale,
            #[cfg(feature = "cuda")]
            scale_values,
            #[cfg(feature = "cuda")]
            device_weights: std::sync::OnceLock::new(),
            in_features,
            out_features,
        })
    }
}

/// CPU reference for the W8A8 dynamic-activation INT8 matmul.
///
/// This mirrors, bit-for-bit in spirit, the CUDA kernel
/// `int8_matmul_w8a8_f16_kernel` (`rllm-kernels/src/cuda/quant_matmul.cu`):
/// for each input row it computes a dynamic per-row (per-token) activation
/// scale from the row's abs-max, quantizes activations to int8, accumulates
/// `int8 * int8 -> int32`, then dequantizes with `act_scale * weight_scale`.
///
/// It is intentionally **not** behind the `cuda` feature so the algorithm the
/// GPU kernel implements can be unit-tested on any host and used as the
/// correctness oracle for the GPU integration test. `qweight` is row-major
/// `[out_features, in_features]`; `weight_scale` is per-output-channel.
pub fn w8a8_matmul_reference_f32(
    x: &[f32],
    rows: usize,
    in_features: usize,
    qweight: &[i8],
    weight_scale: &[f32],
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        let x_row = &x[r * in_features..(r + 1) * in_features];
        let absmax = x_row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let act_scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        let inv_act_scale = 1.0 / act_scale;
        // Quantize the activation row once (the GPU kernel recomputes this per
        // output column; the result is identical, this is just the readable form).
        let qx: Vec<i32> =
            x_row.iter().map(|v| (v * inv_act_scale).round().clamp(-127.0, 127.0) as i32).collect();
        for o in 0..out_features {
            let w_row = &qweight[o * in_features..(o + 1) * in_features];
            let mut acc: i32 = 0;
            for k in 0..in_features {
                acc += qx[k] * w_row[k] as i32;
            }
            output[r * out_features + o] = acc as f32 * act_scale * weight_scale[o];
        }
    }
    output
}

fn quantize_weight_per_channel(weight: &Tensor) -> Result<(QuantTensor, Tensor)> {
    let dims = weight.dims();
    if dims.len() != 2 {
        anyhow::bail!("INT8 linear weight must be rank 2, got shape {dims:?}");
    }
    let out_features = dims[0];
    let in_features = dims[1];
    let rows = weight.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut qdata = Vec::with_capacity(out_features * in_features);
    let mut scales = Vec::with_capacity(out_features);

    for row in rows {
        let absmax = row.iter().fold(0.0f32, |max, value| max.max(value.abs()));
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        scales.push(scale);
        qdata.extend(
            row.into_iter().map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8),
        );
    }

    let qweight =
        QuantTensor::new(qdata, vec![out_features, in_features], weight.device().clone())?;
    let scale = Tensor::from_vec(scales, (out_features,), weight.device())?;
    Ok((qweight, scale))
}

impl LinearMethod for Int8Linear {
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), candle_core::Device::Cuda(_)) {
            return self.apply_cuda(x);
        }

        let weight = self.qweight.dequantize(&self.scale, x.dtype())?;
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::method::WeightSource;

    #[test]
    fn int8_linear_applies_dequantized_weight_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let qweight = QuantTensor::new(vec![2, -4, 8, 3], vec![2, 2], device.clone())?;
        let scale = Tensor::from_vec(vec![0.5f32, 0.25], (2,), &device)?;
        let linear = Int8Linear::new(qweight, scale)?;
        let x = Tensor::from_vec(vec![2.0f32, 3.0, -1.0, 4.0], (2, 2), &device)?;

        let out = linear.apply(&x)?.to_vec2::<f32>()?;

        assert_eq!(out, vec![vec![-4.0, 6.25], vec![-9.0, 1.0]]);
        Ok(())
    }

    #[test]
    fn non_strict_factory_quantizes_float_weights() -> Result<()> {
        let device = Device::Cpu;
        let mut weights = HashMap::from([(
            "linear.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, -2.0, 0.5, 0.25], (2, 2), &device)?,
        )]);
        let mut quantized = HashMap::new();
        let mut source = WeightSource::new(&mut weights, &mut quantized);
        let factory = Int8WeightOnlyFactory::new(Vec::new(), false);

        let linear = factory.build_linear("linear", &mut source)?;
        let x = Tensor::from_vec(vec![2.0f32, -1.0], (1, 2), &device)?;
        let out = linear.apply(&x)?.to_vec2::<f32>()?;

        assert!(linear.weight().is_none());
        assert!((out[0][0] - 4.0).abs() < 0.02);
        assert!((out[0][1] - 0.75).abs() < 0.02);
        Ok(())
    }

    #[test]
    fn w8a8_reference_matches_hand_computed() {
        // qweight [out=2, in=2], row-major; weight_scale per output channel.
        let qweight = vec![2i8, -4, 8, 3];
        let scale = vec![0.5f32, 0.25];
        // Single row whose abs-max is 4 -> act_scale = 4/127.
        let x = vec![2.0f32, 4.0];
        let out = w8a8_matmul_reference_f32(&x, 1, 2, &qweight, &scale, 2);

        // act_scale = 4/127; qx = round(x/act_scale) = round([2,4]*127/4) = [64, 127].
        let act = 4.0f32 / 127.0;
        let qx = [64i32, 127];
        let expect0 = (qx[0] * 2 + qx[1] * -4) as f32 * act * 0.5;
        let expect1 = (qx[0] * 8 + qx[1] * 3) as f32 * act * 0.25;
        assert!((out[0] - expect0).abs() < 1e-4, "{} vs {expect0}", out[0]);
        assert!((out[1] - expect1).abs() < 1e-4, "{} vs {expect1}", out[1]);
    }

    #[test]
    fn w8a8_reference_close_to_w8a16_dequant_path() -> Result<()> {
        // The W8A8 path (also quantizing activations) must stay close to the
        // W8A16 dequant-then-matmul path used on CPU in Phase 1.
        let device = Device::Cpu;
        let qweight =
            QuantTensor::new(vec![10, -20, 30, -40, 50, -60], vec![2, 3], device.clone())?;
        let scale = Tensor::from_vec(vec![0.1f32, 0.2], (2,), &device)?;
        let linear = Int8Linear::new(qweight, scale)?;

        let x_vals = vec![0.7f32, -1.3, 2.1];
        let x = Tensor::from_vec(x_vals.clone(), (1, 3), &device)?;
        let w8a16 = linear.apply(&x)?.to_vec2::<f32>()?;

        let scale_vals = vec![0.1f32, 0.2];
        let qw = vec![10i8, -20, 30, -40, 50, -60];
        let w8a8 = w8a8_matmul_reference_f32(&x_vals, 1, 3, &qw, &scale_vals, 2);

        // Activation quantization adds at most ~1/127 relative error per element.
        for (a, b) in w8a8.iter().zip(w8a16[0].iter()) {
            assert!((a - b).abs() < 0.05 * b.abs().max(1.0), "w8a8 {a} vs w8a16 {b}");
        }
        Ok(())
    }

    #[test]
    fn validates_scale_length_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let qweight = QuantTensor::new(vec![1, 2, 3, 4], vec![2, 2], device.clone())?;
        let scale = Tensor::from_vec(vec![0.5f32], (1,), &device)?;

        let err = match Int8Linear::new(qweight, scale) {
            Ok(_) => anyhow::bail!("expected invalid INT8 scale length to fail"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("INT8 scale must have one value per output channel"));
        Ok(())
    }
}

/// Device-resident int8 weights + per-channel scale.
///
/// Uploaded host→device exactly once (on the first CUDA forward) and shared by
/// every subsequent forward via `Arc`. This is what makes the int8 path a real
/// GPU memory saving: the quantized weights live on the device as a persistent
/// buffer, mirroring vLLM's resident weight Parameter. Only activations are
/// (re)quantized per forward, inside the kernel.
#[cfg(feature = "cuda")]
struct DeviceInt8Weights {
    qweight: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaSlice<i8>>,
    scale: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl Int8Linear {
    /// Return the device-resident weights, uploading them once on first use.
    fn device_weights_for(
        &self,
        device: &candle_core::Device,
    ) -> candle_core::Result<&DeviceInt8Weights> {
        if let Some(weights) = self.device_weights.get() {
            return Ok(weights);
        }
        let cuda = match device {
            candle_core::Device::Cuda(cuda) => cuda,
            _ => {
                return Err(candle_core::Error::Msg(
                    "int8 device weights requested on a non-CUDA device".to_string(),
                ));
            }
        };
        let qweight = cuda.clone_htod(self.qweight.data())?;
        let scale = cuda.clone_htod(self.scale_values.as_slice())?;
        let weights = DeviceInt8Weights {
            qweight: std::sync::Arc::new(qweight),
            scale: std::sync::Arc::new(scale),
        };
        // If another thread won the upload race, keep the already-stored copy.
        let _ = self.device_weights.set(weights);
        Ok(self.device_weights.get().expect("device weights set above"))
    }

    fn apply_cuda(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let original_dtype = x.dtype();
        let x_2d = x.to_dtype(DType::F16)?.reshape((batch, self.in_features))?.contiguous()?;

        let weights = self.device_weights_for(x.device())?;
        let op = Int8MatmulOp {
            qweight: weights.qweight.clone(),
            scale: weights.scale.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
        };
        let out = x_2d.apply_op1_no_bwd(&op)?;
        let out = out.to_dtype(original_dtype)?;
        let mut out_shape = x_shape[..trailing].to_vec();
        out_shape.push(self.out_features);
        out.reshape(out_shape)
    }
}

/// Custom op holding `Arc` handles to the resident device weights (no upload).
#[cfg(feature = "cuda")]
struct Int8MatmulOp {
    qweight: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaSlice<i8>>,
    scale: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    in_features: usize,
    out_features: usize,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for Int8MatmulOp {
    fn name(&self) -> &'static str {
        "rllm-int8-matmul-w8a8"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // The op holds device-resident weights only; on CPU use the
        // feature-independent `w8a8_matmul_reference_f32` oracle instead.
        Err(candle_core::Error::Msg(
            "rllm-int8-matmul-w8a8 runs only on CUDA; use w8a8_matmul_reference_f32 on CPU"
                .to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        if !layout.is_contiguous() {
            return Err(candle_core::Error::Msg(format!(
                "INT8 CUDA matmul input must be contiguous, got layout {layout:?}"
            )));
        }
        let shape = layout.shape();
        let (rows, in_features) = shape.dims2()?;
        if in_features != self.in_features {
            return Err(candle_core::Error::Msg(format!(
                "INT8 CUDA matmul input has {in_features} features, expected {}",
                self.in_features
            )));
        }

        let device = storage.device.clone();
        let stream = device.cuda_stream();
        let input = storage.as_cuda_slice::<half::f16>()?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + shape.elem_count());
        let mut output = unsafe { device.alloc::<half::f16>(rows * self.out_features)? };

        // Named bindings so the `&CudaSlice` outlives the `device_ptr` guards.
        let qweight = self.qweight.as_ref();
        let scale = self.scale.as_ref();

        {
            let (input_ptr, _input_guard) = input.device_ptr(&stream);
            // Resident weights — reused every forward, never re-uploaded.
            let (qweight_ptr, _qweight_guard) = qweight.device_ptr(&stream);
            let (scale_ptr, _scale_guard) = scale.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);

            unsafe {
                rllm_kernels::quant_matmul::int8_matmul_w8a8_f16(
                    input_ptr as *const u16,
                    qweight_ptr as *const i8,
                    scale_ptr as *const f32,
                    output_ptr as *mut u16,
                    rows as i64,
                    self.out_features as i64,
                    self.in_features as i64,
                    stream.cu_stream() as usize,
                )
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            }
        }

        let storage = candle_core::CudaStorage::wrap_cuda_slice(output, device);
        Ok((storage, (rows, self.out_features).into()))
    }
}

// GPU integration test — compiled only with `--features cuda` (needs nvcc), and
// runs only when a CUDA device is actually present, otherwise it skips. Validates
// that the resident-weight CUDA W8A8 path matches the CPU oracle within f16
// tolerance. Run on the RTX box: `cargo test -p rllm-quant --features cuda`.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn cuda_w8a8_matches_cpu_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(_) => return Ok(()), // no GPU on this host; skip.
        };

        let (rows, in_features, out_features) = (4usize, 16usize, 8usize);
        let qdata: Vec<i8> =
            (0..out_features * in_features).map(|i| ((i as i32 % 17) - 8) as i8).collect();
        let scale_vals: Vec<f32> = (0..out_features).map(|i| 0.01 + i as f32 * 0.005).collect();

        let qweight =
            QuantTensor::new(qdata.clone(), vec![out_features, in_features], device.clone())?;
        let scale = Tensor::from_vec(scale_vals.clone(), (out_features,), &device)?;
        let linear = Int8Linear::new(qweight, scale)?;

        let x_vals: Vec<f32> = (0..rows * in_features).map(|i| (i as f32 * 0.03).sin()).collect();
        let x = Tensor::from_vec(x_vals.clone(), (rows, in_features), &device)?;

        // Two forwards: also exercises the upload-once cache (second call must
        // reuse the resident buffer, not re-upload).
        let _ = linear.apply(&x)?;
        let got = linear.apply(&x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let expect = w8a8_matmul_reference_f32(
            &x_vals,
            rows,
            in_features,
            &qdata,
            &scale_vals,
            out_features,
        );

        for r in 0..rows {
            for o in 0..out_features {
                let a = got[r][o];
                let b = expect[r * out_features + o];
                assert!((a - b).abs() < 0.05 * b.abs().max(1.0), "cuda {a} vs ref {b}");
            }
        }
        Ok(())
    }
}
