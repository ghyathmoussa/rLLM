use anyhow::Result;
use candle_core::DType;
use candle_core::Tensor;
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
            in_features,
            out_features,
        })
    }
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

#[cfg(feature = "cuda")]
impl Int8Linear {
    fn apply_cuda(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let original_dtype = x.dtype();
        let x_2d = x.to_dtype(DType::F16)?.reshape((batch, self.in_features))?.contiguous()?;
        let op = Int8MatmulOp {
            qweight: self.qweight.data().to_vec(),
            scale: self.scale_values.clone(),
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

#[cfg(feature = "cuda")]
struct Int8MatmulOp {
    qweight: Vec<i8>,
    scale: Vec<f32>,
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
        Err(candle_core::Error::Msg(
            "INT8 CUDA matmul custom op called with CPU storage".to_string(),
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
        let qweight = device.clone_htod(&self.qweight)?;
        let scale = device.clone_htod(&self.scale)?;
        let mut output = unsafe { device.alloc::<half::f16>(rows * self.out_features)? };

        {
            let (input_ptr, _input_guard) = input.device_ptr(&stream);
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
