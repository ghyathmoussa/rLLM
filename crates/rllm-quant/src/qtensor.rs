use candle_core::{DType, Device, Tensor};

#[derive(Clone)]
pub struct QuantTensor {
    data: Vec<i8>,
    shape: Vec<usize>,
    device: Device,
}

impl QuantTensor {
    pub fn new(data: Vec<i8>, shape: Vec<usize>, device: Device) -> anyhow::Result<Self> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            anyhow::bail!(
                "quant tensor data length {} does not match shape {:?} ({expected} elements)",
                data.len(),
                shape
            );
        }
        Ok(Self { data, shape, device })
    }

    pub fn from_i8_bytes(bytes: &[u8], shape: Vec<usize>, device: &Device) -> anyhow::Result<Self> {
        let data = bytes.iter().map(|byte| *byte as i8).collect::<Vec<_>>();
        Self::new(data, shape, device.clone())
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[i8] {
        &self.data
    }

    pub fn num_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn dequantize(&self, scale: &Tensor, out_dtype: DType) -> candle_core::Result<Tensor> {
        let q = self.data.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let q = Tensor::from_vec(q, self.shape.as_slice(), &self.device)?;
        let scale = scale.to_dtype(DType::F32)?;
        let scale = match scale.dims() {
            [out] if self.shape.len() == 2 && *out == self.shape[0] => scale.reshape((*out, 1))?,
            [out, 1] if self.shape.len() == 2 && *out == self.shape[0] => scale,
            dims => {
                return Err(candle_core::Error::Msg(format!(
                    "unsupported INT8 scale shape {dims:?} for weight shape {:?}",
                    self.shape
                )));
            }
        };
        q.broadcast_mul(&scale)?.to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantizes_i8_with_vector_scale() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let q = QuantTensor::new(vec![-2, -1, 0, 1, 2, 3], vec![2, 3], device.clone()).unwrap();
        let scale = Tensor::from_vec(vec![0.5f32, 0.25], (2,), &device)?;
        let deq = q.dequantize(&scale, DType::F32)?;
        let vals = deq.to_vec2::<f32>()?;
        assert_eq!(vals, vec![vec![-1.0, -0.5, 0.0], vec![0.25, 0.5, 0.75]]);
        Ok(())
    }

    #[test]
    fn dequantizes_i8_with_column_scale() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let q = QuantTensor::new(vec![1, 2, 3, 4], vec![2, 2], device.clone()).unwrap();
        let scale = Tensor::from_vec(vec![2.0f32, 4.0], (2, 1), &device)?;
        let vals = q.dequantize(&scale, DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(vals, vec![vec![2.0, 4.0], vec![12.0, 16.0]]);
        Ok(())
    }

    #[test]
    fn reads_signed_i8_from_raw_bytes() {
        let device = Device::Cpu;
        let q = QuantTensor::from_i8_bytes(&[0_u8, 255, 128, 127], vec![2, 2], &device).unwrap();
        assert_eq!(q.data, vec![0, -1, -128, 127]);
    }
}
