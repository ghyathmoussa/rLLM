use anyhow::Result;
use candle_core::Tensor;

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
            return Ok(Box::new(UnquantizedLinear::new(weight)));
        }

        let qweight = source.remove_quant_tensor(&weight_name)?;
        let scale = source.remove_tensor(&scale_name)?;
        Ok(Box::new(Int8Linear::new(qweight, scale)?))
    }
}

pub struct Int8Linear {
    qweight: QuantTensor,
    scale: Tensor,
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
        Ok(Self { qweight, scale, in_features, out_features })
    }
}

impl LinearMethod for Int8Linear {
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
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
