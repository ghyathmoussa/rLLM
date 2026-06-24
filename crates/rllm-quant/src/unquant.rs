use anyhow::Result;
use candle_core::Tensor;

use crate::method::{LinearMethod, QuantMethodFactory, WeightSource};

pub struct UnquantizedFactory;

impl QuantMethodFactory for UnquantizedFactory {
    fn build_linear(
        &self,
        prefix: &str,
        source: &mut WeightSource<'_>,
    ) -> Result<Box<dyn LinearMethod>> {
        let weight = source.remove_tensor(&format!("{prefix}.weight"))?;
        Ok(Box::new(UnquantizedLinear::new(weight)))
    }
}

pub struct UnquantizedLinear {
    weight: Tensor,
    in_features: usize,
    out_features: usize,
}

impl UnquantizedLinear {
    pub fn new(weight: Tensor) -> Self {
        let dims = weight.dims();
        let out_features = dims[dims.len() - 2];
        let in_features = dims[dims.len() - 1];
        Self { weight, in_features, out_features }
    }
}

impl LinearMethod for UnquantizedLinear {
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_shape = x.dims();
        let trailing = x_shape.len().saturating_sub(1);
        let batch: usize = x_shape[..trailing].iter().product();
        let x_2d = x.reshape((batch, self.in_features))?;
        let out = x_2d.matmul(&self.weight.t()?)?;
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
        Some(&self.weight)
    }
}
