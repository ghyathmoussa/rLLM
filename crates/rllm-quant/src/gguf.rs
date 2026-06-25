use crate::method::{LinearMethod, QuantMethodFactory, WeightSource};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_core::quantized::QMatMul;

pub struct GgufMethodFactory;

impl QuantMethodFactory for GgufMethodFactory {
    fn build_linear(
        &self,
        prefix: &str,
        source: &mut WeightSource<'_>,
    ) -> Result<Box<dyn LinearMethod>> {
        let name = format!("{prefix}.weight");
        let qtensor = source
            .remove_gguf_tensor(&name)
            .with_context(|| format!("loading GGUF layer {name}"))?;

        let shape = qtensor.shape();
        let out_features = shape.dims()[0];
        let in_features = shape.dims()[1];

        let qmatmul = QMatMul::from_arc(qtensor)
            .map_err(|e| anyhow::anyhow!("creating QMatMul for {prefix}: {e}"))?;

        Ok(Box::new(GgufLinear { qmatmul, in_features, out_features }))
    }
}

pub struct GgufLinear {
    qmatmul: QMatMul,
    in_features: usize,
    out_features: usize,
}

impl LinearMethod for GgufLinear {
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        use candle_core::Module;
        self.qmatmul.forward(x)
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
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_gguf_linear_construction_and_apply() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0f32, (16, 32), &device)?;
        let qtensor = QTensor::quantize(&weight, GgmlDType::Q4_0)
            .map_err(|e| anyhow::anyhow!("quantize error: {e}"))?;

        let mut weights = HashMap::new();
        let mut quantized = HashMap::new();
        let mut gguf_weights = HashMap::new();
        gguf_weights
            .insert("model.layers.0.self_attn.q_proj.weight".to_string(), Arc::new(qtensor));

        let mut source =
            WeightSource::new(&mut weights, &mut quantized).with_gguf(&mut gguf_weights);

        let factory = GgufMethodFactory;
        let linear = factory.build_linear("model.layers.0.self_attn.q_proj", &mut source)?;

        assert_eq!(linear.in_features(), 32);
        assert_eq!(linear.out_features(), 16);
        assert!(linear.is_quantized());

        let x = Tensor::randn(0.0f32, 1.0f32, (2, 32), &device)?;
        let out = linear.apply(&x).map_err(|e| anyhow::anyhow!("apply error: {e}"))?;
        assert_eq!(out.dims(), &[2, 16]);

        Ok(())
    }
}
