use std::collections::HashMap;

use anyhow::{Result, bail};
use candle_core::Tensor;
use rllm_core::{
    config::{QuantizationConfig, QuantizationKind},
    dtype::DType as RllmDType,
};

use crate::{
    int8::Int8WeightOnlyFactory, qtensor::QuantTensor, schema::QuantSchema,
    unquant::UnquantizedFactory,
};

pub trait LinearMethod: Send + Sync {
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor>;
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;

    fn weight(&self) -> Option<&Tensor> {
        None
    }

    fn is_quantized(&self) -> bool {
        false
    }
}

pub trait QuantMethodFactory: Send + Sync {
    fn build_linear(
        &self,
        prefix: &str,
        source: &mut WeightSource<'_>,
    ) -> Result<Box<dyn LinearMethod>>;

    fn kv_cache_dtype(&self) -> RllmDType {
        RllmDType::F16
    }
}

pub struct WeightSource<'a> {
    pub weights: &'a mut HashMap<String, Tensor>,
    pub quantized: &'a mut HashMap<String, QuantTensor>,
    pub gguf_weights:
        Option<&'a mut HashMap<String, std::sync::Arc<candle_core::quantized::QTensor>>>,
}

impl<'a> WeightSource<'a> {
    pub fn new(
        weights: &'a mut HashMap<String, Tensor>,
        quantized: &'a mut HashMap<String, QuantTensor>,
    ) -> Self {
        Self { weights, quantized, gguf_weights: None }
    }

    pub fn with_gguf(
        mut self,
        gguf_weights: &'a mut HashMap<String, std::sync::Arc<candle_core::quantized::QTensor>>,
    ) -> Self {
        self.gguf_weights = Some(gguf_weights);
        self
    }

    pub fn remove_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.weights.remove(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))
    }

    pub fn remove_quant_tensor(&mut self, name: &str) -> Result<QuantTensor> {
        self.quantized.remove(name).ok_or_else(|| anyhow::anyhow!("missing quantized {name}"))
    }

    pub fn has_quant_tensor(&self, name: &str) -> bool {
        self.quantized.contains_key(name)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    pub fn remove_gguf_tensor(
        &mut self,
        name: &str,
    ) -> Result<std::sync::Arc<candle_core::quantized::QTensor>> {
        self.gguf_weights
            .as_mut()
            .and_then(|w| w.remove(name))
            .ok_or_else(|| anyhow::anyhow!("missing GGUF tensor {name}"))
    }

    pub fn has_gguf_tensor(&self, name: &str) -> bool {
        self.gguf_weights.as_ref().is_some_and(|w| w.contains_key(name))
    }
}

pub fn factory_from_config(
    config: Option<&QuantizationConfig>,
    checkpoint_schema: Option<&QuantSchema>,
) -> Result<Box<dyn QuantMethodFactory>> {
    if let Some(schema) = checkpoint_schema {
        if schema.is_int8_weight_only() {
            let symmetric = schema.weight_symmetric.unwrap_or(true);
            let strategy = schema.weight_strategy.clone().unwrap_or_else(|| "channel".to_string());
            return Ok(Box::new(Int8WeightOnlyFactory::new(
                schema.ignore.clone(),
                true,
                symmetric,
                strategy,
            )));
        }
        if let Some(strategy) = schema.unsupported_int8_strategy() {
            bail!(
                "unsupported INT8 checkpoint weight strategy {strategy:?}; only per-channel and per-tensor INT8 weights are supported"
            );
        }
    }

    let Some(config) = config else {
        return Ok(Box::new(UnquantizedFactory));
    };

    match config.kind {
        QuantizationKind::None | QuantizationKind::GPTQ | QuantizationKind::AWQ => {
            Ok(Box::new(UnquantizedFactory))
        }
        QuantizationKind::Int8 | QuantizationKind::CompressedTensors => {
            Ok(Box::new(Int8WeightOnlyFactory::new(Vec::new(), false, true, "channel".to_string())))
        }
        QuantizationKind::MXFP8 | QuantizationKind::MXFP4 => {
            let bits = if config.kind == QuantizationKind::MXFP8 { 8 } else { 4 };
            let group_size = config.group_size.unwrap_or(32);
            Ok(Box::new(crate::mxfp::MxfpWeightOnlyFactory::new(bits, group_size)))
        }
        QuantizationKind::NVFP4 => {
            let group_size = config.group_size.unwrap_or(16);
            Ok(Box::new(crate::nvfp::NvfpWeightOnlyFactory::new(group_size)))
        }
        QuantizationKind::Gguf => Ok(Box::new(crate::gguf::GgufMethodFactory)),
        other => bail!("quantization kind {other:?} is not implemented by rllm-quant yet"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::method::WeightSource;

    #[test]
    fn schema_factory_uses_unquantized_for_ignored_linear() -> Result<()> {
        let schema = QuantSchema {
            quant_method: Some("compressed-tensors".into()),
            format: Some("int-quantized".into()),
            weight_num_bits: Some(8),
            weight_strategy: Some("channel".into()),
            weight_symmetric: Some(true),
            ignore: vec!["lm_head".into()],
        };
        let factory = factory_from_config(None, Some(&schema))?;
        let device = Device::Cpu;
        let mut weights = HashMap::from([(
            "lm_head.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &device)?,
        )]);
        let mut quantized = HashMap::new();
        let mut source = WeightSource::new(&mut weights, &mut quantized);

        let linear = factory.build_linear("lm_head", &mut source)?;

        assert!(linear.weight().is_some());
        assert_eq!(linear.in_features(), 2);
        assert_eq!(linear.out_features(), 2);
        Ok(())
    }

    #[test]
    fn schema_factory_accepts_tensor_strategy_int8() -> Result<()> {
        let schema = QuantSchema {
            quant_method: Some("compressed-tensors".into()),
            format: Some("int-quantized".into()),
            weight_num_bits: Some(8),
            weight_strategy: Some("tensor".into()),
            weight_symmetric: Some(true),
            ignore: Vec::new(),
        };

        let factory = factory_from_config(None, Some(&schema))?;
        assert!(factory.kv_cache_dtype() == rllm_core::dtype::DType::INT8);
        Ok(())
    }
}
