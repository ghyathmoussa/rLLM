use serde::{Deserialize, Serialize};

use crate::dtype::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizedWeightFormat {
    Fp8KvCache,
    Gptq,
    Awq,
    Gguf,
    BitsAndBytes,
    Mxfp8,
    Mxfp4,
    Nvfp4,
    Int8,
    Int4,
    CompressedTensors,
    ModelOpt,
    TorchAo,
    Unquantized,
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub format: QuantizedWeightFormat,
    pub bits: Option<u8>,
    pub group_size: Option<usize>,
    pub activation_dtype: DType,
    pub kv_cache_dtype: DType,
}

impl Default for QuantizationPlan {
    fn default() -> Self {
        Self {
            format: QuantizedWeightFormat::Unquantized,
            bits: None,
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        }
    }
}

impl QuantizationPlan {
    pub fn from_config(config: &crate::config::QuantizationConfig) -> Result<Self, String> {
        use crate::config::QuantizationKind;
        let plan = match config.kind {
            QuantizationKind::None => Self::default(),
            QuantizationKind::FP8 => Self::fp8_kv_cache(),
            QuantizationKind::GPTQ => {
                let mut p = Self::int4();
                p.format = QuantizedWeightFormat::Gptq;
                p.group_size = config.group_size.or(p.group_size);
                if let Some(b) = config.bits {
                    p.bits = Some(b as u8);
                }
                p
            }
            QuantizationKind::AWQ => {
                let mut p = Self::int4();
                p.format = QuantizedWeightFormat::Awq;
                p.group_size = config.group_size.or(p.group_size);
                if let Some(b) = config.bits {
                    p.bits = Some(b as u8);
                }
                p
            }
            QuantizationKind::MXFP8 => Self::mxfp8(),
            QuantizationKind::MXFP4 => Self::mxfp4(),
            QuantizationKind::NVFP4 => Self::nvfp4(),
            QuantizationKind::Int8 => {
                let mut p = Self::int8();
                p.group_size = config.group_size;
                p
            }
            QuantizationKind::Int4 => {
                let mut p = Self::int4();
                p.group_size = config.group_size.or(p.group_size);
                p
            }
            QuantizationKind::Gguf => {
                let mut p = Self::default();
                p.format = QuantizedWeightFormat::Gguf;
                p
            }
            QuantizationKind::CompressedTensors => Self::compressed_tensors(),
            QuantizationKind::ModelOpt => Self::model_opt(),
            QuantizationKind::TorchAO => Self::torch_ao(),
            QuantizationKind::BitsAndBytes => {
                let mut p = Self::default();
                p.format = QuantizedWeightFormat::BitsAndBytes;
                p
            }
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn fp8_kv_cache() -> Self {
        Self {
            format: QuantizedWeightFormat::Fp8KvCache,
            bits: Some(8),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::FP8E4M3,
        }
    }

    pub fn mxfp8() -> Self {
        Self {
            format: QuantizedWeightFormat::Mxfp8,
            bits: Some(8),
            group_size: Some(32),
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::FP8E4M3,
        }
    }

    pub fn mxfp4() -> Self {
        Self {
            format: QuantizedWeightFormat::Mxfp4,
            bits: Some(4),
            group_size: Some(32),
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::FP8E4M3,
        }
    }

    pub fn nvfp4() -> Self {
        Self {
            format: QuantizedWeightFormat::Nvfp4,
            bits: Some(4),
            group_size: Some(16),
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::FP8E4M3,
        }
    }

    pub fn int8() -> Self {
        Self {
            format: QuantizedWeightFormat::Int8,
            bits: Some(8),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        }
    }

    pub fn int4() -> Self {
        Self {
            format: QuantizedWeightFormat::Int4,
            bits: Some(4),
            group_size: Some(128),
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        }
    }

    pub fn compressed_tensors() -> Self {
        Self {
            format: QuantizedWeightFormat::CompressedTensors,
            bits: Some(8),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        }
    }

    pub fn model_opt() -> Self {
        Self {
            format: QuantizedWeightFormat::ModelOpt,
            bits: Some(8),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        }
    }

    pub fn torch_ao() -> Self {
        Self {
            format: QuantizedWeightFormat::TorchAo,
            bits: Some(8),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self.format {
            QuantizedWeightFormat::Gptq | QuantizedWeightFormat::Awq => {
                if !matches!(self.bits, Some(2 | 3 | 4 | 8)) {
                    return Err("GPTQ/AWQ quantization requires 2, 3, 4, or 8 bits".into());
                }
                if self.group_size.unwrap_or(0) == 0 {
                    return Err("GPTQ/AWQ quantization requires a positive group size".into());
                }
            }
            QuantizedWeightFormat::Fp8KvCache => {
                if !matches!(self.kv_cache_dtype, DType::FP8E4M3 | DType::FP8E5M2) {
                    return Err("FP8 KV cache requires an FP8 cache dtype".into());
                }
            }
            QuantizedWeightFormat::Mxfp8 => {
                if self.bits != Some(8) {
                    return Err("MXFP8 requires 8 bits".into());
                }
                if self.group_size.unwrap_or(0) == 0 {
                    return Err("MXFP8 requires a positive group size".into());
                }
            }
            QuantizedWeightFormat::Mxfp4 => {
                if self.bits != Some(4) {
                    return Err("MXFP4 requires 4 bits".into());
                }
                if self.group_size.unwrap_or(0) == 0 {
                    return Err("MXFP4 requires a positive group size".into());
                }
            }
            QuantizedWeightFormat::Nvfp4 => {
                if self.bits != Some(4) {
                    return Err("NVFP4 requires 4 bits".into());
                }
                if self.group_size.unwrap_or(0) == 0 {
                    return Err("NVFP4 requires a positive group size".into());
                }
            }
            QuantizedWeightFormat::Int8 => {
                if self.bits != Some(8) {
                    return Err("INT8 requires 8 bits".into());
                }
            }
            QuantizedWeightFormat::Int4 => {
                if self.bits != Some(4) {
                    return Err("INT4 requires 4 bits".into());
                }
            }
            QuantizedWeightFormat::Gguf | QuantizedWeightFormat::BitsAndBytes | QuantizedWeightFormat::CompressedTensors | QuantizedWeightFormat::ModelOpt | QuantizedWeightFormat::TorchAo | QuantizedWeightFormat::Unquantized => {}
        }
        Ok(())
    }

}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraAdapterConfig {
    pub adapter_id: String,
    pub path: String,
    pub rank: usize,
    pub alpha: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraRequestSelection {
    pub adapter_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraCachePolicy {
    pub max_active_adapters: usize,
    pub lru_capacity: usize,
}

impl LoraCachePolicy {
    pub fn admits(&self, active_adapters: usize) -> bool {
        active_adapters < self.max_active_adapters.min(self.lru_capacity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorParallelPlan {
    pub world_size: usize,
    pub rank: usize,
}

impl TensorParallelPlan {
    pub fn shard_range(&self, total_dim: usize) -> std::ops::Range<usize> {
        assert!(self.world_size > 0);
        assert!(self.rank < self.world_size);
        let base = total_dim / self.world_size;
        let rem = total_dim % self.world_size;
        let start = self.rank * base + self.rank.min(rem);
        let len = base + usize::from(self.rank < rem);
        start..start + len
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineParallelPlan {
    pub stages: usize,
    pub stage: usize,
}

impl PipelineParallelPlan {
    pub fn layer_range(&self, num_layers: usize) -> std::ops::Range<usize> {
        TensorParallelPlan { world_size: self.stages, rank: self.stage }.shard_range(num_layers)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataParallelPlan {
    pub replicas: usize,
    pub replica: usize,
}

impl DataParallelPlan {
    pub fn owns_request(&self, request_hash: u64) -> bool {
        self.replicas > 0 && request_hash as usize % self.replicas == self.replica
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteKvFailurePolicy {
    Recompute,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTransferDescriptor {
    pub request_key: u64,
    pub block_ids: Vec<u32>,
    pub source_rank: usize,
    pub target_rank: usize,
    pub failure_policy: RemoteKvFailurePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuKvOffloadPlan {
    pub pinned_bytes: usize,
    pub block_size_bytes: usize,
}

impl CpuKvOffloadPlan {
    pub fn capacity_blocks(&self) -> usize {
        if self.block_size_bytes == 0 { 0 } else { self.pinned_bytes / self.block_size_bytes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_parallel_ranges_cover_uneven_dim() {
        let ranges: Vec<_> =
            (0..3).map(|rank| TensorParallelPlan { world_size: 3, rank }.shard_range(10)).collect();
        assert_eq!(ranges, vec![0..4, 4..7, 7..10]);
    }

    #[test]
    fn quantization_validation_checks_required_fields() {
        let bad = QuantizationPlan {
            format: QuantizedWeightFormat::Gptq,
            bits: Some(4),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::F16,
        };
        assert!(bad.validate().is_err());
        assert!(QuantizationPlan::fp8_kv_cache().validate().is_ok());
        assert!(QuantizationPlan::mxfp8().validate().is_ok());
        assert!(QuantizationPlan::mxfp4().validate().is_ok());
        assert!(QuantizationPlan::nvfp4().validate().is_ok());
        assert!(QuantizationPlan::int8().validate().is_ok());
        assert!(QuantizationPlan::int4().validate().is_ok());
        assert!(QuantizationPlan::compressed_tensors().validate().is_ok());
        assert!(QuantizationPlan::model_opt().validate().is_ok());
        assert!(QuantizationPlan::torch_ao().validate().is_ok());
    }


    #[test]
    fn cpu_offload_capacity_uses_whole_blocks() {
        let plan = CpuKvOffloadPlan { pinned_bytes: 1025, block_size_bytes: 256 };
        assert_eq!(plan.capacity_blocks(), 4);
    }
}
