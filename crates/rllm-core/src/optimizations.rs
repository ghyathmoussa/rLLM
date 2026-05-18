use serde::{Deserialize, Serialize};

use crate::dtype::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizedWeightFormat {
    Fp8KvCache,
    Gptq,
    Awq,
    Gguf,
    BitsAndBytes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub format: QuantizedWeightFormat,
    pub bits: Option<u8>,
    pub group_size: Option<usize>,
    pub activation_dtype: DType,
    pub kv_cache_dtype: DType,
}

impl QuantizationPlan {
    pub fn fp8_kv_cache() -> Self {
        Self {
            format: QuantizedWeightFormat::Fp8KvCache,
            bits: Some(8),
            group_size: None,
            activation_dtype: DType::F16,
            kv_cache_dtype: DType::FP8E4M3,
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
            QuantizedWeightFormat::Gguf | QuantizedWeightFormat::BitsAndBytes => {}
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
    }

    #[test]
    fn cpu_offload_capacity_uses_whole_blocks() {
        let plan = CpuKvOffloadPlan { pinned_bytes: 1025, block_size_bytes: 256 };
        assert_eq!(plan.capacity_blocks(), 4);
    }
}
