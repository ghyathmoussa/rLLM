use serde::{Deserialize, Serialize};

use crate::dtype::DType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_model_len: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub dtype: DType,
    pub quantization: Option<QuantizationConfig>,
    pub tokenizer_mode: TokenizerMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub r#type: String,
    pub factor: f32,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationKind {
    None,
    FP8,
    GPTQ,
    AWQ,
    BitsAndBytes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub kind: QuantizationKind,
    pub group_size: Option<usize>,
    pub bits: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerMode {
    Auto,
    Slow,
    Mistral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub block_size: usize,
    pub hash_block_size: usize,
    pub gpu_memory_utilization: f32,
    pub cpu_swap_bytes: usize,
    pub cache_dtype: DType,
    pub num_gpu_blocks: usize,
    pub enable_prefix_caching: bool,
    pub prefix_hash_algorithm: PrefixHashAlgorithm,
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefixHashAlgorithm {
    Sha256Cbor,
    XxHash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    FCFS,
    Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_num_scheduled_tokens: usize,
    pub long_prefill_token_threshold: usize,
    pub enable_chunked_prefill: bool,
    pub scheduling_policy: SchedulingPolicy,
    pub stream_interval: usize,
    pub async_scheduling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub data_parallel_size: usize,
    pub local_rank: usize,
    pub distributed_backend: DistributedBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedBackend {
    Nccl,
    Gloo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model: ModelConfig,
    pub cache: CacheConfig,
    pub scheduler: SchedulerConfig,
    pub parallel: ParallelConfig,
    pub device: DeviceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_type: DeviceType,
    pub device_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Cuda,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model_config() -> ModelConfig {
        ModelConfig {
            model_id: "meta-llama/Llama-3-8B".into(),
            architecture: "LlamaForCausalLM".into(),
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_model_len: 4096,
            rope_theta: 10000.0,
            rope_scaling: None,
            dtype: DType::F16,
            quantization: None,
            tokenizer_mode: TokenizerMode::Auto,
        }
    }

    fn sample_cache_config() -> CacheConfig {
        CacheConfig {
            block_size: 16,
            hash_block_size: 16,
            gpu_memory_utilization: 0.9,
            cpu_swap_bytes: 0,
            cache_dtype: DType::F16,
            num_gpu_blocks: 1024,
            enable_prefix_caching: false,
            prefix_hash_algorithm: PrefixHashAlgorithm::Sha256Cbor,
            sliding_window: None,
        }
    }

    fn sample_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_num_scheduled_tokens: 4096,
            long_prefill_token_threshold: 2048,
            enable_chunked_prefill: true,
            scheduling_policy: SchedulingPolicy::FCFS,
            stream_interval: 1,
            async_scheduling: false,
        }
    }

    fn sample_parallel_config() -> ParallelConfig {
        ParallelConfig {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            data_parallel_size: 1,
            local_rank: 0,
            distributed_backend: DistributedBackend::Nccl,
        }
    }

    fn sample_engine_config() -> EngineConfig {
        EngineConfig {
            model: sample_model_config(),
            cache: sample_cache_config(),
            scheduler: sample_scheduler_config(),
            parallel: sample_parallel_config(),
            device: DeviceConfig {
                device_type: DeviceType::Cuda,
                device_index: 0,
            },
        }
    }

    fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned>(val: &T) -> T {
        let json = serde_json::to_string(val).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn model_config_serde_roundtrip() {
        let config = sample_model_config();
        let back = roundtrip(&config);
        assert_eq!(config.model_id, back.model_id);
        assert_eq!(config.architecture, back.architecture);
        assert_eq!(config.vocab_size, back.vocab_size);
        assert_eq!(config.hidden_size, back.hidden_size);
        assert_eq!(config.dtype, back.dtype);
    }

    #[test]
    fn model_config_with_rope_scaling_roundtrip() {
        let mut config = sample_model_config();
        config.rope_scaling = Some(RopeScaling {
            r#type: "linear".into(),
            factor: 2.0,
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            original_max_position_embeddings: Some(8192),
        });
        let back = roundtrip(&config);
        assert!(back.rope_scaling.is_some());
        let rs = back.rope_scaling.unwrap();
        assert_eq!(rs.r#type, "linear");
        assert_eq!(rs.factor, 2.0);
    }

    #[test]
    fn model_config_with_quantization_roundtrip() {
        let mut config = sample_model_config();
        config.quantization = Some(QuantizationConfig {
            kind: QuantizationKind::GPTQ,
            group_size: Some(128),
            bits: Some(4),
        });
        let back = roundtrip(&config);
        let q = back.quantization.unwrap();
        assert_eq!(q.kind, QuantizationKind::GPTQ);
        assert_eq!(q.group_size, Some(128));
    }

    #[test]
    fn cache_config_serde_roundtrip() {
        let config = sample_cache_config();
        let back = roundtrip(&config);
        assert_eq!(config.block_size, back.block_size);
        assert_eq!(config.gpu_memory_utilization, back.gpu_memory_utilization);
        assert_eq!(config.cache_dtype, back.cache_dtype);
    }

    #[test]
    fn cache_config_with_prefix_caching_roundtrip() {
        let mut config = sample_cache_config();
        config.enable_prefix_caching = true;
        config.prefix_hash_algorithm = PrefixHashAlgorithm::XxHash;
        config.sliding_window = Some(4096);
        let back = roundtrip(&config);
        assert!(back.enable_prefix_caching);
        assert_eq!(back.prefix_hash_algorithm, PrefixHashAlgorithm::XxHash);
        assert_eq!(back.sliding_window, Some(4096));
    }

    #[test]
    fn scheduler_config_serde_roundtrip() {
        let config = sample_scheduler_config();
        let back = roundtrip(&config);
        assert_eq!(config.max_num_seqs, back.max_num_seqs);
        assert_eq!(config.scheduling_policy, back.scheduling_policy);
        assert!(back.enable_chunked_prefill);
    }

    #[test]
    fn parallel_config_serde_roundtrip() {
        let config = sample_parallel_config();
        let back = roundtrip(&config);
        assert_eq!(config.tensor_parallel_size, back.tensor_parallel_size);
        assert_eq!(config.distributed_backend, back.distributed_backend);
    }

    #[test]
    fn engine_config_serde_roundtrip() {
        let config = sample_engine_config();
        let back = roundtrip(&config);
        assert_eq!(config.model.model_id, back.model.model_id);
        assert_eq!(config.cache.block_size, back.cache.block_size);
        assert_eq!(config.device.device_type, back.device.device_type);
    }

    #[test]
    fn all_enums_roundtrip() {
        assert_eq!(QuantizationKind::None, roundtrip(&QuantizationKind::None));
        assert_eq!(QuantizationKind::FP8, roundtrip(&QuantizationKind::FP8));
        assert_eq!(TokenizerMode::Auto, roundtrip(&TokenizerMode::Auto));
        assert_eq!(TokenizerMode::Slow, roundtrip(&TokenizerMode::Slow));
        assert_eq!(PrefixHashAlgorithm::Sha256Cbor, roundtrip(&PrefixHashAlgorithm::Sha256Cbor));
        assert_eq!(SchedulingPolicy::FCFS, roundtrip(&SchedulingPolicy::FCFS));
        assert_eq!(SchedulingPolicy::Priority, roundtrip(&SchedulingPolicy::Priority));
        assert_eq!(DistributedBackend::Nccl, roundtrip(&DistributedBackend::Nccl));
        assert_eq!(DistributedBackend::Gloo, roundtrip(&DistributedBackend::Gloo));
        assert_eq!(DeviceType::Cpu, roundtrip(&DeviceType::Cpu));
        assert_eq!(DeviceType::Cuda, roundtrip(&DeviceType::Cuda));
    }
}
