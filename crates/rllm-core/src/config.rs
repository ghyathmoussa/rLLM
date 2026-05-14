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
