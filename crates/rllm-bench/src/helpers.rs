use rllm_cache::{
    manager::KVCacheManager,
    spec::{KVCacheConfig, KVCacheSpec},
};
use rllm_core::{
    config::{CacheConfig, PrefixHashAlgorithm, SchedulerConfig, SchedulingPolicy},
    dtype::DType,
    ids::RequestId,
    request::{InferenceRequest, SamplingParams},
};
use rllm_scheduler::Scheduler;

/// Create a test `Scheduler` with reasonable defaults.
pub fn make_test_scheduler(
    block_size: usize,
    num_blocks: usize,
    max_seqs: usize,
    max_budget: usize,
) -> Scheduler {
    make_test_scheduler_with_options(block_size, num_blocks, max_seqs, max_budget, false, false)
}

/// Create a test `Scheduler` with chunked prefill and/or prefix caching.
pub fn make_test_scheduler_with_options(
    block_size: usize,
    num_blocks: usize,
    max_seqs: usize,
    max_budget: usize,
    chunked_prefill: bool,
    prefix_caching: bool,
) -> Scheduler {
    let kv_config = KVCacheConfig {
        num_blocks,
        spec: KVCacheSpec {
            block_size,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            dtype: DType::F16,
            sliding_window: None,
        },
    };
    let kv_mgr = KVCacheManager::new(kv_config, prefix_caching, PrefixHashAlgorithm::Sha256Cbor);

    let sched_config = SchedulerConfig {
        max_num_seqs: max_seqs,
        max_num_batched_tokens: max_budget,
        max_num_scheduled_tokens: max_budget,
        long_prefill_token_threshold: 512,
        enable_chunked_prefill: chunked_prefill,
        scheduling_policy: SchedulingPolicy::FCFS,
        stream_interval: 1,
        async_scheduling: false,
    };

    Scheduler::with_kv_cache_manager(sched_config, block_size, 4096, prefix_caching, kv_mgr)
}

/// Create a test `Scheduler` from a `CacheConfig`.
pub fn make_scheduler_from_cache_config(
    cache_config: &CacheConfig,
    num_blocks: usize,
    max_seqs: usize,
    max_budget: usize,
) -> Scheduler {
    let sched_config = SchedulerConfig {
        max_num_seqs: max_seqs,
        max_num_batched_tokens: max_budget,
        max_num_scheduled_tokens: max_budget,
        long_prefill_token_threshold: 512,
        enable_chunked_prefill: cache_config.enable_prefix_caching,
        scheduling_policy: SchedulingPolicy::FCFS,
        stream_interval: 1,
        async_scheduling: false,
    };
    Scheduler::new(sched_config, cache_config, num_blocks, 4096)
}

/// Create a test `InferenceRequest` with the given prompt length and max output tokens.
pub fn make_inference_request(prompt_len: usize, max_tokens: u32) -> InferenceRequest {
    make_inference_request_with_params(prompt_len, max_tokens, SamplingParams::default())
}

/// Create a test `InferenceRequest` with specific sampling params.
pub fn make_inference_request_with_params(
    prompt_len: usize,
    max_tokens: u32,
    sampling_params: SamplingParams,
) -> InferenceRequest {
    let mut params = sampling_params;
    params.max_tokens = Some(max_tokens);
    InferenceRequest {
        request_id: RequestId::new(),
        prompt: None,
        token_ids: Some((0..prompt_len as u32).collect()),
        messages: None,
        sampling_params: params,
        arrival_time: std::time::Instant::now(),
        priority: 0,
        stream: false,
        cache_salt: None,
    }
}

/// Create a test `KVCacheManager`.
pub fn make_test_kv_cache_manager(block_size: usize, num_blocks: usize) -> KVCacheManager {
    make_test_kv_cache_manager_with_prefix(block_size, num_blocks, false)
}

/// Create a test `KVCacheManager` with optional prefix caching.
pub fn make_test_kv_cache_manager_with_prefix(
    block_size: usize,
    num_blocks: usize,
    enable_prefix_caching: bool,
) -> KVCacheManager {
    let config = KVCacheConfig {
        num_blocks,
        spec: KVCacheSpec {
            block_size,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F16,
            sliding_window: None,
        },
    };
    KVCacheManager::new(config, enable_prefix_caching, PrefixHashAlgorithm::Sha256Cbor)
}
