use std::{collections::HashMap, time::Duration};

use rllm_core::{
    ids::RequestId,
    output::FinishReason,
    request::{InferenceRequest, SamplingParams},
};
use rllm_engine::{AsyncLLMEngine, EngineCore, LLMEngine};
use rllm_executor::executor::{Executor, ExecutorOutput};
use rllm_sampling::{Sampler, SamplingInput};
use rllm_scheduler::Scheduler;

// ── Test Executor ──────────────────────────────────────────────────────────

/// A simple test executor that generates deterministic tokens.
struct TestExecutor {
    vocab_size: usize,
    eos_token_id: u32,
    sampler: Sampler,
    requests: HashMap<RequestId, TestRequestState>,
}

struct TestRequestState {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    sampling_params: SamplingParams,
}

impl TestExecutor {
    fn new(vocab_size: usize, eos_token_id: u32, seed: u64) -> Self {
        Self {
            vocab_size,
            eos_token_id,
            sampler: Sampler::from_seed(seed),
            requests: HashMap::new(),
        }
    }
}

impl Executor for TestExecutor {
    fn initialize(
        &mut self,
        _kv_cache_configs: &[rllm_cache::spec::KVCacheConfig],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn determine_available_memory(&self) -> anyhow::Result<usize> {
        Ok(4 * 1024 * 1024 * 1024)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &rllm_scheduler::SchedulerOutput,
    ) -> anyhow::Result<ExecutorOutput> {
        let scheduled_ids: Vec<RequestId> = scheduler_output
            .scheduled_new
            .iter()
            .chain(scheduler_output.scheduled_cached.iter())
            .chain(scheduler_output.scheduled_running.iter())
            .copied()
            .collect();

        if scheduled_ids.is_empty() {
            return Ok(ExecutorOutput { sampled_token_ids: vec![], logprobs: vec![] });
        }

        let mut sampled_token_ids = Vec::new();
        let mut logprobs = Vec::new();

        for request_id in &scheduled_ids {
            let state = match self.requests.get(request_id) {
                Some(s) => s,
                None => continue,
            };

            let position = state.prompt_token_ids.len() + state.generated_token_ids.len();
            let mut logits = vec![0.0f32; self.vocab_size];
            // Make the token predictable based on position.
            let idx = position % (self.vocab_size - 1);
            logits[idx] = 100.0;

            let mut context = state.prompt_token_ids.clone();
            context.extend_from_slice(&state.generated_token_ids);

            let input = SamplingInput {
                logits,
                params: state.sampling_params.clone(),
                context_token_ids: context,
                num_generated: state.generated_token_ids.len() as u32,
                eos_token_id: self.eos_token_id,
                bad_word_token_ids: vec![],
            };

            let output = self.sampler.sample(&input);
            sampled_token_ids.push(output.token_id);
            logprobs.push(output.logprob);

            if let Some(s) = self.requests.get_mut(request_id) {
                s.generated_token_ids.push(output.token_id);
            }
        }

        Ok(ExecutorOutput { sampled_token_ids, logprobs })
    }

    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) {
        self.requests.insert(
            request_id,
            TestRequestState { prompt_token_ids, generated_token_ids: Vec::new(), sampling_params },
        );
    }

    fn shutdown(&mut self) {
        self.requests.clear();
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────

fn make_test_scheduler(max_seqs: usize, budget: usize) -> Scheduler {
    use rllm_cache::{
        manager::KVCacheManager,
        spec::{KVCacheConfig, KVCacheSpec},
    };
    use rllm_core::{
        config::{PrefixHashAlgorithm, SchedulerConfig, SchedulingPolicy},
        dtype::DType,
    };

    let kv_config = KVCacheConfig {
        num_blocks: 1024,
        spec: KVCacheSpec {
            block_size: 16,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            dtype: DType::F16,
            sliding_window: None,
        },
    };
    let kv_mgr = KVCacheManager::new(kv_config, false, PrefixHashAlgorithm::Sha256Cbor);
    let sched_config = SchedulerConfig {
        max_num_seqs: max_seqs,
        max_num_batched_tokens: budget,
        max_num_scheduled_tokens: budget,
        long_prefill_token_threshold: 512,
        enable_chunked_prefill: false,
        scheduling_policy: SchedulingPolicy::FCFS,
        stream_interval: 1,
        async_scheduling: false,
    };
    Scheduler::with_kv_cache_manager(sched_config, 16, 4096, false, kv_mgr)
}

fn make_request(prompt_len: usize, max_tokens: u32) -> InferenceRequest {
    InferenceRequest {
        request_id: RequestId::new(),
        prompt: None,
        token_ids: Some((0..prompt_len as u32).collect()),
        messages: None,
        sampling_params: SamplingParams {
            max_tokens: Some(max_tokens),
            temperature: 0.0,
            ..Default::default()
        },
        arrival_time: std::time::Instant::now(),
        priority: 0,
        stream: false,
        cache_salt: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// Test: add request -> finished output.
#[test]
fn test_add_request_finished_output() {
    let executor = TestExecutor::new(100, 99, 42);
    let scheduler = make_test_scheduler(16, 256);
    let core = EngineCore::new(Box::new(executor), scheduler, 99);
    let mut engine = LLMEngine::new(core);

    let req = make_request(8, 4);
    engine.add_request(req).unwrap();

    let mut all_finished = false;
    while engine.has_work() {
        let outputs = engine.step();
        for output in &outputs {
            if output.finished {
                all_finished = true;
                assert!(output.usage.completion_tokens > 0);
                assert!(output.usage.completion_tokens <= 4);
            }
        }
    }

    assert!(all_finished, "Request should finish");
}

/// Test: multiple requests all complete.
#[test]
fn test_multiple_requests_all_complete() {
    let executor = TestExecutor::new(100, 99, 42);
    let scheduler = make_test_scheduler(16, 512);
    let core = EngineCore::new(Box::new(executor), scheduler, 99);
    let mut engine = LLMEngine::new(core);

    for _ in 0..8 {
        let req = make_request(16, 8);
        engine.add_request(req).unwrap();
    }

    let outputs = engine.generate(vec![]).unwrap();
    assert_eq!(outputs.len(), 8, "All 8 requests should produce outputs");
    for (i, out) in outputs.iter().enumerate() {
        assert!(out.finished, "Request {} should be finished", i);
    }
}

/// Test: abort request mid-generation.
#[test]
fn test_abort_request() {
    let executor = TestExecutor::new(100, 99, 42);
    let scheduler = make_test_scheduler(16, 256);
    let core = EngineCore::new(Box::new(executor), scheduler, 99);
    let mut engine = LLMEngine::new(core);

    let req = make_request(8, 16);
    let id = req.request_id;
    engine.add_request(req).unwrap();

    // Run a few steps.
    for _ in 0..3 {
        if engine.has_work() {
            engine.step();
        }
    }

    // Abort the request.
    engine.abort_request(id);

    // The engine should no longer have work (since the only request was aborted).
    // The aborted request's output may have already been produced.
    // Verify the engine doesn't crash.
    while engine.has_work() {
        engine.step();
    }
}

/// Test: EOS token stops generation.
#[test]
fn test_eos_stops_generation() {
    let executor = TestExecutor::new(20, 5, 42); // EOS = token 5
    let scheduler = make_test_scheduler(16, 256);
    let core = EngineCore::new(Box::new(executor), scheduler, 5);
    let mut engine = LLMEngine::new(core);

    let req = make_request(4, 16); // max 16 tokens, but should hit EOS first
    engine.add_request(req).unwrap();

    let outputs = engine.generate(vec![]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out = &outputs[0];
    assert!(out.finished);

    // Check finish reason is Stop (EOS).
    if let Some(co) = out.outputs.first() {
        assert_eq!(co.finish_reason, Some(FinishReason::Stop));
    }
}

/// Test: max_tokens limit stops generation.
#[test]
fn test_max_tokens_stops_generation() {
    let executor = TestExecutor::new(100, 99, 42);
    let scheduler = make_test_scheduler(16, 256);
    let core = EngineCore::new(Box::new(executor), scheduler, 99);
    let mut engine = LLMEngine::new(core);

    let max_tokens = 3u32;
    let req = make_request(4, max_tokens);
    engine.add_request(req).unwrap();

    let outputs = engine.generate(vec![]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out = &outputs[0];
    assert!(out.finished);
    assert!(
        out.usage.completion_tokens <= max_tokens,
        "completion_tokens {} should be <= max_tokens {}",
        out.usage.completion_tokens,
        max_tokens
    );
}

/// Test: concurrent async requests.
#[test]
fn test_concurrent_async_requests() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let executor = TestExecutor::new(100, 99, 42);
        let scheduler = make_test_scheduler(16, 512);
        let core = EngineCore::new(Box::new(executor), scheduler, 99);
        let engine = AsyncLLMEngine::new(core);

        // Add multiple requests concurrently.
        for _ in 0..4 {
            let req = make_request(8, 4);
            engine.add_request(req).unwrap();
        }

        // Collect all outputs via the watch channel.
        let rx = engine.output_receiver();
        let mut all_outputs = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);

        loop {
            tokio::task::yield_now().await;
            let batch = rx.borrow().clone();
            if !batch.is_empty() {
                all_outputs.extend(batch);
                let finished_count = all_outputs.iter().filter(|o| o.finished).count();
                if finished_count >= 4 {
                    break;
                }
            }
            if tokio::time::Instant::now() > deadline {
                panic!("Timed out waiting for engine outputs");
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        assert!(!all_outputs.is_empty(), "Engine should produce outputs");
        let finished: Vec<_> = all_outputs.iter().filter(|o| o.finished).collect();
        assert_eq!(finished.len(), 4, "All 4 requests should finish");

        engine.shutdown().unwrap();
    });
}

/// Test: streaming chunks from async engine.
#[test]
fn test_streaming_chunks() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let executor = TestExecutor::new(100, 99, 42);
        let scheduler = make_test_scheduler(16, 256);
        let core = EngineCore::new(Box::new(executor), scheduler, 99);
        let engine = AsyncLLMEngine::new(core);

        let req = make_request(4, 4);
        engine.add_request(req).unwrap();

        let rx = engine.output_receiver();
        let mut total_chunks = 0;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);

        loop {
            tokio::task::yield_now().await;
            let batch = rx.borrow().clone();
            if !batch.is_empty() {
                total_chunks += batch.len();
                if batch.iter().any(|o| o.finished) {
                    break;
                }
            }
            if tokio::time::Instant::now() > deadline {
                panic!("Timed out waiting for streaming chunks");
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        assert!(total_chunks > 0, "Should receive at least one streaming chunk");
        engine.shutdown().unwrap();
    });
}

/// Test: request usage statistics are correct.
#[test]
fn test_usage_statistics() {
    let executor = TestExecutor::new(100, 99, 42);
    let scheduler = make_test_scheduler(16, 256);
    let core = EngineCore::new(Box::new(executor), scheduler, 99);
    let mut engine = LLMEngine::new(core);

    let prompt_len = 8;
    let max_tokens = 5;
    let req = make_request(prompt_len, max_tokens);
    engine.add_request(req).unwrap();

    let outputs = engine.generate(vec![]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out = &outputs[0];

    assert_eq!(out.usage.prompt_tokens, prompt_len as u32);
    assert!(out.usage.completion_tokens > 0);
    assert!(out.usage.completion_tokens <= max_tokens);
    assert_eq!(out.usage.total_tokens, out.usage.prompt_tokens + out.usage.completion_tokens);
}
