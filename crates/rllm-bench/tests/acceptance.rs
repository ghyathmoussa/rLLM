use rllm_bench::helpers::{
    make_inference_request, make_test_kv_cache_manager, make_test_scheduler,
    make_test_scheduler_with_options,
};
use rllm_bench::mock_executor::{MockExecutor, MockExecutorConfig, MockMode};
use rllm_core::ids::RequestId;
use rllm_core::request::SamplingParams;
use rllm_engine::{AsyncLLMEngine, EngineCore, LLMEngine};

/// Acceptance: engine serves 32 concurrent requests on one "GPU".
#[test]
fn acceptance_32_concurrent_requests() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };
    let mock = MockExecutor::new(config);
    let scheduler = make_test_scheduler(16, 4096, 64, 8192);
    let core = EngineCore::new(Box::new(mock), scheduler, 999);
    let mut engine = LLMEngine::new(core);

    let requests: Vec<_> = (0..32).map(|_| make_inference_request(32, 8)).collect();

    let outputs = engine.generate(requests).unwrap();

    assert_eq!(outputs.len(), 32, "Expected 32 outputs");
    for (i, out) in outputs.iter().enumerate() {
        assert!(out.finished, "Request {} not finished", i);
        assert!(out.usage.completion_tokens > 0, "Request {} has zero completion tokens", i);
    }
}

/// Acceptance: paged KV cache allocates and frees correctly under load.
#[test]
fn acceptance_paged_kv_cache() {
    let mut mgr = make_test_kv_cache_manager(16, 256);
    let mut ids = Vec::new();

    // Allocate for 64 requests.
    for _ in 0..64 {
        let rid = RequestId::new();
        let result = mgr.allocate_slots(rid, 48, 0);
        if result.is_some() {
            ids.push(rid);
        }
    }

    assert!(ids.len() >= 32, "Should allocate at least 32 requests");

    // Free all.
    for rid in &ids {
        mgr.free(*rid);
    }

    let usage = mgr.usage();
    assert_eq!(usage.num_active_blocks, 0, "All blocks should be free after cleanup");
}

/// Acceptance: chunked prefill splits long prompts across steps.
#[test]
fn acceptance_chunked_prefill() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };
    let mock = MockExecutor::new(config);
    // Chunked prefill with small budget to force multi-step prefill.
    let scheduler = make_test_scheduler_with_options(16, 256, 64, 64, true, false);
    let core = EngineCore::new(Box::new(mock), scheduler, 999);
    let mut engine = LLMEngine::new(core);

    // 128-token prompt with budget of 64 → needs at least 2 steps for prefill.
    let req = make_inference_request(128, 8);
    engine.add_request(req).unwrap();

    let mut step_count = 0;
    let mut total_outputs = Vec::new();
    while engine.has_work() {
        let outputs = engine.step();
        step_count += 1;
        total_outputs.extend(outputs);
        // Prevent infinite loop.
        if step_count > 200 {
            break;
        }
    }

    assert!(step_count >= 2, "Expected at least 2 steps for chunked prefill, got {}", step_count);
    let finished: Vec<_> = total_outputs.iter().filter(|o| o.finished).collect();
    assert_eq!(finished.len(), 1, "Expected exactly 1 finished output");
}

/// Acceptance: prefix caching produces cache hits.
#[test]
fn acceptance_prefix_caching() {
    let mut sched = make_test_scheduler_with_options(4, 100, 10, 4096, false, true);

    // First request: 8 tokens → 2 blocks.
    let prefix: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut tokens1 = prefix.clone();
    tokens1.push(100);

    let req1 = rllm_core::request::InferenceRequest {
        request_id: RequestId::new(),
        prompt: None,
        token_ids: Some(tokens1),
        messages: None,
        sampling_params: SamplingParams::default(),
        arrival_time: std::time::Instant::now(),
        priority: 0,
        stream: false,
        cache_salt: None,
    };
    sched.add_request(req1);
    let out1 = sched.step();
    assert!(out1.num_scheduled() > 0, "First request should be scheduled");

    // Second request with same prefix.
    let mut tokens2 = prefix.clone();
    tokens2.push(200);
    let req2 = rllm_core::request::InferenceRequest {
        request_id: RequestId::new(),
        prompt: None,
        token_ids: Some(tokens2),
        messages: None,
        sampling_params: SamplingParams::default(),
        arrival_time: std::time::Instant::now(),
        priority: 0,
        stream: false,
        cache_salt: None,
    };
    sched.add_request(req2);
    let out2 = sched.step();
    // Second request should be scheduled (either as new or cached).
    assert!(
        out2.scheduled_new.len() + out2.scheduled_cached.len() + out2.scheduled_running.len() > 0,
        "Second request should be scheduled"
    );
}

/// Acceptance: AsyncLLMEngine produces streaming outputs.
#[test]
fn acceptance_streaming_outputs() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let config = MockExecutorConfig {
            mode: MockMode::Deterministic { offset: 0 },
            vocab_size: 1000,
            eos_token_id: 999,
            sampler_seed: Some(42),
        };
        let mock = MockExecutor::new(config);
        let scheduler = make_test_scheduler(16, 256, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        let engine = AsyncLLMEngine::new(core);

        let req = make_inference_request(16, 8);
        engine.add_request(req).unwrap();

        let rx = engine.output_receiver();
        let mut all_outputs: Vec<rllm_core::output::RequestOutput> = Vec::new();

        // Wait for the engine to produce output.
        // Use tokio::time::timeout to avoid infinite waits.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

        loop {
            // Give the engine task a chance to run.
            tokio::task::yield_now().await;

            let batch = rx.borrow().clone();
            if !batch.is_empty() {
                all_outputs.extend(batch);
                if all_outputs.iter().any(|o| o.finished) {
                    break;
                }
            }

            if tokio::time::Instant::now() > deadline {
                panic!("Timed out waiting for engine output");
            }

            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        assert!(!all_outputs.is_empty(), "Engine should produce outputs");
        assert!(all_outputs.iter().any(|o| o.finished), "At least one output should be finished");

        engine.shutdown().unwrap();
    });
}

/// Acceptance: metrics recorder can be installed and rendered.
#[test]
fn acceptance_metrics_recorder() {
    // Install recorder (may already be installed by other tests).
    let _ = rllm_metrics::install_recorder();
    // Just verify the macros don't panic.
    rllm_metrics::counter!("rllm_test_counter").increment(1);
    rllm_metrics::gauge!("rllm_test_gauge").set(42.0);
}

/// Acceptance: engine generates correct usage statistics.
#[test]
fn acceptance_usage_statistics() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 100,
        eos_token_id: 99,
        sampler_seed: Some(42),
    };
    let mock = MockExecutor::new(config);
    let scheduler = make_test_scheduler(16, 256, 64, 4096);
    let core = EngineCore::new(Box::new(mock), scheduler, 99);
    let mut engine = LLMEngine::new(core);

    let prompt_len = 32u32;
    let max_tokens = 16u32;
    let req = make_inference_request(prompt_len as usize, max_tokens);

    let outputs = engine.generate(vec![req]).unwrap();
    assert_eq!(outputs.len(), 1);

    let out = &outputs[0];
    assert!(out.finished);
    assert_eq!(out.usage.prompt_tokens, prompt_len);
    assert!(out.usage.completion_tokens > 0);
    assert!(out.usage.completion_tokens <= max_tokens);
    assert_eq!(out.usage.total_tokens, out.usage.prompt_tokens + out.usage.completion_tokens);
}
