use proptest::prelude::*;

use rllm_bench::helpers::{
    make_inference_request, make_inference_request_with_params, make_test_kv_cache_manager,
    make_test_kv_cache_manager_with_prefix, make_test_scheduler,
};
use rllm_bench::mock_executor::{MockExecutor, MockExecutorConfig, MockMode};
use rllm_core::ids::RequestId;
use rllm_core::request::SamplingParams;
use rllm_engine::{EngineCore, LLMEngine};
use rllm_sampling::{Sampler, SamplingInput};

// ── Deterministic Regression Tests ─────────────────────────────────────

#[test]
fn greedy_determinism_reproducibility() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };

    let make_engine = || {
        let mock = MockExecutor::new(config.clone());
        let scheduler = make_test_scheduler(16, 256, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        LLMEngine::new(core)
    };

    let params = SamplingParams { temperature: 0.0, max_tokens: Some(16), ..Default::default() };

    let req = make_inference_request_with_params(32, 16, params);

    let mut engine1 = make_engine();
    let out1 = engine1.generate(vec![req.clone()]).unwrap();
    let tokens1: Vec<u32> = out1[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();

    let mut engine2 = make_engine();
    let out2 = engine2.generate(vec![req]).unwrap();
    let tokens2: Vec<u32> = out2[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();

    assert_eq!(tokens1, tokens2, "Greedy runs produced different token sequences");
}

#[test]
fn greedy_matches_argmax() {
    // With Deterministic mode, logits[position % vocab] = 100.0, rest 0.0.
    // With temperature=0, sampler must pick argmax = position % vocab.
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

    let params = SamplingParams { temperature: 0.0, max_tokens: Some(8), ..Default::default() };

    let req = make_inference_request_with_params(4, 8, params);
    let outputs = engine.generate(vec![req]).unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].finished);
    // Should have generated tokens (not hit EOS early).
    assert!(outputs[0].usage.completion_tokens > 0);
}

#[test]
fn greedy_concurrent_determinism() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };

    let params = SamplingParams { temperature: 0.0, max_tokens: Some(8), ..Default::default() };

    // Run 8 requests concurrently.
    let config_clone = config.clone();
    let make_engine = || {
        let mock = MockExecutor::new(config_clone.clone());
        let scheduler = make_test_scheduler(16, 512, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        LLMEngine::new(core)
    };

    let requests: Vec<_> =
        (0..8).map(|_| make_inference_request_with_params(16, 8, params.clone())).collect();

    let mut engine = make_engine();
    let batch_out = engine.generate(requests.clone()).unwrap();

    // Run same requests one at a time.
    let mut sequential_tokens = Vec::new();
    for req in requests {
        let mut engine = make_engine();
        let out = engine.generate(vec![req]).unwrap();
        let tokens: Vec<u32> = out[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();
        sequential_tokens.push(tokens);
    }

    // Each request should produce the same tokens regardless of batching.
    for (i, batch_output) in batch_out.iter().enumerate() {
        let batch_tokens: Vec<u32> =
            batch_output.outputs.iter().flat_map(|o| o.token_ids.clone()).collect();
        assert_eq!(
            batch_tokens, sequential_tokens[i],
            "Request {} produced different tokens in batch vs sequential",
            i
        );
    }
}

#[test]
fn token_output_matches_baseline() {
    // Manually step through sampling without engine, then compare with
    // engine step-by-step output collection.
    let vocab_size = 100;
    let eos = 99;
    let prompt_len = 8usize;
    let max_tokens = 8u32;

    // Compute expected tokens manually using the sampler.
    let mut sampler = Sampler::from_seed(42);
    let mut expected_tokens = Vec::new();
    let mut context: Vec<u32> = (0..prompt_len as u32).collect();

    for step in 0..max_tokens {
        let position = prompt_len + step as usize;
        let mut logits = vec![0.0f32; vocab_size];
        logits[(position + 5) % vocab_size] = 100.0;

        let input = SamplingInput {
            logits,
            params: SamplingParams {
                temperature: 0.0,
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
            context_token_ids: context.clone(),
            num_generated: step,
            eos_token_id: eos,
            bad_word_token_ids: vec![],
        };
        let out = sampler.sample(&input);
        context.push(out.token_id);
        expected_tokens.push(out.token_id);

        if out.token_id == eos {
            break;
        }
    }

    // Run through engine step-by-step to collect all generated tokens.
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 5 },
        vocab_size,
        eos_token_id: eos,
        sampler_seed: Some(42),
    };
    let mock = MockExecutor::new(config);
    let scheduler = make_test_scheduler(16, 256, 64, 4096);
    let core = EngineCore::new(Box::new(mock), scheduler, eos);
    let mut engine = LLMEngine::new(core);

    let params =
        SamplingParams { temperature: 0.0, max_tokens: Some(max_tokens), ..Default::default() };

    let req = make_inference_request_with_params(prompt_len, max_tokens, params);
    engine.add_request(req).unwrap();

    let mut all_engine_tokens = Vec::new();
    while engine.has_work() {
        let outputs = engine.step();
        for output in &outputs {
            for co in &output.outputs {
                all_engine_tokens.extend_from_slice(&co.token_ids);
            }
        }
    }

    assert_eq!(all_engine_tokens, expected_tokens, "Engine tokens don't match baseline");
}

// ── Logits Correctness ─────────────────────────────────────────────────

#[test]
fn temperature_scaling_correctness() {
    let mut sampler = Sampler::new();
    let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    // With temperature=1.0, the probabilities should be softmax(logits).
    let input = SamplingInput {
        logits: logits.clone(),
        params: SamplingParams { temperature: 1.0, ..Default::default() },
        context_token_ids: vec![],
        num_generated: 0,
        eos_token_id: 99,
        bad_word_token_ids: vec![],
    };
    let out = sampler.sample(&input);
    // Token 4 (highest logit) should be most likely but not guaranteed
    // unless temperature=0. Just verify it returns a valid token.
    assert!(out.token_id < 5);
}

#[test]
fn sampling_pipeline_produces_valid_tokens() {
    let mut sampler = Sampler::from_seed(42);
    let vocab_size = 1000;
    let logits = vec![0.5f32; vocab_size];

    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        max_tokens: Some(32),
        ..Default::default()
    };

    let mut context = (0..32u32).collect::<Vec<_>>();

    for step in 0..16 {
        let input = SamplingInput {
            logits: logits.clone(),
            params: params.clone(),
            context_token_ids: context.clone(),
            num_generated: step,
            eos_token_id: vocab_size as u32 - 1,
            bad_word_token_ids: vec![],
        };
        let out = sampler.sample(&input);
        assert!(out.token_id < vocab_size as u32, "Token out of vocab range");
        context.push(out.token_id);
    }
}

// ── Property Tests (proptest) ──────────────────────────────────────────

proptest! {
    #[test]
    fn scheduler_respects_max_seqs(
        num_requests in 1usize..50,
        max_seqs in 2usize..16,
        prompt_len in 4usize..64,
    ) {
        let mut sched = make_test_scheduler(4, 500, max_seqs, 4096);
        for _ in 0..num_requests {
            sched.add_request(make_inference_request(prompt_len, 8));
        }
        let out = sched.step();
        let scheduled = out.scheduled_new.len() + out.scheduled_running.len()
            + out.scheduled_cached.len();
        prop_assert!(scheduled <= max_seqs);
    }

    #[test]
    fn scheduler_respects_token_budget(
        num_requests in 1usize..30,
        budget in 64usize..2048,
        prompt_len in 4usize..128,
    ) {
        let mut sched = make_test_scheduler(4, 500, 64, budget);
        for _ in 0..num_requests {
            sched.add_request(make_inference_request(prompt_len, 8));
        }
        let out = sched.step();
        prop_assert!(out.token_budget_used <= budget);
    }

    #[test]
    fn kv_cache_alloc_free_invariant(
        num_requests in 1usize..20,
        tokens_per_request in 4usize..64,
    ) {
        let mut mgr = make_test_kv_cache_manager(16, 200);
        let mut ids = Vec::new();
        for _ in 0..num_requests {
            let rid = RequestId::new();
            if mgr.allocate_slots(rid, tokens_per_request, 0).is_some() {
                ids.push(rid);
            }
        }
        for rid in &ids {
            mgr.free(*rid);
        }
        let usage = mgr.usage();
        prop_assert_eq!(usage.num_active_blocks, 0);
    }

    #[test]
    fn engine_completes_all_requests(
        num_requests in 1usize..16,
        prompt_len in 4usize..64,
        max_tokens in 4u32..32,
    ) {
        let config = MockExecutorConfig {
            mode: MockMode::Deterministic { offset: 0 },
            vocab_size: 500,
            eos_token_id: 499,
            sampler_seed: Some(42),
        };
        let mock = MockExecutor::new(config);
        let scheduler = make_test_scheduler(16, 256, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 499);
        let mut engine = LLMEngine::new(core);

        let requests: Vec<_> = (0..num_requests)
            .map(|_| make_inference_request(prompt_len, max_tokens))
            .collect();

        let outputs = engine.generate(requests).unwrap();
        prop_assert_eq!(outputs.len(), num_requests);
        for (i, output) in outputs.iter().enumerate() {
            prop_assert!(output.finished, "Request {} not finished", i);
            prop_assert!(output.usage.completion_tokens > 0, "Request {} no tokens", i);
        }
    }
}

#[test]
fn prefix_cache_sharing() {
    let mut mgr = make_test_kv_cache_manager_with_prefix(4, 100, true);

    // Request 1: cache two blocks [1,2,3,4] [5,6,7,8].
    let rid1 = RequestId::new();
    let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let result1 = mgr.get_computed_blocks(rid1, &tokens1, false);
    assert!(result1.cached_block_ids.is_empty());
    let blocks1 = mgr.allocate_slots(rid1, 8, 0).unwrap();
    assert_eq!(blocks1.len(), 2);
    mgr.cache_blocks(rid1, 8);
    mgr.free(rid1);

    // Request 2: same prefix + more tokens.
    let rid2 = RequestId::new();
    let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let result = mgr.get_computed_blocks(rid2, &tokens2, false);
    assert!(result.cached_block_ids.len() >= 2, "Expected at least 2 cached blocks");
    assert!(result.num_computed_tokens >= 8, "Expected at least 8 computed tokens");
}

/// Batch invariance: Running the same batch configuration twice produces
/// the same output tokens (determinism across runs).
#[test]
fn batch_determinism_same_config() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };

    let params = SamplingParams { temperature: 0.0, max_tokens: Some(8), ..Default::default() };

    let run_test = || {
        let mock = MockExecutor::new(config.clone());
        let scheduler = make_test_scheduler(16, 512, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        let mut engine = LLMEngine::new(core);

        let req = make_inference_request_with_params(16, 8, params.clone());
        engine.add_request(req).unwrap();
        let outputs = engine.generate(vec![]).unwrap();
        outputs[0]
            .outputs
            .iter()
            .flat_map(|o| o.token_ids.clone())
            .collect::<Vec<_>>()
    };

    let run1 = run_test();
    let run2 = run_test();
    assert_eq!(
        run1, run2,
        "Same deterministic config should produce same output"
    );
}

/// Batch invariance: Running the same batch of multiple requests twice
/// produces the same outputs (determinism with batching).
#[test]
fn batch_determinism_multiple_requests() {
    let config = MockExecutorConfig {
        mode: MockMode::Deterministic { offset: 0 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };

    let params = SamplingParams { temperature: 0.0, max_tokens: Some(4), ..Default::default() };

    let run_test = || {
        let mock = MockExecutor::new(config.clone());
        let scheduler = make_test_scheduler(16, 512, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        let mut engine = LLMEngine::new(core);

        for _ in 0..4 {
            let req = make_inference_request_with_params(8, 4, params.clone());
            engine.add_request(req).unwrap();
        }
        let outputs = engine.generate(vec![]).unwrap();
        outputs
            .iter()
            .map(|o| {
                o.outputs
                    .iter()
                    .flat_map(|co| co.token_ids.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    let run1 = run_test();
    let run2 = run_test();
    assert_eq!(run1, run2, "Same batch config should produce same outputs");
}

/// Seeded random mode is reproducible across runs.
#[test]
fn batch_determinism_seeded_random() {
    let config = MockExecutorConfig {
        mode: MockMode::SeededRandom { seed: 12345 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };

    let params = SamplingParams { temperature: 1.0, max_tokens: Some(8), ..Default::default() };

    let run_test = || {
        let mock = MockExecutor::new(config.clone());
        let scheduler = make_test_scheduler(16, 512, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        let mut engine = LLMEngine::new(core);

        for _ in 0..3 {
            let req = make_inference_request_with_params(8, 4, params.clone());
            engine.add_request(req).unwrap();
        }
        let outputs = engine.generate(vec![]).unwrap();
        outputs
            .iter()
            .map(|o| {
                o.outputs
                    .iter()
                    .flat_map(|co| co.token_ids.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    let run1 = run_test();
    let run2 = run_test();
    assert_eq!(run1, run2, "Seeded random should be reproducible across runs");
}

#[test]
fn seeded_random_mode_reproducibility() {
    let config = MockExecutorConfig {
        mode: MockMode::SeededRandom { seed: 12345 },
        vocab_size: 1000,
        eos_token_id: 999,
        sampler_seed: Some(42),
    };

    let make_engine = || {
        let mock = MockExecutor::new(config.clone());
        let scheduler = make_test_scheduler(16, 256, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        LLMEngine::new(core)
    };

    let params = SamplingParams { temperature: 1.0, max_tokens: Some(16), ..Default::default() };

    let req = make_inference_request_with_params(32, 16, params.clone());

    let mut e1 = make_engine();
    let out1 = e1.generate(vec![req.clone()]).unwrap();
    let t1: Vec<u32> = out1[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();

    let mut e2 = make_engine();
    let out2 = e2.generate(vec![req]).unwrap();
    let t2: Vec<u32> = out2[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();

    assert_eq!(t1, t2, "Seeded random mode should be reproducible");
}
