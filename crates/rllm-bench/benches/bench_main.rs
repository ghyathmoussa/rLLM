use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use rllm_bench::helpers::{
    make_inference_request, make_test_kv_cache_manager, make_test_scheduler,
};
use rllm_bench::mock_executor::{MockExecutor, MockExecutorConfig, MockMode};
use rllm_bench::workload::{LengthDistribution, SyntheticWorkload, WorkloadConfig};
use rllm_core::ids::RequestId;
use rllm_core::request::SamplingParams;
use rllm_engine::{EngineCore, LLMEngine};
use rllm_sampling::{Sampler, SamplingInput};

// ── Scheduler benchmarks ──────────────────────────────────────────────────

fn bench_scheduler_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_prefill");
    for prompt_len in [32, 128, 512, 1024] {
        group.bench_with_input(
            BenchmarkId::from_parameter(prompt_len),
            &prompt_len,
            |b, &prompt_len| {
                b.iter(|| {
                    let mut sched = make_test_scheduler(16, 4096, 64, 8192);
                    for _ in 0..32 {
                        sched.add_request(make_inference_request(prompt_len, 32));
                    }
                    sched.step()
                })
            },
        );
    }
    group.finish();
}

fn bench_scheduler_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_decode");
    for num_running in [8, 32, 64] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_running),
            &num_running,
            |b, &num_running| {
                b.iter(|| {
                    let mut sched = make_test_scheduler(16, 4096, 128, 8192);
                    for _ in 0..num_running {
                        sched.add_request(make_inference_request(32, 32));
                    }
                    // First step: prefill all.
                    let _ = sched.step();
                    // Second step: decode all.
                    sched.step()
                })
            },
        );
    }
    group.finish();
}

fn bench_scheduler_mixed_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_mixed_batch");
    group.bench_function("16_running_16_new", |b| {
        b.iter(|| {
            let mut sched = make_test_scheduler(16, 4096, 64, 8192);
            // Add 16 "already running" requests.
            for _ in 0..16 {
                sched.add_request(make_inference_request(32, 32));
            }
            let _ = sched.step();
            // Add 16 new requests while 16 are running.
            for _ in 0..16 {
                sched.add_request(make_inference_request(64, 32));
            }
            sched.step()
        })
    });
    group.finish();
}

// ── KV Cache benchmarks ───────────────────────────────────────────────────

fn bench_kv_cache_allocate(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_allocate");
    for num_tokens in [16, 64, 256, 1024] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_tokens),
            &num_tokens,
            |b, &num_tokens| {
                b.iter(|| {
                    let mut mgr = make_test_kv_cache_manager(16, 4096);
                    let rid = RequestId::new();
                    mgr.allocate_slots(rid, num_tokens, 0);
                    mgr.free(rid);
                })
            },
        );
    }
    group.finish();
}

fn bench_kv_cache_prefix_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_prefix_lookup");
    for prefix_len in [16, 64, 256] {
        group.bench_with_input(
            BenchmarkId::from_parameter(prefix_len),
            &prefix_len,
            |b, &prefix_len| {
                b.iter(|| {
                    let mut mgr =
                        rllm_bench::helpers::make_test_kv_cache_manager_with_prefix(16, 4096, true);
                    // Cache a prefix.
                    let rid1 = RequestId::new();
                    let tokens: Vec<u32> = (0..prefix_len as u32).collect();
                    mgr.allocate_slots(rid1, prefix_len, 0);
                    mgr.cache_blocks(rid1, prefix_len);
                    mgr.free(rid1);
                    // Look up.
                    let rid2 = RequestId::new();
                    mgr.get_computed_blocks(rid2, &tokens, false)
                })
            },
        );
    }
    group.finish();
}

// ── Sampling benchmarks ───────────────────────────────────────────────────

fn bench_sampling_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_greedy");
    for vocab_size in [1000, 8000, 32000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &vocab_size| {
                let mut sampler = Sampler::from_seed(42);
                let logits = vec![0.0f32; vocab_size];
                let params = SamplingParams { temperature: 0.0, ..Default::default() };
                b.iter(|| {
                    let input = SamplingInput {
                        logits: logits.clone(),
                        params: params.clone(),
                        context_token_ids: vec![],
                        num_generated: 0,
                        eos_token_id: vocab_size as u32 - 1,
                        bad_word_token_ids: vec![],
                    };
                    sampler.sample(&input)
                })
            },
        );
    }
    group.finish();
}

fn bench_sampling_stochastic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_stochastic");
    for vocab_size in [1000, 32000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &vocab_size| {
                let mut sampler = Sampler::from_seed(42);
                let logits = vec![0.0f32; vocab_size];
                let params = SamplingParams {
                    temperature: 1.0,
                    top_p: 0.9,
                    top_k: 50,
                    ..Default::default()
                };
                b.iter(|| {
                    let input = SamplingInput {
                        logits: logits.clone(),
                        params: params.clone(),
                        context_token_ids: vec![],
                        num_generated: 0,
                        eos_token_id: vocab_size as u32 - 1,
                        bad_word_token_ids: vec![],
                    };
                    sampler.sample(&input)
                })
            },
        );
    }
    group.finish();
}

// ── Engine end-to-end benchmarks ──────────────────────────────────────────

fn bench_engine_step_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_step_prefill");
    for num_requests in [1, 8, 32] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_requests),
            &num_requests,
            |b, &num_requests| {
                b.iter(|| {
                    let config = MockExecutorConfig {
                        mode: MockMode::Deterministic { offset: 0 },
                        vocab_size: 32000,
                        eos_token_id: 31999,
                        sampler_seed: Some(42),
                    };
                    let mock = MockExecutor::new(config);
                    let scheduler = make_test_scheduler(16, 4096, 64, 8192);
                    let core = EngineCore::new(Box::new(mock), scheduler, 31999);
                    let mut engine = LLMEngine::new(core);

                    for _ in 0..num_requests {
                        engine.add_request(make_inference_request(64, 32)).unwrap();
                    }
                    engine.step()
                })
            },
        );
    }
    group.finish();
}

fn bench_engine_step_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_step_decode");
    for num_requests in [1, 8, 32] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_requests),
            &num_requests,
            |b, &num_requests| {
                b.iter(|| {
                    let config = MockExecutorConfig {
                        mode: MockMode::Deterministic { offset: 0 },
                        vocab_size: 32000,
                        eos_token_id: 31999,
                        sampler_seed: Some(42),
                    };
                    let mock = MockExecutor::new(config);
                    let scheduler = make_test_scheduler(16, 4096, 64, 8192);
                    let core = EngineCore::new(Box::new(mock), scheduler, 31999);
                    let mut engine = LLMEngine::new(core);

                    for _ in 0..num_requests {
                        engine.add_request(make_inference_request(32, 32)).unwrap();
                    }
                    // Prefill step.
                    let _ = engine.step();
                    // Decode step (the one we actually measure).
                    engine.step()
                })
            },
        );
    }
    group.finish();
}

fn bench_engine_generate_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_generate_throughput");
    for concurrency in [1, 8, 32, 64] {
        let num_requests = concurrency;
        group.throughput(Throughput::Elements(num_requests as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let config = MockExecutorConfig {
                        mode: MockMode::Deterministic { offset: 0 },
                        vocab_size: 32000,
                        eos_token_id: 31999,
                        sampler_seed: Some(42),
                    };
                    let mock = MockExecutor::new(config);
                    let scheduler = make_test_scheduler(16, 4096, 128, 8192);
                    let core = EngineCore::new(Box::new(mock), scheduler, 31999);
                    let mut engine = LLMEngine::new(core);

                    let workload_config = WorkloadConfig {
                        num_requests: concurrency,
                        input_lengths: LengthDistribution::Fixed(64),
                        output_lengths: LengthDistribution::Fixed(16),
                        concurrency,
                        vocab_size: 32000,
                        seed: 42,
                    };
                    let workload = SyntheticWorkload::generate(&workload_config);
                    engine.generate(workload.requests).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scheduler_prefill,
    bench_scheduler_decode,
    bench_scheduler_mixed_batch,
    bench_kv_cache_allocate,
    bench_kv_cache_prefix_lookup,
    bench_sampling_greedy,
    bench_sampling_stochastic,
    bench_engine_step_prefill,
    bench_engine_step_decode,
    bench_engine_generate_throughput,
);

criterion_main!(benches);
