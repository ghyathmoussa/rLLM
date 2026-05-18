use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main, black_box};

use rllm_bench::helpers::{
    make_inference_request, make_test_kv_cache_manager, make_test_scheduler,
};
use rllm_bench::mock_executor::{MockExecutor, MockExecutorConfig, MockMode};
use rllm_bench::workload::{LengthDistribution, SyntheticWorkload, WorkloadConfig};
use rllm_core::ids::RequestId;
use rllm_core::request::SamplingParams;
use rllm_engine::{EngineCore, LLMEngine};
use rllm_kernels::{AttentionMetadata, AttentionParams};
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

// ── PagedAttention Kernel Benchmarks ─────────────────────────────────────

fn bench_paged_attention_decode_by_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attention_decode");
    let params = AttentionParams::new(32, 8, 128, 16);
    for context_len in [128, 512, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::from_parameter(context_len),
            &context_len,
            |b, &context_len| {
                let num_blocks = params.num_blocks_for_seq_len(context_len);
                let seq_lens: Vec<u32> = vec![context_len as u32];
                let block_tables: Vec<Vec<i32>> = vec![vec![0; num_blocks]];
                let meta = AttentionMetadata::for_decode(seq_lens, block_tables, num_blocks);
                let flat = meta.flatten_block_tables();
                b.iter(|| {
                    let _ = black_box(&meta);
                    let _ = black_box(&flat);
                });
            },
        );
    }
    group.finish();
}

fn bench_paged_attention_prefill_by_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attention_prefill");
    let params = AttentionParams::new(32, 8, 128, 16);
    for context_len in [128, 512, 1024] {
        group.bench_with_input(
            BenchmarkId::from_parameter(context_len),
            &context_len,
            |b, &context_len| {
                let num_blocks = params.num_blocks_for_seq_len(context_len);
                let seq_lens: Vec<u32> = vec![context_len as u32];
                let block_tables: Vec<Vec<i32>> = vec![vec![0; num_blocks]];
                let meta = AttentionMetadata::for_prefill(
                    seq_lens,
                    vec![context_len as u32],
                    block_tables,
                    num_blocks,
                );
                b.iter(|| {
                    let _ = black_box(&meta);
                });
            },
        );
    }
    group.finish();
}

fn bench_kv_cache_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_write");
    let num_tokens = [1, 16, 64, 256];
    for n in &num_tokens {
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            n,
            |b, &num_tokens| {
                // Simulate compute_slot_mappings overhead.
                let positions: Vec<u32> = (0..num_tokens as u32).collect();
                let block_table = vec![0u32; num_tokens.div_ceil(16)];
                let block_size = 16usize;
                b.iter(|| {
                    let slots: Vec<i64> = positions.iter().map(|&pos| {
                        let block_idx = pos as usize / block_size;
                        let offset = pos as usize % block_size;
                        if block_idx < block_table.len() {
                            block_table[block_idx] as i64 * block_size as i64 + offset as i64
                        } else {
                            -1
                        }
                    }).collect();
                    black_box(slots);
                });
            },
        );
    }
    group.finish();
}

fn bench_fused_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_rmsnorm");
    for hidden_size in [4096, 8192] {
        for num_rows in [1, 16, 64] {
            let n_elements = hidden_size * num_rows;
            group.bench_with_input(
                BenchmarkId::new(format!("h{}", hidden_size), num_rows),
                &n_elements,
                |b, &n_elements| {
                    let input = vec![1.0f32; n_elements];
                    let weight = vec![1.0f32; hidden_size];
                    b.iter(|| {
                        let mut output = vec![0.0f32; n_elements];
                        for row in 0..num_rows {
                            let start = row * hidden_size;
                            let row_input = &input[start..start + hidden_size];
                            let row_output = &mut output[start..start + hidden_size];
                            let mean_sq: f32 = row_input.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
                            let rms = 1.0 / (mean_sq + 1e-6).sqrt();
                            for (o, &i) in row_output.iter_mut().zip(row_input.iter()) {
                                *o = i * rms * weight[start % hidden_size];
                            }
                        }
                        black_box(output);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_fused_silu_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_silu_mul");
    for n_elements in [4096, 16384, 65536] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_elements),
            &n_elements,
            |b, &n_elements| {
                let gate: Vec<f32> = (0..n_elements).map(|i| (i as f32) / n_elements as f32).collect();
                let up: Vec<f32> = (0..n_elements).map(|i| 1.0 - (i as f32) / n_elements as f32).collect();
                b.iter(|| {
                    let mut output = vec![0.0f32; n_elements];
                    for i in 0..n_elements {
                        let silu = gate[i] / (1.0 + (-gate[i]).exp());
                        output[i] = silu * up[i];
                    }
                    black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn bench_fused_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_rope");
    for num_tokens in [1, 16, 64] {
        for head_dim in [64, 128] {
            group.bench_with_input(
                BenchmarkId::new(format!("d{}", head_dim), num_tokens),
                &num_tokens,
                |b, &num_tokens| {
                    let n_q_heads = 32;
                    let n_kv_heads = 8;
                    let q_size = num_tokens * n_q_heads * head_dim;
                    let k_size = num_tokens * n_kv_heads * head_dim;
                    let query: Vec<f32> = vec![1.0; q_size];
                    let key: Vec<f32> = vec![1.0; k_size];
                    let positions: Vec<i32> = (0..num_tokens as i32).collect();
                    let rope_theta = 10000.0f32;
                    b.iter(|| {
                        let mut out_q = query.clone();
                        let mut out_k = key.clone();
                        for t in 0..num_tokens {
                            let pos = positions[t] as f32;
                            for d in (0..head_dim).step_by(2) {
                                let freq = 1.0 / rope_theta.powf(d as f32 / head_dim as f32);
                                let angle = pos * freq;
                                let (sin, cos) = angle.sin_cos();
                                for h in 0..n_q_heads {
                                    let idx = t * n_q_heads * head_dim + h * head_dim + d;
                                    let x = query[idx];
                                    let y = query[idx + 1];
                                    out_q[idx] = x * cos - y * sin;
                                    out_q[idx + 1] = x * sin + y * cos;
                                }
                                for h in 0..n_kv_heads {
                                    let idx = t * n_kv_heads * head_dim + h * head_dim + d;
                                    let x = key[idx];
                                    let y = key[idx + 1];
                                    out_k[idx] = x * cos - y * sin;
                                    out_k[idx + 1] = x * sin + y * cos;
                                }
                            }
                        }
                        black_box((out_q, out_k));
                    });
                },
            );
        }
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
    bench_paged_attention_decode_by_context,
    bench_paged_attention_prefill_by_context,
    bench_kv_cache_write,
    bench_fused_rmsnorm,
    bench_fused_silu_mul,
    bench_fused_rope,
);

criterion_main!(benches);
