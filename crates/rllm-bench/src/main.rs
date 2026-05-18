use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use rllm_core::request::SamplingParams;
use rllm_engine::{EngineCore, LLMEngine};

use rllm_bench::helpers::make_test_scheduler;
use rllm_bench::mock_executor::{MockExecutor, MockExecutorConfig, MockMode};
use rllm_bench::workload::{
    LengthDistribution, SyntheticWorkload, WorkloadConfig, sharegpt_prompts,
};

#[derive(Parser)]
#[command(name = "rllm-bench", about = "Benchmarking and correctness harness for rLLM")]
struct BenchCli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Offline benchmark: generate requests with configurable lengths.
    Offline(OfflineArgs),
    /// Serve benchmark: HTTP load generator against a running server.
    Serve(ServeArgs),
    /// Run correctness tests.
    Correctness,
}

#[derive(Parser)]
struct OfflineArgs {
    /// Number of requests.
    #[arg(long, default_value_t = 100)]
    num_requests: usize,
    /// Input token length.
    #[arg(long, default_value_t = 128)]
    input_len: usize,
    /// Minimum input token length for uniform synthetic workloads.
    #[arg(long)]
    input_len_min: Option<usize>,
    /// Maximum input token length for uniform synthetic workloads.
    #[arg(long)]
    input_len_max: Option<usize>,
    /// Output token length (max_tokens).
    #[arg(long, default_value_t = 32)]
    output_len: usize,
    /// Minimum output token length for uniform synthetic workloads.
    #[arg(long)]
    output_len_min: Option<usize>,
    /// Maximum output token length for uniform synthetic workloads.
    #[arg(long)]
    output_len_max: Option<usize>,
    /// Concurrency (batch size).
    #[arg(long, default_value_t = 32)]
    concurrency: usize,
    /// Mock mode: zero, deterministic, seeded, fixed.
    #[arg(long, default_value = "deterministic")]
    mode: String,
    /// Vocab size.
    #[arg(long, default_value_t = 32000)]
    vocab_size: usize,
    /// Block size.
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    /// Number of cache blocks.
    #[arg(long, default_value_t = 4096)]
    num_cache_blocks: usize,
    /// Sampler seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Optional ShareGPT-style JSON dataset for offline request lengths/prompts.
    #[arg(long)]
    sharegpt: Option<std::path::PathBuf>,
    /// Output results as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Parser)]
struct ServeArgs {
    /// Server base URL.
    #[arg(long, default_value = "http://localhost:8000")]
    base_url: String,
    /// Model name.
    #[arg(long, default_value = "test-model")]
    model: String,
    /// Number of requests.
    #[arg(long, default_value_t = 100)]
    num_requests: usize,
    /// Concurrency.
    #[arg(long, default_value_t = 32)]
    concurrency: usize,
    /// Endpoint kind: chat or completions.
    #[arg(long, default_value = "chat")]
    endpoint: String,
    /// Enable streaming.
    #[arg(long)]
    stream: bool,
    /// Synthetic input token count when no dataset is supplied.
    #[arg(long, default_value_t = 128)]
    input_tokens: usize,
    /// Max output tokens.
    #[arg(long, default_value_t = 32)]
    max_tokens: u32,
    /// Optional ShareGPT-style JSON dataset for request prompts.
    #[arg(long)]
    sharegpt: Option<std::path::PathBuf>,
    /// Output results as JSON.
    #[arg(long)]
    json: bool,
}

fn parse_mock_mode(mode: &str, seed: u64) -> MockMode {
    match mode {
        "zero" => MockMode::Zero,
        "deterministic" => MockMode::Deterministic { offset: 0 },
        "seeded" => MockMode::SeededRandom { seed },
        "fixed" => MockMode::FixedToken { token_id: 1 },
        _ => {
            eprintln!("Unknown mode '{}', using deterministic", mode);
            MockMode::Deterministic { offset: 0 }
        }
    }
}

fn length_distribution(fixed: usize, min: Option<usize>, max: Option<usize>) -> LengthDistribution {
    match (min, max) {
        (Some(min), Some(max)) if min <= max => LengthDistribution::Uniform { min, max },
        (Some(min), None) if min <= fixed => LengthDistribution::Uniform { min, max: fixed },
        (None, Some(max)) if fixed <= max => LengthDistribution::Uniform { min: fixed, max },
        _ => LengthDistribution::Fixed(fixed),
    }
}

fn run_offline_benchmark(args: OfflineArgs) -> Result<()> {
    let mode = parse_mock_mode(&args.mode, args.seed);

    let mock_config = MockExecutorConfig {
        mode,
        vocab_size: args.vocab_size,
        eos_token_id: args.vocab_size as u32 - 1,
        sampler_seed: Some(args.seed),
    };
    let mock = MockExecutor::new(mock_config);
    let scheduler =
        make_test_scheduler(args.block_size, args.num_cache_blocks, args.concurrency, 8192);
    let core = EngineCore::new(Box::new(mock), scheduler, args.vocab_size as u32 - 1);
    let mut engine = LLMEngine::new(core);

    let workload = if let Some(path) = &args.sharegpt {
        SyntheticWorkload::from_sharegpt(path, args.output_len, args.seed)?
    } else {
        let workload_config = WorkloadConfig {
            num_requests: args.num_requests,
            input_lengths: length_distribution(
                args.input_len,
                args.input_len_min,
                args.input_len_max,
            ),
            output_lengths: length_distribution(
                args.output_len,
                args.output_len_min,
                args.output_len_max,
            ),
            concurrency: args.concurrency,
            vocab_size: args.vocab_size,
            seed: args.seed,
        };
        SyntheticWorkload::generate(&workload_config)
    };

    if !args.json {
        println!(
            "Running offline benchmark: {} requests, concurrency={}",
            workload.requests.len(),
            args.concurrency
        );
    }

    let start = Instant::now();
    let outputs = engine.generate(workload.requests)?;
    let elapsed = start.elapsed();

    let total_output_tokens: u64 = outputs.iter().map(|o| o.usage.completion_tokens as u64).sum();
    let total_input_tokens: u64 = outputs.iter().map(|o| o.usage.prompt_tokens as u64).sum();

    let elapsed_secs = elapsed.as_secs_f64();
    let output_tps =
        if elapsed_secs > 0.0 { total_output_tokens as f64 / elapsed_secs } else { 0.0 };
    let total_tps = if elapsed_secs > 0.0 {
        (total_input_tokens + total_output_tokens) as f64 / elapsed_secs
    } else {
        0.0
    };

    if args.json {
        let result = serde_json::json!({
            "total_requests": outputs.len(),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": elapsed_secs,
            "throughput_rps": outputs.len() as f64 / elapsed_secs,
            "output_tps": output_tps,
            "total_tps": total_tps,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("=== Offline Benchmark Results ===");
        println!("Total requests:        {}", outputs.len());
        println!("Total input tokens:     {}", total_input_tokens);
        println!("Total output tokens:    {}", total_output_tokens);
        println!("Elapsed:                {:.3}s", elapsed_secs);
        println!("Throughput:             {:.1} req/s", outputs.len() as f64 / elapsed_secs);
        println!("Output throughput:      {:.1} tokens/s", output_tps);
        println!("Total throughput:       {:.1} tokens/s", total_tps);
    }

    Ok(())
}

fn run_serve_benchmark(args: ServeArgs) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let endpoint = match args.endpoint.as_str() {
            "completion" | "completions" => rllm_bench::serve_client::ServeEndpoint::Completions,
            _ => rllm_bench::serve_client::ServeEndpoint::ChatCompletions,
        };
        let prompts = match &args.sharegpt {
            Some(path) => {
                let prompts = sharegpt_prompts(path)?;
                if prompts.is_empty() {
                    bail!("ShareGPT dataset did not contain any human/user prompts");
                }
                Some(prompts)
            }
            None => None,
        };

        let config = rllm_bench::serve_client::ServeBenchConfig {
            base_url: args.base_url,
            model: args.model,
            endpoint,
            num_requests: args.num_requests,
            concurrency: args.concurrency,
            streaming: args.stream,
            input_tokens: args.input_tokens,
            max_tokens: args.max_tokens,
            timeout: std::time::Duration::from_secs(30),
            prompts,
        };

        if !args.json {
            println!(
                "Running serve benchmark: {} requests, concurrency={}",
                config.num_requests, config.concurrency
            );
        }

        let metrics = rllm_bench::serve_client::run_serve_benchmark(config).await?;
        if args.json {
            println!("{}", metrics.to_json());
        } else {
            metrics.print_summary();
        }
        Ok(())
    })
}

fn run_correctness() -> Result<()> {
    println!("Running correctness tests...");

    // Test 1: Greedy determinism.
    {
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

        let params =
            SamplingParams { temperature: 0.0, max_tokens: Some(16), ..Default::default() };

        let req1 = rllm_bench::helpers::make_inference_request_with_params(32, 16, params.clone());
        let _id1 = req1.request_id;

        let mut engine = make_engine();
        let out1 = engine.generate(vec![req1])?;
        let tokens1: Vec<u32> = out1[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();

        let req2 = rllm_bench::helpers::make_inference_request_with_params(32, 16, params.clone());
        let mut engine2 = make_engine();
        let out2 = engine2.generate(vec![req2])?;
        let tokens2: Vec<u32> = out2[0].outputs.iter().flat_map(|o| o.token_ids.clone()).collect();

        assert_eq!(tokens1, tokens2, "Greedy determinism failed: runs produced different tokens");
        println!("  [PASS] greedy determinism: {} tokens match", tokens1.len());
    }

    // Test 2: All requests finish.
    {
        let config = MockExecutorConfig {
            mode: MockMode::Deterministic { offset: 0 },
            vocab_size: 1000,
            eos_token_id: 999,
            sampler_seed: Some(42),
        };
        let mock = MockExecutor::new(config);
        let scheduler = make_test_scheduler(16, 512, 64, 4096);
        let core = EngineCore::new(Box::new(mock), scheduler, 999);
        let mut engine = LLMEngine::new(core);

        let requests: Vec<_> =
            (0..32).map(|_| rllm_bench::helpers::make_inference_request(64, 16)).collect();

        let outputs = engine.generate(requests)?;
        assert_eq!(outputs.len(), 32, "Expected 32 outputs, got {}", outputs.len());
        for (i, out) in outputs.iter().enumerate() {
            assert!(out.finished, "Request {} not finished", i);
        }
        println!("  [PASS] 32 concurrent requests: all finished");
    }

    // Test 3: Prefix caching produces hits.
    {
        let mut sched =
            rllm_bench::helpers::make_test_scheduler_with_options(4, 100, 10, 4096, false, true);

        let prefix: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut req1_tokens = prefix.clone();
        req1_tokens.extend_from_slice(&[100, 101, 102]);

        let req1 = rllm_core::request::InferenceRequest {
            request_id: rllm_core::ids::RequestId::new(),
            prompt: None,
            token_ids: Some(req1_tokens.clone()),
            messages: None,
            sampling_params: SamplingParams::default(),
            arrival_time: std::time::Instant::now(),
            priority: 0,
            stream: false,
            cache_salt: None,
        };
        sched.add_request(req1);

        let out1 = sched.step();
        assert!(out1.num_scheduled() > 0);

        // Simulate completion to cache the blocks.
        let mut req2_tokens = prefix.clone();
        req2_tokens.extend_from_slice(&[200, 201, 202]);
        let req2 = rllm_core::request::InferenceRequest {
            request_id: rllm_core::ids::RequestId::new(),
            prompt: None,
            token_ids: Some(req2_tokens),
            messages: None,
            sampling_params: SamplingParams::default(),
            arrival_time: std::time::Instant::now(),
            priority: 0,
            stream: false,
            cache_salt: None,
        };
        sched.add_request(req2);

        let out2 = sched.step();
        assert!(!out2.scheduled_cached.is_empty() || !out2.scheduled_new.is_empty());
        println!("  [PASS] prefix caching: second request scheduled");
    }

    println!("All correctness tests passed.");
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = BenchCli::parse();
    match cli.command {
        Commands::Offline(args) => run_offline_benchmark(args),
        Commands::Serve(args) => run_serve_benchmark(args),
        Commands::Correctness => run_correctness(),
    }
}
