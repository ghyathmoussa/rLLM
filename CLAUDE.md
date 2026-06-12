# CLAUDE.md

Guidance for working in this repository.

## Project

rLLM is a Rust LLM inference engine inspired by vLLM. It serves Llama-family,
decoder-only causal LMs over an OpenAI-compatible HTTP API, with PagedAttention,
continuous batching, prefix caching, SSE streaming, and CUDA acceleration via
Candle. Single-binary deployment, MSRV 1.85, Rust edition 2024.

## Build, Test, Lint

```bash
cargo build --workspace                 # build all crates (CPU)
cargo build --workspace --features cuda # build with CUDA kernels
cargo test --workspace                  # run tests
cargo clippy --workspace --all-targets -- -D warnings  # CI fails on any warning
cargo fmt --all                         # rustfmt (nightly in CI, edition 2024)
```

CI (`.github/workflows/ci-main.yml`) runs four jobs on PRs to `main`: `fmt`
(nightly `--check`), `clippy` (`-D warnings`), `test`, and `build`. Match all
four before considering work done. Lint config: `clippy.toml`
(cognitive-complexity-threshold = 50), `rustfmt.toml` (max_width 100,
`imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`).

## Running

```bash
# Serve a model (CPU)
cargo run --release -- serve meta-llama/Llama-3.2-1B-Instruct
# Serve with CUDA
cargo run --release --features cuda -- serve meta-llama/Llama-3.2-1B-Instruct --dtype bf16
```

The only CLI subcommand is `serve` (see `Cli` in `rllm-server`). `src/main.rs`
is a thin entry point that parses the CLI and calls `rllm_server::server::serve`.
Examples live in `examples/` (`offline_generate.rs`, `server.rs`).

## Architecture

Workspace of 14 crates under `crates/`. The dependency direction is **strict and
must be preserved** — model code never depends on server/scheduler/engine:

```
server -> engine -> executor -> worker -> {model, kernels, cache, sampling}
```

| Crate | Responsibility |
|-------|----------------|
| `rllm-core` | Configs, errors, IDs, request/response types, sequence state |
| `rllm-tokenizer` | HF tokenizer, chat templates, streaming detokenization |
| `rllm-model` | Model registry, HF config parsing, Llama impl, weight loading |
| `rllm-tensor` | Thin device-tensor abstraction (dtype, shape, CUDA streams) |
| `rllm-kernels` | CUDA kernels + FFI: paged attention, cache ops, RMSNorm, RoPE |
| `rllm-cache` | KV cache specs, block pool, prefix hashing, allocation, eviction |
| `rllm-scheduler` | Waiting/running queues, FCFS, continuous batching, chunked prefill, preemption |
| `rllm-sampling` | Logits processors, top-k/p/min-p, penalties, logprobs |
| `rllm-worker` | Device worker, model runner, batch builder, CUDA graph capture |
| `rllm-executor` | Single-process executor (later multi-process / tensor-parallel) |
| `rllm-engine` | `EngineCore`, sync/async engines, request lifecycle, output processing |
| `rllm-server` | Axum HTTP server, OpenAI API, SSE streaming, CLI, health, metrics |
| `rllm-metrics` | Prometheus (+ optional OpenTelemetry) metrics |
| `rllm-bench` | Offline/online benchmarks: latency, throughput, TTFT, TPOT |

See `docs/architecture.md` for the full runtime flow (request → input processing
→ scheduler → executor → worker forward → sampler → output processor).

## Feature Flags

- `cuda` — CUDA device support and kernels (propagates to tensor/kernels/worker).
- `candle-backend` — use Candle for model/layer bootstrap.

Default build is CPU-only with no features.

## Conventions

- Keep the crate dependency invariant above intact; if a change needs model code
  to reach into engine/server, the design is wrong.
- Custom CUDA kernels are invoked via FFI for hot paths (PagedAttention, fused
  norms, RoPE); Candle handles safe tensor ops and model bootstrap.
- `cargo deny` config is in `deny.toml`; API contract is in `openapi.yaml`.
- Prometheus metrics are exported at `/metrics`; a Grafana dashboard is in
  `grafana_dashboard.json`. Load testing: `scripts/concurrency_test.py`.
- If you strguled in any step, do not hallucinate and make wrong decisions, write what you think and we can discuss togather.

## Main Goal

- Create inference engine that support many models and many platforms (Nvidia RTX series, Nvidia Triton, Nvidia V100, Nvidia H100, Nvidia H200, Nvidia A100,... etc all Nvidia series)
- Support all kernals in Linux first then we will support MacOS and Windows.
- Support All quantization methods, we will start from int8 and we will implement all quantization algorithms.
- Make sure that the process is implemented very well and verifying by testing, if the platform of test does not support the process say that we want to test the process in requisted platform and write the commands that we need.
- Before you running out of limit write what you are thinking in Markdown file so we can resume from where you stopped.
- Every process, methods, and workflows are inspired from vLLM projetc, so make it as reference when I ask you to do somthing