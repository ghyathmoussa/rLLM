<p align="center">
  <img src="docs/images/logo.png" height="80" alt="rLLM Logo">
</p>

<p align="center">
  <strong>A modern, fast, and lightweight Large Language Model (LLM) inference engine built in Rust.</strong>
</p>

<p align="center">
  <a href="https://github.com/ghyathmoussa/rLLM/actions"><img src="https://img.shields.io/github/actions/workflow/status/ghyathmoussa/rLLM/ci-main.yml?style=flat-square" alt="CI"></a>
  <a href="https://crates.io/crates/rllm"><img src="https://img.shields.io/crates/v/rllm?style=flat-square" alt="crates.io"></a>
  <a href="https://github.com/ghyathmoussa/rLLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License"></a>
  <a href="https://rust-lang.github.io/rustup/"><img src="https://img.shields.io/badge/MSRV-1.85-orange?style=flat-square" alt="MSRV"></a>
</p>

---

## Overview

rLLM is an LLM inference engine built in Rust. It leverages Rust's memory safety guarantees, zero-cost abstractions, and freedom from the Python GIL to deliver predictable performance and a single-binary deployment.

What rLLM provides:
- OpenAI-compatible HTTP API (drop-in replacement for OpenAI clients)
- PagedAttention for efficient KV cache management
- Continuous batching for high throughput
- Prefix caching for shared prompt prefixes
- Token streaming via Server-Sent Events
- CUDA-accelerated inference via Candle framework
- Prometheus metrics for monitoring
- Full sampling options (top-k, top-p, min-p, temperature, penalties)
- Llama-family model support

---

## Key Features

- **OpenAI-compatible API** – Drop-in replacement for existing OpenAI client code
- **PagedAttention** – Efficient memory management for KV cache, enabling larger batch sizes
- **Continuous batching** – Dynamically add/remove requests between iterations
- **Prefix caching** – Automatic caching and reuse of common prompt prefixes
- **Streaming** – Token-by-token streaming via Server-Sent Events (SSE)
- **CUDA acceleration** – GPU-accelerated inference via the Candle ML framework
- **Prometheus metrics** – Built-in monitoring: TTFT, TPOT, request rate, token throughput
- **Rich sampling** – Temperature, top-k, top-p, min-p, frequency/presence penalties, logit bias
- **Llama support** – Optimized for Llama-family architectures (Llama 2/3, Mistral, etc.)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ghyathmoussa/rLLM.git
cd rLLM

# Option A: Serve on CPU
cargo run --release -- serve meta-llama/Llama-3.2-1B-Instruct

# Option B: Serve on GPU (CUDA accelerated)
cargo run --release --features cuda -- serve meta-llama/Llama-3.2-1B-Instruct --dtype bf16

# In another terminal, send a request
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"meta-llama/Llama-3.2-1B-Instruct",
    "messages":[{"role":"user","content":"Say hello"}],
    "max_tokens":16,
    "temperature":0
  }'
```

---

## Requirements

- **Rust** 1.85 or newer
- **NVIDIA GPU** with compute capability 7.5+ (Turing or newer; e.g. RTX 20xx,
  Titan RTX, A100, H100). Volta `sm_70` is no longer targeted, as CUDA 13 drops it.
- **CUDA** 12.x toolkit (`nvcc` on `PATH`). **CUDA 13.x is not yet supported** —
  the pinned `cudarc`/`candle` versions only ship bindings up to CUDA 13.2, and a
  13.3 toolkit makes the build fail with `Unsupported cuda toolkit version: 13.3`.
  Install CUDA 12.x (e.g. 12.8) and point `CUDA_HOME`/`PATH` at it; your GPU driver
  can stay newer (drivers are backward compatible).
- **Linux** (Ubuntu 22.04+, RHEL/AlmaLinux/Rocky 8+, CentOS 7+)

> **Building CUDA kernels for a specific GPU:** by default kernels are compiled for
> `sm_75`–`sm_90`. To build only for your card (faster compile) set `CUDA_ARCH`,
> e.g. `export CUDA_ARCH=7.5` for a Titan RTX before `cargo build --features cuda`.

> **Pre-built release binaries** are statically linked (`x86_64-unknown-linux-musl`)
> and **CPU-only** — they run on any Linux without a glibc dependency, but do not use
> the GPU. For GPU inference, build from source with `--features cuda`.

---

## Installation

### Cargo Installation (From Source)

You can compile and install the CLI tool directly to your system path:

* **For CPU execution:**
  ```bash
  cargo install --path .
  ```

* **For GPU execution (with CUDA acceleration):**
  ```bash
  cargo install --path . --features cuda
  ```

### Docker Installation

We provide multi-stage, optimized Docker builds for CPU-only and GPU-accelerated environments:

* **For CPU-only deployment:**
  ```bash
  # Build the container
  docker build -t rllm:latest .

  # Run the container
  docker run -d -p 8000:8000 --name rllm rllm:latest serve --model meta-llama/Llama-3.2-1B-Instruct
  ```

* **For GPU/CUDA deployment:**
  ```bash
  # Build the CUDA container
  docker build -f Dockerfile.cuda -t rllm:cuda .

  # Run the container with GPU access (requires NVIDIA Container Toolkit)
  docker run --gpus all -d -p 8000:8000 --name rllm-cuda rllm:cuda serve --model meta-llama/Llama-3.2-1B-Instruct
  ```

### Local model

* **For CPU:**
  ```bash
  cargo run --release -- serve /path/to/model-directory
  ```

* **For GPU:**
  ```bash
  cargo run --release --features cuda -- serve /path/to/model-directory --dtype bf16
  ```

### Hugging Face models

* **For CPU:**
  ```bash
  # Public models
  cargo run --release -- serve meta-llama/Llama-3.2-1B-Instruct

  # Gated/private models (set HF_TOKEN)
  export HF_TOKEN=hf_xxxxxxxxxxxx
  cargo run --release -- serve meta-llama/Llama-3.2-1B-Instruct
  ```

* **For GPU:**
  ```bash
  # Public models
  cargo run --release --features cuda -- serve meta-llama/Llama-3.2-1B-Instruct --dtype bf16

  # Gated/private models (set HF_TOKEN)
  export HF_TOKEN=hf_xxxxxxxxxxxx
  cargo run --release --features cuda -- serve meta-llama/Llama-3.2-1B-Instruct --dtype bf16
  ```

---

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `model` | (required) | Hugging Face model ID or local path |
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to bind to |
| `--dtype` | `auto` | Data type (`auto`, `fp16`, `bf16`, `fp32`) |
| `--max-model-len` | (auto) | Maximum model context length |
| `--max-num-seqs` | `256` | Maximum concurrent sequences |
| `--max-num-batched-tokens` | `4096` | Maximum batched tokens per step |
| `--gpu-memory-utilization` | `0.9` | GPU memory target (0.01-1.0) |
| `--enable-prefix-caching` | `true` | Enable prefix caching |
| `--api-key` | (none) | API key for authenticated endpoints (env: `RLLM_API_KEY`) |
| `--cors-allowed-origins` | `*` | CORS allowed origins (comma-separated) |
| `--enable-debug-endpoints` | `false` | Enable `/debug/model` endpoint |
| `--max-input-messages` | `256` | Maximum messages in chat requests |
| `--max-input-chars` | `1000000` | Maximum characters in request body |
| `--max-concurrent-requests` | `64` | Maximum concurrent inference requests |
| `--request-timeout-secs` | `120` | Request timeout in seconds |
| `--log-level` | `info` | Log level (`trace`, `debug`, `info`, `warn`, `error`) |

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completions |
| `POST` | `/v1/completions` | Text completions |

### Streaming example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"meta-llama/Llama-3.1-8B-Instruct",
    "messages":[{"role":"user","content":"Count to 5"}],
    "stream": true,
    "max_tokens": 32
  }'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        rLLM                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │ rllm-    │  │ rllm-    │  │ rllm-    │  │ rllm-   ││
│  │ server   │  │ engine   │  │ executor │  │ worker  ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘│
│       │              │              │              │    │
│  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  ┌────┴────┐│
│  │ rllm-    │  │ rllm-    │  │ rllm-    │  │ rllm-   ││
│  │ core     │  │ scheduler│  │ cache    │  │ sampling││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ rllm-    │  │ rllm-    │  │ rllm-    │              │
│  │ tokenizer│  │ tensor   │  │ kernels  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

| Crate | Description |
|-------|-------------|
| `rllm-server` | Axum HTTP server with OpenAI-compatible API |
| `rllm-engine` | Async inference engine with background task loop |
| `rllm-executor` | Orchestrates model execution and sampling |
| `rllm-worker` | Model runner: tensor construction, KV cache ops |
| `rllm-core` | Core types: config, requests, outputs, IDs |
| `rllm-scheduler` | Scheduler: FCFS with continuous batching |
| `rllm-cache` | Prefix-aware KV cache manager |
| `rllm-sampling` | Token sampling: top-k, top-p, penalties, logprobs |
| `rllm-tokenizer` | Async tokenizer pool (HuggingFace tokenizers) |
| `rllm-tensor` | Tensor types and pinned buffer utilities |
| `rllm-kernels` | CUDA kernels: PagedAttention, cache ops |
| `rllm-metrics` | Prometheus metrics recording and description |
| `rllm-model` | Model config loading, weight loading from disk/HF |
| `rllm-bench` | Benchmarking tools and serve client |

---

## Supported Models

| Model                   | Architecture         | Status            |
|-------------------------|----------------------|-------------------|
| Llama 2 7B/13B/70B      | `LlamaForCausalLM`   | ✅ Supported      |
| Llama 3 8B/70B           | `LlamaForCausalLM`   | ✅ Supported      |
| Llama 3.1 8B/70B/405B   | `LlamaForCausalLM`   | ✅ Supported      |
| Mistral 7B               | `LlamaForCausalLM`   | ✅ Supported      |
| Mixtral 8x7B             | `LlamaForCausalLM`   | ✅ Supported      |
| CodeLlama 7B/13B/34B     | `LlamaForCausalLM`   | ✅ Supported      |
| Other Llama-family models | `LlamaForCausalLM`  | ⚠️ Compatible    |

---

## Metrics

rLLM exports the following Prometheus metrics on the `/metrics` endpoint:

| Metric | Type | Description |
|--------|------|-------------|
| `rllm_requests_total` | Counter | Total requests received |
| `rllm_requests_finished_total` | Counter | Finished requests |
| `rllm_generated_tokens_total` | Counter | Generated tokens |
| `rllm_prompt_tokens_total` | Counter | Prompt tokens processed |
| `rllm_http_requests_total` | Counter | HTTP requests received |
| `rllm_http_request_duration_seconds` | Histogram | HTTP request latency |
| `rllm_ttft_seconds` | Histogram | Time to first token |
| `rllm_tpot_seconds` | Histogram | Time per output token |
| `rllm_e2e_latency_seconds` | Histogram | End-to-end request latency |
| `rllm_sampling_duration_seconds` | Histogram | Sampling step duration |
| `rllm_tokens_per_second` | Histogram | Token generation throughput |

## Benchmarking & Concurrency Testing

### Concurrency Load Generator (Python Script)
A helper script is provided in `scripts/concurrency_test.py` to test the concurrency and throughput of the running server without external dependencies:

```bash
# Run with 10 concurrent clients submitting 50 total requests
./scripts/concurrency_test.py --concurrency 10 --total-requests 50 --max-tokens 32

# Run in streaming mode to calculate Time-to-First-Token (TTFT)
./scripts/concurrency_test.py --concurrency 20 --total-requests 100 --max-tokens 64 --stream
```

### Rust Benchmarking Harness (rllm-bench)
For local and simulated offline benchmarks, you can use the built-in `rllm-bench` tool:

```bash
# Run offline benchmark with 100 requests
cargo run --release -p rllm-bench -- offline --num-requests 100 --concurrency 32
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`cargo test --workspace`)
5. Run clippy (`cargo clippy --workspace`)
6. Commit and push
7. Open a Pull Request

---

## License

rLLM is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) – Rust ML framework with CUDA support
- [Axum](https://github.com/tokio-rs/axum) – ergonomic HTTP framework
- [Hugging Face](https://huggingface.co/) – model hub and tokenizers library
