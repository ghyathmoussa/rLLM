# rLLM Architecture

rLLM is a Rust LLM inference engine inspired by [vLLM](https://github.com/vllm-project/vllm),
targeting NVIDIA CUDA with Llama-compatible decoder-only causal language models.

## Crate Graph

```
rllm-core  (configs, errors, IDs, request/response types, sequence state)
  |
  +-- rllm-tokenizer  (HF tokenizer, chat templates, streaming detokenization)
  +-- rllm-tensor     (device tensors, dtype, shape, CUDA streams)
  |     |
  |     +-- rllm-kernels  (CUDA kernels: paged attention, cache ops, RMSNorm, RoPE, fused)
  |
  +-- rllm-model      (model registry, HF config, weight loading, layer definitions)
  +-- rllm-cache      (KV cache specs, block pool, prefix hashing, eviction)
  |     |
  |     +-- rllm-scheduler  (waiting/running queues, FCFS/priority, token budgets, preemption)
  |
  +-- rllm-sampling   (logits processors, top-k/p, penalties, greedy/random, logprobs)
  |
  +-- rllm-worker     (device worker, model runner, batch builder, CUDA graph capture)
        |
        +-- rllm-executor  (single-process executor; later multi-process / tensor-parallel)
              |
              +-- rllm-engine  (EngineCore, sync/async engines, request lifecycle)
                    |
                    +-- rllm-server  (Axum HTTP, OpenAI-compatible API, SSE streaming)

rllm-bench  (offline/online benchmarks: latency, throughput, TTFT, TPOT)
```

### Dependency invariant

Model code does **not** depend on server, scheduler, or engine crates. The
dependency direction is strictly:

```
server -> engine -> executor -> worker -> {model, kernels, cache, sampling}
```

## Crate Responsibilities

| Crate | Responsibility |
|---|---|
| `rllm-core` | Common configs, errors, IDs, request/response types, sequence/request state, token accounting |
| `rllm-tokenizer` | HF tokenizer loading, sync/async tokenization, streaming detokenization, chat template rendering |
| `rllm-model` | Model registry, HF config parsing, Llama model implementation, weight loading |
| `rllm-tensor` | Thin abstraction over device tensors, dtype, shape, strides, CUDA streams |
| `rllm-kernels` | CUDA kernels and FFI wrappers for paged attention, cache ops, RMSNorm, RoPE, SiLU-mul |
| `rllm-cache` | KV cache specs, block pool, block tables, prefix cache hashing, allocation, eviction |
| `rllm-scheduler` | Waiting/running queues, FCFS/priority policy, token budget scheduling, chunked prefill |
| `rllm-sampling` | Logits processors, penalties, top-k/p/min-p, greedy/random sampling, logprobs |
| `rllm-worker` | Device worker, model runner, batch input builder, CUDA graph capture, profiling |
| `rllm-executor` | Single-process executor abstraction; later multi-process and tensor-parallel |
| `rllm-engine` | `EngineCore`, sync engine, async engine, request lifecycle, output processing |
| `rllm-server` | Axum HTTP server, OpenAI-compatible protocol, SSE streaming, metrics, health, CLI |
| `rllm-bench` | Offline and online benchmarks: latency, throughput, TTFT, TPOT, memory |

## Runtime Flow

1. Client sends HTTP or offline generation request.
2. Server parses request into protocol structs.
3. Input processor validates params, renders chat template if needed,
   tokenizes input, assigns request ID, and computes prefix-cache block hashes.
4. Engine core adds request to scheduler.
5. Each engine step:
   1. Scheduler chooses requests under token and sequence budgets.
   2. Scheduler asks KV cache manager for prefix cache hits.
   3. Scheduler allocates new KV blocks or preempts lower-priority requests.
   4. Scheduler emits `SchedulerOutput`.
   5. Executor sends `SchedulerOutput` to worker/model runner.
   6. Model runner builds input tensors, positions, slot mappings, block
      tables, and attention metadata.
   7. Worker executes model forward.
   8. Sampler produces next token IDs and optional logprobs.
   9. Scheduler updates request state, frees finished requests, records stats.
   10. Output processor detokenizes deltas and sends streaming/final outputs.

## Feature Flags

| Flag | Scope | Purpose |
|---|---|---|
| `cpu` | workspace | Enable CPU-only execution path |
| `cuda` | rllm-tensor, rllm-kernels, rllm-worker | Enable CUDA device support and kernels |
| `candle-backend` | rllm-model | Use Candle for model/layer bootstrap |
| `server` | rllm-server | Build the HTTP server binary |
| `bench` | rllm-bench | Build benchmark harness |
| `tracing` | workspace | Structured logging / tracing integration |
| `experimental-tp` | rllm-executor, rllm-worker | Experimental tensor parallel support |

## Tensor Backend Decision

**Candle** for model/layer bootstrap plus custom CUDA kernels for paged
attention and hot fused ops. Candle provides safe Rust tensor operations and
proven model implementations. Custom CUDA kernels are invoked through FFI for
performance-critical paths (PagedAttention, fused norms, RoPE).

Acceptance: model code must not directly depend on server/scheduler crates.

## CUDA Binding Decision

- `cudarc` for CUDA driver/runtime API interaction from Rust.
- `cc` / `cmake` crates for compiling `.cu` kernel source at build time.
- One minimal CUDA kernel launches from Rust in CI when the `cuda` feature is
  enabled.
