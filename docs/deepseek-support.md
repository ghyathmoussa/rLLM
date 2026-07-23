# DeepSeek Support

rLLM has two DeepSeek execution families:

- DeepseekForCausalLM checkpoints (DeepSeek-LLM / DeepSeek-Coder) use the
  Llama-compatible dense decoder.
- DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, and
  DeepseekR1ForCausalLM use the native MLA + MoE decoder.

## Native decoder

The native decoder implements:

- Q-LoRA and non-LoRA query projections
- compressed KV projection and per-head MLA expansion
- decoupled RoPE/no-PE query and key dimensions
- interleaved RoPE, including DeepSeek YaRN scaling
- dense replacement layers
- routed and shared DeepSeek V2 experts
- grouped top-k V2 routing
- V3/R1 sigmoid noaux_tc routing with correction bias
- V3/R1 two-dimensional block-scaled FP8 projections and experts
- tied and untied LM heads
- F16, BF16, FP8, and INT8 paged KV cache dtypes supported by rLLM

MLA writes expanded per-head K and zero-padded V tensors to rLLM's existing
block-addressed GPU cache. This preserves the cache ABI and enables continuous
batching, chunked prefill, and block-hash prefix caching. It uses more cache
memory than a specialized compressed-latent MLA cache.

DeepSeek V2 projections and experts use rLLM's normal linear loader, so
unquantized SafeTensors and compatible GPTQ, AWQ, INT8, MXFP, and
compressed-tensors schemas use their existing GPU paths. DeepSeek V3 and R1
checkpoints with weight_scale_inv use the native block-FP8 CUDA kernels.

## GPU execution

The serving worker requires CUDA and native DeepSeek paged MLA runs directly on
the global GPU cache; it does not require the RLLM_PAGED_ATTENTION opt-in used
by the older Llama path.

Example:

```bash
cargo build --release --features cuda --bin rllm
./target/release/rllm serve deepseek-ai/DeepSeek-V2-Lite \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --dtype bf16
```

## Deployment boundary

The current production executor is single-GPU. The repository's
MultiProcExecutor is scaffold code and does not shard weights or run NCCL
collectives. Consequently, full-size DeepSeek V3 and DeepSeek R1 require a GPU
with enough memory for the whole checkpoint; practical multi-GPU deployment is
not implemented. The decoder and CUDA kernels are wired and tested, but loading
those full-size checkpoints on ordinary single-GPU hardware will fail from
insufficient memory.

DeepSeek R1 distill checkpoints retain their base architecture. Llama-compatible
distills use the Llama path. Architectures not already supported by rLLM, such
as Qwen-based distills, remain unsupported.

DeepSeek V4 is a separate architecture and is not covered by this decoder.
