use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "rllm", version, about = "rLLM: Rust LLM inference engine")]
pub enum Cli {
    /// Serve a model via OpenAI-compatible HTTP API
    Serve(ServeArgs),
}

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Hugging Face model ID or local path
    pub model: String,

    /// Hugging Face tokenizer ID or local path (defaults to model ID/path)
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind to
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Data type for model weights
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Maximum model context length
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Maximum number of concurrent sequences
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// Maximum number of batched tokens per step
    #[arg(long, default_value_t = 4096)]
    pub max_num_batched_tokens: usize,

    /// GPU memory utilization target (0.01 - 1.0)
    #[arg(long, default_value_t = 0.9, value_parser = parse_gpu_utilization)]
    pub gpu_memory_utilization: f32,

    /// Enable prefix caching
    #[arg(long, default_value_t = false)]
    pub enable_prefix_caching: bool,

    /// API key for authenticated endpoints (reads RLLM_API_KEY env var)
    #[arg(long, env = "RLLM_API_KEY")]
    pub api_key: Option<String>,

    /// CORS allowed origins (comma-separated, * for all)
    #[arg(long, default_value = "*")]
    pub cors_allowed_origins: String,

    /// Enable debug endpoints (e.g., /debug/model)
    #[arg(long, default_value_t = false)]
    pub enable_debug_endpoints: bool,

    /// Maximum messages allowed in chat completion request
    #[arg(long, default_value_t = 256)]
    pub max_input_messages: usize,

    /// Maximum characters allowed in input body
    #[arg(long, default_value_t = 1_000_000)]
    pub max_input_chars: usize,

    /// Maximum concurrent inference requests
    #[arg(long, default_value_t = 64)]
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 120)]
    pub request_timeout_secs: u64,

    /// Log level
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Quantization format (none, fp8, mxfp8, mxfp4, nvfp4, int8, int4, compressed-tensors, modelopt, torchao)
    #[arg(long, default_value = "none")]
    pub quantization: String,

    /// Quantization bit width (e.g. 4 or 8)
    #[arg(long)]
    pub quant_bits: Option<usize>,

    /// Quantization group size (e.g. 32 or 128)
    #[arg(long)]
    pub quant_group_size: Option<usize>,

    /// KV Cache data type (auto, f16, bf16, fp8_e4m3, fp8_e5m2)
    #[arg(long, default_value = "auto")]
    pub kv_cache_dtype: String,
}


fn parse_gpu_utilization(s: &str) -> Result<f32, String> {
    let val: f32 = s.parse().map_err(|_| format!("invalid float: {s}"))?;
    if !(0.01..=1.0).contains(&val) {
        return Err(format!("gpu_memory_utilization must be in range 0.01..=1.0, got {val}"));
    }
    Ok(val)
}
