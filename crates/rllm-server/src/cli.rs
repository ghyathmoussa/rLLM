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

    /// GPU memory utilization target (0.0 - 1.0)
    #[arg(long, default_value_t = 0.9)]
    pub gpu_memory_utilization: f32,

    /// Enable prefix caching
    #[arg(long, default_value_t = true)]
    pub enable_prefix_caching: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    pub log_level: String,
}
