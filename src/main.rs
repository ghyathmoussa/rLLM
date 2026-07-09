use std::{fs, path::PathBuf};

use clap::Parser;
use rllm_model::quantize::{GptqExportOptions, quantize_model_to_gptq};
use rllm_server::cli::ServeArgs;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "rllm",
    version,
    about = "rLLM: A high-performance Rust LLM inference engine inspired by vLLM",
    long_about = "rLLM is a high-performance Rust LLM inference engine designed for low-latency, \
                  high-concurrency serving of decoder-only causal language models (Llama family). \
                  It features PagedAttention, continuous batching, prefix caching, and CUDA \
                  acceleration via Candle.\n\n\
                  EXAMPLES:\n  \
                    # Serve Llama model on GPU (CUDA)\n  \
                    rllm serve meta-llama/Llama-3.2-1B-Instruct --dtype bf16\n\n  \
                    # Serve Llama model with INT8 quantization\n  \
                    rllm serve meta-llama/Llama-3.2-1B-Instruct --quantization int8 --quant-bits 8\n\n  \
                    # Quantize a model to 4-bit GPTQ format\n  \
                    rllm quantize meta-llama/Llama-3.2-1B-Instruct --output-dir ./quant-output --calibration-file calibration.txt"
)]
enum Cli {
    /// Serve a model via OpenAI-compatible HTTP API
    ///
    /// Starts an HTTP server hosting the specified Hugging Face model or local checkpoint.
    /// The server exposes standard OpenAI-compatible endpoints like `/v1/chat/completions` and `/v1/models`,
    /// as well as `/health` and Prometheus `/metrics`.
    ///
    /// EXAMPLE:
    ///   rllm serve meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8000
    Serve(Box<ServeArgs>),

    /// Quantize a model checkpoint to GPTQ format
    ///
    /// Runs post-training quantization (PTQ) on a model using calibration prompts to generate
    /// a compressed checkpoint (e.g. 4-bit or 8-bit weights).
    ///
    /// EXAMPLE:
    ///   rllm quantize meta-llama/Llama-3.2-1B-Instruct --output-dir ./quantized --calibration-file calibration.txt
    Quantize(Box<QuantizeArgs>),
}

#[derive(Parser, Debug, Clone)]
struct QuantizeArgs {
    /// Hugging Face model ID or local path
    model: String,

    /// Output directory for the quantized checkpoint
    #[arg(long)]
    output_dir: PathBuf,

    /// Text file with one calibration prompt per line
    #[arg(long)]
    calibration_file: PathBuf,

    /// GPTQ bit width
    #[arg(long, default_value_t = 4)]
    bits: usize,

    /// GPTQ group size
    #[arg(long, default_value_t = 128)]
    group_size: usize,

    /// GPTQ damping percentage applied to Hessian diagonal
    #[arg(long, default_value_t = 0.01)]
    damp_percent: f32,

    /// Enable activation-order column permutation
    #[arg(long, default_value_t = false)]
    act_order: bool,

    /// Maximum number of calibration prompts to use
    #[arg(long, default_value_t = 128)]
    max_calibration_samples: usize,

    /// Maximum sequence length per calibration prompt
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    /// Quantize lm_head in addition to transformer linear layers
    #[arg(long, default_value_t = true)]
    include_lm_head: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cli_log_level = match &cli {
        Cli::Serve(args) => args.log_level.as_str(),
        Cli::Quantize(args) => args.log_level.as_str(),
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(cli_log_level)),
        )
        .init();

    match cli {
        Cli::Serve(args) => {
            tracing::info!(
                model = %args.model,
                host = %args.host,
                port = %args.port,
                "Starting rLLM server"
            );
            rllm_server::server::serve(*args).await?;
        }
        Cli::Quantize(args) => {
            let prompts = load_calibration_prompts(&args.calibration_file)?;
            let opts = GptqExportOptions {
                bits: args.bits,
                group_size: args.group_size,
                damp_percent: args.damp_percent,
                act_order: args.act_order,
                calibration_prompts: prompts,
                max_calibration_samples: args.max_calibration_samples,
                max_seq_len: args.max_seq_len,
                include_lm_head: args.include_lm_head,
            };
            tracing::info!(
                model = %args.model,
                output_dir = %args.output_dir.display(),
                bits = args.bits,
                group_size = args.group_size,
                act_order = args.act_order,
                "Starting GPTQ quantization"
            );
            quantize_model_to_gptq(&args.model, &args.output_dir, &opts)?;
        }
    }

    Ok(())
}

fn load_calibration_prompts(path: &PathBuf) -> anyhow::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let prompts = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if prompts.is_empty() {
        anyhow::bail!("calibration file {} contained no non-empty lines", path.display());
    }
    Ok(prompts)
}
