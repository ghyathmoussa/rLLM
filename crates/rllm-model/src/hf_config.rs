use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

use rllm_core::config::ModelConfig;
use rllm_core::dtype::DType;

#[derive(Debug, Deserialize)]
struct HfConfigJson {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    max_position_embeddings: Option<usize>,
    rope_theta: Option<f32>,
    torch_dtype: Option<String>,
    head_dim: Option<usize>,
}

pub fn parse_hf_config(path: &Path) -> Result<ModelConfig> {
    let content = std::fs::read_to_string(path)?;
    let hf: HfConfigJson = serde_json::from_str(&content)?;

    let architecture = hf
        .architectures
        .and_then(|a| a.into_iter().next())
        .or(hf.model_type)
        .unwrap_or_else(|| "unknown".to_string());

    let hidden_size = hf.hidden_size.unwrap_or(4096);
    let num_attention_heads = hf.num_attention_heads.unwrap_or(32);
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let dtype = match hf.torch_dtype.as_deref() {
        Some("float16") | Some("fp16") => DType::F16,
        Some("bfloat16") | Some("bf16") => DType::BF16,
        Some("float32") | Some("fp32") => DType::F32,
        Some("float8_e4m3fn") => DType::FP8E4M3,
        Some("float8_e5m2") => DType::FP8E5M2,
        _ => DType::F16,
    };

    Ok(ModelConfig {
        model_id: path.parent().unwrap_or(Path::new(".")).to_string_lossy().to_string(),
        architecture,
        vocab_size: hf.vocab_size.unwrap_or(32000),
        hidden_size,
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        num_layers: hf.num_hidden_layers.unwrap_or(32),
        num_attention_heads,
        num_kv_heads,
        head_dim,
        max_model_len: hf.max_position_embeddings.unwrap_or(4096),
        rope_theta: hf.rope_theta.unwrap_or(10000.0),
        rope_scaling: None,
        dtype,
        quantization: None,
        tokenizer_mode: rllm_core::config::TokenizerMode::Auto,
    })
}
