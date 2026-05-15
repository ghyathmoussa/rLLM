use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

use rllm_core::config::ModelConfig;
use rllm_core::dtype::DType;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
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
    rope_theta: Option<f64>,
    torch_dtype: Option<String>,
    head_dim: Option<usize>,
    hidden_act: Option<String>,
    rms_norm_eps: Option<f64>,
}

pub fn parse_hf_config(path: &Path) -> Result<ModelConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading config from {}", path.display()))?;
    let hf: HfConfigJson = serde_json::from_str(&content)
        .with_context(|| format!("parsing config from {}", path.display()))?;

    let architecture = hf
        .architectures
        .and_then(|a| a.into_iter().next())
        .or(hf.model_type)
        .unwrap_or_else(|| "unknown".to_string());

    let hidden_size = hf.hidden_size.unwrap_or(4096);
    let num_attention_heads = hf.num_attention_heads.unwrap_or(32);
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = hf.intermediate_size.unwrap_or(hidden_size * 4);

    validate_config(
        &architecture,
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        head_dim,
    )?;

    let dtype = match hf.torch_dtype.as_deref() {
        Some("float16") | Some("fp16") => DType::F16,
        Some("bfloat16") | Some("bf16") => DType::BF16,
        Some("float32") | Some("fp32") => DType::F32,
        Some("float8_e4m3fn") => DType::FP8E4M3,
        Some("float8_e5m2") => DType::FP8E5M2,
        _ => DType::F16,
    };

    Ok(ModelConfig {
        model_id: path
            .parent()
            .unwrap_or(Path::new("."))
            .to_string_lossy()
            .to_string(),
        architecture,
        vocab_size: hf.vocab_size.unwrap_or(32000),
        hidden_size,
        intermediate_size,
        num_layers: hf.num_hidden_layers.unwrap_or(32),
        num_attention_heads,
        num_kv_heads,
        head_dim,
        max_model_len: hf.max_position_embeddings.unwrap_or(4096),
        rope_theta: hf.rope_theta.unwrap_or(10000.0) as f32,
        rope_scaling: None,
        dtype,
        quantization: None,
        tokenizer_mode: rllm_core::config::TokenizerMode::Auto,
    })
}

fn validate_config(
    architecture: &str,
    hidden_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if hidden_size == 0 {
        anyhow::bail!("hidden_size must be > 0");
    }
    if num_attention_heads == 0 {
        anyhow::bail!("num_attention_heads must be > 0");
    }
    if num_kv_heads == 0 || num_kv_heads > num_attention_heads {
        anyhow::bail!(
            "num_kv_heads must be > 0 and <= num_attention_heads ({num_attention_heads}), got {num_kv_heads}"
        );
    }
    if hidden_size % num_attention_heads != 0 {
        anyhow::bail!(
            "hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        );
    }
    if num_attention_heads % num_kv_heads != 0 {
        anyhow::bail!(
            "num_attention_heads ({num_attention_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
    }
    if head_dim != hidden_size / num_attention_heads {
        anyhow::bail!(
            "head_dim ({head_dim}) doesn't match hidden_size / num_attention_heads ({})",
            hidden_size / num_attention_heads
        );
    }
    match architecture {
        "LlamaForCausalLM" | "MistralForCausalLM" => Ok(()),
        _ => {
            tracing::warn!("unsupported architecture '{architecture}', attempting Llama-compatible loading");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_config_json(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{json}").unwrap();
        f
    }

    #[test]
    fn parse_llama_config() {
        let f = write_config_json(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "torch_dtype": "float16"
            }"#,
        );
        let config = parse_hf_config(f.path()).unwrap();
        assert_eq!(config.architecture, "LlamaForCausalLM");
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn parse_gqa_config() {
        let f = write_config_json(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 128256,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 131072,
                "rope_theta": 500000.0,
                "torch_dtype": "bfloat16"
            }"#,
        );
        let config = parse_hf_config(f.path()).unwrap();
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.dtype, DType::BF16);
    }

    #[test]
    fn rejects_zero_hidden_size() {
        let f = write_config_json(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 0,
                "num_attention_heads": 32
            }"#,
        );
        assert!(parse_hf_config(f.path()).is_err());
    }

    #[test]
    fn rejects_misaligned_kv_heads() {
        let f = write_config_json(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 7
            }"#,
        );
        assert!(parse_hf_config(f.path()).is_err());
    }

    #[test]
    fn accepts_mistral_as_compatible() {
        let f = write_config_json(
            r#"{
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8
            }"#,
        );
        let config = parse_hf_config(f.path()).unwrap();
        assert_eq!(config.architecture, "MistralForCausalLM");
    }
}
