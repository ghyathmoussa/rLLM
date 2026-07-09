use std::path::Path;

use anyhow::{Context, Result};
use rllm_core::{config::ModelConfig, dtype::DType};
use rllm_quant::QuantSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct HfQuantizationConfigJson {
    quant_method: Option<String>,
    bits: Option<usize>,
    group_size: Option<usize>,
}

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
    q_lora_rank: Option<usize>,
    kv_lora_rank: Option<usize>,
    n_routed_experts: Option<usize>,
    n_shared_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    quantization_config: Option<serde_json::Value>,
}

pub fn parse_hf_config(path: &Path) -> Result<ModelConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading config from {}", path.display()))?;
    let hf: HfConfigJson = serde_json::from_str(&content)
        .with_context(|| format!("parsing config from {}", path.display()))?;

    let architecture = hf
        .architectures
        .as_ref()
        .and_then(|a| a.first())
        .cloned()
        .or_else(|| hf.model_type.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = normalize_architecture(&architecture);

    reject_unsupported_deepseek_native_architecture(&architecture, &hf)?;

    let hidden_size = hf.hidden_size.unwrap_or(4096);
    let num_attention_heads = hf.num_attention_heads.unwrap_or(32);
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = hf.intermediate_size.unwrap_or(hidden_size * 4);

    validate_config(&architecture, hidden_size, num_attention_heads, num_kv_heads, head_dim)?;

    let dtype = match hf.torch_dtype.as_deref() {
        Some("float16") | Some("fp16") => DType::F16,
        Some("bfloat16") | Some("bf16") => DType::BF16,
        Some("float32") | Some("fp32") => DType::F32,
        Some("float8_e4m3fn") => DType::FP8E4M3,
        Some("float8_e5m2") => DType::FP8E5M2,
        _ => DType::F16,
    };

    let quantization = hf.quantization_config.as_ref().and_then(|q_val| {
        // 1. Try parsing using the new GPTQ / AWQ logic
        if let Ok(q) = serde_json::from_value::<HfQuantizationConfigJson>(q_val.clone()) {
            if let Some(ref quant_method) = q.quant_method {
                let kind = match quant_method.to_lowercase().as_str() {
                    "gptq" => Some(rllm_core::config::QuantizationKind::GPTQ),
                    "awq" => Some(rllm_core::config::QuantizationKind::AWQ),
                    _ => None,
                };
                if let Some(kind) = kind {
                    return Some(rllm_core::config::QuantizationConfig {
                        kind,
                        group_size: q.group_size,
                        bits: q.bits,
                    });
                }
            }
        }
        // 2. Fall back to QuantSchema::from_hf_value (e.g. for compressed-tensors / int-quantized / INT8)
        QuantSchema::from_hf_value(q_val).and_then(|schema| schema.to_core_config())
    });

    Ok(ModelConfig {
        model_id: path.parent().unwrap_or(Path::new(".")).to_string_lossy().to_string(),
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
        quantization,
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
        "LlamaForCausalLM" | "MistralForCausalLM" | "DeepseekForCausalLM" => Ok(()),
        _ => {
            tracing::warn!(
                "unsupported architecture '{architecture}', attempting Llama-compatible loading"
            );
            Ok(())
        }
    }
}

fn normalize_architecture(architecture: &str) -> String {
    match architecture {
        "llama" | "LlamaModel" | "LLaMAForCausalLM" => "LlamaForCausalLM".to_string(),
        "mistral" | "MistralModel" => "MistralForCausalLM".to_string(),
        "deepseek" | "deepseek_llm" | "DeepSeekForCausalLM" => "DeepseekForCausalLM".to_string(),
        other => other.to_string(),
    }
}

fn reject_unsupported_deepseek_native_architecture(
    architecture: &str,
    hf: &HfConfigJson,
) -> Result<()> {
    let arch_lower = architecture.to_ascii_lowercase();
    let model_type = hf.model_type.as_deref().unwrap_or_default().to_ascii_lowercase();
    let has_mla = hf.q_lora_rank.is_some() || hf.kv_lora_rank.is_some();
    let has_moe = hf.n_routed_experts.is_some()
        || hf.n_shared_experts.is_some()
        || hf.num_experts_per_tok.is_some();
    let is_native_deepseek = arch_lower.contains("deepseekv2")
        || arch_lower.contains("deepseekv3")
        || model_type.contains("deepseek_v2")
        || model_type.contains("deepseek_v3");

    if is_native_deepseek || has_mla || has_moe {
        anyhow::bail!(
            "unsupported DeepSeek native MLA/MoE architecture '{architecture}': \
             rLLM currently supports Llama-compatible dense DeepSeek checkpoints \
             (DeepseekForCausalLM) only; native DeepSeek V2/V3/R1 checkpoints require \
             MLA attention, MLA KV-cache metadata, and DeepSeekMoE execution support"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

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

    #[test]
    fn normalizes_llama_model_type() {
        let f = write_config_json(
            r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8
            }"#,
        );
        let config = parse_hf_config(f.path()).unwrap();
        assert_eq!(config.architecture, "LlamaForCausalLM");
    }

    #[test]
    fn accepts_dense_deepseek_as_llama_compatible() {
        let f = write_config_json(
            r#"{
                "architectures": ["DeepseekForCausalLM"],
                "vocab_size": 102400,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 30,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "torch_dtype": "bfloat16"
            }"#,
        );
        let config = parse_hf_config(f.path()).unwrap();
        assert_eq!(config.architecture, "DeepseekForCausalLM");
        assert_eq!(config.vocab_size, 102400);
        assert_eq!(config.num_layers, 30);
        assert_eq!(config.dtype, DType::BF16);
    }

    #[test]
    fn rejects_native_deepseek_mla_moe_config() {
        let f = write_config_json(
            r#"{
                "architectures": ["DeepseekV2ForCausalLM"],
                "model_type": "deepseek_v2",
                "vocab_size": 102400,
                "hidden_size": 2048,
                "intermediate_size": 10944,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "max_position_embeddings": 163840
            }"#,
        );
        let err = parse_hf_config(f.path()).unwrap_err().to_string();
        assert!(err.contains("unsupported DeepSeek native MLA/MoE architecture"));
    }

    #[test]
    fn parses_int8_quantization_config() {
        let f = write_config_json(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "int-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "num_bits": 8,
                                "strategy": "channel",
                                "symmetric": true
                            }
                        }
                    }
                }
            }"#,
        );
        let config = parse_hf_config(f.path()).unwrap();
        let quant = config.quantization.unwrap();
        assert_eq!(quant.kind, rllm_core::config::QuantizationKind::Int8);
        assert_eq!(quant.bits, Some(8));
    }
}
