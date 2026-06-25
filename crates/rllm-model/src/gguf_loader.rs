use anyhow::{Context, Result};
use candle_core::Device;
use candle_core::quantized::gguf_file::{Content, Value};
use rllm_core::config::{ModelConfig, QuantizationConfig, QuantizationKind, TokenizerMode};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::loader::WeightMap;

fn get_metadata_string(metadata: &HashMap<String, Value>, key: &str) -> Option<String> {
    metadata.get(key).and_then(|v| v.to_string().ok().map(|s| s.to_string()))
}

fn get_metadata_val_as_u64(metadata: &HashMap<String, Value>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(|v| v.to_u64().ok())
}

fn get_metadata_f32(metadata: &HashMap<String, Value>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| v.to_f32().ok())
}

pub fn parse_gguf_config(path: &Path) -> Result<ModelConfig> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening GGUF file {}", path.display()))?;
    let content =
        Content::read(&mut file).map_err(|e| anyhow::anyhow!("reading GGUF content: {e}"))?;

    let arch = get_metadata_string(&content.metadata, "general.architecture")
        .unwrap_or_else(|| "llama".to_string());

    let num_layers = get_metadata_val_as_u64(&content.metadata, &format!("{arch}.block_count"))
        .unwrap_or(32) as usize;
    let hidden_size =
        get_metadata_val_as_u64(&content.metadata, &format!("{arch}.embedding_length"))
            .unwrap_or(4096) as usize;
    let num_attention_heads =
        get_metadata_val_as_u64(&content.metadata, &format!("{arch}.attention.head_count"))
            .unwrap_or(32) as usize;
    let num_kv_heads =
        get_metadata_val_as_u64(&content.metadata, &format!("{arch}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads as u64) as usize;
    let intermediate_size =
        get_metadata_val_as_u64(&content.metadata, &format!("{arch}.feed_forward_length"))
            .unwrap_or((hidden_size * 4) as u64) as usize;
    let max_model_len =
        get_metadata_val_as_u64(&content.metadata, &format!("{arch}.context_length"))
            .unwrap_or(4096) as usize;
    let rope_theta =
        get_metadata_f32(&content.metadata, &format!("{arch}.rope.freq_base")).unwrap_or(10000.0);

    let head_dim = hidden_size / num_attention_heads;

    let architecture = match arch.as_str() {
        "llama" => "LlamaForCausalLM".to_string(),
        "mistral" => "MistralForCausalLM".to_string(),
        other => {
            tracing::warn!("Unknown GGUF architecture: {}, defaulting to LlamaForCausalLM", other);
            "LlamaForCausalLM".to_string()
        }
    };

    let vocab_size =
        if let Some(Value::Array(tokens)) = content.metadata.get("tokenizer.ggml.tokens") {
            tokens.len()
        } else {
            32000
        };

    Ok(ModelConfig {
        model_id: path.to_string_lossy().to_string(),
        architecture,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_layers,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
        rope_theta,
        rope_scaling: None,
        dtype: rllm_core::dtype::DType::F16,
        quantization: Some(QuantizationConfig {
            kind: QuantizationKind::Gguf,
            group_size: None,
            bits: None,
        }),
        tokenizer_mode: TokenizerMode::Auto,
    })
}

pub fn map_gguf_name_to_hf(gguf_name: &str) -> String {
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }

    if gguf_name.starts_with("blk.") {
        let parts: Vec<&str> = gguf_name.split('.').collect();
        if parts.len() >= 3 {
            let layer_idx = parts[1];
            let suffix = parts[2..].join(".");
            let hf_suffix = match suffix.as_str() {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                other => other,
            };
            return format!("model.layers.{layer_idx}.{hf_suffix}");
        }
    }

    gguf_name.to_string()
}

pub fn load_gguf_weights(path: &Path, device: &Device) -> Result<WeightMap> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening GGUF file {}", path.display()))?;
    let content =
        Content::read(&mut file).map_err(|e| anyhow::anyhow!("reading GGUF content: {e}"))?;

    let mut weights = HashMap::new();
    let mut gguf_weights = HashMap::new();

    for name in content.tensor_infos.keys() {
        let qtensor = content
            .tensor(&mut file, name, device)
            .map_err(|e| anyhow::anyhow!("loading GGUF tensor {name}: {e}"))?;

        let hf_name = map_gguf_name_to_hf(name);

        let is_quantized = !matches!(
            qtensor.dtype(),
            candle_core::quantized::GgmlDType::F32
                | candle_core::quantized::GgmlDType::F16
                | candle_core::quantized::GgmlDType::BF16
        );

        let should_keep_quantized =
            is_quantized && (hf_name.ends_with(".proj.weight") || hf_name == "lm_head.weight");

        if should_keep_quantized {
            gguf_weights.insert(hf_name, Arc::new(qtensor));
        } else {
            let tensor = qtensor
                .dequantize(device)
                .map_err(|e| anyhow::anyhow!("dequantizing tensor {name}: {e}"))?;
            weights.insert(hf_name, tensor);
        }
    }

    Ok(WeightMap {
        weights,
        quantized: HashMap::new(),
        gguf_weights,
        quant_schema: None,
        device: device.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_gguf_name_to_hf() {
        assert_eq!(map_gguf_name_to_hf("token_embd.weight"), "model.embed_tokens.weight");
        assert_eq!(map_gguf_name_to_hf("output.weight"), "lm_head.weight");
        assert_eq!(map_gguf_name_to_hf("output_norm.weight"), "model.norm.weight");
        assert_eq!(
            map_gguf_name_to_hf("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_gguf_name_to_hf("blk.5.ffn_gate.weight"),
            "model.layers.5.mlp.gate_proj.weight"
        );
        assert_eq!(
            map_gguf_name_to_hf("blk.12.attn_norm.weight"),
            "model.layers.12.input_layernorm.weight"
        );
    }
}
