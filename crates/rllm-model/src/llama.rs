#[cfg(feature = "candle-backend")]
use anyhow::{Context, Result};
#[cfg(feature = "candle-backend")]
use candle_core::{D, Device, Tensor};
#[cfg(feature = "candle-backend")]
use rllm_core::config::ModelConfig;

#[cfg(feature = "candle-backend")]
use crate::layers::{Linear, LlamaAttention, LlamaDecoderLayer, LlamaMLP, RmsNorm};
#[cfg(feature = "candle-backend")]
use crate::loader::WeightMap;
#[cfg(feature = "candle-backend")]
use crate::registry::{CausalLM, Model};
#[cfg(feature = "candle-backend")]
use crate::rope::RotaryEmbedding;

#[cfg(feature = "candle-backend")]
pub struct LlamaForCausalLM {
    model: LlamaModel,
    config: ModelConfig,
}

#[cfg(feature = "candle-backend")]
impl LlamaForCausalLM {
    pub fn factory(_config: &ModelConfig) -> Result<Box<dyn CausalLM>> {
        anyhow::bail!(
            "LlamaForCausalLM::factory requires a loaded model. \
             Use LlamaForCausalLM::from_weights() instead."
        );
    }

    pub fn from_weights(config: ModelConfig, weights: WeightMap) -> Result<Self> {
        let device = weights.device.clone();
        let model = LlamaModel::new(&config, weights, &device)
            .context("building Llama model from weights")?;

        Ok(Self { model, config })
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }
}

#[cfg(feature = "candle-backend")]
impl Model for LlamaForCausalLM {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self.model.lm_head.forward(&hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(logits)
    }

    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        let hidden = self.model.forward_paged(input_ids, positions, gpu_kv_cache, attn_meta)?;
        let logits = self.model.lm_head.forward(&hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(logits)
    }
}

#[cfg(feature = "candle-backend")]
impl CausalLM for LlamaForCausalLM {
    fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let device = self.device();

        // Prefill
        let input_ids = Tensor::new(prompt.to_vec(), device)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .reshape((1, prompt.len()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let mut kv_cache = vec![None; self.config.num_layers];

        let logits = self.forward(&input_ids, &positions, &mut kv_cache)?;
        let seq_len = logits.dim(D::Minus2).map_err(|e| anyhow::anyhow!("{e}"))?;
        let last_logits =
            logits.narrow(D::Minus2, seq_len - 1, 1).map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut tokens = prompt.to_vec();
        let mut next_token = argmax(&last_logits)?;

        if tokens.len() >= max_tokens {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Decode loop
        let num_decode_steps = max_tokens.saturating_sub(tokens.len());
        for _step in 0..num_decode_steps {
            let input_ids = Tensor::new(&[next_token], device)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .reshape((1, 1))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let pos = tokens.len() - 1;

            let logits = self.forward(&input_ids, &[pos], &mut kv_cache)?;
            next_token = argmax(&logits)?;
            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

#[cfg(feature = "candle-backend")]
fn argmax(logits: &Tensor) -> Result<u32> {
    let (batch, seq, vocab) = logits.dims3().map_err(|e| anyhow::anyhow!("{e}"))?;
    let flat = logits
        .reshape((batch * seq, vocab))
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let vals = flat.to_vec2::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
    let best = vals[0]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    Ok(best)
}

// ── LlamaModel (transformer backbone, no LM head) ───────────────────────

#[cfg(feature = "candle-backend")]
pub struct LlamaModel {
    embed_tokens: Linear,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    #[allow(dead_code)]
    config: ModelConfig,
    device: Device,
}

#[cfg(feature = "candle-backend")]
impl LlamaModel {
    pub fn new(config: &ModelConfig, mut weights: WeightMap, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        // Embedding
        let embed_weight = weights
            .weights
            .remove("model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("missing model.embed_tokens.weight"))?;
        let embed_tokens = Linear::new(embed_weight);

        // LM head - may be tied with embedding
        let has_lm_head = weights.weights.contains_key("lm_head.weight");
        let lm_head_weight = if has_lm_head {
            weights.weights.remove("lm_head.weight").unwrap()
        } else {
            tracing::info!("lm_head is tied with embed_tokens, reusing embedding weight");
            embed_tokens.weight().clone()
        };
        let lm_head = Linear::new(lm_head_weight);

        let rms_norm_eps = 1e-6;

        // Build decoder layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");

            let q_proj = weights
                .weights
                .remove(&format!("{prefix}.self_attn.q_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.q_proj.weight"))?;
            let k_proj = weights
                .weights
                .remove(&format!("{prefix}.self_attn.k_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.k_proj.weight"))?;
            let v_proj = weights
                .weights
                .remove(&format!("{prefix}.self_attn.v_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.v_proj.weight"))?;
            let o_proj = weights
                .weights
                .remove(&format!("{prefix}.self_attn.o_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.o_proj.weight"))?;

            let attn = LlamaAttention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                num_heads,
                num_kv_heads,
                head_dim,
            );

            let gate_proj = weights
                .weights
                .remove(&format!("{prefix}.mlp.gate_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.mlp.gate_proj.weight"))?;
            let up_proj = weights
                .weights
                .remove(&format!("{prefix}.mlp.up_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.mlp.up_proj.weight"))?;
            let down_proj = weights
                .weights
                .remove(&format!("{prefix}.mlp.down_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.mlp.down_proj.weight"))?;
            let mlp = LlamaMLP::new(gate_proj, up_proj, down_proj);

            let input_ln_w = weights
                .weights
                .remove(&format!("{prefix}.input_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.input_layernorm.weight"))?;
            let post_attn_ln_w = weights
                .weights
                .remove(&format!("{prefix}.post_attention_layernorm.weight"))
                .ok_or_else(|| {
                    anyhow::anyhow!("missing {prefix}.post_attention_layernorm.weight")
                })?;

            layers.push(LlamaDecoderLayer::new(
                attn,
                mlp,
                RmsNorm::new(input_ln_w, rms_norm_eps),
                RmsNorm::new(post_attn_ln_w, rms_norm_eps),
            ));
        }

        // Final norm
        let final_norm_w = weights
            .weights
            .remove("model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("missing model.norm.weight"))?;
        let norm = RmsNorm::new(final_norm_w, rms_norm_eps);

        // Rotary embeddings
        let rope = RotaryEmbedding::new(head_dim, config.max_model_len, config.rope_theta, device)
            .map_err(|e| anyhow::anyhow!("creating rotary embeddings: {e}"))?;

        tracing::info!(
            "LlamaModel: {} layers, {} heads ({} KV heads), head_dim={}, hidden={}, vocab={}",
            config.num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            config.vocab_size,
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config: config.clone(),
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let hidden_states = embedding_lookup(self.embed_tokens.weight(), input_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut hidden_states = hidden_states;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward(&hidden_states, positions, &mut kv_cache[i], &self.rope)
                .map_err(|e| anyhow::anyhow!("layer {i}: {e}"))?;
        }

        self.norm.forward(&hidden_states).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Paged forward pass using PagedAttention kernels for all layers.
    pub fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        gpu_kv_cache: &rllm_kernels::cache_ops::GpuKVCache,
        attn_meta: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        let hidden_states = embedding_lookup(self.embed_tokens.weight(), input_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut hidden_states = hidden_states;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward_paged(&hidden_states, positions, gpu_kv_cache, attn_meta, i, &self.rope)
                .map_err(|e| anyhow::anyhow!("layer {i}: {e}"))?;
        }

        self.norm.forward(&hidden_states).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(feature = "candle-backend")]
fn embedding_lookup(weight: &Tensor, ids: &Tensor) -> candle_core::Result<Tensor> {
    let id_vec = ids.flatten_all()?.to_vec1::<u32>()?;
    let bsz = ids.dim(0)?;
    let seq = ids.dim(1)?;
    let hidden = weight.dim(D::Minus1)?;
    let indices = Tensor::from_vec(id_vec, (bsz * seq,), ids.device())?;
    let embedded = weight.index_select(&indices, 0)?;
    embedded.reshape((bsz, seq, hidden))
}

#[cfg(all(test, feature = "candle-backend"))]
mod tests {
    use std::collections::HashMap;

    use candle_core::DType;

    use super::*;

    fn toy_config() -> ModelConfig {
        ModelConfig {
            model_id: "test-llama".into(),
            architecture: "LlamaForCausalLM".into(),
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            max_model_len: 256,
            rope_theta: 10000.0,
            rope_scaling: None,
            dtype: rllm_core::dtype::DType::F32,
            quantization: None,
            tokenizer_mode: rllm_core::config::TokenizerMode::Auto,
        }
    }

    fn build_toy_model(config: &ModelConfig) -> LlamaForCausalLM {
        let device = Device::Cpu;
        let mut weights = HashMap::new();

        weights.insert(
            "model.embed_tokens.weight".into(),
            Tensor::randn(0.0f32, 1.0f32, (config.vocab_size, config.hidden_size), &device)
                .unwrap(),
        );
        weights.insert(
            "model.norm.weight".into(),
            Tensor::ones(config.hidden_size, DType::F32, &device).unwrap(),
        );

        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let h = config.hidden_size;
            let ih = config.intermediate_size;
            let nq = config.num_attention_heads * config.head_dim;
            let nkv = config.num_kv_heads * config.head_dim;

            weights.insert(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (nq, h), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (nkv, h), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.self_attn.v_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (nkv, h), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (h, nq), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.mlp.gate_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (ih, h), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.mlp.up_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (ih, h), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.mlp.down_proj.weight"),
                Tensor::randn(0.0f32, 0.02f32, (h, ih), &device).unwrap(),
            );
            weights.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::ones(h, DType::F32, &device).unwrap(),
            );
            weights.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::ones(h, DType::F32, &device).unwrap(),
            );
        }

        let weight_map = WeightMap { weights, device: device.clone() };
        LlamaForCausalLM::from_weights(config.clone(), weight_map).unwrap()
    }

    #[test]
    fn forward_shape_is_batch_seq_vocab() -> Result<()> {
        let config = toy_config();
        let model = build_toy_model(&config);
        let device = model.device();

        let input_ids = Tensor::new(vec![1u32, 2, 3, 4, 5], device)?.reshape((1, 5))?;
        let positions: Vec<usize> = (0..5).collect();
        let mut kv_cache = vec![None; config.num_layers];

        let logits = model.forward(&input_ids, &positions, &mut kv_cache)?;
        assert_eq!(
            logits.dims(),
            &[1, 5, config.vocab_size],
            "forward output shape should be [batch, seq, vocab]"
        );

        for kv in &kv_cache {
            assert!(kv.is_some(), "KV cache should be populated after prefill");
        }
        Ok(())
    }

    #[test]
    fn decode_step_extends_kv_cache() -> Result<()> {
        let config = toy_config();
        let model = build_toy_model(&config);
        let device = model.device();

        let input_ids = Tensor::new(vec![1u32, 2, 3], device)?.reshape((1, 3))?;
        let mut kv_cache = vec![None; config.num_layers];

        model.forward(&input_ids, &[0, 1, 2], &mut kv_cache)?;
        let prefilled_len = kv_cache[0].as_ref().unwrap().0.dim(2)?;
        assert_eq!(prefilled_len, 3);

        let next_id = Tensor::new(vec![4u32], device)?.reshape((1, 1))?;
        model.forward(&next_id, &[3], &mut kv_cache)?;
        let decoded_len = kv_cache[0].as_ref().unwrap().0.dim(2)?;
        assert_eq!(decoded_len, 4, "KV cache should grow after decode step");
        Ok(())
    }

    #[test]
    fn greedy_generation_is_stable() -> Result<()> {
        let config = toy_config();
        let model = build_toy_model(&config);

        let prompt = vec![1u32, 2, 3];
        let gen1 = model.generate(&prompt, 10)?;
        let gen2 = model.generate(&prompt, 10)?;
        assert_eq!(gen1, gen2, "greedy generation should be deterministic");
        assert!(gen1.len() <= 13);
        Ok(())
    }

    #[test]
    fn embedding_lookup_shape() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0f32, (64, 32), &device)?;
        let ids = Tensor::new(vec![1u32, 5, 10], &device)?.reshape((1, 3))?;
        let out = embedding_lookup(&weight, &ids)?;
        assert_eq!(out.dims(), &[1, 3, 32]);
        Ok(())
    }
}
