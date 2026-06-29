#[cfg(feature = "candle-backend")]
use anyhow::{Context, Result};
#[cfg(feature = "candle-backend")]
use candle_core::{D, DType, Device, Tensor};
#[cfg(feature = "candle-backend")]
use rllm_core::config::ModelConfig;
#[cfg(feature = "candle-backend")]
use rllm_quant::{WeightSource, factory_from_config};

#[cfg(feature = "candle-backend")]
use crate::gptq::GptqCalibration;
#[cfg(feature = "candle-backend")]
use crate::layers::{
    Linear, LlamaAttention, LlamaDecoderLayer, LlamaMLP, RmsNorm, causal_mask, repeat_kv,
};
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

    pub fn collect_gptq_calibrations(
        &self,
        calibration_token_batches: &[Vec<u32>],
        include_lm_head: bool,
    ) -> Result<std::collections::BTreeMap<String, GptqCalibration>> {
        let mut calibrations = std::collections::BTreeMap::new();
        for token_ids in calibration_token_batches {
            if token_ids.is_empty() {
                continue;
            }
            let input_ids = Tensor::new(token_ids.clone(), self.device())
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .reshape((1, token_ids.len()))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let positions: Vec<usize> = (0..token_ids.len()).collect();
            self.model.collect_gptq_calibrations(
                &input_ids,
                &positions,
                include_lm_head,
                &mut calibrations,
            )?;
        }
        Ok(calibrations)
    }
}

#[cfg(feature = "candle-backend")]
impl Model for LlamaForCausalLM {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn quantized_layer_count(&self) -> usize {
        self.model.quantized_layer_count
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
    pub quantized_layer_count: usize,
}

#[cfg(feature = "candle-backend")]
fn load_linear(
    prefix: &str,
    weights: &mut WeightMap,
    config: &ModelConfig,
    device: &Device,
) -> Result<Linear> {
    let qweight_name = format!("{prefix}.qweight");
    if weights.weights.contains_key(&qweight_name) {
        let qweight = weights
            .weights
            .remove(&qweight_name)
            .ok_or_else(|| anyhow::anyhow!("missing {qweight_name}"))?;
        let qzeros = weights
            .weights
            .remove(&format!("{prefix}.qzeros"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}.qzeros"))?;
        let scales = weights
            .weights
            .remove(&format!("{prefix}.scales"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}.scales"))?;

        let bits = config.quantization.as_ref().and_then(|q| q.bits).unwrap_or(4);
        let group_size = config.quantization.as_ref().and_then(|q| q.group_size).unwrap_or(128);

        let in_features = qweight.dim(0)? * 8;

        if let Some(rllm_core::config::QuantizationKind::AWQ) = config.quantization.as_ref().map(|q| q.kind) {
            Ok(Linear::new_awq(qweight, qzeros, scales, bits, group_size))
        } else {
            let g_idx = if let Some(g) = weights.weights.remove(&format!("{prefix}.g_idx")) {
                g
            } else {
                let g_idx_vec: Vec<u32> = (0..in_features).map(|r| (r / group_size) as u32).collect();
                Tensor::from_vec(g_idx_vec, (in_features,), device)?
            };
            Ok(Linear::new_gptq(qweight, qzeros, scales, g_idx, bits, group_size))
        }
    } else {
        let quant_factory =
            factory_from_config(config.quantization.as_ref(), weights.quant_schema.as_ref())?;
        let mut source = WeightSource::new(&mut weights.weights, &mut weights.quantized)
            .with_gguf(&mut weights.gguf_weights);
        let method = quant_factory.build_linear(prefix, &mut source)?;
        Ok(Linear::from_method(method))
    }
}

#[cfg(feature = "candle-backend")]
impl LlamaModel {
    pub fn new(config: &ModelConfig, mut weights: WeightMap, device: &Device) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let _quant_factory =
            factory_from_config(config.quantization.as_ref(), weights.quant_schema.as_ref())
                .context("building quantization method factory")?;

        // When INT8 quantization is active, weights may have been loaded to CPU
        // to avoid allocating full BF16 tensors on GPU.  Non-quantized tensors
        // (embeddings, layernorms) must be moved to the target device before use.
        let to_device = |t: Tensor| -> candle_core::Result<Tensor> {
            if t.device().is_cpu() && !device.is_cpu() { t.to_device(device) } else { Ok(t) }
        };

        // Embedding
        let embed_weight = weights
            .weights
            .remove("model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("missing model.embed_tokens.weight"))?;
        let embed_weight = to_device(embed_weight).map_err(|e| anyhow::anyhow!("{e}"))?;
        let embed_tokens = Linear::new(embed_weight);

        // LM head - may be tied with embedding
        let lm_head = if weights.weights.contains_key("lm_head.qweight")
            || weights.weights.contains_key("lm_head.weight")
        {
            load_linear("lm_head", &mut weights, config, device)?
        } else {
            tracing::info!("lm_head is tied with embed_tokens, reusing embedding weight");
            Linear::new(embed_tokens.weight()?.clone())
        };

        let rms_norm_eps = 1e-6;

        // Build decoder layers
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut quantized_linears = usize::from(lm_head.is_quantized());
        let mut unquantized_linears = usize::from(!lm_head.is_quantized());
        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");

            let q_proj =
                load_linear(&format!("{prefix}.self_attn.q_proj"), &mut weights, config, device)?;
            let k_proj =
                load_linear(&format!("{prefix}.self_attn.k_proj"), &mut weights, config, device)?;
            let v_proj =
                load_linear(&format!("{prefix}.self_attn.v_proj"), &mut weights, config, device)?;
            let o_proj =
                load_linear(&format!("{prefix}.self_attn.o_proj"), &mut weights, config, device)?;

            let attn = LlamaAttention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                num_heads,
                num_kv_heads,
                head_dim,
            );

            let gate_proj =
                load_linear(&format!("{prefix}.mlp.gate_proj"), &mut weights, config, device)?;
            let up_proj =
                load_linear(&format!("{prefix}.mlp.up_proj"), &mut weights, config, device)?;
            let down_proj =
                load_linear(&format!("{prefix}.mlp.down_proj"), &mut weights, config, device)?;
            let mlp = LlamaMLP::new(gate_proj, up_proj, down_proj);

            let linears = [
                attn.q_proj(),
                attn.k_proj(),
                attn.v_proj(),
                attn.o_proj(),
                mlp.gate_proj(),
                mlp.up_proj(),
                mlp.down_proj(),
            ];
            for lin in linears {
                if lin.is_quantized() {
                    quantized_linears += 1;
                } else {
                    unquantized_linears += 1;
                }
            }

            let input_ln_w = weights
                .weights
                .remove(&format!("{prefix}.input_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.input_layernorm.weight"))?;
            let input_ln_w = to_device(input_ln_w).map_err(|e| anyhow::anyhow!("{e}"))?;
            let post_attn_ln_w = weights
                .weights
                .remove(&format!("{prefix}.post_attention_layernorm.weight"))
                .ok_or_else(|| {
                    anyhow::anyhow!("missing {prefix}.post_attention_layernorm.weight")
                })?;
            let post_attn_ln_w = to_device(post_attn_ln_w).map_err(|e| anyhow::anyhow!("{e}"))?;

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
        let final_norm_w = to_device(final_norm_w).map_err(|e| anyhow::anyhow!("{e}"))?;
        let norm = RmsNorm::new(final_norm_w, rms_norm_eps);

        // Release any remaining weight allocations before building the model.
        // After this point, only the tensors inside layers/norm/embed/lm_head
        // should be alive.  Log unconsumed names as a diagnostic.
        weights.shrink_to_fit();
        if !weights.is_empty() {
            tracing::warn!(
                unconsumed = ?weights.unconsumed_names(),
                "some weights were not consumed during model construction"
            );
        }
        // Log if we performed CPU→GPU transfer for non-quantized weights
        // (happens when INT8 quantization is active with CUDA).
        let is_int8 = config
            .quantization
            .as_ref()
            .is_some_and(|q| q.kind == rllm_core::config::QuantizationKind::Int8);
        if is_int8 && !device.is_cpu() {
            tracing::info!(
                "transferred non-quantized weights from CPU to GPU after INT8 quantization"
            );
        }

        // Log quantization mode, bits, and strategy at startup.
        if let Some(ref q) = config.quantization {
            let strategy = weights
                .quant_schema
                .as_ref()
                .and_then(|s| s.weight_strategy.clone())
                .unwrap_or_else(|| "channel".to_string());
            tracing::info!(
                kind = ?q.kind,
                bits = ?q.bits,
                strategy = %strategy,
                "Quantization configured at startup"
            );
        } else {
            tracing::info!("Quantization not configured at startup (running in full precision)");
        }

        drop(weights);

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
        tracing::info!(
            quantized_linears,
            unquantized_linears,
            embed_tokens_quantized = false,
            lm_head_quantized = lm_head.is_quantized(),
            "Llama quantization summary"
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config: config.clone(),
            device: device.clone(),
            quantized_layer_count: quantized_linears,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let hidden_states = embedding_lookup(self.embed_tokens.weight()?, input_ids)
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
        let hidden_states = embedding_lookup(self.embed_tokens.weight()?, input_ids)
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

    pub fn collect_gptq_calibrations(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        include_lm_head: bool,
        calibrations: &mut std::collections::BTreeMap<String, GptqCalibration>,
    ) -> Result<()> {
        let mut hidden_states = embedding_lookup(self.embed_tokens.weight()?, input_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = collect_decoder_layer_calibration(
                layer,
                &hidden_states,
                positions,
                &self.rope,
                &format!("model.layers.{i}"),
                calibrations,
            )
            .map_err(|e| anyhow::anyhow!("layer {i}: {e}"))?;
        }

        let hidden_states =
            self.norm.forward(&hidden_states).map_err(|e| anyhow::anyhow!("{e}"))?;
        if include_lm_head {
            observe_linear_input("lm_head", &self.lm_head, &hidden_states, calibrations)?;
        }
        Ok(())
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

#[cfg(feature = "candle-backend")]
fn collect_decoder_layer_calibration(
    layer: &LlamaDecoderLayer,
    hidden_states: &Tensor,
    positions: &[usize],
    rope: &RotaryEmbedding,
    prefix: &str,
    calibrations: &mut std::collections::BTreeMap<String, GptqCalibration>,
) -> Result<Tensor> {
    let residual = hidden_states.clone();
    let normed = layer.input_layernorm().forward(hidden_states)?;
    let attn_out = collect_attention_calibration(
        layer.self_attn(),
        &normed,
        positions,
        rope,
        prefix,
        calibrations,
    )?;
    let hidden_states = (residual + attn_out)?;

    let residual = hidden_states.clone();
    let normed = layer.post_attention_layernorm().forward(&hidden_states)?;
    let mlp_out = collect_mlp_calibration(layer.mlp(), &normed, prefix, calibrations)?;
    Ok((residual + mlp_out)?)
}

#[cfg(feature = "candle-backend")]
fn collect_attention_calibration(
    attn: &LlamaAttention,
    hidden_states: &Tensor,
    positions: &[usize],
    rope: &RotaryEmbedding,
    prefix: &str,
    calibrations: &mut std::collections::BTreeMap<String, GptqCalibration>,
) -> Result<Tensor> {
    let (bsz, seq_len, _) = hidden_states.dims3()?;

    observe_linear_input(
        &format!("{prefix}.self_attn.q_proj"),
        attn.q_proj(),
        hidden_states,
        calibrations,
    )?;
    let q = attn.q_proj().forward(hidden_states)?;
    observe_linear_input(
        &format!("{prefix}.self_attn.k_proj"),
        attn.k_proj(),
        hidden_states,
        calibrations,
    )?;
    let k = attn.k_proj().forward(hidden_states)?;
    observe_linear_input(
        &format!("{prefix}.self_attn.v_proj"),
        attn.v_proj(),
        hidden_states,
        calibrations,
    )?;
    let v = attn.v_proj().forward(hidden_states)?;

    let q = q.reshape((bsz, seq_len, attn.num_heads(), attn.head_dim()))?.transpose(1, 2)?;
    let k = k.reshape((bsz, seq_len, attn.num_kv_heads(), attn.head_dim()))?.transpose(1, 2)?;
    let v = v.reshape((bsz, seq_len, attn.num_kv_heads(), attn.head_dim()))?.transpose(1, 2)?;
    let (q, k) = rope.apply(&q, &k, positions)?;
    let (k, v) = if attn.num_kv_heads() < attn.num_heads() {
        let n_rep = attn.num_heads() / attn.num_kv_heads();
        (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?)
    } else {
        (k, v)
    };

    let scale = 1.0f32 / (attn.head_dim() as f32).sqrt();
    let attn_weights =
        q.matmul(&k.t()?)?.broadcast_mul(&Tensor::new(scale, q.device())?.to_dtype(q.dtype())?)?;
    let attn_weights = if seq_len > 1 {
        let mask = causal_mask(seq_len, q.device())?.to_dtype(q.dtype())?;
        attn_weights.broadcast_add(&mask)?
    } else {
        attn_weights
    };
    let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
    let attn_output = attn_weights.matmul(&v)?;
    let attn_output =
        attn_output.transpose(1, 2)?.reshape((bsz, seq_len, attn.num_heads() * attn.head_dim()))?;

    observe_linear_input(
        &format!("{prefix}.self_attn.o_proj"),
        attn.o_proj(),
        &attn_output,
        calibrations,
    )?;
    attn.o_proj().forward(&attn_output).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(feature = "candle-backend")]
fn collect_mlp_calibration(
    mlp: &LlamaMLP,
    hidden_states: &Tensor,
    prefix: &str,
    calibrations: &mut std::collections::BTreeMap<String, GptqCalibration>,
) -> Result<Tensor> {
    observe_linear_input(
        &format!("{prefix}.mlp.gate_proj"),
        mlp.gate_proj(),
        hidden_states,
        calibrations,
    )?;
    let gate = mlp.gate_proj().forward(hidden_states)?;
    observe_linear_input(
        &format!("{prefix}.mlp.up_proj"),
        mlp.up_proj(),
        hidden_states,
        calibrations,
    )?;
    let up = mlp.up_proj().forward(hidden_states)?;
    let gate = gate.silu()?;
    let down_input = gate.broadcast_mul(&up)?;
    observe_linear_input(
        &format!("{prefix}.mlp.down_proj"),
        mlp.down_proj(),
        &down_input,
        calibrations,
    )?;
    mlp.down_proj().forward(&down_input).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(feature = "candle-backend")]
fn observe_linear_input(
    name: &str,
    linear: &Linear,
    input: &Tensor,
    calibrations: &mut std::collections::BTreeMap<String, GptqCalibration>,
) -> Result<()> {
    let weight = linear.weight()?;
    let in_features = weight.dim(D::Minus1)?;
    let input = input.to_dtype(DType::F32)?;
    let numel = input.dims().iter().product::<usize>();
    let samples = input.reshape((numel / in_features, in_features))?.to_vec2::<f32>()?;
    let calib =
        calibrations.entry(name.to_string()).or_insert_with(|| GptqCalibration::new(in_features));
    calib.observe(&samples)
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

    fn build_toy_weight_map(config: &ModelConfig, include_lm_head: bool) -> WeightMap {
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
        if include_lm_head {
            weights.insert(
                "lm_head.weight".into(),
                Tensor::randn(0.0f32, 0.02f32, (config.vocab_size, config.hidden_size), &device)
                    .unwrap(),
            );
        }

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

        WeightMap {
            weights,
            quantized: HashMap::new(),
            gguf_weights: HashMap::new(),
            quant_schema: None,
            device: device.clone(),
        }
    }

    fn build_toy_model(config: &ModelConfig) -> LlamaForCausalLM {
        let weight_map = build_toy_weight_map(config, false);
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

    #[test]
    fn standalone_lm_head_is_quantized_when_int8_requested() -> Result<()> {
        let mut config = toy_config();
        config.quantization = Some(rllm_core::config::QuantizationConfig {
            kind: rllm_core::config::QuantizationKind::Int8,
            group_size: None,
            bits: Some(8),
        });
        let weight_map = build_toy_weight_map(&config, true);
        let model = LlamaForCausalLM::from_weights(config, weight_map)?;

        assert!(model.model.lm_head.is_quantized());
        assert!(!model.model.embed_tokens.is_quantized());
        Ok(())
    }
}
