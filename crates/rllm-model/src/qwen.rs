use std::{collections::BTreeSet, path::Path};

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{D, DType, Device, Tensor};
use candle_transformers::fused_moe::FusedMoeGGUF;
use rllm_core::config::ModelConfig;
use serde::Deserialize;

use crate::{
    layers::{Linear, LlamaAttention, RmsNorm},
    llama::load_linear,
    loader::WeightMap,
    registry::{CausalLM, Model},
    rope::RotaryEmbedding,
};

/// Supported text-only Qwen decoder families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenArchitecture {
    Qwen2,
    Qwen2Moe,
    Qwen3,
    Qwen3Moe,
}

impl QwenArchitecture {
    pub fn resolve(identifier: &str) -> Result<Self> {
        match identifier {
            "Qwen2ForCausalLM" | "qwen2" => Ok(Self::Qwen2),
            "Qwen2MoeForCausalLM" | "qwen2_moe" => Ok(Self::Qwen2Moe),
            "Qwen3ForCausalLM" | "qwen3" => Ok(Self::Qwen3),
            "Qwen3MoeForCausalLM" | "qwen3_moe" => Ok(Self::Qwen3Moe),
            unsupported => anyhow::bail!(
                "unsupported Qwen architecture '{unsupported}'; supported text architectures are \
                 Qwen2ForCausalLM, Qwen2MoeForCausalLM, Qwen3ForCausalLM, and \
                 Qwen3MoeForCausalLM"
            ),
        }
    }

    fn is_qwen3(self) -> bool {
        matches!(self, Self::Qwen3 | Self::Qwen3Moe)
    }

    fn is_moe(self) -> bool {
        matches!(self, Self::Qwen2Moe | Self::Qwen3Moe)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenCapabilities {
    pub paged_attention: bool,
    pub tensor_parallel: bool,
    pub pipeline_parallel: bool,
    pub lora: bool,
    pub moe: bool,
}

impl QwenCapabilities {
    pub fn for_architecture(architecture: QwenArchitecture) -> Self {
        Self {
            paged_attention: true,
            tensor_parallel: false,
            pipeline_parallel: false,
            lora: false,
            moe: architecture.is_moe(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: Option<String>,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    hidden_act: String,
    rms_norm_eps: f64,
    rope_theta: f64,
    max_position_embeddings: usize,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    attention_dropout: f64,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "one")]
    decoder_sparse_step: usize,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    norm_topk_prob: bool,
}

fn one() -> usize {
    1
}

impl QwenConfig {
    pub fn from_json(content: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(content).context("deserializing Qwen config")?;
        config.validate_fields()?;
        Ok(config)
    }

    pub fn architecture(&self) -> Result<QwenArchitecture> {
        let identifier = self
            .architectures
            .first()
            .map(String::as_str)
            .or(self.model_type.as_deref())
            .ok_or_else(|| {
                anyhow::anyhow!("Qwen config requires 'architectures' or 'model_type'")
            })?;
        QwenArchitecture::resolve(identifier)
    }

    /// Build the Qwen-specific configuration carried by a GGUF checkpoint.
    pub fn from_gguf(path: &Path, core: &ModelConfig) -> Result<Self> {
        let architecture = QwenArchitecture::resolve(&core.architecture)?;
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("opening GGUF file {}", path.display()))?;
        let content =
            Content::read(&mut file).map_err(|e| anyhow::anyhow!("reading GGUF content: {e}"))?;
        let gguf_arch = content
            .metadata
            .get("general.architecture")
            .and_then(|value| value.to_string().ok())
            .ok_or_else(|| anyhow::anyhow!("GGUF metadata is missing general.architecture"))?;
        let rms_norm_key = format!("{gguf_arch}.attention.layer_norm_rms_epsilon");
        let rms_norm_eps = content
            .metadata
            .get(&rms_norm_key)
            .and_then(|value| match value {
                Value::F32(value) => Some(*value as f64),
                Value::F64(value) => Some(*value),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("GGUF metadata is missing {rms_norm_key}"))?;
        let tie_word_embeddings = !content.tensor_infos.contains_key("output.weight");
        let metadata_usize = |suffix: &str| {
            let key = format!("{gguf_arch}.{suffix}");
            content
                .metadata
                .get(&key)
                .and_then(|value| value.to_u64().ok())
                .map(|value| value as usize)
        };
        let num_experts = metadata_usize("expert_count");
        let num_experts_per_tok = metadata_usize("expert_used_count");
        let moe_intermediate_size = metadata_usize("expert_feed_forward_length");
        let shared_expert_intermediate_size =
            metadata_usize("expert_shared_feed_forward_length").filter(|&width| width > 0);

        let config = Self {
            architectures: vec![core.architecture.clone()],
            model_type: None,
            vocab_size: core.vocab_size,
            hidden_size: core.hidden_size,
            intermediate_size: core.intermediate_size,
            num_hidden_layers: core.num_layers,
            num_attention_heads: core.num_attention_heads,
            num_key_value_heads: core.num_kv_heads,
            head_dim: Some(core.head_dim),
            hidden_act: "silu".to_string(),
            rms_norm_eps,
            rope_theta: core.rope_theta as f64,
            max_position_embeddings: core.max_model_len,
            rope_scaling: None,
            attention_bias: None,
            attention_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: None,
            tie_word_embeddings,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            // Qwen3 normalizes selected routes; Qwen2-MoE combines routed and
            // separately gated shared experts without route renormalization.
            norm_topk_prob: architecture == QwenArchitecture::Qwen3Moe,
        };
        config.validate_fields()?;
        Ok(config)
    }

    fn head_dim(&self) -> Result<usize> {
        if let Some(head_dim) = self.head_dim {
            return Ok(head_dim);
        }
        self.hidden_size
            .checked_div(self.num_attention_heads)
            .ok_or_else(|| anyhow::anyhow!("num_attention_heads must be greater than zero"))
    }

    fn attention_bias(&self, architecture: QwenArchitecture) -> bool {
        if architecture == QwenArchitecture::Qwen2 || architecture == QwenArchitecture::Qwen2Moe {
            true
        } else {
            self.attention_bias.unwrap_or(false)
        }
    }

    fn validate_fields(&self) -> Result<()> {
        let architecture = self.architecture()?;
        let label = self
            .architectures
            .first()
            .map(String::as_str)
            .or(self.model_type.as_deref())
            .unwrap_or("Qwen");
        for (field, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                anyhow::bail!("{label}: field '{field}' has value 0; expected a positive integer");
            }
        }
        if self.num_key_value_heads > self.num_attention_heads
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            anyhow::bail!(
                "{label}: field 'num_key_value_heads' has value {}; expected a positive divisor of num_attention_heads ({})",
                self.num_key_value_heads,
                self.num_attention_heads
            );
        }
        let head_dim = self.head_dim()?;
        if head_dim == 0 || head_dim % 2 != 0 {
            anyhow::bail!(
                "{label}: field 'head_dim' has value {head_dim}; expected a positive even integer"
            );
        }
        self.num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
            anyhow::anyhow!("{label}: num_attention_heads * head_dim overflows usize")
        })?;
        self.num_key_value_heads.checked_mul(head_dim).ok_or_else(|| {
            anyhow::anyhow!("{label}: num_key_value_heads * head_dim overflows usize")
        })?;
        if !architecture.is_qwen3() && self.hidden_size % self.num_attention_heads != 0 {
            anyhow::bail!(
                "{label}: hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size,
                self.num_attention_heads
            );
        }
        if self.hidden_act != "silu" {
            anyhow::bail!(
                "{label}: field 'hidden_act' has value '{}'; only 'silu' is supported",
                self.hidden_act
            );
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            anyhow::bail!(
                "{label}: field 'rms_norm_eps' has value {}; expected a finite positive number",
                self.rms_norm_eps
            );
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            anyhow::bail!(
                "{label}: field 'rope_theta' has value {}; expected a finite positive number",
                self.rope_theta
            );
        }
        if self.rope_scaling.is_some() {
            anyhow::bail!(
                "{label}: field 'rope_scaling' is configured, but scaled Qwen RoPE is not supported"
            );
        }
        if self.use_sliding_window {
            anyhow::bail!(
                "{label}: field 'use_sliding_window' is true (sliding_window={:?}), but sliding-window attention is not fully implemented in rLLM",
                self.sliding_window
            );
        }
        if self.attention_dropout != 0.0 {
            anyhow::bail!(
                "{label}: field 'attention_dropout' has value {}; inference requires 0",
                self.attention_dropout
            );
        }
        if architecture.is_moe() {
            let experts = self
                .num_experts
                .ok_or_else(|| anyhow::anyhow!("{label}: MoE field 'num_experts' is required"))?;
            let top_k = self.num_experts_per_tok.ok_or_else(|| {
                anyhow::anyhow!("{label}: MoE field 'num_experts_per_tok' is required")
            })?;
            let width = self.moe_intermediate_size.ok_or_else(|| {
                anyhow::anyhow!("{label}: MoE field 'moe_intermediate_size' is required")
            })?;
            if experts == 0 || top_k == 0 || top_k > experts || width == 0 {
                anyhow::bail!(
                    "{label}: invalid MoE fields num_experts={experts}, num_experts_per_tok={top_k}, moe_intermediate_size={width}; expected experts > 0, 0 < top-k <= experts, and width > 0"
                );
            }
            if self.decoder_sparse_step == 0 {
                anyhow::bail!(
                    "{label}: field 'decoder_sparse_step' has value 0; expected a positive integer"
                );
            }
            if self.mlp_only_layers.iter().any(|&layer| layer >= self.num_hidden_layers) {
                anyhow::bail!(
                    "{label}: field 'mlp_only_layers' contains an index outside num_hidden_layers ({})",
                    self.num_hidden_layers
                );
            }
        }
        Ok(())
    }

    fn validate_core(&self, core: &ModelConfig) -> Result<()> {
        let architecture = self.architecture()?;
        if architecture != QwenArchitecture::resolve(&core.architecture)? {
            anyhow::bail!(
                "Qwen config architecture does not match resolved core architecture '{}'",
                core.architecture
            );
        }
        let expected = [
            ("vocab_size", self.vocab_size, core.vocab_size),
            ("hidden_size", self.hidden_size, core.hidden_size),
            ("intermediate_size", self.intermediate_size, core.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers, core.num_layers),
            ("num_attention_heads", self.num_attention_heads, core.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads, core.num_kv_heads),
            ("head_dim", self.head_dim()?, core.head_dim),
        ];
        for (field, config_value, core_value) in expected {
            if config_value != core_value {
                anyhow::bail!(
                    "{}: field '{field}' has checkpoint value {config_value}, but runtime config has {core_value}",
                    core.architecture
                );
            }
        }
        Ok(())
    }
}

pub struct QwenForCausalLM {
    model: QwenModel,
    config: ModelConfig,
}

impl QwenForCausalLM {
    pub fn factory(_config: &ModelConfig) -> Result<Box<dyn CausalLM>> {
        anyhow::bail!("QwenForCausalLM::factory requires loaded weights")
    }

    pub fn parse_config(path: &Path) -> Result<QwenConfig> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading Qwen config from {}", path.display()))?;
        QwenConfig::from_json(&content)
            .with_context(|| format!("parsing Qwen config from {}", path.display()))
    }

    pub fn from_weights(
        config: ModelConfig,
        qwen_config: QwenConfig,
        weights: WeightMap,
    ) -> Result<Self> {
        qwen_config.validate_core(&config)?;
        let model = QwenModel::new(&config, &qwen_config, weights)?;
        Ok(Self { model, config })
    }
}

impl Model for QwenForCausalLM {
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
        self.model.forward(input_ids, positions, kv_cache)
    }

    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        cache: &rllm_kernels::cache_ops::GpuKVCache,
        metadata: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        self.model.forward_paged(input_ids, positions, cache, metadata)
    }
}

impl CausalLM for QwenForCausalLM {
    fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            anyhow::bail!("cannot generate from an empty prompt");
        }
        let mut tokens = prompt.to_vec();
        if tokens.len() >= max_tokens {
            return Ok(tokens);
        }
        let mut cache = (0..self.config.num_layers).map(|_| None).collect::<Vec<_>>();
        let input = Tensor::new(prompt, self.model.device())?.reshape((1, prompt.len()))?;
        let positions = (0..prompt.len()).collect::<Vec<_>>();
        let mut logits = self.forward(&input, &positions, &mut cache)?;
        while tokens.len() < max_tokens {
            let last = logits.narrow(1, logits.dim(1)? - 1, 1)?;
            let next = argmax(&last)?;
            tokens.push(next);
            if tokens.len() == max_tokens {
                break;
            }
            let position = tokens.len() - 1;
            let input = Tensor::new(&[next], self.model.device())?.reshape((1, 1))?;
            logits = self.forward(&input, &[position], &mut cache)?;
        }
        Ok(tokens)
    }
}

struct DenseMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl DenseMlp {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(hidden)?.silu()?;
        let up = self.up.forward(hidden)?;
        self.down.forward(&gate.broadcast_mul(&up)?).map_err(Into::into)
    }

    fn quantized_count(&self) -> usize {
        usize::from(self.gate.is_quantized())
            + usize::from(self.up.is_quantized())
            + usize::from(self.down.is_quantized())
    }
}

enum QwenExperts {
    Dense { router: Tensor, experts: Vec<DenseMlp>, top_k: usize, normalize: bool },
    Gguf(FusedMoeGGUF),
}

struct QwenMoe {
    experts: QwenExperts,
    shared_expert: Option<DenseMlp>,
    shared_expert_gate: Option<Linear>,
}

impl QwenMoe {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let shape = hidden.shape().clone();
        let width = hidden.dim(D::Minus1)?;
        let tokens = hidden.elem_count() / width;
        let flat = hidden.reshape((tokens, width))?;
        let mut output = match &self.experts {
            QwenExperts::Gguf(experts) => {
                // Match vLLM route selection: vector kernels serve decode and
                // small batches; grouped prefill kernels amortize setup above 64 tokens.
                experts
                    .forward(hidden, tokens > 64)
                    .context("executing grouped GGUF Qwen experts")?
            }
            QwenExperts::Dense { router, experts, top_k, normalize } => {
                let logits =
                    flat.to_dtype(DType::F32)?.matmul(&router.to_dtype(DType::F32)?.t()?)?;
                let probabilities = candle_nn::ops::softmax_last_dim(&logits)?;
                let ids = probabilities
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, *top_k)?
                    .contiguous()?;
                let mut route_weights = probabilities.gather(&ids, D::Minus1)?;
                if *normalize && *top_k > 1 {
                    route_weights = route_weights
                        .broadcast_div(&(route_weights.sum_keepdim(D::Minus1)? + 1e-20)?)?;
                }

                let cpu_ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
                let cpu_weights = route_weights
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?;
                let mut routed = flat.zeros_like()?;
                for (expert_id, expert) in experts.iter().enumerate() {
                    let mut token_indices = Vec::new();
                    let mut weights = Vec::new();
                    for token in 0..tokens {
                        for route in 0..*top_k {
                            if cpu_ids[token][route] as usize == expert_id {
                                token_indices.push(token as u32);
                                weights.push(cpu_weights[token][route]);
                            }
                        }
                    }
                    if token_indices.is_empty() {
                        continue;
                    }
                    let indices = Tensor::from_vec(token_indices, weights.len(), hidden.device())?;
                    let selected = expert.forward(&flat.index_select(&indices, 0)?)?;
                    let weights =
                        Tensor::from_vec(weights, (indices.elem_count(), 1), hidden.device())?
                            .to_dtype(selected.dtype())?;
                    routed = routed.index_add(&indices, &selected.broadcast_mul(&weights)?, 0)?;
                }
                routed.reshape(&shape)?
            }
        };

        if let Some(shared) = &self.shared_expert {
            let mut shared_output = shared.forward(hidden)?;
            if let Some(gate) = &self.shared_expert_gate {
                let gate = candle_nn::ops::sigmoid(&gate.forward(hidden)?)?
                    .to_dtype(shared_output.dtype())?;
                shared_output = shared_output.broadcast_mul(&gate)?;
            }
            output = (output + shared_output)?;
        }
        Ok(output)
    }

    fn quantized_count(&self) -> usize {
        let routed = match &self.experts {
            QwenExperts::Dense { experts, .. } => {
                experts.iter().map(DenseMlp::quantized_count).sum::<usize>()
            }
            QwenExperts::Gguf(_) => 3,
        };
        routed
            + self.shared_expert.as_ref().map(DenseMlp::quantized_count).unwrap_or(0)
            + self.shared_expert_gate.as_ref().map(|x| usize::from(x.is_quantized())).unwrap_or(0)
    }
}

enum FeedForward {
    Dense(DenseMlp),
    Moe(QwenMoe),
}

impl FeedForward {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden),
            Self::Moe(moe) => moe.forward(hidden),
        }
    }
    fn quantized_count(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.quantized_count(),
            Self::Moe(moe) => moe.quantized_count(),
        }
    }
}

struct QwenDecoderLayer {
    attention: LlamaAttention,
    feed_forward: FeedForward,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
}

impl QwenDecoderLayer {
    fn forward(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        cache: &mut Option<(Tensor, Tensor)>,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let attention =
            self.attention.forward(&self.input_norm.forward(hidden)?, positions, cache, rope)?;
        let hidden = (hidden + attention)?;
        let feed_forward =
            self.feed_forward.forward(&self.post_attention_norm.forward(&hidden)?)?;
        (hidden + feed_forward).map_err(Into::into)
    }

    fn forward_paged(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        cache: &rllm_kernels::cache_ops::GpuKVCache,
        metadata: &rllm_kernels::AttentionMetadata,
        layer: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let attention = self.attention.forward_paged(
            &self.input_norm.forward(hidden)?,
            positions,
            cache,
            metadata,
            layer,
            rope,
        )?;
        let hidden = (hidden + attention)?;
        let feed_forward =
            self.feed_forward.forward(&self.post_attention_norm.forward(&hidden)?)?;
        (hidden + feed_forward).map_err(Into::into)
    }
}

struct QwenModel {
    embed_tokens: Linear,
    layers: Vec<QwenDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    device: Device,
    quantized_layer_count: usize,
}

impl QwenModel {
    fn new(core: &ModelConfig, config: &QwenConfig, mut weights: WeightMap) -> Result<Self> {
        let device = weights.device.clone();
        let architecture = config.architecture()?;
        let embed_weight = take_tensor(&mut weights, "model.embed_tokens.weight", &device)?;
        let embed_tokens = Linear::new(embed_weight);
        let lm_head = if has_linear(&weights, "lm_head") {
            load_linear("lm_head", &mut weights, core, &device)?
        } else if config.tie_word_embeddings {
            Linear::new(embed_tokens.weight()?.clone())
        } else {
            anyhow::bail!(
                "{}: missing lm_head.weight while tie_word_embeddings is false",
                core.architecture
            );
        };

        let mlp_only = config.mlp_only_layers.iter().copied().collect::<BTreeSet<_>>();
        let mut layers = Vec::with_capacity(core.num_layers);
        let mut quantized = usize::from(lm_head.is_quantized());
        for layer in 0..core.num_layers {
            let prefix = format!("model.layers.{layer}");
            let projection_bias = config.attention_bias(architecture);
            let q = load_projection(
                &format!("{prefix}.self_attn.q_proj"),
                &mut weights,
                core,
                &device,
                projection_bias,
            )?;
            let k = load_projection(
                &format!("{prefix}.self_attn.k_proj"),
                &mut weights,
                core,
                &device,
                projection_bias,
            )?;
            let v = load_projection(
                &format!("{prefix}.self_attn.v_proj"),
                &mut weights,
                core,
                &device,
                projection_bias,
            )?;
            let o = load_projection(
                &format!("{prefix}.self_attn.o_proj"),
                &mut weights,
                core,
                &device,
                architecture.is_qwen3() && projection_bias,
            )?;
            let mut attention = LlamaAttention::new(
                q,
                k,
                v,
                o,
                core.num_attention_heads,
                core.num_kv_heads,
                core.head_dim,
            );
            if architecture.is_qwen3() {
                attention = attention.with_qk_norm(
                    RmsNorm::new(
                        take_tensor(
                            &mut weights,
                            &format!("{prefix}.self_attn.q_norm.weight"),
                            &device,
                        )?,
                        config.rms_norm_eps,
                    ),
                    RmsNorm::new(
                        take_tensor(
                            &mut weights,
                            &format!("{prefix}.self_attn.k_norm.weight"),
                            &device,
                        )?,
                        config.rms_norm_eps,
                    ),
                );
            }
            for projection in
                [attention.q_proj(), attention.k_proj(), attention.v_proj(), attention.o_proj()]
            {
                quantized += usize::from(projection.is_quantized());
            }

            let sparse = architecture.is_moe()
                && !mlp_only.contains(&layer)
                && (layer + 1) % config.decoder_sparse_step == 0;
            let feed_forward = if sparse {
                FeedForward::Moe(load_moe(
                    &format!("{prefix}.mlp"),
                    core,
                    config,
                    architecture,
                    &mut weights,
                    &device,
                )?)
            } else {
                FeedForward::Dense(load_mlp(&format!("{prefix}.mlp"), core, &mut weights, &device)?)
            };
            quantized += feed_forward.quantized_count();
            layers.push(QwenDecoderLayer {
                attention,
                feed_forward,
                input_norm: RmsNorm::new(
                    take_tensor(
                        &mut weights,
                        &format!("{prefix}.input_layernorm.weight"),
                        &device,
                    )?,
                    config.rms_norm_eps,
                ),
                post_attention_norm: RmsNorm::new(
                    take_tensor(
                        &mut weights,
                        &format!("{prefix}.post_attention_layernorm.weight"),
                        &device,
                    )?,
                    config.rms_norm_eps,
                ),
            });
        }
        let norm = RmsNorm::new(
            take_tensor(&mut weights, "model.norm.weight", &device)?,
            config.rms_norm_eps,
        );
        if !weights.is_empty() {
            tracing::warn!(unconsumed = ?weights.unconsumed_names(), "unconsumed Qwen checkpoint tensors");
        }
        let rope = RotaryEmbedding::new(
            core.head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            &device,
        )
        .context("creating Qwen rotary embeddings")?;
        tracing::info!(
            architecture = ?architecture,
            layers = core.num_layers,
            q_heads = core.num_attention_heads,
            kv_heads = core.num_kv_heads,
            head_dim = core.head_dim,
            quantized_linears = quantized,
            "loaded Qwen causal language model"
        );
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            device,
            quantized_layer_count: quantized,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        if cache.len() != self.layers.len() {
            anyhow::bail!(
                "Qwen KV cache has {} layers; expected {}",
                cache.len(),
                self.layers.len()
            );
        }
        let mut hidden = embedding_lookup(self.embed_tokens.weight()?, input_ids)?;
        for (layer, decoder) in self.layers.iter().enumerate() {
            hidden = decoder
                .forward(&hidden, positions, &mut cache[layer], &self.rope)
                .with_context(|| format!("Qwen layer {layer}"))?;
        }
        self.lm_head.forward(&self.norm.forward(&hidden)?).map_err(Into::into)
    }

    fn forward_paged(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        cache: &rllm_kernels::cache_ops::GpuKVCache,
        metadata: &rllm_kernels::AttentionMetadata,
    ) -> Result<Tensor> {
        let mut hidden = embedding_lookup(self.embed_tokens.weight()?, input_ids)?;
        for (layer, decoder) in self.layers.iter().enumerate() {
            hidden = decoder
                .forward_paged(&hidden, positions, cache, metadata, layer, &self.rope)
                .with_context(|| format!("Qwen layer {layer}"))?;
        }
        self.lm_head.forward(&self.norm.forward(&hidden)?).map_err(Into::into)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

fn load_projection(
    prefix: &str,
    weights: &mut WeightMap,
    config: &ModelConfig,
    device: &Device,
    bias: bool,
) -> Result<Linear> {
    let projection = load_linear(prefix, weights, config, device)
        .with_context(|| format!("loading Qwen projection {prefix}"))?;
    if bias {
        Ok(projection.with_bias(take_tensor(weights, &format!("{prefix}.bias"), device)?))
    } else {
        Ok(projection)
    }
}

fn load_mlp(
    prefix: &str,
    config: &ModelConfig,
    weights: &mut WeightMap,
    device: &Device,
) -> Result<DenseMlp> {
    Ok(DenseMlp {
        gate: load_linear(&format!("{prefix}.gate_proj"), weights, config, device)?,
        up: load_linear(&format!("{prefix}.up_proj"), weights, config, device)?,
        down: load_linear(&format!("{prefix}.down_proj"), weights, config, device)?,
    })
}

fn load_moe(
    prefix: &str,
    core: &ModelConfig,
    config: &QwenConfig,
    architecture: QwenArchitecture,
    weights: &mut WeightMap,
    device: &Device,
) -> Result<QwenMoe> {
    let num_experts = config.num_experts.ok_or_else(|| anyhow::anyhow!("missing num_experts"))?;
    let top_k =
        config.num_experts_per_tok.ok_or_else(|| anyhow::anyhow!("missing num_experts_per_tok"))?;
    let expert_width = config
        .moe_intermediate_size
        .ok_or_else(|| anyhow::anyhow!("missing moe_intermediate_size"))?;

    let packed_gate_name = format!("{prefix}.experts.gate_proj.weight");
    let packed_up_name = format!("{prefix}.experts.up_proj.weight");
    let packed_down_name = format!("{prefix}.experts.down_proj.weight");
    let has_packed_experts = weights.gguf_weights.contains_key(&packed_gate_name)
        || weights.weights.contains_key(&packed_gate_name);

    let experts = if has_packed_experts {
        if !matches!(device, Device::Cuda(_)) {
            anyhow::bail!(
                "Qwen GGUF MoE requires CUDA grouped-expert kernels; device is {device:?}"
            );
        }
        let gate_experts = take_gguf_experts(
            weights,
            &packed_gate_name,
            &[num_experts, expert_width, core.hidden_size],
        )?;
        let up_experts = take_gguf_experts(
            weights,
            &packed_up_name,
            &[num_experts, expert_width, core.hidden_size],
        )?;
        let down_experts = take_gguf_experts(
            weights,
            &packed_down_name,
            &[num_experts, core.hidden_size, expert_width],
        )?;
        let router =
            take_tensor(weights, &format!("{prefix}.gate.weight"), device)?.to_dtype(DType::F32)?;
        if router.dims() != [num_experts, core.hidden_size] {
            anyhow::bail!(
                "Qwen GGUF MoE router {prefix}.gate.weight has shape {:?}; expected [{num_experts}, {}]",
                router.dims(),
                core.hidden_size
            );
        }
        QwenExperts::Gguf(FusedMoeGGUF {
            gate: candle_nn::Linear::new(router, None),
            gate_experts,
            up_experts,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: config.norm_topk_prob,
            num_experts_per_tok: top_k,
            dtype: DType::F16,
        })
    } else {
        let mut dense_experts = Vec::with_capacity(num_experts);
        for expert in 0..num_experts {
            dense_experts.push(
                load_mlp(&format!("{prefix}.experts.{expert}"), core, weights, device)
                    .with_context(|| format!("loading Qwen expert {expert}"))?,
            );
        }
        QwenExperts::Dense {
            router: take_tensor(weights, &format!("{prefix}.gate.weight"), device)?,
            experts: dense_experts,
            top_k,
            normalize: config.norm_topk_prob,
        }
    };

    let (shared_expert, shared_expert_gate) = if architecture == QwenArchitecture::Qwen2Moe {
        if config.shared_expert_intermediate_size.unwrap_or(0) == 0 {
            (None, None)
        } else {
            (
                Some(load_mlp(&format!("{prefix}.shared_expert"), core, weights, device)?),
                Some(load_linear(&format!("{prefix}.shared_expert_gate"), weights, core, device)?),
            )
        }
    } else {
        (None, None)
    };

    Ok(QwenMoe { experts, shared_expert, shared_expert_gate })
}

fn take_gguf_experts(
    weights: &mut WeightMap,
    name: &str,
    expected_shape: &[usize],
) -> Result<std::sync::Arc<QTensor>> {
    let tensor = weights.gguf_weights.remove(name).ok_or_else(|| {
        if weights.weights.contains_key(name) {
            anyhow::anyhow!(
                "Qwen GGUF MoE tensor {name} is not quantized; grouped CUDA experts require Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, or Q8_0"
            )
        } else {
            anyhow::anyhow!("missing Qwen GGUF MoE tensor {name}")
        }
    })?;
    if tensor.shape().dims() != expected_shape {
        anyhow::bail!(
            "Qwen GGUF MoE tensor {name} has shape {:?}; expected {expected_shape:?}",
            tensor.shape().dims()
        );
    }
    if !matches!(
        tensor.dtype(),
        GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    ) {
        anyhow::bail!(
            "Qwen GGUF MoE tensor {name} uses {:?}; grouped CUDA experts support Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0",
            tensor.dtype()
        );
    }
    Ok(tensor)
}

fn take_tensor(weights: &mut WeightMap, name: &str, device: &Device) -> Result<Tensor> {
    let tensor = weights
        .weights
        .remove(name)
        .ok_or_else(|| anyhow::anyhow!("missing Qwen checkpoint tensor '{name}'"))?;
    if tensor.device().is_cpu() && !device.is_cpu() {
        tensor.to_device(device).map_err(Into::into)
    } else {
        Ok(tensor)
    }
}

fn has_linear(weights: &WeightMap, prefix: &str) -> bool {
    weights.weights.contains_key(&format!("{prefix}.weight"))
        || weights.weights.contains_key(&format!("{prefix}.qweight"))
        || weights.quantized.contains_key(&format!("{prefix}.weight"))
        || weights.gguf_weights.contains_key(&format!("{prefix}.weight"))
}

fn embedding_lookup(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let values = ids.flatten_all()?.to_vec1::<u32>()?;
    let batch = ids.dim(0)?;
    let tokens = ids.dim(1)?;
    let hidden = weight.dim(D::Minus1)?;
    let indices = Tensor::from_vec(values, batch * tokens, ids.device())?;
    weight.index_select(&indices, 0)?.reshape((batch, tokens, hidden)).map_err(Into::into)
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let values =
        logits.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    Ok(values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rllm_core::{config::TokenizerMode, dtype::DType as CoreDType};

    use super::*;

    fn qwen_json(architecture: &str, model_type: &str, moe: bool) -> String {
        let moe_fields = if moe {
            r#", "decoder_sparse_step": 1, "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8,
                "num_experts": 2, "num_experts_per_tok": 1, "norm_topk_prob": true"#
        } else {
            ""
        };
        format!(
            r#"{{
            "architectures": ["{architecture}"], "model_type": "{model_type}",
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 8, "hidden_act": "silu",
            "rms_norm_eps": 0.000001, "rope_theta": 10000.0,
            "max_position_embeddings": 32, "tie_word_embeddings": true{moe_fields}
        }}"#
        )
    }

    fn core(architecture: &str) -> ModelConfig {
        ModelConfig {
            model_id: "toy".into(),
            architecture: architecture.into(),
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            max_model_len: 32,
            rope_theta: 10000.0,
            rope_scaling: None,
            dtype: CoreDType::F32,
            quantization: None,
            tokenizer_mode: TokenizerMode::Auto,
        }
    }

    fn weights(architecture: QwenArchitecture, moe: bool) -> Result<WeightMap> {
        let device = Device::Cpu;
        let mut values = HashMap::new();
        let mut add = |name: &str, shape: &[usize]| -> Result<()> {
            values.insert(name.to_string(), Tensor::zeros(shape, DType::F32, &device)?);
            Ok(())
        };
        add("model.embed_tokens.weight", &[32, 16])?;
        add("model.norm.weight", &[16])?;
        add("model.layers.0.input_layernorm.weight", &[16])?;
        add("model.layers.0.post_attention_layernorm.weight", &[16])?;
        add("model.layers.0.self_attn.q_proj.weight", &[16, 16])?;
        add("model.layers.0.self_attn.k_proj.weight", &[8, 16])?;
        add("model.layers.0.self_attn.v_proj.weight", &[8, 16])?;
        add("model.layers.0.self_attn.o_proj.weight", &[16, 16])?;
        if architecture.is_qwen3() {
            add("model.layers.0.self_attn.q_norm.weight", &[8])?;
            add("model.layers.0.self_attn.k_norm.weight", &[8])?;
        } else {
            add("model.layers.0.self_attn.q_proj.bias", &[16])?;
            add("model.layers.0.self_attn.k_proj.bias", &[8])?;
            add("model.layers.0.self_attn.v_proj.bias", &[8])?;
        }
        if moe {
            add("model.layers.0.mlp.gate.weight", &[2, 16])?;
            for expert in 0..2 {
                add(&format!("model.layers.0.mlp.experts.{expert}.gate_proj.weight"), &[8, 16])?;
                add(&format!("model.layers.0.mlp.experts.{expert}.up_proj.weight"), &[8, 16])?;
                add(&format!("model.layers.0.mlp.experts.{expert}.down_proj.weight"), &[16, 8])?;
            }
            if architecture == QwenArchitecture::Qwen2Moe {
                add("model.layers.0.mlp.shared_expert.gate_proj.weight", &[8, 16])?;
                add("model.layers.0.mlp.shared_expert.up_proj.weight", &[8, 16])?;
                add("model.layers.0.mlp.shared_expert.down_proj.weight", &[16, 8])?;
                add("model.layers.0.mlp.shared_expert_gate.weight", &[1, 16])?;
            }
        } else {
            add("model.layers.0.mlp.gate_proj.weight", &[24, 16])?;
            add("model.layers.0.mlp.up_proj.weight", &[24, 16])?;
            add("model.layers.0.mlp.down_proj.weight", &[16, 24])?;
        }
        Ok(WeightMap {
            weights: values,
            quantized: HashMap::new(),
            gguf_weights: HashMap::new(),
            quant_schema: None,
            device,
        })
    }

    #[test]
    fn resolves_supported_architectures_and_rejects_hybrids() {
        assert_eq!(QwenArchitecture::resolve("qwen2").unwrap(), QwenArchitecture::Qwen2);
        assert_eq!(
            QwenArchitecture::resolve("Qwen3MoeForCausalLM").unwrap(),
            QwenArchitecture::Qwen3Moe
        );
        assert!(QwenArchitecture::resolve("Qwen3NextForCausalLM").is_err());
    }

    #[test]
    fn validates_qwen3_independent_head_dimension() {
        let json = qwen_json("Qwen3ForCausalLM", "qwen3", false)
            .replace("\"hidden_size\": 16", "\"hidden_size\": 12");
        assert!(QwenConfig::from_json(&json).is_ok());
    }

    #[test]
    fn rejects_sliding_window_and_rope_scaling() {
        let sliding = qwen_json("Qwen2ForCausalLM", "qwen2", false).replace(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": true, \"use_sliding_window\": true",
        );
        assert!(QwenConfig::from_json(&sliding).is_err());
        let scaling = qwen_json("Qwen2ForCausalLM", "qwen2", false)
            .replace("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": true, \"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0}");
        assert!(QwenConfig::from_json(&scaling).is_err());
    }

    #[test]
    fn dense_qwen2_and_qwen3_forward() -> Result<()> {
        for (name, model_type, architecture) in [
            ("Qwen2ForCausalLM", "qwen2", QwenArchitecture::Qwen2),
            ("Qwen3ForCausalLM", "qwen3", QwenArchitecture::Qwen3),
        ] {
            let qwen = QwenConfig::from_json(&qwen_json(name, model_type, false))?;
            let model =
                QwenForCausalLM::from_weights(core(name), qwen, weights(architecture, false)?)?;
            let input = Tensor::new(&[[1u32, 2]], &Device::Cpu)?;
            let mut cache = vec![None];
            let logits = model.forward(&input, &[0, 1], &mut cache)?;
            assert_eq!(logits.dims(), &[1, 2, 32]);
        }
        Ok(())
    }

    #[test]
    fn qwen2_and_qwen3_moe_forward() -> Result<()> {
        for (name, model_type, architecture) in [
            ("Qwen2MoeForCausalLM", "qwen2_moe", QwenArchitecture::Qwen2Moe),
            ("Qwen3MoeForCausalLM", "qwen3_moe", QwenArchitecture::Qwen3Moe),
        ] {
            let qwen = QwenConfig::from_json(&qwen_json(name, model_type, true))?;
            let model =
                QwenForCausalLM::from_weights(core(name), qwen, weights(architecture, true)?)?;
            let input = Tensor::new(&[[1u32]], &Device::Cpu)?;
            let mut cache = vec![None];
            let logits = model.forward(&input, &[0], &mut cache)?;
            assert_eq!(logits.dims(), &[1, 1, 32]);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gguf_moe_cuda_tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn grouped_kernels_match_dequantized_reference() -> Result<()> {
        const HIDDEN: usize = 256;
        const EXPERT_WIDTH: usize = 256;
        const EXPERTS: usize = 2;

        let device = Device::new_cuda(0)?;
        let make_weight = |phase: f32| -> Result<Tensor> {
            let values = (0..EXPERTS * EXPERT_WIDTH * HIDDEN)
                .map(|index| ((index as f32 * 0.013 + phase).sin()) * 0.02)
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (EXPERTS, EXPERT_WIDTH, HIDDEN), &Device::Cpu)
                .map_err(Into::into)
        };
        let gate_cpu = make_weight(0.0)?;
        let up_cpu = make_weight(0.7)?;
        let down_cpu = make_weight(1.3)?;
        let gate_experts = Arc::new(QTensor::quantize_onto(&gate_cpu, GgmlDType::Q8_0, &device)?);
        let up_experts = Arc::new(QTensor::quantize_onto(&up_cpu, GgmlDType::Q8_0, &device)?);
        let down_experts = Arc::new(QTensor::quantize_onto(&down_cpu, GgmlDType::Q8_0, &device)?);

        let mut router_values = vec![0f32; EXPERTS * HIDDEN];
        router_values[0] = 1.0;
        router_values[HIDDEN] = -1.0;
        let router = Tensor::from_vec(router_values, (EXPERTS, HIDDEN), &device)?;
        let fused = FusedMoeGGUF {
            gate: candle_nn::Linear::new(router, None),
            gate_experts: Arc::clone(&gate_experts),
            up_experts: Arc::clone(&up_experts),
            down_experts: Arc::clone(&down_experts),
            act: candle_nn::Activation::Silu,
            norm_topk_prob: true,
            num_experts_per_tok: 1,
            dtype: DType::F16,
        };

        for tokens in [2usize, 65] {
            let mut input_values = vec![0f32; tokens * HIDDEN];
            for token in 0..tokens {
                input_values[token * HIDDEN] = if token % 2 == 0 { 1.0 } else { -1.0 };
                for column in 1..HIDDEN {
                    input_values[token * HIDDEN + column] =
                        ((token * HIDDEN + column) as f32 * 0.007).cos() * 0.1;
                }
            }
            let input = Tensor::from_vec(input_values, (1, tokens, HIDDEN), &device)?;
            let actual = fused.forward(&input, tokens > 64)?;

            let gate = gate_experts.dequantize(&device)?;
            let up = up_experts.dequantize(&device)?;
            let down = down_experts.dequantize(&device)?;
            let flat = input.reshape((tokens, HIDDEN))?;
            let mut reference_tokens = Vec::with_capacity(tokens);
            for token in 0..tokens {
                let expert = token % EXPERTS;
                let x = flat.narrow(0, token, 1)?;
                let gate_w = gate.narrow(0, expert, 1)?.squeeze(0)?;
                let up_w = up.narrow(0, expert, 1)?.squeeze(0)?;
                let down_w = down.narrow(0, expert, 1)?.squeeze(0)?;
                let activated =
                    x.matmul(&gate_w.t()?)?.silu()?.broadcast_mul(&x.matmul(&up_w.t()?)?)?;
                reference_tokens.push(activated.matmul(&down_w.t()?)?);
            }
            let references = reference_tokens.iter().collect::<Vec<_>>();
            let reference = Tensor::cat(&references, 0)?.reshape((1, tokens, HIDDEN))?;
            let max_error = (actual - reference)?.abs()?.max_all()?.to_vec0::<f32>()?;
            assert!(max_error < 2e-3, "grouped GGUF MoE max error {max_error} for {tokens} tokens");
        }
        Ok(())
    }
}
