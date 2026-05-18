use std::path::Path;

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rllm_core::ids::RequestId;
use rllm_core::request::{InferenceRequest, SamplingParams};

use crate::helpers::make_inference_request_with_params;

#[derive(Debug, serde::Deserialize)]
struct ShareGptConversation {
    conversations: Vec<ShareGptTurn>,
}

#[derive(Debug, serde::Deserialize)]
struct ShareGptTurn {
    from: String,
    value: String,
}

/// Distribution of sequence lengths for workload generation.
#[derive(Debug, Clone)]
pub enum LengthDistribution {
    /// Fixed length for all requests.
    Fixed(usize),
    /// Uniform random in [min, max].
    Uniform { min: usize, max: usize },
}

impl LengthDistribution {
    fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        match self {
            LengthDistribution::Fixed(n) => *n,
            LengthDistribution::Uniform { min, max } => rng.random_range(*min..=*max),
        }
    }
}

/// Configuration for synthetic workload generation.
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    pub num_requests: usize,
    pub input_lengths: LengthDistribution,
    pub output_lengths: LengthDistribution,
    pub concurrency: usize,
    pub vocab_size: usize,
    pub seed: u64,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_requests: 100,
            input_lengths: LengthDistribution::Fixed(128),
            output_lengths: LengthDistribution::Fixed(32),
            concurrency: 32,
            vocab_size: 32000,
            seed: 42,
        }
    }
}

/// A generated workload: a set of `InferenceRequest`s ready for submission.
pub struct SyntheticWorkload {
    pub requests: Vec<InferenceRequest>,
    pub config: WorkloadConfig,
}

impl SyntheticWorkload {
    /// Generate a synthetic workload from config.
    pub fn generate(config: &WorkloadConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let requests: Vec<InferenceRequest> = (0..config.num_requests)
            .map(|_| {
                let input_len = config.input_lengths.sample(&mut rng);
                let output_len = config.output_lengths.sample(&mut rng);
                let params = SamplingParams {
                    max_tokens: Some(output_len as u32),
                    temperature: 0.0,
                    ..Default::default()
                };
                make_inference_request_with_params(input_len, output_len as u32, params)
            })
            .collect();

        Self { requests, config: config.clone() }
    }

    /// Load a workload from a ShareGPT-format JSON file.
    ///
    /// Expects an array of objects with `conversations` containing
    /// `human`/`gpt` turns. Maps conversation lengths to token counts
    /// using a simple heuristic (1 token ≈ 4 characters).
    pub fn from_sharegpt(path: &Path, max_output: usize, seed: u64) -> Result<Self> {
        let prompts = sharegpt_prompts(path)?;
        if prompts.is_empty() {
            anyhow::bail!("ShareGPT dataset did not contain any human/user prompts");
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut requests = Vec::new();

        for prompt_text in prompts {
            // Approximate token count: ~4 chars per token.
            let input_len = (prompt_text.len() / 4).max(4);
            let output_len = rng.random_range(4..=max_output);

            let params = SamplingParams {
                max_tokens: Some(output_len as u32),
                temperature: 0.0,
                ..Default::default()
            };

            requests.push(InferenceRequest {
                request_id: RequestId::new(),
                prompt: Some(prompt_text),
                token_ids: Some((0..input_len as u32).collect()),
                messages: None,
                sampling_params: params,
                arrival_time: std::time::Instant::now(),
                priority: 0,
                stream: false,
                cache_salt: None,
            });
        }

        let num_requests = requests.len();
        Ok(Self {
            requests,
            config: WorkloadConfig {
                num_requests,
                input_lengths: LengthDistribution::Fixed(128),
                output_lengths: LengthDistribution::Fixed(max_output),
                concurrency: 32,
                vocab_size: 32000,
                seed,
            },
        })
    }
}

/// Load first human/user prompts from a ShareGPT-style JSON file.
pub fn sharegpt_prompts(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let conversations: Vec<ShareGptConversation> = serde_json::from_str(&content)?;

    let prompts = conversations
        .into_iter()
        .filter_map(|conv| {
            conv.conversations
                .into_iter()
                .find(|turn| turn.from == "human" || turn.from == "user")
                .map(|turn| turn.value)
        })
        .filter(|prompt| !prompt.trim().is_empty())
        .collect();

    Ok(prompts)
}

/// Build a deterministic synthetic prompt with roughly `input_tokens` tokens.
pub fn synthetic_prompt(input_tokens: usize, index: usize) -> String {
    let mut words = Vec::with_capacity(input_tokens.max(1));
    words.push(format!("request-{index}"));
    for i in 1..input_tokens.max(1) {
        words.push(format!("tok{i}"));
    }
    words.join(" ")
}
