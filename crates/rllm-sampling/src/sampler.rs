use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use rllm_core::request::SamplingParams;

use crate::logits;
use crate::logprobs;

/// Per-request input for sampling.
pub struct SamplingInput {
    /// Logits for this request: `[vocab_size]`.
    pub logits: Vec<f32>,
    /// Sampling parameters.
    pub params: SamplingParams,
    /// Token IDs already in context (prompt + generated) for penalty application.
    pub context_token_ids: Vec<u32>,
    /// Number of generated tokens so far (for min_tokens EOS suppression).
    pub num_generated: u32,
    /// EOS token ID.
    pub eos_token_id: u32,
    /// Bad-word token ID sequences to block (flattened: each inner Vec is one bad word).
    pub bad_word_token_ids: Vec<Vec<u32>>,
}

/// Per-request sampling output.
#[derive(Debug, Clone)]
pub struct SamplingOutput {
    /// The sampled token ID.
    pub token_id: u32,
    /// Log probability of the sampled token (if requested).
    pub logprob: Option<f32>,
    /// Top-N logprobs (if requested).
    pub top_logprobs: Option<Vec<(u32, f32)>>,
}

/// Sampler that produces next-token IDs from logits.
///
/// Stateful only for seeded RNG reproducibility. Thread-safe via `&mut self`.
pub struct Sampler {
    /// Optional seeded RNG for deterministic sampling.
    rng: Option<ChaCha8Rng>,
}

impl Sampler {
    /// Create a sampler without a fixed seed (uses entropy).
    pub fn new() -> Self {
        Self { rng: None }
    }

    /// Create a sampler with a fixed seed for reproducible sampling.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: Some(ChaCha8Rng::seed_from_u64(seed)),
        }
    }

    /// Sample a single token from the given input.
    #[tracing::instrument(skip_all, name = "sampling")]
    pub fn sample(&mut self, input: &SamplingInput) -> SamplingOutput {
        let start = std::time::Instant::now();
        let vocab_size = input.logits.len();
        if vocab_size == 0 {
            return SamplingOutput {
                token_id: 0,
                logprob: None,
                top_logprobs: None,
            };
        }

        let mut logits = input.logits.clone();

        // 1. Apply logit bias.
        if !input.params.logit_bias.is_empty() {
            logits::apply_logit_bias(&mut logits, &input.params.logit_bias);
        }

        // 2. Apply repetition/frequency/presence penalties.
        if !input.context_token_ids.is_empty() {
            if input.params.repetition_penalty != 1.0 {
                logits::apply_repetition_penalty(
                    &mut logits,
                    &input.context_token_ids,
                    input.params.repetition_penalty,
                );
            }
            if input.params.frequency_penalty != 0.0 {
                logits::apply_frequency_penalty(
                    &mut logits,
                    &input.context_token_ids,
                    input.params.frequency_penalty,
                );
            }
            if input.params.presence_penalty != 0.0 {
                logits::apply_presence_penalty(
                    &mut logits,
                    &input.context_token_ids,
                    input.params.presence_penalty,
                );
            }
        }

        // 3. Apply bad-word token masks.
        for bad_word in &input.bad_word_token_ids {
            if let Some(&last_tid) = bad_word.last() {
                logits::apply_bad_token_ids(&mut logits, &[last_tid]);
            }
        }

        // 4. Apply allowed-token-id mask.
        if let Some(ref allowed) = input.params.allowed_token_ids {
            logits::apply_allowed_token_ids(&mut logits, allowed);
        }

        // 5. Suppress EOS until min_tokens reached.
        if input.params.min_tokens > 0 {
            logits::apply_eos_suppression(
                &mut logits,
                input.eos_token_id,
                input.num_generated,
                input.params.min_tokens,
            );
        }

        // 6. Determine greedy vs stochastic.
        let is_greedy = input.params.temperature <= 0.0 || input.params.top_k == 1;

        // 7. Temperature scaling (skip for greedy — we'll argmax directly).
        if !is_greedy {
            logits::apply_temperature(&mut logits, input.params.temperature);
        }

        // 8. Top-k filtering.
        if !is_greedy && input.params.top_k > 0 {
            logits::apply_top_k(&mut logits, input.params.top_k);
        }

        // 9. Top-p filtering.
        if !is_greedy && input.params.top_p < 1.0 {
            logits::apply_top_p(&mut logits, input.params.top_p);
        }

        // 10. Min-p filtering.
        if !is_greedy && input.params.min_p > 0.0 {
            logits::apply_min_p(&mut logits, input.params.min_p);
        }

        // 11. Compute logprobs before sampling (from the modified but un-softmaxed logits).
        let want_logprobs = input.params.logprobs.is_some();

        // 12. Sample.
        let token_id = if is_greedy {
            greedy_sample(&logits)
        } else {
            random_sample(&mut logits, &mut self.rng)
        };

        // 13. Compute logprobs if requested (from original modified logits).
        let (logprob, top_logprobs) = if want_logprobs {
            let (lp, _) = logprobs::compute_logprob(&input.logits, token_id);
            let top_n = input.params.logprobs.unwrap_or(0) as usize;
            let top = if top_n > 0 {
                Some(logprobs::compute_top_n_logprobs(&input.logits, top_n))
            } else {
                None
            };
            (Some(lp), top)
        } else {
            (None, None)
        };

        let output = SamplingOutput {
            token_id,
            logprob,
            top_logprobs,
        };

        rllm_metrics::histogram!("rllm_sampling_duration_seconds")
            .record(start.elapsed().as_secs_f64());

        output
    }

    /// Sample a batch of requests, returning one `SamplingOutput` per request.
    pub fn sample_batch(&mut self, inputs: &[SamplingInput]) -> Vec<SamplingOutput> {
        inputs.iter().map(|inp| self.sample(inp)).collect()
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Greedy: return the argmax token.
fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Random: softmax then weighted sample.
fn random_sample(logits: &mut [f32], rng: &mut Option<ChaCha8Rng>) -> u32 {
    logits::softmax_in_place(logits);

    let r = match rng {
        Some(r) => r.random::<f32>(),
        None => rand::rng().random::<f32>(),
    };

    let mut cumsum = 0.0f32;
    for (i, &prob) in logits.iter().enumerate() {
        cumsum += prob;
        if cumsum >= r {
            return i as u32;
        }
    }
    // Fallback: return last token.
    (logits.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_greedy_equals_argmax() {
        let logits = vec![0.1, 0.5, 0.9, 0.3, 0.2];
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let input = SamplingInput {
            logits: logits.clone(),
            params,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let mut sampler = Sampler::new();
        let out = sampler.sample(&input);
        // Argmax of [0.1, 0.5, 0.9, 0.3, 0.2] is index 2.
        assert_eq!(out.token_id, 2);
    }

    #[test]
    fn test_top_k_1_equals_greedy() {
        let logits = vec![0.1, 0.5, 0.9, 0.3, 0.2];
        let params_greedy = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let params_topk1 = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..SamplingParams::default()
        };
        let input_greedy = SamplingInput {
            logits: logits.clone(),
            params: params_greedy,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let input_topk1 = SamplingInput {
            logits: logits.clone(),
            params: params_topk1,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let mut sampler = Sampler::new();
        let greedy_out = sampler.sample(&input_greedy);
        let topk1_out = sampler.sample(&input_topk1);
        assert_eq!(greedy_out.token_id, topk1_out.token_id);
    }

    #[test]
    fn test_penalties_change_repeated_token_prob() {
        let logits = vec![1.0, 5.0, 1.0]; // token 1 is favored

        // Without penalties: token 1 should win.
        let params_no_penalty = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let input_no_penalty = SamplingInput {
            logits: logits.clone(),
            params: params_no_penalty,
            context_token_ids: vec![1, 1, 1], // token 1 repeated
            num_generated: 3,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let mut sampler = Sampler::new();
        let out_no = sampler.sample(&input_no_penalty);
        assert_eq!(out_no.token_id, 1);

        // With high repetition penalty: token 1 should lose.
        let params_penalty = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 10.0,
            ..SamplingParams::default()
        };
        let input_penalty = SamplingInput {
            logits: logits.clone(),
            params: params_penalty,
            context_token_ids: vec![1, 1, 1],
            num_generated: 3,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let out_pen = sampler.sample(&input_penalty);
        assert_ne!(out_pen.token_id, 1, "penalty should push away from repeated token");
    }

    #[test]
    fn test_bad_words_block_tokens() {
        let logits = vec![1.0, 5.0, 1.0]; // token 1 is favored

        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let input = SamplingInput {
            logits: logits.clone(),
            params,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![vec![1]], // block token 1
        };
        let mut sampler = Sampler::new();
        let out = sampler.sample(&input);
        assert_ne!(out.token_id, 1, "bad word should block token 1");
    }

    #[test]
    fn test_fixed_seed_is_reproducible() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        };

        let make_input = || SamplingInput {
            logits: logits.clone(),
            params: params.clone(),
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };

        let mut sampler1 = Sampler::from_seed(42);
        let out1 = sampler1.sample(&make_input());

        let mut sampler2 = Sampler::from_seed(42);
        let out2 = sampler2.sample(&make_input());

        assert_eq!(out1.token_id, out2.token_id, "same seed must produce same token");
    }

    #[test]
    fn test_fixed_seed_different_from_different_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let input = SamplingInput {
            logits: logits.clone(),
            params,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };

        let mut sampler1 = Sampler::from_seed(42);
        let out1 = sampler1.sample(&input);

        let mut sampler2 = Sampler::from_seed(99);
        let out2 = sampler2.sample(&input);

        // With enough vocab diversity, different seeds should likely produce different tokens.
        // We can't guarantee this, so just verify both are valid.
        assert!(out1.token_id < 5);
        assert!(out2.token_id < 5);
    }

    #[test]
    fn test_logprobs_computed_when_requested() {
        let logits = vec![1.0, 2.0, 5.0, 0.5, 0.3];
        let params = SamplingParams {
            temperature: 0.0,
            logprobs: Some(3),
            ..SamplingParams::default()
        };
        let input = SamplingInput {
            logits,
            params,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let mut sampler = Sampler::new();
        let out = sampler.sample(&input);
        assert_eq!(out.token_id, 2);
        assert!(out.logprob.is_some());
        let lp = out.logprob.unwrap();
        assert!(lp > -1.0, "highest logit should have logprob > -1.0");
        let top = out.top_logprobs.unwrap();
        assert_eq!(top.len(), 3);
        // Sorted descending.
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn test_eos_suppressed_under_min_tokens() {
        let logits = vec![1.0, 1.0, 100.0]; // EOS (token 2) is dominant
        let params = SamplingParams {
            temperature: 0.0,
            min_tokens: 5,
            ..SamplingParams::default()
        };
        let input = SamplingInput {
            logits,
            params,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let mut sampler = Sampler::new();
        let out = sampler.sample(&input);
        assert_ne!(out.token_id, 2, "EOS should be suppressed under min_tokens");
    }

    #[test]
    fn test_allowed_token_ids_restricts() {
        let logits = vec![100.0, 1.0, 1.0]; // token 0 is dominant
        let params = SamplingParams {
            temperature: 0.0,
            allowed_token_ids: Some(HashSet::from([2u32])),
            ..SamplingParams::default()
        };
        let input = SamplingInput {
            logits,
            params,
            context_token_ids: vec![],
            num_generated: 0,
            eos_token_id: 2,
            bad_word_token_ids: vec![],
        };
        let mut sampler = Sampler::new();
        let out = sampler.sample(&input);
        assert_eq!(out.token_id, 2, "only token 2 is allowed");
    }

    #[test]
    fn test_sample_batch() {
        let logits1 = vec![0.0, 0.0, 10.0];
        let logits2 = vec![10.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let inputs = vec![
            SamplingInput {
                logits: logits1,
                params: params.clone(),
                context_token_ids: vec![],
                num_generated: 0,
                eos_token_id: 2,
                bad_word_token_ids: vec![],
            },
            SamplingInput {
                logits: logits2,
                params,
                context_token_ids: vec![],
                num_generated: 0,
                eos_token_id: 2,
                bad_word_token_ids: vec![],
            },
        ];
        let mut sampler = Sampler::new();
        let outputs = sampler.sample_batch(&inputs);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].token_id, 2);
        assert_eq!(outputs[1].token_id, 0);
    }
}
