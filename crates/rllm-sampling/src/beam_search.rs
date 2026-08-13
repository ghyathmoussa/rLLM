use rllm_core::request::SamplingParams;
use thiserror::Error;

use crate::{
    logits,
    sampler::{SamplingInput, prepare_logits},
};

/// Configuration for deterministic beam search.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchConfig {
    /// Number of live hypotheses retained after each decoding step.
    pub beam_width: usize,
    /// Maximum number of generated tokens in each hypothesis.
    pub max_tokens: usize,
    /// Exponent used to normalize final cumulative log probability.
    pub length_penalty: f32,
    /// Stop as soon as `beam_width` completed hypotheses are available.
    pub early_stopping: bool,
    /// Treat EOS as an ordinary token.
    pub ignore_eos: bool,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 1,
            max_tokens: 16,
            length_penalty: 1.0,
            early_stopping: false,
            ignore_eos: false,
        }
    }
}

impl BeamSearchConfig {
    pub fn validate(&self) -> Result<(), BeamSearchError> {
        if self.beam_width == 0 {
            return Err(BeamSearchError::InvalidConfig("beam_width must be at least 1".into()));
        }
        if self.max_tokens == 0 {
            return Err(BeamSearchError::InvalidConfig("max_tokens must be at least 1".into()));
        }
        if !self.length_penalty.is_finite() {
            return Err(BeamSearchError::InvalidConfig("length_penalty must be finite".into()));
        }
        self.beam_width
            .checked_mul(2)
            .ok_or_else(|| BeamSearchError::InvalidConfig("beam_width is too large".into()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamFinishReason {
    Eos,
    StopToken,
    Length,
}

/// One beam-search hypothesis. `token_ids` contains generated tokens only.
#[derive(Debug, Clone, PartialEq)]
pub struct Beam {
    pub token_ids: Vec<u32>,
    pub cumulative_logprob: f32,
    pub finish_reason: Option<BeamFinishReason>,
}

impl Beam {
    /// Length-normalized score used to rank completed hypotheses.
    pub fn score(&self, length_penalty: f32) -> f32 {
        let length = self.token_ids.len().max(1) as f32;
        self.cumulative_logprob / length.powf(length_penalty)
    }

    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchStep {
    pub active_beams: Vec<Beam>,
    /// Parent in the previous `active_beams` slice for each surviving beam.
    pub active_parent_indices: Vec<usize>,
    pub completed_beams: Vec<Beam>,
    pub finished: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BeamSearchError {
    #[error("invalid beam search configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid sampling parameters: {0}")]
    InvalidSamplingParams(String),
    #[error("expected logits for {expected} active beams, received {actual}")]
    LogitBatchSize { expected: usize, actual: usize },
    #[error("beam {beam_index} has an empty vocabulary")]
    EmptyVocabulary { beam_index: usize },
    #[error("beam {beam_index} vocabulary exceeds u32 token ID capacity")]
    VocabularyTooLarge { beam_index: usize },
    #[error("beam {beam_index} has no finite candidate logits after processing")]
    NoFiniteCandidates { beam_index: usize },
    #[error("beam search is already finished")]
    AlreadyFinished,
}

/// Stateful beam-search decoder.
///
/// The caller runs the model for [`BeamSearch::active_beams`] and supplies one
/// logits vector per active beam to [`BeamSearch::advance`]. This keeps model
/// execution and KV-cache ownership outside the sampler.
pub struct BeamSearch {
    config: BeamSearchConfig,
    prompt_token_ids: Vec<u32>,
    active_beams: Vec<Beam>,
    completed_beams: Vec<Beam>,
    finished: bool,
}

impl BeamSearch {
    pub fn new(
        config: BeamSearchConfig,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self, BeamSearchError> {
        config.validate()?;
        Ok(Self {
            config,
            prompt_token_ids,
            active_beams: vec![Beam {
                token_ids: Vec::new(),
                cumulative_logprob: 0.0,
                finish_reason: None,
            }],
            completed_beams: Vec::new(),
            finished: false,
        })
    }

    pub fn config(&self) -> &BeamSearchConfig {
        &self.config
    }

    pub fn active_beams(&self) -> &[Beam] {
        &self.active_beams
    }

    pub fn completed_beams(&self) -> &[Beam] {
        &self.completed_beams
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Expand and prune one decoding step.
    #[tracing::instrument(skip_all, name = "beam_search_step")]
    pub fn advance(
        &mut self,
        logits_batch: &[Vec<f32>],
        params: &SamplingParams,
        eos_token_id: u32,
        bad_word_token_ids: &[Vec<u32>],
    ) -> Result<BeamSearchStep, BeamSearchError> {
        if self.finished {
            return Err(BeamSearchError::AlreadyFinished);
        }
        params
            .validate()
            .map_err(|error| BeamSearchError::InvalidSamplingParams(error.to_string()))?;
        validate_supported_sampling_params(params)?;
        if logits_batch.len() != self.active_beams.len() {
            return Err(BeamSearchError::LogitBatchSize {
                expected: self.active_beams.len(),
                actual: logits_batch.len(),
            });
        }

        let candidates_per_beam = self.config.beam_width.saturating_mul(2);
        let mut candidates =
            Vec::with_capacity(self.active_beams.len().saturating_mul(candidates_per_beam));
        for (beam_index, (beam, model_logits)) in
            self.active_beams.iter().zip(logits_batch).enumerate()
        {
            if model_logits.is_empty() {
                return Err(BeamSearchError::EmptyVocabulary { beam_index });
            }
            if model_logits.len() > u32::MAX as usize {
                return Err(BeamSearchError::VocabularyTooLarge { beam_index });
            }

            let mut context_token_ids = self.prompt_token_ids.clone();
            context_token_ids.extend_from_slice(&beam.token_ids);
            let input = SamplingInput {
                logits: model_logits.clone(),
                params: params.clone(),
                context_token_ids,
                num_generated: beam.token_ids.len() as u32,
                eos_token_id,
                bad_word_token_ids: bad_word_token_ids.to_vec(),
            };
            let mut processed = prepare_logits(&input, params.temperature > 0.0, true);
            if beam.token_ids.len() < params.min_tokens as usize {
                logits::apply_bad_token_ids(&mut processed, &params.stop_token_ids);
            }
            for (token_id, logprob) in top_logprobs(&processed, candidates_per_beam, beam_index)? {
                let mut token_ids = beam.token_ids.clone();
                token_ids.push(token_id);
                candidates.push(Candidate {
                    beam: Beam {
                        token_ids,
                        cumulative_logprob: beam.cumulative_logprob + logprob,
                        finish_reason: None,
                    },
                    parent_index: beam_index,
                    token_id,
                });
            }
        }

        candidates.sort_by(compare_candidates);
        candidates.truncate(candidates_per_beam);
        let mut next_active = Vec::with_capacity(self.config.beam_width);
        let mut next_active_parent_indices = Vec::with_capacity(self.config.beam_width);
        for mut candidate in candidates {
            let generated = candidate.beam.token_ids.len();
            let is_eos =
                candidate.token_id == eos_token_id && !self.config.ignore_eos && !params.ignore_eos;
            let is_stop = params.stop_token_ids.contains(&candidate.token_id);
            candidate.beam.finish_reason = if is_eos {
                Some(BeamFinishReason::Eos)
            } else if is_stop {
                Some(BeamFinishReason::StopToken)
            } else if generated >= self.config.max_tokens {
                Some(BeamFinishReason::Length)
            } else {
                None
            };

            if candidate.beam.is_finished() {
                self.completed_beams.push(candidate.beam);
            } else if next_active.len() < self.config.beam_width {
                next_active.push(candidate.beam);
                next_active_parent_indices.push(candidate.parent_index);
            }
        }

        sort_final_beams(&mut self.completed_beams, self.config.length_penalty);
        self.completed_beams.truncate(self.config.beam_width);
        self.active_beams = next_active;
        self.finished = self.active_beams.is_empty()
            || (self.config.early_stopping && self.completed_beams.len() >= self.config.beam_width);

        Ok(BeamSearchStep {
            active_beams: self.active_beams.clone(),
            active_parent_indices: next_active_parent_indices,
            completed_beams: self.completed_beams.clone(),
            finished: self.finished,
        })
    }

    /// Return the best current hypotheses using final length-normalized scores.
    pub fn best_beams(&self) -> Vec<Beam> {
        let mut beams = self.completed_beams.clone();
        if beams.len() < self.config.beam_width {
            beams.extend(self.active_beams.iter().cloned());
        }
        sort_final_beams(&mut beams, self.config.length_penalty);
        beams.truncate(self.config.beam_width);
        beams
    }
}

#[derive(Debug)]
struct Candidate {
    beam: Beam,
    parent_index: usize,
    token_id: u32,
}

fn compare_candidates(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.beam
        .cumulative_logprob
        .total_cmp(&a.beam.cumulative_logprob)
        .then_with(|| a.parent_index.cmp(&b.parent_index))
        .then_with(|| a.token_id.cmp(&b.token_id))
}

fn validate_supported_sampling_params(params: &SamplingParams) -> Result<(), BeamSearchError> {
    let unsupported = if params.n != 1 {
        Some("n must be 1; beam_width controls the number of hypotheses")
    } else if params.best_of.is_some() {
        Some("best_of is not used by beam search; configure beam_width instead")
    } else if !params.stop.is_empty() {
        Some("string stop conditions require detokenization; use stop_token_ids")
    } else if params.structured_outputs.is_some() {
        Some("structured outputs require grammar state that beam search does not own")
    } else if params.speculative_decoding.as_ref().is_some_and(|config| config.enabled) {
        Some("speculative decoding cannot be combined with beam search")
    } else {
        None
    };
    unsupported
        .map_or(Ok(()), |message| Err(BeamSearchError::InvalidSamplingParams(message.into())))
}

fn sort_final_beams(beams: &mut [Beam], length_penalty: f32) {
    beams.sort_by(|a, b| {
        b.score(length_penalty)
            .total_cmp(&a.score(length_penalty))
            .then_with(|| a.token_ids.cmp(&b.token_ids))
    });
}

fn top_logprobs(
    logits: &[f32],
    limit: usize,
    beam_index: usize,
) -> Result<Vec<(u32, f32)>, BeamSearchError> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .ok_or(BeamSearchError::NoFiniteCandidates { beam_index })?;
    let normalizer = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - max).exp())
        .sum::<f32>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(BeamSearchError::NoFiniteCandidates { beam_index });
    }
    let log_normalizer = max + normalizer.ln();
    let mut scores = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(token_id, value)| (token_id as u32, value - log_normalizer))
        .collect::<Vec<_>>();
    scores.sort_by(|(token_a, score_a), (token_b, score_b)| {
        score_b.total_cmp(score_a).then_with(|| token_a.cmp(token_b))
    });
    scores.truncate(limit.min(scores.len()));
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn greedy_params() -> SamplingParams {
        SamplingParams { temperature: 0.0, ..SamplingParams::default() }
    }

    #[test]
    fn rejects_invalid_config_and_logit_batch() {
        assert!(matches!(
            BeamSearch::new(
                BeamSearchConfig { beam_width: 0, ..BeamSearchConfig::default() },
                vec![]
            ),
            Err(BeamSearchError::InvalidConfig(_))
        ));
        let mut search = BeamSearch::new(BeamSearchConfig::default(), vec![]).unwrap();
        assert_eq!(
            search.advance(&[], &greedy_params(), 2, &[]),
            Err(BeamSearchError::LogitBatchSize { expected: 1, actual: 0 })
        );
    }

    #[test]
    fn rejects_sampling_modes_that_need_external_state() {
        for params in [
            SamplingParams { best_of: Some(2), ..greedy_params() },
            SamplingParams { stop: vec!["stop".into()], ..greedy_params() },
            SamplingParams { speculative_decoding: Some(Default::default()), ..greedy_params() },
        ] {
            let mut search = BeamSearch::new(BeamSearchConfig::default(), vec![]).unwrap();
            assert!(matches!(
                search.advance(&[vec![1.0]], &params, 9, &[]),
                Err(BeamSearchError::InvalidSamplingParams(_))
            ));
        }
    }

    #[test]
    fn retains_highest_probability_beams() {
        let config = BeamSearchConfig { beam_width: 2, max_tokens: 2, ..Default::default() };
        let mut search = BeamSearch::new(config, vec![10]).unwrap();
        let first = search.advance(&[vec![3.0, 2.0, -10.0]], &greedy_params(), 2, &[]).unwrap();
        assert_eq!(first.active_beams.len(), 2);
        assert_eq!(first.active_beams[0].token_ids, vec![0]);
        assert_eq!(first.active_beams[1].token_ids, vec![1]);
        assert_eq!(first.active_parent_indices, vec![0, 0]);

        let second = search
            .advance(&[vec![0.0, 4.0, -10.0], vec![5.0, 0.0, -10.0]], &greedy_params(), 2, &[])
            .unwrap();
        assert!(second.finished);
        assert!(second.active_parent_indices.is_empty());
        let best = search.best_beams();
        assert_eq!(best[0].token_ids, vec![0, 1]);
        assert_eq!(best.len(), 2);
        assert!(best.iter().all(|beam| beam.finish_reason == Some(BeamFinishReason::Length)));
    }

    #[test]
    fn eos_and_stop_tokens_complete_beams() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 4,
            early_stopping: true,
            ..Default::default()
        };
        let mut search = BeamSearch::new(config, vec![]).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            stop_token_ids: vec![1],
            ..SamplingParams::default()
        };
        let step = search.advance(&[vec![0.0, 3.0, 4.0]], &params, 2, &[]).unwrap();
        assert!(step.finished);
        assert_eq!(step.completed_beams.len(), 2);
        assert!(
            step.completed_beams
                .iter()
                .any(|beam| { beam.finish_reason == Some(BeamFinishReason::Eos) })
        );
        assert!(
            step.completed_beams
                .iter()
                .any(|beam| { beam.finish_reason == Some(BeamFinishReason::StopToken) })
        );
    }

    #[test]
    fn applies_allowed_token_mask() {
        let config = BeamSearchConfig { beam_width: 2, ..Default::default() };
        let mut search = BeamSearch::new(config.clone(), vec![]).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            allowed_token_ids: Some(HashSet::from([1u32, 2u32])),
            ..SamplingParams::default()
        };
        let step = search.advance(&[vec![100.0, 2.0, 1.0]], &params, 9, &[]).unwrap();
        assert_eq!(step.active_beams[0].token_ids, vec![1]);
        assert_eq!(step.active_beams[1].token_ids, vec![2]);
    }

    #[test]
    fn length_penalty_changes_final_ranking() {
        let short = Beam {
            token_ids: vec![1],
            cumulative_logprob: -1.0,
            finish_reason: Some(BeamFinishReason::Eos),
        };
        let long = Beam {
            token_ids: vec![1, 2, 3, 4],
            cumulative_logprob: -2.0,
            finish_reason: Some(BeamFinishReason::Length),
        };
        assert!(short.score(0.0) > long.score(0.0));
        assert!(long.score(1.0) > short.score(1.0));
    }

    #[test]
    fn rejects_empty_and_non_finite_logits() {
        let config = BeamSearchConfig::default();
        let mut empty = BeamSearch::new(config.clone(), vec![]).unwrap();
        assert_eq!(
            empty.advance(&[vec![]], &greedy_params(), 0, &[]),
            Err(BeamSearchError::EmptyVocabulary { beam_index: 0 })
        );
        let mut non_finite = BeamSearch::new(config, vec![]).unwrap();
        assert_eq!(
            non_finite.advance(&[vec![f32::NAN, f32::NEG_INFINITY]], &greedy_params(), 0, &[]),
            Err(BeamSearchError::NoFiniteCandidates { beam_index: 0 })
        );
    }
}
