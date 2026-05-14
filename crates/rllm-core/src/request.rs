use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::ids::RequestId;
use crate::error::{CoreError, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputKind {
    Text,
    TokenIds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub n: u32,
    pub best_of: Option<u32>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    pub max_tokens: Option<u32>,
    pub min_tokens: u32,
    pub logprobs: Option<u32>,
    pub prompt_logprobs: Option<u32>,
    pub detokenize: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    pub output_kind: OutputKind,
    pub logit_bias: std::collections::HashMap<u32, f32>,
    pub allowed_token_ids: Option<HashSet<u32>>,
    pub bad_words: Vec<String>,
    pub structured_outputs: Option<StructuredOutputParams>,
    pub skip_reading_prefix_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredOutputParams {
    pub json_schema: Option<serde_json::Value>,
    pub regex: Option<String>,
    pub grammar: Option<String>,
    pub choice: Option<Vec<String>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            best_of: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            seed: None,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            max_tokens: Some(16),
            min_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
            detokenize: true,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            include_stop_str_in_output: false,
            output_kind: OutputKind::Text,
            logit_bias: std::collections::HashMap::new(),
            allowed_token_ids: None,
            bad_words: Vec::new(),
            structured_outputs: None,
            skip_reading_prefix_cache: false,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            return Err(CoreError::InvalidSamplingParams("n must be >= 1".into()));
        }
        if self.temperature < 0.0 {
            return Err(CoreError::InvalidSamplingParams("temperature must be >= 0".into()));
        }
        if !(0.0..=2.0).contains(&self.presence_penalty) {
            return Err(CoreError::InvalidSamplingParams(
                "presence_penalty must be in [0, 2]".into(),
            ));
        }
        if !(0.0..=2.0).contains(&self.frequency_penalty) {
            return Err(CoreError::InvalidSamplingParams(
                "frequency_penalty must be in [0, 2]".into(),
            ));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(CoreError::InvalidSamplingParams(
                "repetition_penalty must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(CoreError::InvalidSamplingParams("top_p must be in [0, 1]".into()));
        }
        if self.temperature == 0.0 && self.n > 1 {
            return Err(CoreError::InvalidSamplingParams(
                "n must be 1 when temperature is 0 (greedy)".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: RequestId,
    pub prompt: Option<String>,
    pub token_ids: Option<Vec<u32>>,
    pub messages: Option<Vec<ChatMessage>>,
    pub sampling_params: SamplingParams,
    pub arrival_time: std::time::Instant,
    pub priority: i32,
    pub stream: bool,
    pub cache_salt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestStatus {
    Waiting,
    Running,
    WaitingForRemoteKV,
    Preempted,
    FinishedStopped,
    FinishedLength,
    FinishedAborted,
    FinishedError,
}

impl RequestStatus {
    pub fn is_finished(self) -> bool {
        matches!(
            self,
            Self::FinishedStopped
                | Self::FinishedLength
                | Self::FinishedAborted
                | Self::FinishedError
        )
    }

    pub fn can_transition_to(self, next: RequestStatus) -> bool {
        match (self, next) {
            (Self::Waiting, Self::Running | Self::FinishedAborted) => true,
            (Self::Running, Self::Running
                | Self::Waiting
                | Self::Preempted
                | Self::FinishedStopped
                | Self::FinishedLength
                | Self::FinishedAborted
                | Self::FinishedError) => true,
            (Self::Preempted, Self::Waiting | Self::FinishedAborted) => true,
            (Self::WaitingForRemoteKV, Self::Running | Self::FinishedAborted) => true,
            _ => false,
        }
    }

    pub fn transition(self, next: RequestStatus) -> Result<RequestStatus> {
        if self.can_transition_to(next) {
            Ok(next)
        } else {
            Err(CoreError::InvalidStatusTransition {
                from: format!("{self:?}"),
                to: format!("{next:?}"),
            })
        }
    }
}
