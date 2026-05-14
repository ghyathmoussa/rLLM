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

#[cfg(test)]
mod tests {
    use super::*;

    // --- SamplingParams validation ---

    #[test]
    fn default_params_are_valid() {
        assert!(SamplingParams::default().validate().is_ok());
    }

    #[test]
    fn n_zero_rejected() {
        let mut p = SamplingParams::default();
        p.n = 0;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("n must be >= 1"));
    }

    #[test]
    fn negative_temperature_rejected() {
        let mut p = SamplingParams::default();
        p.temperature = -0.5;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("temperature must be >= 0"));
    }

    #[test]
    fn temperature_zero_ok() {
        let mut p = SamplingParams::default();
        p.temperature = 0.0;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn temperature_zero_with_n_gt_1_rejected() {
        let mut p = SamplingParams::default();
        p.temperature = 0.0;
        p.n = 3;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("n must be 1 when temperature is 0"));
    }

    #[test]
    fn presence_penalty_out_of_range() {
        let mut p = SamplingParams::default();
        p.presence_penalty = 2.5;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("presence_penalty"));

        p.presence_penalty = -0.1;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("presence_penalty"));
    }

    #[test]
    fn frequency_penalty_out_of_range() {
        let mut p = SamplingParams::default();
        p.frequency_penalty = -1.0;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("frequency_penalty"));
    }

    #[test]
    fn repetition_penalty_zero_rejected() {
        let mut p = SamplingParams::default();
        p.repetition_penalty = 0.0;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("repetition_penalty"));
    }

    #[test]
    fn top_p_out_of_range() {
        let mut p = SamplingParams::default();
        p.top_p = 1.5;
        let err = p.validate().unwrap_err();
        assert!(err.to_string().contains("top_p"));
    }

    #[test]
    fn sampling_params_serde_roundtrip() {
        let p = SamplingParams::default();
        let json = serde_json::to_string(&p).unwrap();
        let back: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p.n, back.n);
        assert_eq!(p.temperature, back.temperature);
        assert_eq!(p.top_p, back.top_p);
        assert_eq!(p.output_kind, back.output_kind);
    }

    #[test]
    fn sampling_params_with_all_fields_roundtrip() {
        let p = SamplingParams {
            n: 3,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.05,
            presence_penalty: 0.5,
            frequency_penalty: 0.3,
            repetition_penalty: 1.2,
            max_tokens: Some(512),
            min_tokens: 10,
            stop: vec!["\n".into(), "END".into()],
            stop_token_ids: vec![2, 50256],
            logprobs: Some(5),
            seed: Some(42),
            ..SamplingParams::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.n, 3);
        assert_eq!(back.stop.len(), 2);
        assert_eq!(back.seed, Some(42));
    }

    // --- RequestStatus transitions ---

    #[test]
    fn finished_statuses() {
        assert!(RequestStatus::FinishedStopped.is_finished());
        assert!(RequestStatus::FinishedLength.is_finished());
        assert!(RequestStatus::FinishedAborted.is_finished());
        assert!(RequestStatus::FinishedError.is_finished());
        assert!(!RequestStatus::Waiting.is_finished());
        assert!(!RequestStatus::Running.is_finished());
        assert!(!RequestStatus::Preempted.is_finished());
    }

    #[test]
    fn waiting_to_running() {
        assert!(RequestStatus::Waiting.can_transition_to(RequestStatus::Running));
        assert_eq!(
            RequestStatus::Waiting.transition(RequestStatus::Running).unwrap(),
            RequestStatus::Running
        );
    }

    #[test]
    fn waiting_to_aborted() {
        assert!(RequestStatus::Waiting.can_transition_to(RequestStatus::FinishedAborted));
    }

    #[test]
    fn running_to_finished_stopped() {
        assert!(RequestStatus::Running.can_transition_to(RequestStatus::FinishedStopped));
    }

    #[test]
    fn running_to_finished_length() {
        assert!(RequestStatus::Running.can_transition_to(RequestStatus::FinishedLength));
    }

    #[test]
    fn running_to_finished_error() {
        assert!(RequestStatus::Running.can_transition_to(RequestStatus::FinishedError));
    }

    #[test]
    fn running_preempted_to_waiting() {
        assert!(RequestStatus::Running.can_transition_to(RequestStatus::Preempted));
        assert!(RequestStatus::Preempted.can_transition_to(RequestStatus::Waiting));
    }

    #[test]
    fn running_can_loop() {
        assert!(RequestStatus::Running.can_transition_to(RequestStatus::Running));
    }

    #[test]
    fn waiting_for_remote_kv_transitions() {
        assert!(RequestStatus::WaitingForRemoteKV.can_transition_to(RequestStatus::Running));
        assert!(RequestStatus::WaitingForRemoteKV.can_transition_to(RequestStatus::FinishedAborted));
    }

    #[test]
    fn invalid_transitions_rejected() {
        // Cannot go from Waiting to FinishedStopped directly
        assert!(!RequestStatus::Waiting.can_transition_to(RequestStatus::FinishedStopped));
        assert!(RequestStatus::Waiting.transition(RequestStatus::FinishedStopped).is_err());

        // Cannot go from finished to anything
        for finished in [
            RequestStatus::FinishedStopped,
            RequestStatus::FinishedLength,
            RequestStatus::FinishedAborted,
            RequestStatus::FinishedError,
        ] {
            for target in [
                RequestStatus::Waiting,
                RequestStatus::Running,
                RequestStatus::Preempted,
                RequestStatus::WaitingForRemoteKV,
            ] {
                assert!(!finished.can_transition_to(target));
            }
        }
    }

    #[test]
    fn request_status_serde_roundtrip() {
        for status in [
            RequestStatus::Waiting,
            RequestStatus::Running,
            RequestStatus::WaitingForRemoteKV,
            RequestStatus::Preempted,
            RequestStatus::FinishedStopped,
            RequestStatus::FinishedLength,
            RequestStatus::FinishedAborted,
            RequestStatus::FinishedError,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: RequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }

    // --- InferenceRequest ---

    #[test]
    fn inference_request_construction() {
        let req = InferenceRequest {
            request_id: RequestId::new(),
            prompt: Some("Hello world".into()),
            token_ids: None,
            messages: None,
            sampling_params: SamplingParams::default(),
            arrival_time: std::time::Instant::now(),
            priority: 0,
            stream: false,
            cache_salt: None,
        };
        assert!(req.prompt.is_some());
        assert!(!req.stream);
        assert_eq!(req.priority, 0);
    }
}
