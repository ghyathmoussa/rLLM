use serde::{Deserialize, Serialize};

use crate::{ids::RequestId, request::RequestStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestState {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub num_computed_tokens: usize,
    pub block_hashes: Vec<u64>,
    pub status: RequestStatus,
    pub output_text: String,
    pub stop_state: StopState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopState {
    None,
    StopString(StringIndex),
    StopToken,
    LengthLimit,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringIndex(pub usize);

impl RequestState {
    pub fn num_tokens(&self) -> usize {
        self.prompt_token_ids.len() + self.generated_token_ids.len()
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RequestId;

    fn sample_state() -> RequestState {
        RequestState {
            request_id: RequestId::new(),
            prompt_token_ids: vec![1, 2, 3, 4],
            generated_token_ids: vec![5, 6],
            num_computed_tokens: 6,
            block_hashes: Vec::new(),
            status: RequestStatus::Running,
            output_text: "Hello".into(),
            stop_state: StopState::None,
        }
    }

    #[test]
    fn num_tokens_counts_prompt_and_generated() {
        let state = sample_state();
        assert_eq!(state.num_tokens(), 6);
    }

    #[test]
    fn is_finished_delegates_to_status() {
        let mut state = sample_state();
        assert!(!state.is_finished());
        state.status = RequestStatus::FinishedStopped;
        assert!(state.is_finished());
    }

    #[test]
    fn request_state_serde_roundtrip() {
        let state = sample_state();
        let json = serde_json::to_string(&state).unwrap();
        let back: RequestState = serde_json::from_str(&json).unwrap();
        assert_eq!(state.prompt_token_ids, back.prompt_token_ids);
        assert_eq!(state.generated_token_ids, back.generated_token_ids);
        assert_eq!(state.num_computed_tokens, back.num_computed_tokens);
        assert_eq!(state.status, back.status);
        assert_eq!(state.output_text, back.output_text);
    }

    #[test]
    fn stop_state_serde_roundtrip() {
        for ss in [
            StopState::None,
            StopState::StopString(StringIndex(2)),
            StopState::StopToken,
            StopState::LengthLimit,
            StopState::Aborted,
        ] {
            let json = serde_json::to_string(&ss).unwrap();
            let back: StopState = serde_json::from_str(&json).unwrap();
            assert_eq!(ss, back);
        }
    }
}
