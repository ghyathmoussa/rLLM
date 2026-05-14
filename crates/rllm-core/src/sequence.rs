use serde::{Deserialize, Serialize};

use crate::ids::RequestId;
use crate::request::RequestStatus;

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
