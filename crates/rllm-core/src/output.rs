use serde::{Deserialize, Serialize};

use crate::ids::RequestId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    pub request_id: RequestId,
    pub outputs: Vec<CompletionOutput>,
    pub finished: bool,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionOutput {
    pub index: u32,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<Vec<Logprob>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub request_id: RequestId,
    pub outputs: Vec<CompletionOutput>,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Aborted,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprob {
    pub token_id: u32,
    pub logprob: f32,
    pub text: Option<String>,
}
