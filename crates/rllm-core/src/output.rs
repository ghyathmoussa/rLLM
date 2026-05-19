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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RequestId;

    #[test]
    fn usage_fields() {
        let usage = Usage { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 };
        assert_eq!(usage.prompt_tokens + usage.completion_tokens, usage.total_tokens);
    }

    #[test]
    fn usage_serde_roundtrip() {
        let usage = Usage { prompt_tokens: 5, completion_tokens: 15, total_tokens: 20 };
        let json = serde_json::to_string(&usage).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, back);
    }

    #[test]
    fn finish_reason_serde_roundtrip() {
        for reason in
            [FinishReason::Stop, FinishReason::Length, FinishReason::Aborted, FinishReason::Error]
        {
            let json = serde_json::to_string(&reason).unwrap();
            let back: FinishReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, back);
        }
    }

    #[test]
    fn request_output_serde_roundtrip() {
        let output = RequestOutput {
            request_id: RequestId::new(),
            outputs: vec![CompletionOutput {
                index: 0,
                text: "Hello world".into(),
                token_ids: vec![1, 2, 3],
                finish_reason: Some(FinishReason::Stop),
                logprobs: Some(vec![Logprob {
                    token_id: 1,
                    logprob: -0.5,
                    text: Some("Hello".into()),
                }]),
            }],
            finished: true,
            usage: Usage { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: RequestOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output.request_id, back.request_id);
        assert_eq!(output.outputs.len(), back.outputs.len());
        assert_eq!(back.outputs[0].text, "Hello world");
        assert!(back.finished);
    }

    #[test]
    fn stream_chunk_serde_roundtrip() {
        let chunk = StreamChunk {
            request_id: RequestId::new(),
            outputs: vec![CompletionOutput {
                index: 0,
                text: "Hi".into(),
                token_ids: vec![10],
                finish_reason: None,
                logprobs: None,
            }],
            finished: false,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk.request_id, back.request_id);
        assert!(!back.finished);
    }
}
