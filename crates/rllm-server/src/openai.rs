//! OpenAI-compatible protocol types and conversion functions.

use serde::{Deserialize, Serialize};

// ── Request types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptInput,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Single(String),
    Multiple(Vec<String>),
}

// ── Response types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Model list ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

// ── Error ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

// ── Health ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

// ── Conversion helpers ─────────────────────────────────────────────────────

use rllm_core::output::RequestOutput;
use rllm_core::request::SamplingParams;
use uuid::Uuid;

/// Generate a completion ID (e.g., "cmpl-abc123").
pub fn generate_completion_id(prefix: &str) -> String {
    format!("{}-{}", prefix, Uuid::new_v4().as_simple())
}

/// Get the current unix timestamp.
pub fn now_timestamp() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Convert a ChatCompletionRequest's sampling fields into rllm SamplingParams.
pub fn chat_request_to_sampling_params(req: &ChatCompletionRequest) -> SamplingParams {
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(mt) = req.max_tokens {
        params.max_tokens = Some(mt);
    }
    if let Some(pp) = req.presence_penalty {
        params.presence_penalty = pp;
    }
    if let Some(fp) = req.frequency_penalty {
        params.frequency_penalty = fp;
    }
    if let Some(seed) = req.seed {
        params.seed = Some(seed);
    }
    if let Some(n) = req.n {
        params.n = n;
    }
    if let Some(true) = req.logprobs {
        params.logprobs = req.top_logprobs.or(Some(1));
    }
    params.stop = extract_stop_strings(&req.stop);
    params
}

/// Convert a CompletionRequest's sampling fields into rllm SamplingParams.
pub fn completion_request_to_sampling_params(req: &CompletionRequest) -> SamplingParams {
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }
    if let Some(mt) = req.max_tokens {
        params.max_tokens = Some(mt);
    }
    if let Some(pp) = req.presence_penalty {
        params.presence_penalty = pp;
    }
    if let Some(fp) = req.frequency_penalty {
        params.frequency_penalty = fp;
    }
    if let Some(seed) = req.seed {
        params.seed = Some(seed);
    }
    if let Some(n) = req.n {
        params.n = n;
    }
    if let Some(lp) = req.logprobs {
        params.logprobs = Some(lp);
    }
    params.stop = extract_stop_strings(&req.stop);
    params
}

fn extract_stop_strings(stop: &Option<StopSequence>) -> Vec<String> {
    match stop {
        Some(StopSequence::Single(s)) => vec![s.clone()],
        Some(StopSequence::Multiple(v)) => v.clone(),
        None => vec![],
    }
}

/// Convert engine RequestOutput to a ChatCompletionResponse.
pub fn request_output_to_chat_completion(
    output: &RequestOutput,
    model: &str,
) -> ChatCompletionResponse {
    let id = generate_completion_id("chatcmpl");
    let created = now_timestamp();

    let choices: Vec<ChatChoice> = output
        .outputs
        .iter()
        .enumerate()
        .map(|(i, co)| {
            let finish_reason = co.finish_reason.map(|r| match r {
                rllm_core::output::FinishReason::Stop => "stop".to_string(),
                rllm_core::output::FinishReason::Length => "length".to_string(),
                rllm_core::output::FinishReason::Aborted => "stop".to_string(),
                rllm_core::output::FinishReason::Error => "stop".to_string(),
            });
            ChatChoice {
                index: i as u32,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: co.text.clone(),
                },
                finish_reason,
            }
        })
        .collect();

    ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: UsageInfo {
            prompt_tokens: output.usage.prompt_tokens,
            completion_tokens: output.usage.completion_tokens,
            total_tokens: output.usage.total_tokens,
        },
    }
}

/// Convert engine RequestOutput to a CompletionResponse.
pub fn request_output_to_completion(output: &RequestOutput, model: &str) -> CompletionResponse {
    let id = generate_completion_id("cmpl");
    let created = now_timestamp();

    let choices: Vec<CompletionChoice> = output
        .outputs
        .iter()
        .enumerate()
        .map(|(i, co)| {
            let finish_reason = co.finish_reason.map(|r| match r {
                rllm_core::output::FinishReason::Stop => "stop".to_string(),
                rllm_core::output::FinishReason::Length => "length".to_string(),
                rllm_core::output::FinishReason::Aborted => "stop".to_string(),
                rllm_core::output::FinishReason::Error => "stop".to_string(),
            });
            CompletionChoice { index: i as u32, text: co.text.clone(), finish_reason }
        })
        .collect();

    CompletionResponse {
        id,
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: UsageInfo {
            prompt_tokens: output.usage.prompt_tokens,
            completion_tokens: output.usage.completion_tokens,
            total_tokens: output.usage.total_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_serde_roundtrip() {
        let req = ChatCompletionRequest {
            model: "test-model".into(),
            messages: vec![
                ChatMessage { role: "system".into(), content: "You are helpful.".into() },
                ChatMessage { role: "user".into(), content: "Hello!".into() },
            ],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: Some(true),
            stop: Some(StopSequence::Single("\n".into())),
            top_p: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "test-model");
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.temperature, Some(0.7));
    }

    #[test]
    fn completion_request_serde_roundtrip() {
        let req = CompletionRequest {
            model: "test-model".into(),
            prompt: PromptInput::Single("Hello world".into()),
            max_tokens: Some(50),
            suffix: None,
            temperature: None,
            top_p: None,
            stream: None,
            stop: None,
            n: None,
            logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "test-model");
    }

    #[test]
    fn error_response_serialization() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "Model not found".into(),
                error_type: "invalid_request_error".into(),
                code: Some("model_not_found".into()),
            },
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("invalid_request_error"));
    }

    #[test]
    fn generate_completion_id_format() {
        let id = generate_completion_id("chatcmpl");
        assert!(id.starts_with("chatcmpl-"));
        assert!(id.len() > "chatcmpl-".len());
    }
}
