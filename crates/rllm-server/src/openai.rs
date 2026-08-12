//! OpenAI-compatible protocol types and conversion functions.

use rllm_core::request::StructuredOutputParams;
use serde::{Deserialize, Serialize};

// ── Request types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// OpenAI represents function arguments as a JSON-encoded string.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: NamedFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedFunction {
    pub name: String,
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
    #[serde(default)]
    pub structured_outputs: Option<StructuredOutputParams>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: ResponseFormatJsonSchema },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormatJsonSchema {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    #[serde(default)]
    pub strict: Option<bool>,
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
    #[serde(default)]
    pub structured_outputs: Option<StructuredOutputParams>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_time: Option<f64>,
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
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_time: Option<f64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_time: Option<f64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedChatOutput {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
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

use rllm_core::{output::RequestOutput, request::SamplingParams};
use uuid::Uuid;

/// Generate a completion ID (e.g., "cmpl-abc123").
pub fn generate_completion_id(prefix: &str) -> String {
    format!("{}-{}", prefix, Uuid::new_v4().as_simple())
}

pub fn generate_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().as_simple())
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
    params.structured_outputs =
        request_structured_outputs(req.structured_outputs.clone(), req.response_format.as_ref());
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
    params.structured_outputs =
        request_structured_outputs(req.structured_outputs.clone(), req.response_format.as_ref());
    params
}

pub fn validate_structured_output_request(
    structured_outputs: &Option<StructuredOutputParams>,
    response_format: &Option<ResponseFormat>,
) -> Result<(), String> {
    if structured_outputs.is_some() && response_format.is_some() {
        return Err("structured_outputs and response_format cannot both be set".into());
    }
    Ok(())
}

pub fn validate_tool_call_request(req: &ChatCompletionRequest) -> Result<(), String> {
    let tools = req.tools.as_deref().unwrap_or_default();
    if req.tools.as_ref().is_some_and(Vec::is_empty) {
        return Err("tools must contain at least one function".into());
    }
    if tools.len() > 128 {
        return Err("tools cannot contain more than 128 functions".into());
    }

    let mut names = std::collections::HashSet::with_capacity(tools.len());
    for tool in tools {
        if tool.tool_type != "function" {
            return Err("only tools with type \"function\" are supported".into());
        }
        validate_function_name(&tool.function.name)?;
        if !names.insert(tool.function.name.as_str()) {
            return Err(format!("duplicate tool function name: {}", tool.function.name));
        }
        if let Some(parameters) = &tool.function.parameters
            && !parameters.is_object()
        {
            return Err(format!(
                "parameters for tool {} must be a JSON object",
                tool.function.name
            ));
        }
    }

    match req.tool_choice.as_ref() {
        None | Some(ToolChoice::Mode(ToolChoiceMode::None)) => {}
        Some(ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::Required)) => {
            if tools.is_empty() {
                return Err("tools must be set when tool_choice is auto or required".into());
            }
        }
        Some(ToolChoice::Named(choice)) => {
            if choice.tool_type != "function" {
                return Err("named tool_choice must have type \"function\"".into());
            }
            validate_function_name(&choice.function.name)?;
            if !names.contains(choice.function.name.as_str()) {
                return Err("the named tool_choice does not match any supplied tool".into());
            }
        }
    }

    for message in &req.messages {
        validate_tool_message(message)?;
    }
    Ok(())
}

pub fn tools_for_chat_template(
    req: &ChatCompletionRequest,
) -> Result<Option<Vec<serde_json::Value>>, String> {
    validate_tool_call_request(req)?;
    let Some(tools) = req.tools.as_ref() else {
        return Ok(None);
    };
    if matches!(req.tool_choice.as_ref(), Some(ToolChoice::Mode(ToolChoiceMode::None))) {
        return Ok(None);
    }

    let named = match req.tool_choice.as_ref() {
        Some(ToolChoice::Named(choice)) => Some(choice.function.name.as_str()),
        _ => None,
    };
    tools
        .iter()
        .filter(|tool| named.is_none_or(|name| tool.function.name == name))
        .map(|tool| serde_json::to_value(tool).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub fn messages_for_chat_template(
    req: &ChatCompletionRequest,
) -> Result<Vec<serde_json::Value>, String> {
    validate_tool_call_request(req)?;
    req.messages
        .iter()
        .map(|message| {
            let mut value = serde_json::to_value(message).map_err(|error| error.to_string())?;
            if let Some(tool_calls) =
                value.get_mut("tool_calls").and_then(|calls| calls.as_array_mut())
            {
                for tool_call in tool_calls {
                    let Some(arguments) = tool_call
                        .get_mut("function")
                        .and_then(|function| function.get_mut("arguments"))
                    else {
                        continue;
                    };
                    if let Some(encoded) = arguments.as_str() {
                        *arguments = serde_json::from_str(encoded).map_err(|_| {
                            "assistant tool call arguments must contain valid JSON".to_string()
                        })?;
                    }
                }
            }
            Ok(value)
        })
        .collect()
}

pub fn tool_choice_for_chat_template(
    req: &ChatCompletionRequest,
) -> Result<Option<serde_json::Value>, String> {
    validate_tool_call_request(req)?;
    if req.tools.is_none() {
        return Ok(None);
    }
    let choice = req.tool_choice.clone().unwrap_or(ToolChoice::Mode(ToolChoiceMode::Auto));
    serde_json::to_value(choice).map(Some).map_err(|error| error.to_string())
}

fn validate_function_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        return Err("tool function names must contain between 1 and 64 characters".into());
    }
    if !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')) {
        return Err(format!(
            "invalid tool function name {name:?}: use only letters, numbers, underscores, or hyphens"
        ));
    }
    Ok(())
}

fn validate_tool_message(message: &ChatMessage) -> Result<(), String> {
    match message.role.as_str() {
        "system" | "user" => {
            if message.content.is_none() {
                return Err(format!("{} messages require content", message.role));
            }
            if message.tool_calls.is_some() || message.tool_call_id.is_some() {
                return Err(format!("{} messages cannot contain tool call fields", message.role));
            }
        }
        "assistant" => {
            if message.content.is_none() && message.tool_calls.as_ref().is_none_or(Vec::is_empty) {
                return Err("assistant messages require content or tool_calls".into());
            }
            if message.tool_call_id.is_some() {
                return Err("assistant messages cannot contain tool_call_id".into());
            }
            if let Some(tool_calls) = &message.tool_calls {
                for tool_call in tool_calls {
                    if tool_call.id.is_empty() {
                        return Err("assistant tool calls require a non-empty id".into());
                    }
                    if tool_call.tool_type != "function" {
                        return Err(
                            "only assistant tool calls with type \"function\" are supported".into(),
                        );
                    }
                    validate_function_name(&tool_call.function.name)?;
                    serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                        .map_err(|_| {
                            "assistant tool call arguments must contain valid JSON".to_string()
                        })?;
                }
            }
        }
        "tool" => {
            if message.content.is_none() {
                return Err("tool messages require content".into());
            }
            if message.tool_call_id.as_deref().is_none_or(str::is_empty) {
                return Err("tool messages require a non-empty tool_call_id".into());
            }
            if message.tool_calls.is_some() {
                return Err("tool messages cannot contain tool_calls".into());
            }
        }
        role => return Err(format!("unsupported chat message role: {role}")),
    }
    Ok(())
}

fn request_structured_outputs(
    structured_outputs: Option<StructuredOutputParams>,
    response_format: Option<&ResponseFormat>,
) -> Option<StructuredOutputParams> {
    structured_outputs.or_else(|| match response_format {
        Some(ResponseFormat::JsonObject) => Some(StructuredOutputParams {
            json_schema: None,
            json_object: Some(true),
            xml: None,
            regex: None,
            grammar: None,
            choice: None,
        }),
        Some(ResponseFormat::JsonSchema { json_schema }) => Some(StructuredOutputParams {
            json_schema: Some(json_schema.schema.clone()),
            json_object: None,
            xml: None,
            regex: None,
            grammar: None,
            choice: None,
        }),
        Some(ResponseFormat::Text) | None => None,
    })
}

fn extract_stop_strings(stop: &Option<StopSequence>) -> Vec<String> {
    match stop {
        Some(StopSequence::Single(s)) => vec![s.clone()],
        Some(StopSequence::Multiple(v)) => v.clone(),
        None => vec![],
    }
}

/// Parse model-native function-call text into the OpenAI response shape.
///
/// Extraction is gated by the tools supplied with the request and validates
/// every parsed function name against that list. This prevents an ordinary
/// JSON answer from being mistaken for a tool call.
pub fn parse_chat_output(text: &str, tools: Option<&[ChatCompletionTool]>) -> ParsedChatOutput {
    let Some(tools) = tools.filter(|tools| !tools.is_empty()) else {
        return ParsedChatOutput { content: Some(text.to_string()), tool_calls: None };
    };
    let allowed_names = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<std::collections::HashSet<_>>();

    let Some((values, content)) = tool_call_json_values(text) else {
        return ParsedChatOutput { content: Some(text.to_string()), tool_calls: None };
    };
    let Some(raw_calls) = values_to_tool_calls(values) else {
        return ParsedChatOutput { content: Some(text.to_string()), tool_calls: None };
    };

    let mut tool_calls = Vec::with_capacity(raw_calls.len());
    for raw in raw_calls {
        let Some((id, name, arguments)) = normalize_tool_call(raw) else {
            return ParsedChatOutput { content: Some(text.to_string()), tool_calls: None };
        };
        if !allowed_names.contains(name.as_str()) {
            return ParsedChatOutput { content: Some(text.to_string()), tool_calls: None };
        }
        tool_calls.push(ToolCall {
            id: id.unwrap_or_else(generate_tool_call_id),
            tool_type: "function".to_string(),
            function: FunctionCall { name, arguments },
        });
    }

    if tool_calls.is_empty() {
        ParsedChatOutput { content: Some(text.to_string()), tool_calls: None }
    } else {
        ParsedChatOutput { content, tool_calls: Some(tool_calls) }
    }
}

/// Convert complete tool calls into a valid OpenAI streaming delta.
/// Arguments may be emitted in one chunk; clients concatenate them exactly as
/// they do with token-sized argument deltas.
pub fn tool_calls_to_deltas(tool_calls: &[ToolCall]) -> Vec<ToolCallDelta> {
    tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| ToolCallDelta {
            index: index as u32,
            id: Some(call.id.clone()),
            tool_type: Some(call.tool_type.clone()),
            function: Some(FunctionCallDelta {
                name: Some(call.function.name.clone()),
                arguments: Some(call.function.arguments.clone()),
            }),
        })
        .collect()
}

fn tool_call_json_values(text: &str) -> Option<(Vec<serde_json::Value>, Option<String>)> {
    const OPEN_TAG: &str = "<tool_call>";
    const CLOSE_TAG: &str = "</tool_call>";
    const PYTHON_TAG: &str = "<|python_tag|>";

    if text.contains(OPEN_TAG) {
        let mut values = Vec::new();
        let mut remaining = String::new();
        let mut cursor = 0;
        while let Some(relative_start) = text[cursor..].find(OPEN_TAG) {
            let start = cursor + relative_start;
            remaining.push_str(&text[cursor..start]);
            let body_start = start + OPEN_TAG.len();
            let relative_end = text[body_start..].find(CLOSE_TAG)?;
            let end = body_start + relative_end;
            values.extend(parse_json_sequence(&text[body_start..end])?);
            cursor = end + CLOSE_TAG.len();
        }
        remaining.push_str(&text[cursor..]);
        return Some((values, nonempty_content(&remaining)));
    }

    if let Some(start) = text.find(PYTHON_TAG) {
        let content = nonempty_content(&text[..start]);
        let values = parse_json_sequence(&text[start + PYTHON_TAG.len()..])?;
        return Some((values, content));
    }

    let trimmed = strip_json_code_fence(text.trim());
    parse_json_sequence(trimmed).map(|values| (values, None))
}

fn strip_json_code_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```json").or_else(|| text.strip_prefix("```")) else {
        return text;
    };
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

fn parse_json_sequence(text: &str) -> Option<Vec<serde_json::Value>> {
    let values = serde_json::Deserializer::from_str(text)
        .into_iter::<serde_json::Value>()
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!values.is_empty()).then_some(values)
}

fn values_to_tool_calls(values: Vec<serde_json::Value>) -> Option<Vec<serde_json::Value>> {
    let mut calls = Vec::new();
    for value in values {
        match value {
            serde_json::Value::Array(items) => calls.extend(items),
            serde_json::Value::Object(mut object) => {
                if let Some(serde_json::Value::Array(items)) = object.remove("tool_calls") {
                    calls.extend(items);
                } else {
                    calls.push(serde_json::Value::Object(object));
                }
            }
            _ => return None,
        }
    }
    (!calls.is_empty()).then_some(calls)
}

fn normalize_tool_call(value: serde_json::Value) -> Option<(Option<String>, String, String)> {
    let object = value.as_object()?;
    let id = object.get("id").and_then(serde_json::Value::as_str).map(str::to_string);
    let call = object.get("function").and_then(serde_json::Value::as_object).unwrap_or(object);
    let name = call.get("name")?.as_str()?.to_string();
    let arguments = call
        .get("arguments")
        .or_else(|| call.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let arguments = match arguments {
        serde_json::Value::String(encoded) => {
            serde_json::from_str::<serde_json::Value>(&encoded).ok()?;
            encoded
        }
        value @ (serde_json::Value::Object(_) | serde_json::Value::Array(_)) => {
            serde_json::to_string(&value).ok()?
        }
        _ => return None,
    };
    Some((id.filter(|id| !id.is_empty()), name, arguments))
}

fn nonempty_content(text: &str) -> Option<String> {
    let content = text.trim();
    (!content.is_empty()).then(|| content.to_string())
}

/// Convert engine RequestOutput to a ChatCompletionResponse without tool extraction.
pub fn request_output_to_chat_completion(
    output: &RequestOutput,
    model: &str,
) -> ChatCompletionResponse {
    request_output_to_chat_completion_with_tools(output, model, None)
}

/// Convert engine output and extract calls that match the request's tools.
pub fn request_output_to_chat_completion_with_tools(
    output: &RequestOutput,
    model: &str,
    tools: Option<&[ChatCompletionTool]>,
) -> ChatCompletionResponse {
    let id = generate_completion_id("chatcmpl");
    let created = now_timestamp();

    let choices: Vec<ChatChoice> = output
        .outputs
        .iter()
        .enumerate()
        .map(|(i, co)| {
            let parsed = parse_chat_output(&co.text, tools);
            let finish_reason = if parsed.tool_calls.is_some() {
                Some("tool_calls".to_string())
            } else {
                co.finish_reason.map(|r| match r {
                    rllm_core::output::FinishReason::Stop => "stop".to_string(),
                    rllm_core::output::FinishReason::Length => "length".to_string(),
                    rllm_core::output::FinishReason::Aborted => "stop".to_string(),
                    rllm_core::output::FinishReason::Error => "stop".to_string(),
                })
            };
            ChatChoice {
                index: i as u32,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content: parsed.content,
                    tool_calls: parsed.tool_calls,
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
        generation_time: None,
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
        generation_time: None,
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
                ChatMessage {
                    role: "system".into(),
                    content: Some("You are helpful.".into()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
                ChatMessage {
                    role: "user".into(),
                    content: Some("Hello!".into()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
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
            structured_outputs: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "test-model");
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.temperature, Some(0.7));
    }

    #[test]
    fn tool_request_parses_and_defaults_template_choice_to_auto() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather in Istanbul?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }]
        }))
        .unwrap();

        validate_tool_call_request(&req).unwrap();
        assert_eq!(tools_for_chat_template(&req).unwrap().unwrap().len(), 1);
        assert_eq!(tool_choice_for_chat_template(&req).unwrap(), Some(serde_json::json!("auto")));
    }

    #[test]
    fn named_tool_choice_filters_template_tools() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "function", "function": {"name": "get_time"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }))
        .unwrap();

        let tools = tools_for_chat_template(&req).unwrap().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn prior_tool_calls_are_normalized_for_model_templates() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Istanbul\"}"
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}
            ]
        }))
        .unwrap();

        let messages = messages_for_chat_template(&req).unwrap();
        assert_eq!(messages[1]["tool_calls"][0]["function"]["arguments"]["city"], "Istanbul");
    }

    #[test]
    fn rejects_unknown_named_tool_choice() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": {"type": "function", "function": {"name": "missing"}}
        }))
        .unwrap();
        assert!(validate_tool_call_request(&req).is_err());
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
            structured_outputs: None,
            response_format: None,
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

    fn output_tools() -> Vec<ChatCompletionTool> {
        vec![
            serde_json::from_value(serde_json::json!({
                "type": "function",
                "function": {"name": "get_weather"}
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "type": "function",
                "function": {"name": "get_time"}
            }))
            .unwrap(),
        ]
    }

    #[test]
    fn parses_llama_json_tool_call() {
        let tools = output_tools();
        let parsed = parse_chat_output(
            r#"{"name":"get_weather","parameters":{"city":"Istanbul"}}"#,
            Some(&tools),
        );
        assert_eq!(parsed.content, None);
        let calls = parsed.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"city": "Istanbul"})
        );
    }

    #[test]
    fn parses_parallel_hermes_tool_calls_and_preserves_content() {
        let tools = output_tools();
        let parsed = parse_chat_output(
            concat!(
                "I will check both.\n",
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Istanbul\"}}</tool_call>\n",
                "<tool_call>{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}</tool_call>"
            ),
            Some(&tools),
        );
        assert_eq!(parsed.content.as_deref(), Some("I will check both."));
        let calls = parsed.tool_calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn leaves_unknown_or_malformed_calls_as_content() {
        let tools = output_tools();
        for text in [
            r#"{"name":"delete_everything","arguments":{}}"#,
            r#"<tool_call>{"name":"get_weather","arguments":</tool_call>"#,
            r#"{"answer":42}"#,
        ] {
            let parsed = parse_chat_output(text, Some(&tools));
            assert_eq!(parsed.content.as_deref(), Some(text));
            assert!(parsed.tool_calls.is_none());
        }
    }

    #[test]
    fn response_converter_emits_tool_calls_finish_reason() {
        let output = RequestOutput {
            request_id: rllm_core::ids::RequestId::new(),
            outputs: vec![rllm_core::output::CompletionOutput {
                index: 0,
                text: r#"<|python_tag|>{"name":"get_weather","arguments":{"city":"Istanbul"}}"#
                    .into(),
                token_ids: vec![],
                finish_reason: Some(rllm_core::output::FinishReason::Stop),
                logprobs: None,
            }],
            finished: true,
            usage: rllm_core::output::Usage {
                prompt_tokens: 10,
                completion_tokens: 8,
                total_tokens: 18,
            },
        };
        let tools = output_tools();
        let response =
            request_output_to_chat_completion_with_tools(&output, "test-model", Some(&tools));
        let choice = &response.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert!(choice.message.content.is_none());
        assert_eq!(choice.message.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn streaming_delta_has_openai_tool_call_shape() {
        let tools = output_tools();
        let parsed = parse_chat_output(
            r#"[{"name":"get_weather","arguments":{"city":"Istanbul"}}]"#,
            Some(&tools),
        );
        let delta = ChunkDelta {
            role: None,
            content: None,
            tool_calls: parsed.tool_calls.as_deref().map(tool_calls_to_deltas),
        };
        let json = serde_json::to_value(delta).unwrap();
        assert_eq!(json["tool_calls"][0]["index"], 0);
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "get_weather");
    }
}
