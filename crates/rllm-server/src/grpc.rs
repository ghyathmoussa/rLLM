#![allow(clippy::result_large_err)]

use std::{pin::Pin, time::Duration};

use async_stream::try_stream;
use rllm_core::{ids::RequestId, output::FinishReason, request::InferenceRequest};
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::{
    openai::{
        self, PromptInput, StopSequence, chat_request_to_sampling_params,
        completion_request_to_sampling_params, generate_completion_id, messages_for_chat_template,
        now_timestamp, tool_choice_for_chat_template, tools_for_chat_template,
        validate_tool_call_request,
    },
    server::{
        AppState, EngineCompletion, ModelRuntime, finish_reason_to_openai, submit_and_collect,
    },
};

pub mod pb {
    tonic::include_proto!("rllm.v1");
}

use pb::inference_service_server::InferenceService;

#[derive(Clone)]
pub struct GrpcInferenceService {
    state: AppState,
}

impl GrpcInferenceService {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    fn auth_error<T>(&self, request: &Request<T>) -> Option<Status> {
        let expected = self.state.api_key.as_deref()?;

        if metadata_value_matches(request.metadata(), "x-api-key", expected)
            || bearer_matches(request.metadata(), expected)
        {
            return None;
        }

        Some(Status::unauthenticated("missing or invalid API key"))
    }
}

#[tonic::async_trait]
impl InferenceService for GrpcInferenceService {
    type StreamChatCompletionStream =
        Pin<Box<dyn Stream<Item = Result<pb::ChatCompletionChunk, Status>> + Send + 'static>>;
    type StreamCompletionStream =
        Pin<Box<dyn Stream<Item = Result<pb::CompletionChunk, Status>> + Send + 'static>>;

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        Ok(Response::new(pb::HealthResponse { status: "ok".to_string() }))
    }

    async fn list_models(
        &self,
        request: Request<pb::ListModelsRequest>,
    ) -> Result<Response<pb::ModelListResponse>, Status> {
        if let Some(status) = self.auth_error(&request) {
            return Err(status);
        }
        let now = now_timestamp();
        Ok(Response::new(pb::ModelListResponse {
            object: "list".to_string(),
            data: vec![pb::ModelInfo {
                id: self.state.model_name().to_string(),
                object: "model".to_string(),
                created: now,
                owned_by: "rllm".to_string(),
            }],
        }))
    }

    async fn chat_completion(
        &self,
        request: Request<pb::ChatCompletionRequest>,
    ) -> Result<Response<pb::ChatCompletionResponse>, Status> {
        if let Some(status) = self.auth_error(&request) {
            return Err(status);
        }
        let started = std::time::Instant::now();
        rllm_metrics::counter!("rllm_grpc_requests_total").increment(1);

        let req = proto_to_chat_request(request.into_inner()).map_err(Status::invalid_argument)?;
        if req.messages.is_empty() {
            return Err(Status::invalid_argument("messages must not be empty"));
        }
        if req.messages.len() > self.state.max_input_messages() {
            return Err(Status::invalid_argument(format!(
                "too many messages: max is {}",
                self.state.max_input_messages()
            )));
        }

        let Some(runtime) = self.state.runtime() else {
            return Ok(Response::new(empty_chat_response(self.state.model_name(), started)));
        };

        let response_tools = response_tools(&req);
        let completion = run_chat_completion(
            runtime,
            req,
            Duration::from_secs(self.state.request_timeout_secs()),
        )
        .await?;
        rllm_metrics::histogram!("rllm_grpc_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());

        Ok(Response::new(chat_completion_response(
            self.state.model_name(),
            completion,
            response_tools.as_deref(),
        )))
    }

    async fn stream_chat_completion(
        &self,
        request: Request<pb::ChatCompletionRequest>,
    ) -> Result<Response<Self::StreamChatCompletionStream>, Status> {
        if let Some(status) = self.auth_error(&request) {
            return Err(status);
        }
        let started = std::time::Instant::now();
        rllm_metrics::counter!("rllm_grpc_requests_total").increment(1);

        let req = proto_to_chat_request(request.into_inner()).map_err(Status::invalid_argument)?;
        if req.messages.is_empty() {
            return Err(Status::invalid_argument("messages must not be empty"));
        }
        if req.messages.len() > self.state.max_input_messages() {
            return Err(Status::invalid_argument(format!(
                "too many messages: max is {}",
                self.state.max_input_messages()
            )));
        }

        let model = self.state.model_name().to_string();
        let runtime = self.state.runtime();
        let response_tools = response_tools(&req);
        let stream = try_stream! {
            if let Some(runtime) = runtime {
                let mut receiver = start_chat_stream(runtime.clone(), req).await?;
                let id = generate_completion_id("chatcmpl");
                let created = now_timestamp();
                yield pb::ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![pb::ChunkChoice {
                        index: 0,
                        delta: Some(pb::ChunkDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: Vec::new(),
                        }),
                        finish_reason: None,
                    }],
                    generation_time: None,
                };

                let elapsed_start = std::time::Instant::now();
                let mut buffered_tool_token_ids = Vec::new();
                while let Some(output) = receiver.recv().await {
                    let mut chunk_text = String::new();
                    for completion in &output.outputs {
                        if response_tools.is_some() {
                            buffered_tool_token_ids.extend_from_slice(&completion.token_ids);
                        } else if !completion.token_ids.is_empty() {
                            let text = runtime
                                .tokenizer
                                .decode(completion.token_ids.clone(), true)
                                .await
                                .map_err(|e| Status::internal(format!("failed to decode output tokens: {e}")))?;
                            chunk_text.push_str(&text);
                        }
                    }

                    let engine_finish_reason = output
                        .outputs
                        .first()
                        .and_then(|completion| completion.finish_reason)
                        .map(finish_reason_to_openai);

                    if response_tools.is_some() {
                        if output.finished {
                            let buffered_tool_text = runtime
                                .tokenizer
                                .decode(buffered_tool_token_ids.clone(), true)
                                .await
                                .map_err(|e| Status::internal(format!("failed to decode tool call: {e}")))?;
                            let parsed = openai::parse_chat_output(
                                &buffered_tool_text,
                                response_tools.as_deref(),
                            );
                            let has_tool_calls = parsed.tool_calls.is_some();
                            yield pb::ChatCompletionChunk {
                                id: id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![pb::ChunkChoice {
                                    index: 0,
                                    delta: Some(pb::ChunkDelta {
                                        role: None,
                                        content: parsed.content,
                                        tool_calls: parsed
                                            .tool_calls
                                            .as_deref()
                                            .map(tool_call_deltas_to_proto)
                                            .unwrap_or_default(),
                                    }),
                                    finish_reason: if has_tool_calls {
                                        Some("tool_calls".to_string())
                                    } else {
                                        engine_finish_reason
                                    },
                                }],
                                generation_time: Some(elapsed_start.elapsed().as_secs_f64()),
                            };
                        }
                    } else if !chunk_text.is_empty() || engine_finish_reason.is_some() {
                        yield pb::ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![pb::ChunkChoice {
                                index: 0,
                                delta: Some(pb::ChunkDelta {
                                    role: None,
                                    content: Some(chunk_text),
                                    tool_calls: Vec::new(),
                                }),
                                finish_reason: engine_finish_reason,
                            }],
                            generation_time: Some(elapsed_start.elapsed().as_secs_f64()),
                        };
                    }

                    if output.finished {
                        break;
                    }
                }
            } else {
                yield pb::ChatCompletionChunk {
                    id: generate_completion_id("chatcmpl"),
                    object: "chat.completion.chunk".to_string(),
                    created: now_timestamp(),
                    model,
                    choices: vec![pb::ChunkChoice {
                        index: 0,
                        delta: Some(pb::ChunkDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: Vec::new(),
                        }),
                        finish_reason: Some("stop".to_string()),
                    }],
                    generation_time: Some(started.elapsed().as_secs_f64()),
                };
            }
            rllm_metrics::histogram!("rllm_grpc_request_duration_seconds")
                .record(started.elapsed().as_secs_f64());
        };

        Ok(Response::new(Box::pin(stream) as Self::StreamChatCompletionStream))
    }

    async fn completion(
        &self,
        request: Request<pb::CompletionRequest>,
    ) -> Result<Response<pb::CompletionResponse>, Status> {
        if let Some(status) = self.auth_error(&request) {
            return Err(status);
        }
        let started = std::time::Instant::now();
        rllm_metrics::counter!("rllm_grpc_requests_total").increment(1);

        let req = proto_to_completion_request(request.into_inner());
        let Some(runtime) = self.state.runtime() else {
            return Ok(Response::new(empty_completion_response(self.state.model_name(), started)));
        };

        let completion = crate::server::run_text_completion(
            runtime,
            req,
            Duration::from_secs(self.state.request_timeout_secs()),
        )
        .await
        .map_err(internal_status)?;
        rllm_metrics::histogram!("rllm_grpc_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());

        Ok(Response::new(completion_response(self.state.model_name(), completion)))
    }

    async fn stream_completion(
        &self,
        request: Request<pb::CompletionRequest>,
    ) -> Result<Response<Self::StreamCompletionStream>, Status> {
        if let Some(status) = self.auth_error(&request) {
            return Err(status);
        }
        let started = std::time::Instant::now();
        rllm_metrics::counter!("rllm_grpc_requests_total").increment(1);

        let req = proto_to_completion_request(request.into_inner());
        let model = self.state.model_name().to_string();
        let runtime = self.state.runtime();
        let stream = try_stream! {
            if let Some(runtime) = runtime {
                let prompt = match &req.prompt {
                    PromptInput::Single(text) => text.clone(),
                    PromptInput::Multiple(items) => items.join("\n"),
                };
                let token_ids = runtime
                    .tokenizer
                    .encode(prompt.clone(), true)
                    .await
                    .map_err(|e| Status::internal(format!("failed to tokenize completion prompt: {e}")))?;
                let sampling_params = completion_request_to_sampling_params(&req);
                sampling_params
                    .validate()
                    .map_err(|e| Status::invalid_argument(format!("invalid sampling params: {e}")))?;

                let mut receiver = runtime
                    .engine
                    .add_request_stream(InferenceRequest {
                        request_id: RequestId::new(),
                        prompt: Some(prompt),
                        token_ids: Some(token_ids.clone()),
                        messages: None,
                        sampling_params,
                        arrival_time: std::time::Instant::now(),
                        priority: 0,
                        stream: true,
                        cache_salt: None,
                    })
                    .map_err(internal_status)?;

                let id = generate_completion_id("cmpl");
                let created = now_timestamp();
                let elapsed_start = std::time::Instant::now();
                while let Some(output) = receiver.recv().await {
                    let mut chunk_text = String::new();
                    for completion in &output.outputs {
                        if !completion.token_ids.is_empty() {
                            let text = runtime
                                .tokenizer
                                .decode(completion.token_ids.clone(), true)
                                .await
                                .map_err(|e| Status::internal(format!("failed to decode output tokens: {e}")))?;
                            chunk_text.push_str(&text);
                        }
                    }
                    let finish_reason = output
                        .outputs
                        .first()
                        .and_then(|completion| completion.finish_reason)
                        .map(finish_reason_to_openai);
                    yield pb::CompletionChunk {
                        id: id.clone(),
                        object: "text_completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![pb::CompletionChoice {
                            index: 0,
                            text: chunk_text,
                            finish_reason,
                        }],
                        usage: Some(usage_to_proto(openai::UsageInfo {
                            prompt_tokens: output.usage.prompt_tokens,
                            completion_tokens: output.usage.completion_tokens,
                            total_tokens: output.usage.total_tokens,
                        })),
                        finished: output.finished,
                        generation_time: Some(elapsed_start.elapsed().as_secs_f64()),
                    };
                    if output.finished {
                        break;
                    }
                }
            } else {
                yield pb::CompletionChunk {
                    id: generate_completion_id("cmpl"),
                    object: "text_completion.chunk".to_string(),
                    created: now_timestamp(),
                    model,
                    choices: vec![pb::CompletionChoice {
                        index: 0,
                        text: String::new(),
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Some(pb::UsageInfo {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    }),
                    finished: true,
                    generation_time: Some(started.elapsed().as_secs_f64()),
                };
            }
            rllm_metrics::histogram!("rllm_grpc_request_duration_seconds")
                .record(started.elapsed().as_secs_f64());
        };

        Ok(Response::new(Box::pin(stream) as Self::StreamCompletionStream))
    }
}

async fn run_chat_completion(
    runtime: ModelRuntime,
    req: openai::ChatCompletionRequest,
    timeout: Duration,
) -> Result<EngineCompletion, Status> {
    validate_tool_call_request(&req).map_err(Status::invalid_argument)?;
    let messages = messages_for_chat_template(&req).map_err(Status::invalid_argument)?;
    let tools = tools_for_chat_template(&req).map_err(Status::invalid_argument)?;
    let tool_choice = tool_choice_for_chat_template(&req).map_err(Status::invalid_argument)?;
    let prompt = runtime
        .tokenizer
        .render_chat_with_tools(
            messages,
            tools,
            tool_choice,
            req.parallel_tool_calls.unwrap_or(true),
            true,
        )
        .await
        .map_err(|e| Status::internal(format!("failed to render chat template: {e}")))?;
    let token_ids = runtime
        .tokenizer
        .encode(prompt.clone(), false)
        .await
        .map_err(|e| Status::internal(format!("failed to tokenize chat prompt: {e}")))?;
    let sampling_params = chat_request_to_sampling_params(&req);
    sampling_params
        .validate()
        .map_err(|e| Status::invalid_argument(format!("invalid sampling params: {e}")))?;

    submit_and_collect(runtime, Some(prompt), token_ids, sampling_params, timeout)
        .await
        .map_err(internal_status)
}

async fn start_chat_stream(
    runtime: ModelRuntime,
    req: openai::ChatCompletionRequest,
) -> Result<tokio::sync::mpsc::UnboundedReceiver<rllm_core::output::RequestOutput>, Status> {
    validate_tool_call_request(&req).map_err(Status::invalid_argument)?;
    let messages = messages_for_chat_template(&req).map_err(Status::invalid_argument)?;
    let tools = tools_for_chat_template(&req).map_err(Status::invalid_argument)?;
    let tool_choice = tool_choice_for_chat_template(&req).map_err(Status::invalid_argument)?;
    let prompt = runtime
        .tokenizer
        .render_chat_with_tools(
            messages,
            tools,
            tool_choice,
            req.parallel_tool_calls.unwrap_or(true),
            true,
        )
        .await
        .map_err(|e| Status::internal(format!("failed to render chat template: {e}")))?;
    let token_ids = runtime
        .tokenizer
        .encode(prompt.clone(), false)
        .await
        .map_err(|e| Status::internal(format!("failed to tokenize chat prompt: {e}")))?;
    let sampling_params = chat_request_to_sampling_params(&req);
    sampling_params
        .validate()
        .map_err(|e| Status::invalid_argument(format!("invalid sampling params: {e}")))?;

    runtime
        .engine
        .add_request_stream(InferenceRequest {
            request_id: RequestId::new(),
            prompt: Some(prompt),
            token_ids: Some(token_ids),
            messages: None,
            sampling_params,
            arrival_time: std::time::Instant::now(),
            priority: 0,
            stream: true,
            cache_salt: None,
        })
        .map_err(internal_status)
}

fn proto_to_chat_request(
    req: pb::ChatCompletionRequest,
) -> Result<openai::ChatCompletionRequest, String> {
    let messages = req
        .messages
        .into_iter()
        .map(|msg| {
            let tool_calls = if msg.tool_calls.is_empty() {
                None
            } else {
                Some(
                    msg.tool_calls
                        .into_iter()
                        .map(proto_to_tool_call)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            };
            Ok(openai::ChatMessage {
                role: msg.role,
                content: msg.content,
                name: msg.name,
                tool_call_id: msg.tool_call_id,
                tool_calls,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(
            req.tools
                .into_iter()
                .map(|tool| {
                    let function =
                        tool.function.ok_or_else(|| "tools require a function".to_string())?;
                    let parameters = function
                        .parameters_json
                        .map(|json| {
                            serde_json::from_str(&json)
                                .map_err(|error| format!("invalid tool parameters_json: {error}"))
                        })
                        .transpose()?;
                    Ok(openai::ChatCompletionTool {
                        tool_type: tool.r#type,
                        function: openai::FunctionDefinition {
                            name: function.name,
                            description: function.description,
                            parameters,
                            strict: function.strict,
                        },
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        )
    };

    let tool_choice = req
        .tool_choice
        .and_then(|choice| choice.choice)
        .map(|choice| match choice {
            pb::tool_choice::Choice::Mode(mode) => match mode.as_str() {
                "none" => Ok(openai::ToolChoice::Mode(openai::ToolChoiceMode::None)),
                "auto" => Ok(openai::ToolChoice::Mode(openai::ToolChoiceMode::Auto)),
                "required" => Ok(openai::ToolChoice::Mode(openai::ToolChoiceMode::Required)),
                _ => Err("tool_choice mode must be none, auto, or required".to_string()),
            },
            pb::tool_choice::Choice::Named(named) => {
                Ok(openai::ToolChoice::Named(openai::NamedToolChoice {
                    tool_type: named.r#type,
                    function: openai::NamedFunction { name: named.function_name },
                }))
            }
        })
        .transpose()?;

    Ok(openai::ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stream: Some(false),
        stop: stop_sequence(req.stop),
        n: req.n,
        logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        structured_outputs: None,
        response_format: None,
        tools,
        tool_choice,
        parallel_tool_calls: req.parallel_tool_calls,
    })
}

fn proto_to_tool_call(call: pb::ToolCall) -> Result<openai::ToolCall, String> {
    let function = call.function.ok_or_else(|| "tool calls require a function".to_string())?;
    Ok(openai::ToolCall {
        id: call.id,
        tool_type: call.r#type,
        function: openai::FunctionCall { name: function.name, arguments: function.arguments },
    })
}

fn proto_to_completion_request(req: pb::CompletionRequest) -> openai::CompletionRequest {
    openai::CompletionRequest {
        model: req.model,
        prompt: PromptInput::Single(req.prompt),
        suffix: None,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stream: Some(false),
        stop: stop_sequence(req.stop),
        n: req.n,
        logprobs: req.logprobs,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        structured_outputs: None,
        response_format: None,
    }
}

fn stop_sequence(stop: Vec<String>) -> Option<StopSequence> {
    if stop.is_empty() { None } else { Some(StopSequence::Multiple(stop)) }
}

fn chat_completion_response(
    model: &str,
    completion: EngineCompletion,
    tools: Option<&[openai::ChatCompletionTool]>,
) -> pb::ChatCompletionResponse {
    let parsed = openai::parse_chat_output(&completion.text, tools);
    let has_tool_calls = parsed.tool_calls.is_some();
    pb::ChatCompletionResponse {
        id: generate_completion_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![pb::ChatChoice {
            index: 0,
            message: Some(pb::ChatResponseMessage {
                role: "assistant".to_string(),
                content: parsed.content,
                tool_calls: parsed
                    .tool_calls
                    .as_deref()
                    .map(tool_calls_to_proto)
                    .unwrap_or_default(),
            }),
            finish_reason: Some(if has_tool_calls {
                "tool_calls".to_string()
            } else {
                completion.finish_reason
            }),
        }],
        usage: Some(usage_to_proto(completion.usage)),
        generation_time: Some(completion.generation_time),
    }
}

fn response_tools(req: &openai::ChatCompletionRequest) -> Option<Vec<openai::ChatCompletionTool>> {
    if matches!(
        req.tool_choice.as_ref(),
        Some(openai::ToolChoice::Mode(openai::ToolChoiceMode::None))
    ) {
        None
    } else {
        req.tools.clone()
    }
}

fn tool_calls_to_proto(calls: &[openai::ToolCall]) -> Vec<pb::ToolCall> {
    calls
        .iter()
        .map(|call| pb::ToolCall {
            id: call.id.clone(),
            r#type: call.tool_type.clone(),
            function: Some(pb::FunctionCall {
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
            }),
        })
        .collect()
}

fn tool_call_deltas_to_proto(calls: &[openai::ToolCall]) -> Vec<pb::ToolCallDelta> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| pb::ToolCallDelta {
            index: index as u32,
            id: Some(call.id.clone()),
            r#type: Some(call.tool_type.clone()),
            function: Some(pb::FunctionCallDelta {
                name: Some(call.function.name.clone()),
                arguments: Some(call.function.arguments.clone()),
            }),
        })
        .collect()
}

fn completion_response(model: &str, completion: EngineCompletion) -> pb::CompletionResponse {
    pb::CompletionResponse {
        id: generate_completion_id("cmpl"),
        object: "text_completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![pb::CompletionChoice {
            index: 0,
            text: completion.text,
            finish_reason: Some(completion.finish_reason),
        }],
        usage: Some(usage_to_proto(completion.usage)),
        generation_time: Some(completion.generation_time),
    }
}

fn empty_chat_response(model: &str, started: std::time::Instant) -> pb::ChatCompletionResponse {
    pb::ChatCompletionResponse {
        id: generate_completion_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![pb::ChatChoice {
            index: 0,
            message: Some(pb::ChatResponseMessage {
                role: "assistant".to_string(),
                content: Some(String::new()),
                tool_calls: Vec::new(),
            }),
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(pb::UsageInfo { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }),
        generation_time: Some(started.elapsed().as_secs_f64()),
    }
}

fn empty_completion_response(model: &str, started: std::time::Instant) -> pb::CompletionResponse {
    pb::CompletionResponse {
        id: generate_completion_id("cmpl"),
        object: "text_completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![pb::CompletionChoice {
            index: 0,
            text: String::new(),
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(pb::UsageInfo { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }),
        generation_time: Some(started.elapsed().as_secs_f64()),
    }
}

fn usage_to_proto(usage: openai::UsageInfo) -> pb::UsageInfo {
    pb::UsageInfo {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

fn internal_status(error: anyhow::Error) -> Status {
    tracing::error!(?error, "gRPC request failed");
    Status::internal("An internal error occurred while processing your request.")
}

fn metadata_value_matches(
    metadata: &tonic::metadata::MetadataMap,
    key: &'static str,
    expected: &str,
) -> bool {
    metadata.get(key).and_then(|value| value.to_str().ok()).is_some_and(|actual| {
        subtle::ConstantTimeEq::ct_eq(actual.as_bytes(), expected.as_bytes()).into()
    })
}

fn bearer_matches(metadata: &tonic::metadata::MetadataMap, expected: &str) -> bool {
    metadata
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|actual| actual.strip_prefix("Bearer "))
        .is_some_and(|token| {
            subtle::ConstantTimeEq::ct_eq(token.as_bytes(), expected.as_bytes()).into()
        })
}

#[allow(dead_code)]
fn _finish_reason_to_proto(reason: FinishReason) -> String {
    finish_reason_to_openai(reason)
}

#[cfg(test)]
mod tests {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use tokio_stream::StreamExt;
    use tonic::{Code, transport::Server};

    use super::*;

    fn test_state(api_key: Option<&str>) -> AppState {
        let recorder = PrometheusBuilder::new().build_recorder();
        let mut state = AppState::new("test-model".to_string(), recorder.handle());
        state.api_key = api_key.map(ToOwned::to_owned);
        state
    }

    async fn spawn_local_grpc_server()
    -> (String, tokio::sync::oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let service = pb::inference_service_server::InferenceServiceServer::new(
            GrpcInferenceService::new(test_state(None)),
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let incoming = async_stream::stream! {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => yield Ok::<_, std::io::Error>(stream),
                    Err(error) => {
                        yield Err(error);
                        break;
                    }
                }
            }
        };

        let handle = tokio::spawn(async move {
            let result = Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = shutdown_rx.await;
                })
                .await;
            if let Err(error) = result {
                panic!("local gRPC server failed: {error}");
            }
        });

        (format!("http://{addr}"), shutdown_tx, handle)
    }

    fn chat_request(content: impl Into<String>) -> pb::ChatCompletionRequest {
        pb::ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![pb::ChatMessage {
                role: "user".to_string(),
                content: Some(content.into()),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }],
            temperature: None,
            top_p: None,
            max_tokens: Some(4),
            stop: vec![],
            n: None,
            logprobs: None,
            top_logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
        }
    }

    #[tokio::test]
    async fn list_models_requires_api_key_when_configured() {
        let service = GrpcInferenceService::new(test_state(Some("secret")));
        let err = service.list_models(Request::new(pb::ListModelsRequest {})).await.unwrap_err();
        assert_eq!(err.code(), Code::Unauthenticated);
    }

    #[tokio::test]
    async fn list_models_accepts_bearer_token() {
        let service = GrpcInferenceService::new(test_state(Some("secret")));
        let mut request = Request::new(pb::ListModelsRequest {});
        request.metadata_mut().insert("authorization", "Bearer secret".parse().unwrap());

        let response = service.list_models(request).await.unwrap().into_inner();
        assert_eq!(response.object, "list");
        assert_eq!(response.data[0].id, "test-model");
    }

    #[tokio::test]
    async fn chat_completion_uses_placeholder_without_runtime() {
        let service = GrpcInferenceService::new(test_state(None));
        let response = service
            .chat_completion(Request::new(chat_request("hello")))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.choices[0].message.as_ref().unwrap().role, "assistant");
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn stream_completion_uses_placeholder_without_runtime() {
        let service = GrpcInferenceService::new(test_state(None));
        let mut stream = service
            .stream_completion(Request::new(pb::CompletionRequest {
                model: "test-model".to_string(),
                prompt: "hello".to_string(),
                temperature: None,
                top_p: None,
                max_tokens: Some(4),
                stop: vec![],
                n: None,
                logprobs: None,
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
            }))
            .await
            .unwrap()
            .into_inner();

        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.object, "text_completion.chunk");
        assert!(chunk.finished);
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn local_grpc_server_handles_concurrent_chat_requests() {
        let (endpoint, shutdown, server_handle) = spawn_local_grpc_server().await;

        let mut client =
            pb::inference_service_client::InferenceServiceClient::connect(endpoint.clone())
                .await
                .unwrap();
        let health = client.health(pb::HealthRequest {}).await.unwrap().into_inner();
        assert_eq!(health.status, "ok");

        let mut tasks = Vec::new();
        for index in 0..8 {
            let endpoint = endpoint.clone();
            tasks.push(tokio::spawn(async move {
                let mut client =
                    pb::inference_service_client::InferenceServiceClient::connect(endpoint)
                        .await
                        .unwrap();
                let response = client
                    .chat_completion(chat_request(format!("hello {index}")))
                    .await
                    .unwrap()
                    .into_inner();
                assert_eq!(response.object, "chat.completion");
                assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
            }));
        }

        for task in tasks {
            task.await.unwrap();
        }

        let _ = shutdown.send(());
        server_handle.await.unwrap();
    }

    #[test]
    fn proto_tool_request_maps_all_fields() {
        let mut req = chat_request("Weather in Istanbul?");
        req.messages.push(pb::ChatMessage {
            role: "assistant".to_string(),
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: vec![pb::ToolCall {
                id: "call_weather".to_string(),
                r#type: "function".to_string(),
                function: Some(pb::FunctionCall {
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Istanbul"}"#.to_string(),
                }),
            }],
        });
        req.tools = vec![pb::ChatCompletionTool {
            r#type: "function".to_string(),
            function: Some(pb::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters_json: Some(r#"{"type":"object"}"#.to_string()),
                strict: Some(true),
            }),
        }];
        req.tool_choice = Some(pb::ToolChoice {
            choice: Some(pb::tool_choice::Choice::Named(pb::NamedToolChoice {
                r#type: "function".to_string(),
                function_name: "get_weather".to_string(),
            })),
        });
        req.parallel_tool_calls = Some(false);

        let mapped = proto_to_chat_request(req).unwrap();
        validate_tool_call_request(&mapped).unwrap();
        assert_eq!(mapped.tools.as_ref().unwrap()[0].function.name, "get_weather");
        assert!(matches!(mapped.tool_choice, Some(openai::ToolChoice::Named(_))));
        assert_eq!(mapped.messages[1].tool_calls.as_ref().unwrap()[0].id, "call_weather");
        assert_eq!(mapped.parallel_tool_calls, Some(false));
    }
}
