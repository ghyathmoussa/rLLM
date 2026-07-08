use std::{pin::Pin, time::Duration};

use async_stream::try_stream;
use rllm_core::{ids::RequestId, output::FinishReason, request::InferenceRequest};
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::{
    openai::{
        self, PromptInput, StopSequence, chat_request_to_sampling_params,
        completion_request_to_sampling_params, generate_completion_id, now_timestamp,
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

    fn authorize<T>(&self, request: &Request<T>) -> Result<(), Status> {
        let Some(expected) = self.state.api_key.as_deref() else {
            return Ok(());
        };

        if metadata_value_matches(request.metadata(), "x-api-key", expected)
            || bearer_matches(request.metadata(), expected)
        {
            return Ok(());
        }

        Err(Status::unauthenticated("missing or invalid API key"))
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
        self.authorize(&request)?;
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
        self.authorize(&request)?;
        let started = std::time::Instant::now();
        rllm_metrics::counter!("rllm_grpc_requests_total").increment(1);

        let req = proto_to_chat_request(request.into_inner());
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

        let completion = run_chat_completion(
            runtime,
            req,
            Duration::from_secs(self.state.request_timeout_secs()),
        )
        .await?;
        rllm_metrics::histogram!("rllm_grpc_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());

        Ok(Response::new(chat_completion_response(self.state.model_name(), completion)))
    }

    async fn stream_chat_completion(
        &self,
        request: Request<pb::ChatCompletionRequest>,
    ) -> Result<Response<Self::StreamChatCompletionStream>, Status> {
        self.authorize(&request)?;
        let started = std::time::Instant::now();
        rllm_metrics::counter!("rllm_grpc_requests_total").increment(1);

        let req = proto_to_chat_request(request.into_inner());
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
                        }),
                        finish_reason: None,
                    }],
                    generation_time: None,
                };

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

                    if !chunk_text.is_empty() || finish_reason.is_some() {
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
                                }),
                                finish_reason,
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
        self.authorize(&request)?;
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
        self.authorize(&request)?;
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
    let messages = req
        .messages
        .iter()
        .map(|msg| rllm_core::request::ChatMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        })
        .collect::<Vec<_>>();
    let prompt = runtime
        .tokenizer
        .render_chat(messages, true)
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
    let messages = req
        .messages
        .iter()
        .map(|msg| rllm_core::request::ChatMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        })
        .collect::<Vec<_>>();
    let prompt = runtime
        .tokenizer
        .render_chat(messages, true)
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

fn proto_to_chat_request(req: pb::ChatCompletionRequest) -> openai::ChatCompletionRequest {
    openai::ChatCompletionRequest {
        model: req.model,
        messages: req
            .messages
            .into_iter()
            .map(|msg| openai::ChatMessage { role: msg.role, content: msg.content })
            .collect(),
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
    }
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
    }
}

fn stop_sequence(stop: Vec<String>) -> Option<StopSequence> {
    if stop.is_empty() { None } else { Some(StopSequence::Multiple(stop)) }
}

fn chat_completion_response(
    model: &str,
    completion: EngineCompletion,
) -> pb::ChatCompletionResponse {
    pb::ChatCompletionResponse {
        id: generate_completion_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![pb::ChatChoice {
            index: 0,
            message: Some(pb::ChatResponseMessage {
                role: "assistant".to_string(),
                content: completion.text,
            }),
            finish_reason: Some(completion.finish_reason),
        }],
        usage: Some(usage_to_proto(completion.usage)),
        generation_time: Some(completion.generation_time),
    }
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
                content: String::new(),
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
    use tonic::Code;

    use super::*;

    fn test_state(api_key: Option<&str>) -> AppState {
        let recorder = PrometheusBuilder::new().build_recorder();
        let mut state = AppState::new("test-model".to_string(), recorder.handle());
        state.api_key = api_key.map(ToOwned::to_owned);
        state
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
            .chat_completion(Request::new(pb::ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![pb::ChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
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
            }))
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
}
