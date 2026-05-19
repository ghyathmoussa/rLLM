use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderValue, Method};
use axum::middleware::from_fn_with_state;
use axum::response::sse::Event;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use metrics_exporter_prometheus::PrometheusHandle;
use rllm_cache::spec::{KVCacheConfig, KVCacheSpec};
use rllm_core::config::{
    CacheConfig, ModelConfig, PrefixHashAlgorithm, SchedulerConfig, SchedulingPolicy,
};
use rllm_core::dtype::DType;
use rllm_core::ids::RequestId;
use rllm_core::output::FinishReason;
use rllm_core::request::InferenceRequest;
use rllm_engine::{AsyncLLMEngine, EngineCore};
use rllm_executor::{Executor, UniProcExecutor};
use rllm_model::{hf_config, loader};
use rllm_scheduler::Scheduler;
use rllm_tokenizer::pool::AsyncTokenizerPool;
use rllm_tokenizer::tokenizer::Tokenizer;
use rllm_worker::Worker;
use serde::Serialize;
use tokio::net::TcpListener;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::auth;
use crate::cli::ServeArgs;
use crate::openai::*;

const BLOCK_SIZE: usize = 16;

#[derive(Clone)]
#[allow(dead_code)]
struct ModelRuntime {
    engine: Arc<AsyncLLMEngine>,
    tokenizer: Arc<AsyncTokenizerPool>,
    model_dir: String,
    architecture: String,
    device: String,
}

/// Application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    model_name: String,
    runtime: Option<ModelRuntime>,
    /// Prometheus metrics handle for the `/metrics` endpoint.
    metrics_handle: PrometheusHandle,
    /// Optional API key for authenticated endpoints.
    pub api_key: Option<String>,
    /// CORS allowed origins configuration.
    cors_allowed_origins: String,
    /// Enable debug endpoints.
    enable_debug_endpoints: bool,
    /// Maximum concurrent requests.
    max_concurrent_requests: usize,
    /// Request timeout in seconds.
    request_timeout_secs: u64,
    /// Maximum messages allowed in chat completion request.
    max_input_messages: usize,
}

impl AppState {
    /// Test/placeholder state. The real server path uses `with_runtime`.
    pub fn new(model_name: String, metrics_handle: PrometheusHandle) -> Self {
        Self {
            model_name,
            runtime: None,
            metrics_handle,
            api_key: None,
            cors_allowed_origins: "*".to_string(),
            enable_debug_endpoints: false,
            max_concurrent_requests: 64,
            request_timeout_secs: 120,
            max_input_messages: 256,
        }
    }

    fn with_runtime(
        model_name: String,
        metrics_handle: PrometheusHandle,
        runtime: ModelRuntime,
        api_key: Option<String>,
        cors_allowed_origins: String,
        enable_debug_endpoints: bool,
        max_concurrent_requests: usize,
        request_timeout_secs: u64,
        max_input_messages: usize,
    ) -> Self {
        Self {
            model_name,
            runtime: Some(runtime),
            metrics_handle,
            api_key,
            cors_allowed_origins,
            enable_debug_endpoints,
            max_concurrent_requests,
            request_timeout_secs,
            max_input_messages,
        }
    }
}

/// Build the router with all routes and middleware.
pub fn build_router(state: AppState) -> Router {
    let cors = build_cors_layer(&state.cors_allowed_origins);

    let v1_router = Router::new()
        .route("/models", get(list_models_handler))
        .route("/chat/completions", post(chat_completions_handler))
        .route("/completions", post(completions_handler))
        .layer(ConcurrencyLimitLayer::new(state.max_concurrent_requests))
        .route_layer(from_fn_with_state(state.clone(), auth::auth_middleware));

    let mut router = Router::new()
        .route("/", get(docs_handler))
        .route("/docs", get(docs_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .nest("/v1", v1_router)
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    if state.enable_debug_endpoints {
        router = router.route("/debug/model", get(debug_model_handler));
    }

    router.with_state(state)
}

fn build_cors_layer(origins: &str) -> CorsLayer {
    if origins == "*" {
        return CorsLayer::permissive();
    }
    let origins: Vec<HeaderValue> = origins
        .split(',')
        .map(|s| s.trim().parse().expect("invalid CORS origin"))
        .collect();
    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
}

/// Build and run the HTTP server.
pub async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    let metrics_handle = rllm_metrics::install_recorder();
    rllm_metrics::describe_metrics();

    let runtime = build_runtime(&args).await?;
    let state = AppState::with_runtime(
        args.model.clone(),
        metrics_handle,
        runtime,
        args.api_key.clone(),
        args.cors_allowed_origins.clone(),
        args.enable_debug_endpoints,
        args.max_concurrent_requests,
        args.request_timeout_secs,
        args.max_input_messages,
    );
    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);
    tracing::info!("OpenAI-compatible docs available at http://{}/docs", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn build_runtime(args: &ServeArgs) -> Result<ModelRuntime> {
    let model_ref = args.model.clone();
    let args = args.clone();
    tokio::task::spawn_blocking(move || build_runtime_blocking(&model_ref, &args))
        .await
        .context("joining model initialization task")?
}

fn build_runtime_blocking(model_ref: &str, args: &ServeArgs) -> Result<ModelRuntime> {
    tracing::info!(model = %model_ref, "initializing real inference runtime");

    let model_dir = loader::resolve_model_dir(model_ref)
        .with_context(|| format!("resolving model reference {model_ref}"))?;
    let mut model_config = hf_config::parse_hf_config(&model_dir.join("config.json"))
        .with_context(|| format!("parsing model config from {}", model_dir.display()))?;
    model_config.model_id = model_dir.to_string_lossy().to_string();
    if let Some(max_len) = args.max_model_len {
        model_config.max_model_len = max_len;
    }
    model_config.dtype = parse_dtype(&args.dtype).unwrap_or(model_config.dtype);

    let tokenizer = load_tokenizer(model_ref, &model_dir)?;
    let eos_token_id = tokenizer.eos_token_id().unwrap_or(model_config.vocab_size as u32);
    let tokenizer = Arc::new(AsyncTokenizerPool::new(tokenizer, 4));

    let kv_config = kv_cache_config(&model_config, args);
    let cache_config = cache_config(args, &model_config);
    let scheduler_config = scheduler_config(args);
    let scheduler = Scheduler::new(
        scheduler_config,
        &cache_config,
        kv_config.num_blocks,
        model_config.max_model_len,
    );

    let worker = Worker::new(0, model_config.clone(), rllm_tensor::Device::cuda(0), BLOCK_SIZE);
    let mut executor = UniProcExecutor::new(worker);
    executor.set_eos_token_id(eos_token_id);
    executor.worker_mut().initialize_rng_seed(0).context("initializing worker RNG")?;
    executor.initialize(&[kv_config]).context("initializing executor and loading model weights")?;

    let device = "cuda:0".to_string();
    let architecture = model_config.architecture.clone();
    let core = EngineCore::new(Box::new(executor), scheduler, eos_token_id);
    let engine = Arc::new(AsyncLLMEngine::new(core));

    tracing::info!(
        model = %model_ref,
        model_dir = %model_dir.display(),
        architecture = %architecture,
        device = %device,
        eos_token_id,
        "real inference runtime initialized"
    );

    Ok(ModelRuntime {
        engine,
        tokenizer,
        model_dir: model_dir.to_string_lossy().to_string(),
        architecture,
        device,
    })
}

fn load_tokenizer(model_ref: &str, model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_json = model_dir.join("tokenizer.json");
    if tokenizer_json.exists() {
        tracing::info!(path = %tokenizer_json.display(), "loading tokenizer from local model dir");
        return Tokenizer::from_file(&tokenizer_json.to_string_lossy());
    }

    if Path::new(model_ref).is_dir() {
        anyhow::bail!("tokenizer.json not found in local model directory {}", model_dir.display());
    }

    tracing::info!(model = %model_ref, "loading tokenizer from Hugging Face");
    Tokenizer::from_model_id(model_ref)
}

fn parse_dtype(dtype: &str) -> Option<DType> {
    match dtype {
        "auto" => None,
        "float16" | "fp16" | "f16" => Some(DType::F16),
        "bfloat16" | "bf16" => Some(DType::BF16),
        "float32" | "fp32" | "f32" => Some(DType::F32),
        _ => None,
    }
}

fn cache_config(args: &ServeArgs, model_config: &ModelConfig) -> CacheConfig {
    CacheConfig {
        block_size: BLOCK_SIZE,
        hash_block_size: BLOCK_SIZE,
        gpu_memory_utilization: args.gpu_memory_utilization,
        cpu_swap_bytes: 0,
        cache_dtype: model_config.dtype,
        num_gpu_blocks: num_cache_blocks(args, model_config.max_model_len),
        enable_prefix_caching: args.enable_prefix_caching,
        prefix_hash_algorithm: PrefixHashAlgorithm::Sha256Cbor,
        sliding_window: None,
    }
}

fn scheduler_config(args: &ServeArgs) -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: args.max_num_seqs,
        max_num_batched_tokens: args.max_num_batched_tokens,
        max_num_scheduled_tokens: args.max_num_batched_tokens,
        long_prefill_token_threshold: 2048,
        enable_chunked_prefill: true,
        scheduling_policy: SchedulingPolicy::FCFS,
        stream_interval: 1,
        async_scheduling: false,
    }
}

fn kv_cache_config(model_config: &ModelConfig, args: &ServeArgs) -> KVCacheConfig {
    KVCacheConfig {
        num_blocks: num_cache_blocks(args, model_config.max_model_len),
        spec: KVCacheSpec {
            block_size: BLOCK_SIZE,
            num_layers: model_config.num_layers,
            num_kv_heads: model_config.num_kv_heads,
            head_dim: model_config.head_dim,
            dtype: model_config.dtype,
            sliding_window: None,
        },
    }
}

fn num_cache_blocks(args: &ServeArgs, max_model_len: usize) -> usize {
    let blocks_per_seq = max_model_len.div_ceil(BLOCK_SIZE).max(1);
    args.max_num_seqs.saturating_mul(blocks_per_seq).max(1)
}

// ── Handlers ───────────────────────────────────────────────────────────────

async fn docs_handler() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html>
<head><title>rLLM OpenAI API</title></head>
<body>
<h1>rLLM OpenAI-compatible API</h1>
<pre>
GET  /health
GET  /metrics
GET  /debug/model
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions

curl http://localhost:8000/debug/model

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Say hello"}],"max_tokens":16,"temperature":0}'
</pre>
</body>
</html>"#,
    )
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string() })
}

/// Prometheus metrics endpoint.
async fn metrics_handler(State(state): State<AppState>) -> String {
    let rendered = state.metrics_handle.render();
    if rendered.is_empty() {
        "# no metrics recorded yet\n".to_string()
    } else {
        rendered
    }
}

#[derive(Serialize)]
struct DebugModelResponse {
    model: String,
    loaded: bool,
    model_dir: Option<String>,
    architecture: Option<String>,
    device: Option<String>,
}

async fn debug_model_handler(State(state): State<AppState>) -> Json<DebugModelResponse> {
    let runtime = state.runtime.as_ref();
    Json(DebugModelResponse {
        model: state.model_name.clone(),
        loaded: runtime.is_some(),
        model_dir: Some(state.model_name),
        architecture: runtime.map(|rt| rt.architecture.clone()),
        device: runtime.map(|rt| rt.device.clone()),
    })
}

#[tracing::instrument(skip(state), name = "http_list_models")]
async fn list_models_handler(State(state): State<AppState>) -> Json<ModelListResponse> {
    let now = now_timestamp();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "rllm".to_string(),
        }],
    })
}

#[tracing::instrument(skip(state), name = "http_chat_completions")]
async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let started = std::time::Instant::now();
    rllm_metrics::counter!("rllm_http_requests_total").increment(1);

    if req.messages.is_empty() {
        return typed_error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            started,
        );
    }

    if req.messages.len() > state.max_input_messages {
        return typed_error_response(
            axum::http::StatusCode::BAD_REQUEST,
            &format!("too many messages: max is {}", state.max_input_messages),
            "invalid_request_error",
            started,
        );
    }

    let is_stream = req.stream.unwrap_or(false);
    let model = state.model_name.clone();
    let Some(runtime) = state.runtime.clone() else {
        if is_stream {
            return placeholder_chat_stream_response(model, started);
        }
        return placeholder_chat_response(&state.model_name, started);
    };

    let result = run_chat_completion(runtime, req, Duration::from_secs(state.request_timeout_secs)).await;

    if is_stream {
        let sse_stream = async_stream::stream! {
            let id = generate_completion_id("chatcmpl");
            let created = now_timestamp();
            let role_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta { role: Some("assistant".into()), content: None },
                    finish_reason: None,
                }],
            };
            yield Ok::<_, std::convert::Infallible>(
                Event::default().data(serialize_sse(&role_chunk))
            );

            if let Ok(completion) = result {
                let content_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta { role: None, content: Some(completion.text) },
                        finish_reason: None,
                    }],
                };
                yield Ok(Event::default().data(serialize_sse(&content_chunk)));
            }

            let done_chunk = ChatCompletionChunk {
                id,
                object: "chat.completion.chunk".to_string(),
                created,
                model,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta { role: None, content: None },
                    finish_reason: Some("stop".into()),
                }],
            };
            yield Ok(Event::default().data(serialize_sse(&done_chunk)));
            yield Ok(Event::default().data("[DONE]"));
        };

        rllm_metrics::histogram!("rllm_http_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());
        return axum::response::Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    match result {
        Ok(completion) => {
            let response = ChatCompletionResponse {
                id: generate_completion_id("chatcmpl"),
                object: "chat.completion".to_string(),
                created: now_timestamp(),
                model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".into(),
                        content: completion.text,
                    },
                    finish_reason: Some(completion.finish_reason),
                }],
                usage: completion.usage,
            };
            rllm_metrics::histogram!("rllm_http_request_duration_seconds")
                .record(started.elapsed().as_secs_f64());
            Json(response).into_response()
        }
        Err(err) => {
            tracing::error!("chat completion error: {:?}", err);
            error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, "An internal error occurred while processing your request.", started)
        }
    }
}

#[tracing::instrument(skip(state), name = "http_completions")]
async fn completions_handler(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let started = std::time::Instant::now();
    rllm_metrics::counter!("rllm_http_requests_total").increment(1);

    if let PromptInput::Multiple(items) = &req.prompt {
        if items.len() > 16 {
            return typed_error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "too many prompt variants: max is 16",
                "invalid_request_error",
                started,
            );
        }
    }

    let Some(runtime) = state.runtime.clone() else {
        return placeholder_completion_response(&state.model_name, started);
    };

    match run_text_completion(runtime, req, Duration::from_secs(state.request_timeout_secs)).await {
        Ok(completion) => {
            let response = CompletionResponse {
                id: generate_completion_id("cmpl"),
                object: "text_completion".to_string(),
                created: now_timestamp(),
                model: state.model_name,
                choices: vec![CompletionChoice {
                    index: 0,
                    text: completion.text,
                    finish_reason: Some(completion.finish_reason),
                }],
                usage: completion.usage,
            };
            rllm_metrics::histogram!("rllm_http_request_duration_seconds")
                .record(started.elapsed().as_secs_f64());
            Json(response).into_response()
        }
        Err(err) => {
            tracing::error!("completion error: {:?}", err);
            error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, "An internal error occurred while processing your request.", started)
        }
    }
}

struct EngineCompletion {
    text: String,
    usage: UsageInfo,
    finish_reason: String,
}

async fn run_chat_completion(
    runtime: ModelRuntime,
    req: ChatCompletionRequest,
    timeout: Duration,
) -> Result<EngineCompletion> {
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
        .render_chat(messages.clone(), true)
        .await
        .context("rendering chat template")?;
    let token_ids =
        runtime.tokenizer.encode(prompt.clone(), false).await.context("tokenizing chat prompt")?;
    let sampling_params = chat_request_to_sampling_params(&req);
    sampling_params.validate().context("invalid sampling params")?;

    submit_and_collect(runtime, Some(prompt), token_ids, sampling_params, timeout).await
}

async fn run_text_completion(
    runtime: ModelRuntime,
    req: CompletionRequest,
    timeout: Duration,
) -> Result<EngineCompletion> {
    let prompt = match &req.prompt {
        PromptInput::Single(text) => text.clone(),
        PromptInput::Multiple(items) => items.join("\n"),
    };
    let token_ids = runtime
        .tokenizer
        .encode(prompt.clone(), true)
        .await
        .context("tokenizing completion prompt")?;
    let sampling_params = completion_request_to_sampling_params(&req);
    sampling_params.validate().context("invalid sampling params")?;

    submit_and_collect(runtime, Some(prompt), token_ids, sampling_params, timeout).await
}

async fn submit_and_collect(
    runtime: ModelRuntime,
    prompt: Option<String>,
    token_ids: Vec<u32>,
    sampling_params: rllm_core::request::SamplingParams,
    timeout: Duration,
) -> Result<EngineCompletion> {
    let request_id = RequestId::new();
    let max_tokens = sampling_params.max_tokens.unwrap_or(16);
    tracing::info!(
        request_id = ?request_id,
        prompt_tokens = token_ids.len(),
        max_tokens,
        "submitting request to inference engine"
    );

    let mut output_rx = runtime.engine.output_receiver();
    runtime.engine.add_request(InferenceRequest {
        request_id,
        prompt,
        token_ids: Some(token_ids.clone()),
        messages: None,
        sampling_params,
        arrival_time: std::time::Instant::now(),
        priority: 0,
        stream: false,
        cache_salt: None,
    })?;

    let mut generated_ids = Vec::new();
    let mut finish_reason = "length".to_string();

    let collect = async {
        loop {
            output_rx.changed().await.context("engine output channel closed")?;
            for output in output_rx.borrow_and_update().iter() {
                if output.request_id != request_id {
                    continue;
                }
                for completion in &output.outputs {
                    generated_ids.extend_from_slice(&completion.token_ids);
                    if let Some(reason) = completion.finish_reason {
                        finish_reason = finish_reason_to_openai(reason);
                    }
                }
                if output.finished {
                    return Ok::<(), anyhow::Error>(());
                }
            }
        }
    };
    tokio::time::timeout(timeout, collect)
        .await
        .context("request timed out waiting for model output")??;

    let text = runtime
        .tokenizer
        .decode(generated_ids.clone(), true)
        .await
        .context("decoding generated tokens")?;
    tracing::info!(
        request_id = ?request_id,
        completion_tokens = generated_ids.len(),
        finish_reason = %finish_reason,
        "request completed"
    );

    Ok(EngineCompletion {
        text,
        usage: UsageInfo {
            prompt_tokens: u32::try_from(token_ids.len()).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(generated_ids.len()).unwrap_or(u32::MAX),
            total_tokens: u32::try_from(token_ids.len()).unwrap_or(u32::MAX)
                .saturating_add(u32::try_from(generated_ids.len()).unwrap_or(u32::MAX)),
        },
        finish_reason,
    })
}

fn serialize_sse<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|e| {
        tracing::error!("SSE serialization failed: {}", e);
        r#"{"error":"internal serialization error"}"#.to_string()
    })
}

fn finish_reason_to_openai(reason: FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Aborted | FinishReason::Error => "stop",
    }
    .to_string()
}

fn error_response(
    status: axum::http::StatusCode,
    message: &str,
    started: std::time::Instant,
) -> axum::response::Response {
    typed_error_response(status, message, "server_error", started)
}

fn typed_error_response(
    status: axum::http::StatusCode,
    message: &str,
    error_type: &str,
    started: std::time::Instant,
) -> axum::response::Response {
    rllm_metrics::histogram!("rllm_http_request_duration_seconds")
        .record(started.elapsed().as_secs_f64());
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.into(),
                code: None,
            },
        }),
    )
        .into_response()
}

fn placeholder_chat_stream_response(
    model: String,
    started: std::time::Instant,
) -> axum::response::Response {
    let sse_stream = async_stream::stream! {
        let done_chunk = ChatCompletionChunk {
            id: generate_completion_id("chatcmpl"),
            object: "chat.completion.chunk".to_string(),
            created: now_timestamp(),
            model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta { role: Some("assistant".into()), content: None },
                finish_reason: Some("stop".into()),
            }],
        };
        yield Ok::<_, std::convert::Infallible>(
            Event::default().data(serialize_sse(&done_chunk))
        );
        yield Ok(Event::default().data("[DONE]"));
    };

    rllm_metrics::histogram!("rllm_http_request_duration_seconds")
        .record(started.elapsed().as_secs_f64());
    axum::response::Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

fn placeholder_chat_response(model: &str, started: std::time::Instant) -> axum::response::Response {
    let response = ChatCompletionResponse {
        id: generate_completion_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage { role: "assistant".into(), content: String::new() },
            finish_reason: Some("stop".into()),
        }],
        usage: UsageInfo { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
    rllm_metrics::histogram!("rllm_http_request_duration_seconds")
        .record(started.elapsed().as_secs_f64());
    Json(response).into_response()
}

fn placeholder_completion_response(
    model: &str,
    started: std::time::Instant,
) -> axum::response::Response {
    let response = CompletionResponse {
        id: generate_completion_id("cmpl"),
        object: "text_completion".to_string(),
        created: now_timestamp(),
        model: model.to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            text: String::new(),
            finish_reason: Some("stop".into()),
        }],
        usage: UsageInfo { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
    rllm_metrics::histogram!("rllm_http_request_duration_seconds")
        .record(started.elapsed().as_secs_f64());
    Json(response).into_response()
}
