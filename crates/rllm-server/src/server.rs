use std::{path::Path, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{HeaderValue, Method, header},
    middleware::from_fn_with_state,
    response::{Html, IntoResponse, sse::Event},
    routing::{get, post},
};
use metrics_exporter_prometheus::PrometheusHandle;
use rllm_cache::spec::{KVCacheConfig, KVCacheSpec};
use rllm_core::{
    config::{CacheConfig, ModelConfig, PrefixHashAlgorithm, SchedulerConfig, SchedulingPolicy},
    dtype::DType,
    ids::RequestId,
    output::FinishReason,
    request::InferenceRequest,
};
use rllm_engine::{AsyncLLMEngine, EngineCore};
use rllm_executor::{Executor, UniProcExecutor};
use rllm_model::{hf_config, loader};
use rllm_scheduler::Scheduler;
use rllm_tokenizer::{pool::AsyncTokenizerPool, tokenizer::Tokenizer};
use rllm_worker::Worker;
use serde::Serialize;
use tokio::net::TcpListener;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{auth, cli::ServeArgs, openai::*};

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

    #[allow(clippy::too_many_arguments)]
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
        .route("/doc", get(docs_handler))
        .route("/openapi.yaml", get(openapi_handler))
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
    let origins: Vec<HeaderValue> =
        origins.split(',').map(|s| s.trim().parse().expect("invalid CORS origin")).collect();
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
    model_config.quantization = parse_quantization(&args.quantization, args.quant_bits, args.quant_group_size);

    let tokenizer_ref = args.tokenizer.as_deref().unwrap_or(model_ref);
    let tokenizer = load_tokenizer(tokenizer_ref, &model_dir)?;
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

fn parse_quantization(quant_str: &str, bits: Option<usize>, group_size: Option<usize>) -> Option<rllm_core::config::QuantizationConfig> {
    use rllm_core::config::{QuantizationConfig, QuantizationKind};
    let kind = match quant_str.to_lowercase().as_str() {
        "none" => return None,
        "fp8" => QuantizationKind::FP8,
        "mxfp8" => QuantizationKind::MXFP8,
        "mxfp4" => QuantizationKind::MXFP4,
        "nvfp4" => QuantizationKind::NVFP4,
        "int8" => QuantizationKind::Int8,
        "int4" => QuantizationKind::Int4,
        "gptq" => QuantizationKind::GPTQ,
        "awq" => QuantizationKind::AWQ,
        "gguf" => QuantizationKind::Gguf,
        "compressed-tensors" | "compressed_tensors" => QuantizationKind::CompressedTensors,
        "modelopt" => QuantizationKind::ModelOpt,
        "torchao" => QuantizationKind::TorchAO,
        _ => return None,
    };
    Some(QuantizationConfig {
        kind,
        group_size,
        bits,
    })
}

fn cache_config(args: &ServeArgs, model_config: &ModelConfig) -> CacheConfig {
    let cache_dtype = match args.kv_cache_dtype.to_lowercase().as_str() {
        "f16" => rllm_core::dtype::DType::F16,
        "bf16" => rllm_core::dtype::DType::BF16,
        "fp8_e4m3" | "fp8-e4m3" | "e4m3" => rllm_core::dtype::DType::FP8E4M3,
        "fp8_e5m2" | "fp8-e5m2" | "e5m2" => rllm_core::dtype::DType::FP8E5M2,
        _ => {
            if let Some(ref q) = model_config.quantization {
                let plan = rllm_core::optimizations::QuantizationPlan::from_config(q).unwrap_or_default();
                plan.kv_cache_dtype
            } else {
                model_config.dtype
            }
        }
    };

    CacheConfig {
        block_size: BLOCK_SIZE,
        hash_block_size: BLOCK_SIZE,
        gpu_memory_utilization: args.gpu_memory_utilization,
        cpu_swap_bytes: 0,
        cache_dtype,
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
    let cache_dtype = match args.kv_cache_dtype.to_lowercase().as_str() {
        "f16" => rllm_core::dtype::DType::F16,
        "bf16" => rllm_core::dtype::DType::BF16,
        "fp8_e4m3" | "fp8-e4m3" | "e4m3" => rllm_core::dtype::DType::FP8E4M3,
        "fp8_e5m2" | "fp8-e5m2" | "e5m2" => rllm_core::dtype::DType::FP8E5M2,
        _ => {
            if let Some(ref q) = model_config.quantization {
                let plan = rllm_core::optimizations::QuantizationPlan::from_config(q).unwrap_or_default();
                plan.kv_cache_dtype
            } else {
                model_config.dtype
            }
        }
    };

    KVCacheConfig {
        num_blocks: num_cache_blocks(args, model_config.max_model_len),
        spec: KVCacheSpec {
            block_size: BLOCK_SIZE,
            num_layers: model_config.num_layers,
            num_kv_heads: model_config.num_kv_heads,
            head_dim: model_config.head_dim,
            dtype: cache_dtype,
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
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>rLLM API Documentation</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5/favicon-32x32.png" sizes="32x32" />
  <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5/favicon-16x16.png" sizes="16x16" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" charset="UTF-8"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/openapi.yaml',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>"#,
    )
}

async fn openapi_handler() -> impl axum::response::IntoResponse {
    ([(axum::http::header::CONTENT_TYPE, "text/yaml")], include_str!("../../../openapi.yaml"))
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string() })
}

/// Prometheus metrics endpoint.
async fn metrics_handler(State(state): State<AppState>) -> String {
    let rendered = state.metrics_handle.render();
    if rendered.is_empty() { "# no metrics recorded yet\n".to_string() } else { rendered }
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

    let messages = req
        .messages
        .iter()
        .map(|msg| rllm_core::request::ChatMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        })
        .collect::<Vec<_>>();
    let prompt = match runtime.tokenizer.render_chat(messages.clone(), true).await {
        Ok(p) => p,
        Err(e) => {
            return typed_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to render chat template: {e}"),
                "internal_error",
                started,
            );
        }
    };
    let token_ids = match runtime.tokenizer.encode(prompt.clone(), false).await {
        Ok(ids) => ids,
        Err(e) => {
            return typed_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to tokenize chat prompt: {e}"),
                "internal_error",
                started,
            );
        }
    };
    let sampling_params = chat_request_to_sampling_params(&req);
    if let Err(e) = sampling_params.validate() {
        return typed_error_response(
            axum::http::StatusCode::BAD_REQUEST,
            &format!("invalid sampling params: {e}"),
            "invalid_request_error",
            started,
        );
    }

    if is_stream {
        let inference_req = InferenceRequest {
            request_id: RequestId::new(),
            prompt: Some(prompt.clone()),
            token_ids: Some(token_ids),
            messages: None,
            sampling_params,
            arrival_time: std::time::Instant::now(),
            priority: 0,
            stream: true,
            cache_salt: None,
        };
        let mut receiver = match runtime.engine.add_request_stream(inference_req) {
            Ok(rx) => rx,
            Err(e) => {
                return typed_error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("failed to submit stream request: {e}"),
                    "internal_error",
                    started,
                );
            }
        };

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
                generation_time: None,
            };
            yield Ok::<_, std::convert::Infallible>(
                Event::default().data(serialize_sse(&role_chunk))
            );

            let elapsed_start = std::time::Instant::now();
            while let Some(output) = receiver.recv().await {
                let mut chunk_text = String::new();
                for completion in &output.outputs {
                    if !completion.token_ids.is_empty() {
                        if let Ok(text) = runtime.tokenizer.decode(completion.token_ids.clone(), true).await {
                            chunk_text.push_str(&text);
                        }
                    }
                }

                let finish_reason = output.outputs.first().and_then(|c| c.finish_reason).map(finish_reason_to_openai);

                if !chunk_text.is_empty() || finish_reason.is_some() {
                    let content_chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta { role: None, content: Some(chunk_text) },
                            finish_reason,
                        }],
                        generation_time: Some(elapsed_start.elapsed().as_secs_f64()),
                    };
                    yield Ok(Event::default().data(serialize_sse(&content_chunk)));
                }

                if output.finished {
                    break;
                }
            }

            yield Ok(Event::default().data("[DONE]"));
        };

        rllm_metrics::histogram!("rllm_http_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());
        return axum::response::Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    let result = submit_and_collect(
        runtime,
        Some(prompt),
        token_ids,
        sampling_params,
        Duration::from_secs(state.request_timeout_secs),
    )
    .await;

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
                generation_time: Some(completion.generation_time),
            };
            rllm_metrics::histogram!("rllm_http_request_duration_seconds")
                .record(started.elapsed().as_secs_f64());
            Json(response).into_response()
        }
        Err(err) => {
            tracing::error!("chat completion error: {:?}", err);
            error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "An internal error occurred while processing your request.",
                started,
            )
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
                generation_time: Some(completion.generation_time),
            };
            rllm_metrics::histogram!("rllm_http_request_duration_seconds")
                .record(started.elapsed().as_secs_f64());
            Json(response).into_response()
        }
        Err(err) => {
            tracing::error!("completion error: {:?}", err);
            error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "An internal error occurred while processing your request.",
                started,
            )
        }
    }
}

struct EngineCompletion {
    text: String,
    usage: UsageInfo,
    finish_reason: String,
    generation_time: f64,
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
    let start_time = std::time::Instant::now();
    let request_id = RequestId::new();
    let max_tokens = sampling_params.max_tokens.unwrap_or(16);
    tracing::info!(
        request_id = ?request_id,
        prompt_tokens = token_ids.len(),
        max_tokens,
        "submitting request to inference engine"
    );

    let mut receiver = runtime.engine.add_request_stream(InferenceRequest {
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
        while let Some(output) = receiver.recv().await {
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
        anyhow::bail!("Engine closed output stream before finishing request")
    };
    tokio::time::timeout(timeout, collect)
        .await
        .context("request timed out waiting for model output")??;

    let text = runtime
        .tokenizer
        .decode(generated_ids.clone(), true)
        .await
        .context("decoding generated tokens")?;
    let duration = start_time.elapsed();
    tracing::info!(
        request_id = ?request_id,
        completion_tokens = generated_ids.len(),
        finish_reason = %finish_reason,
        duration = ?duration,
        "request completed"
    );

    Ok(EngineCompletion {
        text,
        usage: UsageInfo {
            prompt_tokens: u32::try_from(token_ids.len()).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(generated_ids.len()).unwrap_or(u32::MAX),
            total_tokens: u32::try_from(token_ids.len())
                .unwrap_or(u32::MAX)
                .saturating_add(u32::try_from(generated_ids.len()).unwrap_or(u32::MAX)),
        },
        finish_reason,
        generation_time: duration.as_secs_f64(),
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
            generation_time: Some(started.elapsed().as_secs_f64()),
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
        generation_time: Some(started.elapsed().as_secs_f64()),
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
        generation_time: Some(started.elapsed().as_secs_f64()),
    };
    rllm_metrics::histogram!("rllm_http_request_duration_seconds")
        .record(started.elapsed().as_secs_f64());
    Json(response).into_response()
}
