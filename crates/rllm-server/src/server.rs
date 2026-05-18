use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use axum::routing::{get, post};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::cli::ServeArgs;
use crate::openai::*;

/// Application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    model_name: String,
    // The engine is behind Arc<AsyncLLMEngine> for sharing across handlers.
    // For now, we store the config info needed for responses.
    #[expect(dead_code)]
    engine: Arc<AsyncLLMEngineWrapper>,
    /// Prometheus metrics handle for the `/metrics` endpoint.
    metrics_handle: PrometheusHandle,
}

/// Wrapper to allow cloning the engine handle.
/// In production, this would wrap the AsyncLLMEngine directly.
/// For now, it holds the channel endpoints.
pub struct AsyncLLMEngineWrapper {
    // Placeholder: in production this would be the AsyncLLMEngine.
    // For now, we simulate the engine interaction.
}

impl AppState {
    pub fn new(model_name: String, metrics_handle: PrometheusHandle) -> Self {
        Self { model_name, engine: Arc::new(AsyncLLMEngineWrapper {}), metrics_handle }
    }
}

/// Build the router with all routes and middleware.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Build and run the HTTP server.
pub async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    // Install metrics recorder and describe all metrics.
    let metrics_handle = rllm_metrics::install_recorder();
    rllm_metrics::describe_metrics();

    let model_name = args.model.clone();
    let state = AppState::new(model_name, metrics_handle);

    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ── Handlers ───────────────────────────────────────────────────────────────

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string() })
}

/// Prometheus metrics endpoint.
async fn metrics_handler(State(state): State<AppState>) -> String {
    state.metrics_handle.render()
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
    let request_span = tracing::info_span!("http_request", path = "/v1/chat/completions");
    let _guard = request_span.enter();
    let model = state.model_name.clone();

    // Validate request.
    if req.messages.is_empty() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "messages must not be empty".into(),
                error_type: "invalid_request_error".into(),
                code: None,
            },
        };
        rllm_metrics::histogram!("rllm_http_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());
        return (axum::http::StatusCode::BAD_REQUEST, Json(err)).into_response();
    }

    let is_stream = req.stream.unwrap_or(false);

    if is_stream {
        // SSE streaming response.
        let sse_stream = async_stream::stream! {
            let id = generate_completion_id("chatcmpl");
            let created = now_timestamp();

            // Initial chunk with role.
            let role_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            yield Ok::<_, std::convert::Infallible>(
                Event::default().data(serde_json::to_string(&role_chunk).unwrap())
            );

            // Placeholder: in production, stream tokens from engine.
            // For now, send a final chunk.
            let done_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&done_chunk).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };

        rllm_metrics::histogram!("rllm_http_request_duration_seconds")
            .record(started.elapsed().as_secs_f64());
        return axum::response::Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    // Non-streaming: build a placeholder response.
    // In production, this would submit to the engine and collect the full output.
    let response = ChatCompletionResponse {
        id: generate_completion_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: now_timestamp(),
        model: model.clone(),
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

#[tracing::instrument(skip(state), name = "http_completions")]
async fn completions_handler(
    State(state): State<AppState>,
    Json(_req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let started = std::time::Instant::now();
    rllm_metrics::counter!("rllm_http_requests_total").increment(1);
    let request_span = tracing::info_span!("http_request", path = "/v1/completions");
    let _guard = request_span.enter();
    let model = state.model_name.clone();

    let response = CompletionResponse {
        id: generate_completion_id("cmpl"),
        object: "text_completion".to_string(),
        created: now_timestamp(),
        model: model.clone(),
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
