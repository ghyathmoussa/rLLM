use std::sync::OnceLock;

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use rllm_server::{openai::*, server::AppState};
use tower::ServiceExt;

static APP: OnceLock<axum::Router> = OnceLock::new();

fn make_app() -> &'static axum::Router {
    APP.get_or_init(|| {
        let recorder = rllm_metrics::install_recorder();
        rllm_metrics::describe_metrics();
        let state = AppState::new("test-model".into(), recorder);
        rllm_server::server::build_router(state)
    })
}

fn json_body(value: serde_json::Value) -> Body {
    Body::from(serde_json::to_string(&value).unwrap())
}

async fn get(uri: &str) -> ResponseX {
    let resp = make_app()
        .clone()
        .oneshot(Request::builder().method(Method::GET).uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    ResponseX(resp)
}

async fn post_json(uri: &str, body: serde_json::Value) -> ResponseX {
    let resp = make_app()
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header("content-type", "application/json")
                .body(json_body(body))
                .unwrap(),
        )
        .await
        .unwrap();
    ResponseX(resp)
}

struct ResponseX(axum::response::Response);
impl ResponseX {
    fn status(&self) -> StatusCode {
        self.0.status()
    }
    fn headers(&self) -> &axum::http::HeaderMap {
        self.0.headers()
    }
    async fn text(self) -> String {
        let bytes = axum::body::to_bytes(self.0.into_body(), usize::MAX).await.unwrap();
        String::from_utf8_lossy(&bytes).to_string()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_server_health_endpoint() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let resp = get("/health").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let health: HealthResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(health.status, "ok");
    });
}

#[test]
fn test_server_metrics_endpoint() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let resp = get("/metrics").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        assert!(!body.is_empty());
    });
}

#[test]
fn test_server_list_models() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let resp = get("/v1/models").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let models: ModelListResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(models.object, "list");
        assert!(!models.data.is_empty());
        assert_eq!(models.data[0].id, "test-model");
    });
}

#[test]
fn test_chat_completions_basic() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        });
        let resp = post_json("/v1/chat/completions", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let chat_resp: ChatCompletionResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(chat_resp.object, "chat.completion");
        assert!(!chat_resp.choices.is_empty());
        assert_eq!(chat_resp.choices[0].message.role, "assistant");
    });
}

#[test]
fn test_chat_completions_empty_messages_error() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": []
        });
        let resp = post_json("/v1/chat/completions", body).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await;
        let err: ErrorResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(err.error.error_type, "invalid_request_error");
        assert!(err.error.message.contains("empty"));
    });
}

#[test]
fn test_chat_completions_invalid_json() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({ "model": "test-model" });
        let resp = post_json("/v1/chat/completions", body).await;
        assert!(
            resp.status() == StatusCode::OK || resp.status() == StatusCode::UNPROCESSABLE_ENTITY
        );
    });
}

#[test]
fn test_completions_basic() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "Hello world",
            "max_tokens": 10
        });
        let resp = post_json("/v1/completions", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let comp_resp: CompletionResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(comp_resp.object, "text_completion");
        assert!(!comp_resp.choices.is_empty());
    });
}

#[test]
fn test_chat_completions_with_all_params() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "n": 1,
            "stream": false,
            "stop": ["\n", "stop"]
        });
        let resp = post_json("/v1/chat/completions", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let chat_resp: ChatCompletionResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(chat_resp.model, "test-model");
        assert_eq!(chat_resp.choices.len(), 1);
    });
}

#[test]
fn test_chat_completions_streaming_response() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Stream test"}
            ],
            "stream": true,
            "max_tokens": 5
        });
        let resp = post_json("/v1/chat/completions", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type =
            resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
        assert!(
            content_type.contains("text/event-stream"),
            "Expected SSE content type, got: {}",
            content_type
        );
        let body = resp.text().await;
        assert!(body.contains("data:"), "SSE response should contain data events");
        assert!(body.contains("[DONE]"), "SSE should end with [DONE]");
    });
}

#[test]
fn test_chat_completions_streaming_chunks_are_valid_json() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "JSON test"}
            ],
            "stream": true,
            "max_tokens": 3
        });
        let resp = post_json("/v1/chat/completions", body).await;
        let body = resp.text().await;
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                let chunk: ChatCompletionChunk = serde_json::from_str(data).unwrap_or_else(|e| {
                    panic!("Invalid SSE data JSON: {} - data: {}", e, data);
                });
                assert_eq!(chunk.object, "chat.completion.chunk");
            }
        }
    });
}

#[test]
fn test_model_list_endpoint() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let resp = get("/v1/models").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let models: ModelListResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(models.object, "list");
        assert_eq!(models.data.len(), 1);
        assert_eq!(models.data[0].id, "test-model");
        assert_eq!(models.data[0].object, "model");
        assert_eq!(models.data[0].owned_by, "rllm");
    });
}

#[test]
fn test_completions_with_stop_sequences() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "Complete this sentence:",
            "max_tokens": 50,
            "stop": ["\n", "."],
            "temperature": 0.5
        });
        let resp = post_json("/v1/completions", body).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await;
        let comp_resp: CompletionResponse = serde_json::from_str(&body).unwrap();
        assert_eq!(comp_resp.object, "text_completion");
        assert!(comp_resp.usage.total_tokens == 0);
    });
}

#[test]
fn test_multiple_rounds_no_crash() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let app = make_app().clone();
        for i in 0..5 {
            let body = serde_json::json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": format!("Request {}", i)}
                ],
                "max_tokens": 5
            });
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(json_body(body))
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "Request {} failed", i);
        }
    });
}
