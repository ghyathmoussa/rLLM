use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use rllm_server::server::AppState;
use tower::ServiceExt;

fn metrics_handle() -> metrics_exporter_prometheus::PrometheusHandle {
    static HANDLE: std::sync::OnceLock<metrics_exporter_prometheus::PrometheusHandle> =
        std::sync::OnceLock::new();
    HANDLE.get_or_init(rllm_metrics::install_recorder).clone()
}

fn tool_request(tool_choice: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the weather in Istanbul?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }
            }
        }],
        "tool_choice": tool_choice,
        "parallel_tool_calls": false
    })
}

async fn send(body: serde_json::Value) -> StatusCode {
    let state = AppState::new("test-model".into(), metrics_handle());
    let response = rllm_server::server::build_router(state)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    response.status()
}

#[tokio::test]
async fn accepts_named_tool_choice() {
    let body = tool_request(serde_json::json!({
        "type": "function",
        "function": {"name": "get_weather"}
    }));
    assert_eq!(send(body).await, StatusCode::OK);
}

#[tokio::test]
async fn rejects_named_tool_choice_not_in_tools() {
    let body = tool_request(serde_json::json!({
        "type": "function",
        "function": {"name": "missing"}
    }));
    assert_eq!(send(body).await, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn rejects_empty_tools_array() {
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": []
    });
    assert_eq!(send(body).await, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn accepts_tool_call_conversation_history() {
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Weather in Istanbul?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Istanbul\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": "{\"temperature\":24,\"condition\":\"sunny\"}"
            },
            {"role": "user", "content": "Summarize that result."}
        ]
    });
    assert_eq!(send(body).await, StatusCode::OK);
}
