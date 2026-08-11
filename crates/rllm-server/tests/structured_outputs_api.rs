use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use rllm_server::server::AppState;
use tower::ServiceExt;

#[tokio::test]
async fn response_format_json_schema_maps_to_internal_constraint() {
    let request: rllm_server::openai::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "answer"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            }
        }))
        .unwrap();
    let params = rllm_server::openai::chat_request_to_sampling_params(&request);
    let structured = params.structured_outputs.unwrap();
    assert!(structured.json_schema.is_some());
    assert!(structured.validate().is_ok());
}

#[tokio::test]
async fn rejects_response_format_with_structured_outputs() {
    let recorder = rllm_metrics::install_recorder();
    let state = AppState::new("test-model".into(), recorder);
    let app = rllm_server::server::build_router(state);
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "answer"}],
        "response_format": {"type": "json_object"},
        "structured_outputs": {"json_object": true}
    });
    let response = app
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
