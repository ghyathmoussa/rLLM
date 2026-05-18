use std::time::{Duration, Instant};

use anyhow::Result;
use rllm_core::output::RequestOutput;

use crate::metrics::{BenchmarkMetrics, LatencyStats};
use crate::workload::synthetic_prompt;

/// Which endpoint to benchmark.
#[derive(Debug, Clone)]
pub enum ServeEndpoint {
    ChatCompletions,
    Completions,
}

/// Configuration for the serve (HTTP) benchmark client.
#[derive(Debug, Clone)]
pub struct ServeBenchConfig {
    pub base_url: String,
    pub model: String,
    pub endpoint: ServeEndpoint,
    pub num_requests: usize,
    pub concurrency: usize,
    pub streaming: bool,
    pub input_tokens: usize,
    pub max_tokens: u32,
    pub timeout: Duration,
    pub prompts: Option<Vec<String>>,
}

/// Result of a single HTTP request benchmark.
struct ServeRequestResult {
    ttft: Option<Duration>,
    e2e_latency: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    error: Option<String>,
}

/// Placeholder response parsing for non-streaming completions.
#[derive(serde::Deserialize)]
struct CompletionResponse {
    usage: UsageInfo,
}

#[derive(serde::Deserialize)]
struct UsageInfo {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Run the serve benchmark against a running rLLM server.
///
/// This function requires `reqwest` (dev-dependency) and a running server.
/// It sends concurrent HTTP requests and measures TTFT, TPOT, and throughput.
pub async fn run_serve_benchmark(config: ServeBenchConfig) -> Result<BenchmarkMetrics> {
    if config.prompts.as_ref().is_some_and(Vec::is_empty) {
        anyhow::bail!("serve benchmark prompts cannot be empty");
    }

    let client = reqwest::Client::builder().timeout(config.timeout).build()?;

    let endpoint_path = match config.endpoint {
        ServeEndpoint::ChatCompletions => "/v1/chat/completions",
        ServeEndpoint::Completions => "/v1/completions",
    };
    let url = format!("{}{}", config.base_url.trim_end_matches('/'), endpoint_path);

    let start = Instant::now();
    let mut results: Vec<ServeRequestResult> = Vec::with_capacity(config.num_requests);

    // Send requests in batches of `concurrency`.
    let mut sent = 0;
    while sent < config.num_requests {
        let batch_size = config.concurrency.min(config.num_requests - sent);
        let mut handles = Vec::with_capacity(batch_size);

        for request_index in sent..sent + batch_size {
            let client = client.clone();
            let url = url.clone();
            let model = config.model.clone();
            let endpoint = config.endpoint.clone();
            let max_tokens = config.max_tokens;
            let streaming = config.streaming;
            let input_tokens = config.input_tokens;
            let prompt = config
                .prompts
                .as_ref()
                .and_then(|prompts| prompts.get(request_index % prompts.len()))
                .cloned()
                .unwrap_or_else(|| synthetic_prompt(input_tokens, request_index));

            let handle = tokio::spawn(async move {
                let req_start = Instant::now();

                let body = match endpoint {
                    ServeEndpoint::ChatCompletions => serde_json::json!({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": streaming,
                    }),
                    ServeEndpoint::Completions => serde_json::json!({
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "stream": streaming,
                    }),
                };

                let resp: Result<reqwest::Response, reqwest::Error> =
                    client.post(&url).json(&body).send().await;

                match resp {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        if status != 200 {
                            let text: String = resp.text().await.unwrap_or_default();
                            return ServeRequestResult {
                                ttft: None,
                                e2e_latency: req_start.elapsed(),
                                prompt_tokens: 0,
                                completion_tokens: 0,
                                error: Some(format!("HTTP {}: {}", status, text)),
                            };
                        }

                        if streaming {
                            // For streaming, measure TTFT as time to first chunk.
                            // We just consume the full body for simplicity.
                            let ttft = None; // Would need byte-level SSE parsing for true TTFT
                            let _ = resp.bytes().await;
                            ServeRequestResult {
                                ttft,
                                e2e_latency: req_start.elapsed(),
                                prompt_tokens: 0,
                                completion_tokens: max_tokens,
                                error: None,
                            }
                        } else {
                            let ttft = Some(req_start.elapsed());
                            let body: String = resp.text().await.unwrap_or_default();
                            let parsed: Result<CompletionResponse, _> = serde_json::from_str(&body);
                            match parsed {
                                Ok(resp) => ServeRequestResult {
                                    ttft,
                                    e2e_latency: req_start.elapsed(),
                                    prompt_tokens: resp.usage.prompt_tokens,
                                    completion_tokens: resp.usage.completion_tokens,
                                    error: None,
                                },
                                Err(_) => ServeRequestResult {
                                    ttft,
                                    e2e_latency: req_start.elapsed(),
                                    prompt_tokens: 0,
                                    completion_tokens: max_tokens,
                                    error: None,
                                },
                            }
                        }
                    }
                    Err(e) => ServeRequestResult {
                        ttft: None,
                        e2e_latency: req_start.elapsed(),
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        error: Some(e.to_string()),
                    },
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(ServeRequestResult {
                    ttft: None,
                    e2e_latency: Duration::ZERO,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    error: Some(e.to_string()),
                }),
            }
        }

        sent += batch_size;
    }

    let elapsed = start.elapsed();

    let errors: Vec<&str> = results.iter().filter_map(|r| r.error.as_deref()).collect();
    if !errors.is_empty() {
        tracing::warn!("{} requests had errors", errors.len());
    }

    let successful: Vec<&ServeRequestResult> =
        results.iter().filter(|r| r.error.is_none()).collect();

    let total_input_tokens: u64 = successful.iter().map(|r| r.prompt_tokens as u64).sum();
    let total_output_tokens: u64 = successful.iter().map(|r| r.completion_tokens as u64).sum();
    let total_requests = successful.len();
    let elapsed_seconds = elapsed.as_secs_f64();

    let throughput_rps =
        if elapsed_seconds > 0.0 { total_requests as f64 / elapsed_seconds } else { 0.0 };
    let output_tps =
        if elapsed_seconds > 0.0 { total_output_tokens as f64 / elapsed_seconds } else { 0.0 };
    let total_tps = if elapsed_seconds > 0.0 {
        (total_input_tokens + total_output_tokens) as f64 / elapsed_seconds
    } else {
        0.0
    };

    let e2e_samples: Vec<f64> = successful.iter().map(|r| r.e2e_latency.as_secs_f64()).collect();
    let ttft_samples: Vec<Duration> = successful.iter().filter_map(|r| r.ttft).collect();
    let tpot_samples: Vec<Duration> = successful
        .iter()
        .filter(|r| r.completion_tokens > 0)
        .map(|r| r.e2e_latency / r.completion_tokens.max(1))
        .collect();

    // Build dummy RequestOutputs for compatibility with BenchmarkMetrics.
    let dummy_outputs: Vec<RequestOutput> = successful
        .iter()
        .map(|_| RequestOutput {
            request_id: rllm_core::ids::RequestId::new(),
            outputs: vec![],
            finished: true,
            usage: rllm_core::output::Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
        .collect();

    let mut metrics =
        BenchmarkMetrics::from_results(&dummy_outputs, elapsed, &ttft_samples, &tpot_samples);
    metrics.total_requests = total_requests;
    metrics.total_input_tokens = total_input_tokens;
    metrics.total_output_tokens = total_output_tokens;
    metrics.throughput_rps = throughput_rps;
    metrics.output_tps = output_tps;
    metrics.total_tps = total_tps;
    metrics.latencies = LatencyStats::from_samples(e2e_samples);

    Ok(metrics)
}
