//! Metrics infrastructure for rLLM.
//!
//! Provides a Prometheus-compatible metrics recorder and convenience macros
//! for recording counters, gauges, and histograms throughout the inference
//! pipeline. An optional `otel` feature adds OpenTelemetry trace export.

// Re-export the metrics facade macros so downstream crates only need
// `rllm-metrics` as a dependency.
pub use metrics::{
    counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Install the global Prometheus metrics recorder.
///
/// Must be called once at startup (before any metrics are recorded).
/// Returns a [`PrometheusHandle`] whose `render()` method produces the
/// Prometheus text exposition format for the `/metrics` endpoint.
pub fn install_recorder() -> PrometheusHandle {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::set_global_recorder(recorder).expect("failed to install Prometheus metrics recorder");
    handle
}

/// Convenience: render all registered metrics in Prometheus text format.
pub fn render_prometheus(handle: &PrometheusHandle) -> String {
    handle.render()
}

/// Register all rLLM metric descriptions with the metrics recorder.
///
/// Call this after `install_recorder()` so that Prometheus metric help
/// text is available from the start.
pub fn describe_metrics() {
    // Counters
    describe_counter!("rllm_prompt_tokens_total", "Total prompt tokens processed");
    describe_counter!("rllm_generated_tokens_total", "Total generated tokens");
    describe_counter!("rllm_requests_total", "Total requests received");
    describe_counter!("rllm_http_requests_total", "Total HTTP requests received");
    describe_counter!("rllm_requests_finished_total", "Total finished requests");
    describe_counter!("rllm_preemptions_total", "Total request preemptions");
    describe_counter!(
        "rllm_prefix_cache_hit_tokens_total",
        "Total tokens served from prefix cache"
    );
    describe_counter!("rllm_prefix_cache_lookups_total", "Total prefix cache lookups");
    describe_counter!(
        "rllm_prefix_cache_hits_total",
        "Total prefix cache lookups with at least one cached token"
    );

    // Gauges
    describe_gauge!("rllm_requests_running", "Currently running requests");
    describe_gauge!("rllm_requests_waiting", "Currently waiting requests");
    describe_gauge!("rllm_kv_cache_usage_ratio", "Fraction of KV cache blocks in use");
    describe_gauge!("rllm_kv_cache_active_blocks", "Active KV cache blocks");
    describe_gauge!("rllm_kv_cache_total_blocks", "Total KV cache blocks");
    describe_gauge!("rllm_prefix_cache_entries", "Number of cached prefix hashes");
    describe_gauge!(
        "rllm_prefix_cache_hit_rate",
        "Fraction of prefix cache lookups with at least one cached token"
    );
    describe_gauge!("rllm_scheduler_budget_used", "Scheduler token budget used this step");
    describe_gauge!("rllm_scheduler_budget_max", "Scheduler token budget maximum");
    describe_gauge!(
        "rllm_scheduler_budget_utilization",
        "Fraction of scheduler token budget consumed this step"
    );
    describe_gauge!("rllm_gpu_memory_allocated_bytes", "GPU memory allocated by rLLM");
    describe_gauge!(
        "rllm_tokens_per_second",
        "Most recently observed completed-request token throughput"
    );

    // Histograms
    describe_histogram!("rllm_ttft_seconds", "Time to first token (seconds)");
    describe_histogram!("rllm_tpot_seconds", "Time per output token (seconds)");
    describe_histogram!("rllm_e2e_latency_seconds", "End-to-end request latency (seconds)");
    describe_histogram!(
        "rllm_http_request_duration_seconds",
        "HTTP request handler latency (seconds)"
    );
    describe_histogram!(
        "rllm_scheduler_step_duration_seconds",
        "Scheduler step duration (seconds)"
    );
    describe_histogram!(
        "rllm_model_forward_duration_seconds",
        "Model forward pass duration (seconds)"
    );
    describe_histogram!("rllm_sampling_duration_seconds", "Token sampling duration (seconds)");

    // Initialize all counters and gauges to default/0 so they are exported immediately by Prometheus
    counter!("rllm_prompt_tokens_total").increment(0);
    counter!("rllm_generated_tokens_total").increment(0);
    counter!("rllm_requests_total").increment(0);
    counter!("rllm_http_requests_total").increment(0);
    counter!("rllm_requests_finished_total").increment(0);
    counter!("rllm_preemptions_total").increment(0);
    counter!("rllm_prefix_cache_hit_tokens_total").increment(0);
    counter!("rllm_prefix_cache_lookups_total").increment(0);
    counter!("rllm_prefix_cache_hits_total").increment(0);

    gauge!("rllm_requests_running").set(0.0);
    gauge!("rllm_requests_waiting").set(0.0);
    gauge!("rllm_kv_cache_usage_ratio").set(0.0);
    gauge!("rllm_kv_cache_active_blocks").set(0.0);
    gauge!("rllm_kv_cache_total_blocks").set(0.0);
    gauge!("rllm_prefix_cache_entries").set(0.0);
    gauge!("rllm_prefix_cache_hit_rate").set(0.0);
    gauge!("rllm_scheduler_budget_used").set(0.0);
    gauge!("rllm_scheduler_budget_max").set(0.0);
    gauge!("rllm_scheduler_budget_utilization").set(0.0);
    gauge!("rllm_gpu_memory_allocated_bytes").set(0.0);
    gauge!("rllm_tokens_per_second").set(0.0);
}

/// Record a scheduler queue and token-budget snapshot.
pub fn record_scheduler_snapshot(
    num_running: usize,
    num_waiting: usize,
    budget_used: usize,
    budget_max: usize,
) {
    gauge!("rllm_requests_running").set(num_running as f64);
    gauge!("rllm_requests_waiting").set(num_waiting as f64);
    gauge!("rllm_scheduler_budget_used").set(budget_used as f64);
    gauge!("rllm_scheduler_budget_max").set(budget_max as f64);
    let utilization = if budget_max == 0 { 0.0 } else { budget_used as f64 / budget_max as f64 };
    gauge!("rllm_scheduler_budget_utilization").set(utilization);
}

/// Record a KV cache usage snapshot.
pub fn record_kv_cache_usage(
    num_total_blocks: usize,
    num_active_blocks: usize,
    num_cached_hashes: usize,
) {
    gauge!("rllm_kv_cache_active_blocks").set(num_active_blocks as f64);
    gauge!("rllm_kv_cache_total_blocks").set(num_total_blocks as f64);
    gauge!("rllm_prefix_cache_entries").set(num_cached_hashes as f64);
    let usage_ratio = if num_total_blocks == 0 {
        0.0
    } else {
        num_active_blocks as f64 / num_total_blocks as f64
    };
    gauge!("rllm_kv_cache_usage_ratio").set(usage_ratio);
}

/// Record a prefix-cache lookup and keep hit-rate gauges current.
pub fn record_prefix_cache_lookup(hit_tokens: usize, total_lookups: usize, total_hits: usize) {
    counter!("rllm_prefix_cache_lookups_total").increment(1);
    if hit_tokens > 0 {
        counter!("rllm_prefix_cache_hits_total").increment(1);
        counter!("rllm_prefix_cache_hit_tokens_total").increment(hit_tokens as u64);
    }
    let hit_rate = if total_lookups == 0 { 0.0 } else { total_hits as f64 / total_lookups as f64 };
    gauge!("rllm_prefix_cache_hit_rate").set(hit_rate);
}

/// Record GPU memory allocated by lower-level workers/kernels.
pub fn record_gpu_memory_allocated(bytes: usize) {
    gauge!("rllm_gpu_memory_allocated_bytes").set(bytes as f64);
}

/// Record completed-request throughput.
pub fn record_tokens_per_second(tokens: u32, elapsed_seconds: f64) {
    if elapsed_seconds > 0.0 {
        gauge!("rllm_tokens_per_second").set(tokens as f64 / elapsed_seconds);
    }
}

// ── Debug dump helpers ──────────────────────────────────────────────────────

/// Produce a human-readable summary of request states for debugging.
pub fn debug_request_state_summary(
    num_waiting: usize,
    num_running: usize,
    num_finished: usize,
    num_active: usize,
) -> String {
    format!(
        "Request state: waiting={} running={} finished={} active={}",
        num_waiting, num_running, num_finished, num_active
    )
}

/// Produce a human-readable KV cache block table summary for debugging.
pub fn debug_kv_cache_summary(
    num_total_blocks: usize,
    num_active_blocks: usize,
    num_free_blocks: usize,
    num_cached_hashes: usize,
    num_tracked_requests: usize,
) -> String {
    format!(
        "KV cache: total={} active={} free={} cached_hashes={} tracked_requests={}",
        num_total_blocks,
        num_active_blocks,
        num_free_blocks,
        num_cached_hashes,
        num_tracked_requests
    )
}

/// Produce a scheduler output debug dump (used on model failure).
pub fn debug_scheduler_output_dump(
    scheduled_new: usize,
    scheduled_cached: usize,
    scheduled_running: usize,
    budget_used: usize,
    preempted: usize,
    finished: usize,
) -> String {
    format!(
        "Scheduler output: new={} cached={} running={} budget_used={} preempted={} finished={}",
        scheduled_new, scheduled_cached, scheduled_running, budget_used, preempted, finished
    )
}

/// Produce a CUDA launch-parameter summary for debug logs.
pub fn debug_cuda_launch_params(
    kernel: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: usize,
    stream: usize,
) -> String {
    format!(
        "CUDA launch: kernel={} grid=({}, {}, {}) block=({}, {}, {}) shared_mem_bytes={} stream=0x{:x}",
        kernel, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared_mem_bytes, stream
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_recorder_and_render() {
        let handle = install_recorder();
        describe_metrics();

        // Record some test metrics.
        counter!("rllm_prompt_tokens_total").increment(100);
        gauge!("rllm_requests_running").set(5.0);

        let output = render_prometheus(&handle);
        assert!(output.contains("rllm_prompt_tokens_total"));
        assert!(output.contains("rllm_requests_running"));
    }

    #[test]
    fn debug_summaries_format() {
        let req = debug_request_state_summary(3, 2, 10, 5);
        assert!(req.contains("waiting=3"));
        assert!(req.contains("running=2"));

        let kv = debug_kv_cache_summary(100, 30, 70, 5, 2);
        assert!(kv.contains("total=100"));
        assert!(kv.contains("active=30"));

        let sched = debug_scheduler_output_dump(1, 2, 3, 50, 0, 1);
        assert!(sched.contains("new=1"));
        assert!(sched.contains("running=3"));

        let launch = debug_cuda_launch_params("paged_attention", (2, 1, 1), (128, 1, 1), 0, 0);
        assert!(launch.contains("kernel=paged_attention"));
    }
}
