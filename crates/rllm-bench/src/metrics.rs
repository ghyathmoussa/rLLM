use std::time::Duration;

use rllm_core::output::RequestOutput;
use serde::Serialize;

/// Statistics computed from a collection of latency samples.
#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

impl LatencyStats {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                median: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let median = percentile(&samples, 50.0);
        let p90 = percentile(&samples, 90.0);
        let p95 = percentile(&samples, 95.0);
        let p99 = percentile(&samples, 99.0);
        let min = samples[0];
        let max = samples[n - 1];

        Self { mean, median, p90, p95, p99, min, max }
    }

    pub fn from_durations(durations: &[Duration]) -> Self {
        Self::from_samples(durations.iter().map(|d| d.as_secs_f64()).collect())
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Collected metrics from a benchmark run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkMetrics {
    pub total_requests: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub elapsed_seconds: f64,
    pub throughput_rps: f64,
    pub output_tps: f64,
    pub total_tps: f64,
    pub latencies: LatencyStats,
    pub ttft: LatencyStats,
    pub tpot: LatencyStats,
}

impl BenchmarkMetrics {
    /// Compute metrics from request outputs and timing data.
    pub fn from_results(
        outputs: &[RequestOutput],
        elapsed: Duration,
        ttfts: &[Duration],
        tpots: &[Duration],
    ) -> Self {
        let total_requests = outputs.len();
        let total_input_tokens: u64 = outputs.iter().map(|o| o.usage.prompt_tokens as u64).sum();
        let total_output_tokens: u64 =
            outputs.iter().map(|o| o.usage.completion_tokens as u64).sum();
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

        // Per-request e2e latencies (approximate from usage + elapsed).
        let e2e_latencies: Vec<f64> = outputs
            .iter()
            .map(|o| {
                let tokens = o.usage.completion_tokens.max(1) as f64;
                // Approximate: elapsed * (this request's share of total tokens)
                let share = tokens / total_output_tokens.max(1) as f64;
                elapsed_seconds * share
            })
            .collect();

        Self {
            total_requests,
            total_input_tokens,
            total_output_tokens,
            elapsed_seconds,
            throughput_rps,
            output_tps,
            total_tps,
            latencies: LatencyStats::from_samples(e2e_latencies),
            ttft: LatencyStats::from_durations(ttfts),
            tpot: LatencyStats::from_durations(tpots),
        }
    }

    /// Print a summary to stdout.
    pub fn print_summary(&self) {
        println!("=== Benchmark Results ===");
        println!("Total requests:        {}", self.total_requests);
        println!("Total input tokens:     {}", self.total_input_tokens);
        println!("Total output tokens:    {}", self.total_output_tokens);
        println!("Elapsed:                {:.3}s", self.elapsed_seconds);
        println!("Throughput:             {:.1} req/s", self.throughput_rps);
        println!("Output throughput:      {:.1} tokens/s", self.output_tps);
        println!("Total throughput:       {:.1} tokens/s", self.total_tps);
        println!();
        println!(
            "E2E Latency:  mean={:.3}s median={:.3}s p95={:.3}s p99={:.3}s",
            self.latencies.mean, self.latencies.median, self.latencies.p95, self.latencies.p99
        );
        println!(
            "TTFT:         mean={:.3}s median={:.3}s p95={:.3}s p99={:.3}s",
            self.ttft.mean, self.ttft.median, self.ttft.p95, self.ttft.p99
        );
        println!(
            "TPOT:         mean={:.4}s median={:.4}s p95={:.4}s p99={:.4}s",
            self.tpot.mean, self.tpot.median, self.tpot.p95, self.tpot.p99
        );
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}
