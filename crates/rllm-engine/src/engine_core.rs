use std::collections::HashMap;

use anyhow::Result;
use rllm_core::ids::RequestId;
use rllm_core::output::{CompletionOutput, FinishReason, RequestOutput, Usage};
use rllm_core::request::{InferenceRequest, SamplingParams};
use rllm_executor::Executor;
use rllm_scheduler::Scheduler;

/// Per-request state tracked by the engine core across steps.
struct EngineRequest {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    sampling_params: SamplingParams,
    #[expect(dead_code)]
    prompt_text: Option<String>,
    max_tokens: u32,
    /// Arrival time (from the original InferenceRequest) for TTFT/e2e latency.
    arrival_time: std::time::Instant,
    /// Whether the first token has been generated (for TTFT).
    first_token_generated: bool,
}

impl EngineRequest {
    #[expect(dead_code)]
    fn num_tokens(&self) -> usize {
        self.prompt_token_ids.len() + self.generated_token_ids.len()
    }
}

/// Core engine that owns the executor and scheduler, orchestrating inference steps.
///
/// Each call to `step()` runs one iteration:
/// 1. Scheduler decides which requests to schedule
/// 2. Executor runs the model forward pass and sampling
/// 3. Engine processes outputs (check stopping conditions, build responses)
pub struct EngineCore {
    executor: Box<dyn Executor>,
    scheduler: Scheduler,
    requests: HashMap<RequestId, EngineRequest>,
    eos_token_id: u32,
}

impl EngineCore {
    pub fn new(executor: Box<dyn Executor>, scheduler: Scheduler, eos_token_id: u32) -> Self {
        Self { executor, scheduler, requests: HashMap::new(), eos_token_id }
    }

    /// Add a new inference request.
    ///
    /// Tokenizes the prompt if token IDs are not already provided,
    /// registers with the scheduler and executor.
    pub fn add_request(&mut self, request: InferenceRequest) -> Result<()> {
        let request_id = request.request_id;
        let token_ids = request.token_ids.clone().unwrap_or_default();
        let max_tokens = request.sampling_params.max_tokens.unwrap_or(16);
        let sampling_params = request.sampling_params.clone();
        let prompt_text = request.prompt.clone();

        self.executor.add_request(request_id, token_ids.clone(), sampling_params.clone());

        let engine_req = EngineRequest {
            prompt_token_ids: token_ids,
            generated_token_ids: Vec::new(),
            sampling_params,
            prompt_text,
            max_tokens,
            arrival_time: request.arrival_time,
            first_token_generated: false,
        };

        self.requests.insert(request_id, engine_req);
        self.scheduler.add_request(request);

        rllm_metrics::counter!("rllm_requests_total").increment(1);

        Ok(())
    }

    /// Abort a request.
    pub fn abort_request(&mut self, request_id: RequestId) {
        self.scheduler.abort_request(request_id);
        self.requests.remove(&request_id);
    }

    /// Run one engine step. Returns outputs for requests that produced tokens
    /// this step, including any that have finished.
    #[tracing::instrument(skip_all, name = "engine_step")]
    pub fn step(&mut self) -> Vec<RequestOutput> {
        // 1. Schedule.
        let scheduler_output = self.scheduler.step();

        if scheduler_output.num_scheduled() == 0 {
            return vec![];
        }

        // 2. Execute model.
        let exec_result = match self.executor.execute_model(&scheduler_output) {
            Ok(output) => output,
            Err(e) => {
                tracing::error!("Executor error: {}", e);
                tracing::debug!(
                    "{}",
                    rllm_metrics::debug_scheduler_output_dump(
                        scheduler_output.scheduled_new.len(),
                        scheduler_output.scheduled_cached.len(),
                        scheduler_output.scheduled_running.len(),
                        scheduler_output.token_budget_used,
                        scheduler_output.preempted.len(),
                        scheduler_output.finished.len(),
                    )
                );
                tracing::debug!(
                    "{}",
                    self.scheduler.debug_request_state_summary(scheduler_output.finished.len())
                );
                tracing::debug!("{}", self.scheduler.debug_kv_cache_summary());
                return vec![];
            }
        };

        // Record token throughput metrics.
        let n_sampled = exec_result.sampled_token_ids.len() as u64;
        rllm_metrics::counter!("rllm_generated_tokens_total").increment(n_sampled);

        // 3. Process outputs.
        let mut outputs = Vec::new();

        // Pair sampled tokens with their request IDs.
        let scheduled_ids: Vec<RequestId> = scheduler_output
            .scheduled_new
            .iter()
            .chain(scheduler_output.scheduled_running.iter())
            .copied()
            .collect();

        for (i, request_id) in scheduled_ids.iter().enumerate() {
            let token_id = exec_result.sampled_token_ids.get(i).copied().unwrap_or(0);

            let finished = if let Some(req) = self.requests.get_mut(request_id) {
                req.generated_token_ids.push(token_id);

                // TTFT: record time from arrival to first generated token.
                if !req.first_token_generated {
                    req.first_token_generated = true;
                    let ttft = req.arrival_time.elapsed().as_secs_f64();
                    rllm_metrics::histogram!("rllm_ttft_seconds").record(ttft);
                }

                // Check stopping conditions.
                let reached_eos = token_id == self.eos_token_id && !req.sampling_params.ignore_eos;
                let reached_length = req.generated_token_ids.len() >= req.max_tokens as usize;
                let hit_stop_token =
                    req.sampling_params.stop_token_ids.contains(&token_id);

                if reached_eos || hit_stop_token {
                    Some(FinishReason::Stop)
                } else if reached_length {
                    Some(FinishReason::Length)
                } else {
                    None
                }
            } else {
                None
            };

            // Build output for this request.
            let req = self.requests.get(request_id);
            if let Some(req) = req {
                let prompt_tokens = req.prompt_token_ids.len() as u32;
                let completion_tokens = req.generated_token_ids.len() as u32;

                // Record prompt tokens on first output.
                if completion_tokens == 1 {
                    rllm_metrics::counter!("rllm_prompt_tokens_total")
                        .increment(prompt_tokens as u64);
                }

                let finish_reason = finished;

                outputs.push(RequestOutput {
                    request_id: *request_id,
                    outputs: vec![CompletionOutput {
                        index: 0,
                        text: String::new(),
                        token_ids: vec![token_id],
                        finish_reason,
                        logprobs: None,
                    }],
                    finished: finished.is_some(),
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                });
            }
        }

        // 4. Clean up finished requests.
        let finished_ids: Vec<RequestId> =
            outputs.iter().filter(|o| o.finished).map(|o| o.request_id).collect();

        for id in &finished_ids {
            if let Some(req) = self.requests.remove(id) {
                // Record e2e latency and finished count.
                let elapsed = req.arrival_time.elapsed().as_secs_f64();
                rllm_metrics::histogram!("rllm_e2e_latency_seconds").record(elapsed);
                rllm_metrics::counter!("rllm_requests_finished_total").increment(1);
                rllm_metrics::record_tokens_per_second(
                    req.generated_token_ids.len() as u32,
                    elapsed,
                );
                // TPOT approximation: e2e / num_generated_tokens.
                if req.generated_token_ids.len() > 1 {
                    let tpot = elapsed / req.generated_token_ids.len() as f64;
                    rllm_metrics::histogram!("rllm_tpot_seconds").record(tpot);
                }
            }
        }

        outputs
    }

    /// Check if there is any pending work.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Get the number of active requests.
    pub fn num_active_requests(&self) -> usize {
        self.requests.len()
    }

    /// Reset the prefix cache.
    pub fn reset_prefix_cache(&mut self) -> Result<(), String> {
        self.scheduler.reset_prefix_cache()
    }
}
