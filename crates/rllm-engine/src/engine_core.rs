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
    prompt_text: Option<String>,
    max_tokens: u32,
}

impl EngineRequest {
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
    pub fn new(
        executor: Box<dyn Executor>,
        scheduler: Scheduler,
        eos_token_id: u32,
    ) -> Self {
        Self {
            executor,
            scheduler,
            requests: HashMap::new(),
            eos_token_id,
        }
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

        self.executor.add_request(
            request_id,
            token_ids.clone(),
            sampling_params.clone(),
        );

        let engine_req = EngineRequest {
            prompt_token_ids: token_ids,
            generated_token_ids: Vec::new(),
            sampling_params,
            prompt_text,
            max_tokens,
        };

        self.requests.insert(request_id, engine_req);
        self.scheduler.add_request(request);

        Ok(())
    }

    /// Abort a request.
    pub fn abort_request(&mut self, request_id: RequestId) {
        self.scheduler.abort_request(request_id);
        self.requests.remove(&request_id);
    }

    /// Run one engine step. Returns outputs for requests that produced tokens
    /// this step, including any that have finished.
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
                return vec![];
            }
        };

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

                // Check stopping conditions.
                let reached_eos = token_id == self.eos_token_id
                    && !req.sampling_params.ignore_eos;
                let reached_length = req.generated_token_ids.len() >= req.max_tokens as usize;
                let hit_stop_token = req
                    .sampling_params
                    .stop_token_ids
                    .iter()
                    .any(|&st| st == token_id);

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

                let finish_reason = finished;

                outputs.push(RequestOutput {
                    request_id: *request_id,
                    outputs: vec![CompletionOutput {
                        index: 0,
                        text: String::new(), // Detokenization done by output processor
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
        let finished_ids: Vec<RequestId> = outputs
            .iter()
            .filter(|o| o.finished)
            .map(|o| o.request_id)
            .collect();

        for id in &finished_ids {
            // Tell scheduler the request is finished by aborting it.
            // The scheduler will handle the state transition.
            self.requests.remove(id);
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
