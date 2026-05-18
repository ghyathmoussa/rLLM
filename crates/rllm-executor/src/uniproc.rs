use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::ids::RequestId;
use rllm_core::request::SamplingParams;
use rllm_scheduler::SchedulerOutput;
use rllm_sampling::{SamplingInput, Sampler};
use rllm_worker::Worker;

use crate::executor::{Executor, ExecutorOutput};

/// Single-process executor that owns one worker in the same process.
///
/// Delegates all calls directly to the worker. In the future, a
/// `MultiProcExecutor` will use IPC to coordinate multiple workers.
pub struct UniProcExecutor {
    worker: Worker,
    sampler: Sampler,
    eos_token_id: u32,
}

impl UniProcExecutor {
    pub fn new(mut worker: Worker) -> Self {
        let eos_token_id = worker.model_config().vocab_size as u32; // placeholder
        let sampler = worker.take_sampler().unwrap_or_default();
        Self {
            worker,
            sampler,
            eos_token_id,
        }
    }

    /// Get a reference to the underlying worker.
    pub fn worker(&self) -> &Worker {
        &self.worker
    }

    /// Get a mutable reference to the underlying worker.
    pub fn worker_mut(&mut self) -> &mut Worker {
        &mut self.worker
    }

    /// Set the EOS token ID (should be called after model loading).
    pub fn set_eos_token_id(&mut self, eos_token_id: u32) {
        self.eos_token_id = eos_token_id;
    }
}

impl Executor for UniProcExecutor {
    fn initialize(&mut self, kv_cache_configs: &[KVCacheConfig]) -> Result<()> {
        if let Some(config) = kv_cache_configs.first() {
            self.worker.initialize_kv_cache(config)?;
        }
        Ok(())
    }

    fn determine_available_memory(&self) -> Result<usize> {
        self.worker.determine_available_memory()
    }

    #[tracing::instrument(skip_all, name = "model_forward")]
    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput> {
        let start = std::time::Instant::now();

        // 1. Build input tensors from scheduler output.
        let batch = self.worker.model_runner_mut().build_tensors(scheduler_output)?;

        if batch.num_seqs == 0 {
            return Ok(ExecutorOutput {
                sampled_token_ids: vec![],
                logprobs: vec![],
            });
        }

        // 2. Build attention metadata.
        let _attn_meta = self.worker.model_runner().build_attention_metadata(&batch);

        // 3. Model forward pass.
        //    Without candle-backend, generate dummy logits so the sampling
        //    pipeline can be tested. With candle-backend, the real model
        //    forward would produce logits here.
        let vocab_size = self.worker.model_runner().vocab_size();

        // 4. For each request, extract logits and sample a token.
        let mut sampled_token_ids = Vec::with_capacity(batch.num_seqs);
        let mut logprobs = Vec::with_capacity(batch.num_seqs);

        for i in 0..batch.num_seqs {
            let request_id = batch.request_ids[i];
            let n_tokens = batch.tokens_per_seq[i];
            let is_prefill = batch.is_prefill[i];

            // Get or create logits for this request.
            // In production, these come from the model forward pass.
            // For now, use uniform dummy logits (all zeros → random sampling).
            let logits = if is_prefill {
                // During prefill, we only sample from the last token's logits.
                // Store dummy logits for the full prefill, but sample from the last position.
                vec![0.0f32; vocab_size]
            } else {
                // Decode: single token logits.
                vec![0.0f32; vocab_size]
            };

            // Get sampling params for this request.
            let sampling_params = self
                .worker
                .model_runner()
                .get_sampling_params(&request_id)
                .cloned()
                .unwrap_or_default();

            // Build context token IDs for penalty application.
            let (prompt_ids, generated_ids) = self
                .worker
                .model_runner()
                .get_context_token_ids(&request_id)
                .unwrap_or_default();

            let mut context_token_ids = prompt_ids.clone();
            context_token_ids.extend_from_slice(&generated_ids);

            let num_generated = generated_ids.len() as u32;

            let sampling_input = SamplingInput {
                logits,
                params: sampling_params.clone(),
                context_token_ids,
                num_generated,
                eos_token_id: self.eos_token_id,
                bad_word_token_ids: vec![],
            };

            let output = self.sampler.sample(&sampling_input);
            sampled_token_ids.push(output.token_id);
            logprobs.push(output.logprob);

            // Update model runner state.
            if is_prefill {
                // For prefill, advance computed tokens by the number of
                // prefill tokens. The sampled token from the last position
                // will be stored as a generated token.
                self.worker
                    .model_runner_mut()
                    .advance_computed(&request_id, n_tokens)?;
                // Store the sampled token as the first generated token.
                self.worker
                    .model_runner_mut()
                    .store_generated_token(&request_id, output.token_id)?;
            } else {
                // Decode: store the generated token.
                self.worker
                    .model_runner_mut()
                    .store_generated_token(&request_id, output.token_id)?;
            }
        }

        // 5. Async output copy.
        let copied_ids = self.worker.model_runner_mut().async_output_copy(&sampled_token_ids)?;
        self.worker
            .model_runner_mut()
            .cache_execute_model_state(copied_ids);

        rllm_metrics::histogram!("rllm_model_forward_duration_seconds")
            .record(start.elapsed().as_secs_f64());

        Ok(ExecutorOutput {
            sampled_token_ids,
            logprobs,
        })
    }

    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) {
        self.worker
            .model_runner_mut()
            .add_request(request_id, prompt_token_ids.clone());
        self.worker
            .model_runner_mut()
            .set_sampling_params(request_id, sampling_params);
    }

    fn shutdown(&mut self) {
        tracing::info!(worker_id = self.worker.id, "Executor shutting down");
    }
}
