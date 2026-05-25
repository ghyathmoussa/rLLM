use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::{ids::RequestId, request::SamplingParams};
use rllm_sampling::{Sampler, SamplingInput};
use rllm_scheduler::SchedulerOutput;
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
        Self { worker, sampler, eos_token_id }
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
        self.worker.initialize_cuda_device()?;
        self.worker.load_model_weights()?;
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
            return Ok(ExecutorOutput { sampled_token_ids: vec![], logprobs: vec![] });
        }

        // 2. Build attention metadata.
        let attn_meta = self.worker.model_runner().build_attention_metadata(&batch);

        // 3. Model forward pass.
        let vocab_size = self.worker.model_runner().vocab_size();
        let mut sampled_token_ids = Vec::with_capacity(batch.num_seqs);
        let mut logprobs = Vec::with_capacity(batch.num_seqs);

        // Try CUDA Graph replay first (Phase 2 optimization) for decode-only iterations.
        #[cfg(feature = "candle-backend")]
        let mut batched_logits = None;

        #[cfg(feature = "candle-backend")]
        if batch.num_prefill_tokens == 0 && self.worker.has_loaded_model() && self.worker.gpu_kv_cache().is_some() {
            let batch_size = batch.num_decode_tokens;
            if let Some(graph) = self.worker.cuda_graphs.get_graph_for_batch(batch_size) {
                let _ = graph;
                #[cfg(has_cuda)]
                {
                    if let (Some(ref input_tensor), Some(ref logits_tensor)) = (&graph.input_ids, &graph.logits) {
                        unsafe {
                            let ptr = input_tensor.as_ptr() as *mut std::ffi::c_void;
                            let bytes = batch.token_ids.len() * std::mem::size_of::<u32>();
                            let res = cudarc::driver::sys::cudaMemcpy(
                                ptr,
                                batch.token_ids.as_ptr() as *const std::ffi::c_void,
                                bytes,
                                1, // HostToDevice
                            );
                            if res == cudarc::driver::sys::cudaError::cudaSuccess {
                                if let Ok(_) = graph.replay() {
                                    batched_logits = Some(logits_tensor.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fall back to eager batched paged forward if graph path is not active/available.
        #[cfg(feature = "candle-backend")]
        if batched_logits.is_none() && self.worker.has_loaded_model() && self.worker.gpu_kv_cache().is_some() {
            let device = self.worker.worker_model_device();
            if let Some(device) = device {
                let input_ids = candle_core::Tensor::new(&batch.token_ids[..], device)?
                    .to_dtype(candle_core::DType::U32)?
                    .reshape((1, batch.token_ids.len()))?;
                let positions: Vec<usize> = batch.positions.iter().map(|&p| p as usize).collect();

                match self.worker.forward_paged_batch(&input_ids, &positions, &attn_meta) {
                    Ok(logits) => {
                        batched_logits = Some(logits);
                    }
                    Err(e) => {
                        tracing::warn!("Paged forward failed, falling back to legacy: {e}");
                    }
                }
            }
        }

        #[cfg(not(feature = "candle-backend"))]
        let batched_logits: Option<()> = None;

        // 4. For each request, extract logits and sample a token.
        // Token offset for extracting per-request logits from batched output.
        let mut token_offset = 0usize;

        for i in 0..batch.num_seqs {
            let request_id = batch.request_ids[i];
            let n_tokens = batch.tokens_per_seq[i];
            let is_prefill = batch.is_prefill[i];

            // Get sampling params for this request.
            let sampling_params = self
                .worker
                .model_runner()
                .get_sampling_params(&request_id)
                .cloned()
                .unwrap_or_default();

            // Build context token IDs for penalty application.
            let (prompt_ids, generated_ids) =
                self.worker.model_runner().get_context_token_ids(&request_id).unwrap_or_default();

            let mut context_token_ids = prompt_ids.clone();
            context_token_ids.extend_from_slice(&generated_ids);

            let num_generated = generated_ids.len() as u32;

            let (token_id, logprob) = {
                let mut logits_vec = None;

                #[cfg(feature = "candle-backend")]
                {
                    // Try batched logits first (paged path).
                    if let Some(ref all_logits) = batched_logits {
                        // Extract this request's last-token logits from the batched output.
                        // all_logits shape: [1, total_tokens, vocab_size]
                        let last_token_idx = token_offset + n_tokens - 1;
                        if let Ok(req_logits) = all_logits
                            .narrow(1, last_token_idx, 1)
                            .and_then(|t| t.reshape((all_logits.dim(2)?,)))
                            .and_then(|t| t.to_dtype(candle_core::DType::F32))
                            .and_then(|t| t.to_vec1::<f32>())
                        {
                            logits_vec = Some(req_logits);
                        }
                    }

                    // Fallback: legacy per-request forward if batched path unavailable.
                    if logits_vec.is_none() && self.worker.has_loaded_model() {
                        let tokens_to_run = if is_prefill {
                            let start = self.worker.model_runner().num_computed(&request_id);
                            let end = start + n_tokens;
                            let prompt_ids = self.worker.model_runner().get_context_token_ids(&request_id)
                                .map(|(p, _)| p)
                                .unwrap_or_default();
                            prompt_ids[start..end].to_vec()
                        } else {
                            let last_token = self.worker.model_runner().get_context_token_ids(&request_id)
                                .map(|(_, g)| g.last().copied())
                                .flatten()
                                .unwrap_or_else(|| {
                                    self.worker.model_runner().get_context_token_ids(&request_id)
                                        .map(|(p, _)| p.last().copied())
                                        .flatten()
                                        .unwrap_or(0)
                                });
                            vec![last_token]
                        };

                        let pos_usize: Vec<usize> = (0..tokens_to_run.len())
                            .map(|j| self.worker.model_runner().num_computed(&request_id) + j)
                            .collect();

                        let logits = self.worker.execute_model_step(&request_id, &tokens_to_run, &pos_usize)?;
                        if let Some(logits) = logits {
                            let seq_len = logits.dim(1)?;
                            let vocab_dim = logits.dim(2)?;
                            let last_logits = logits.narrow(1, seq_len - 1, 1)?
                                .reshape((vocab_dim,))?
                                .to_dtype(candle_core::DType::F32)?
                                .to_vec1::<f32>()?;
                            logits_vec = Some(last_logits);
                        }
                    }
                }

                let logits = match logits_vec {
                    Some(l) => l,
                    None => vec![0.0f32; vocab_size],
                };

                let sampling_input = SamplingInput {
                    logits,
                    params: sampling_params.clone(),
                    context_token_ids,
                    num_generated,
                    eos_token_id: self.eos_token_id,
                    bad_word_token_ids: vec![],
                };
                let output = self.sampler.sample(&sampling_input);
                (output.token_id, output.logprob)
            };

            sampled_token_ids.push(token_id);
            logprobs.push(logprob);

            // Update model runner state.
            if is_prefill {
                self.worker.model_runner_mut().advance_computed(&request_id, n_tokens)?;
                self.worker.model_runner_mut().store_generated_token(&request_id, token_id)?;
            } else {
                self.worker.model_runner_mut().store_generated_token(&request_id, token_id)?;
            }

            token_offset += n_tokens;
        }

        // 5. Async output copy.
        let copied_ids = self.worker.model_runner_mut().async_output_copy(&sampled_token_ids)?;
        self.worker.model_runner_mut().cache_execute_model_state(copied_ids);

        rllm_metrics::histogram!("rllm_model_forward_duration_seconds")
            .record(start.elapsed().as_secs_f64());

        Ok(ExecutorOutput { sampled_token_ids, logprobs })
    }

    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) {
        self.worker.model_runner_mut().add_request(request_id, prompt_token_ids.clone());
        self.worker.model_runner_mut().set_sampling_params(request_id, sampling_params);
    }

    fn shutdown(&mut self) {
        tracing::info!(worker_id = self.worker.id, "Executor shutting down");
    }
}
