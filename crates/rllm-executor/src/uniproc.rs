use std::collections::HashMap;

use anyhow::Result;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::{ids::RequestId, request::SamplingParams};
#[cfg(feature = "structured-outputs")]
use rllm_sampling::StructuredOutputManager;
use rllm_sampling::{NGramProposer, Sampler, SamplingInput, SpeculativeProposer};
use rllm_scheduler::SchedulerOutput;
use rllm_worker::Worker;

use crate::executor::{Executor, ExecutorOutput, PerRequestOutput};

/// Helper: find the index of the maximum value in a slice.
fn argmax_impl(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Single-process executor that owns one worker in the same process.
///
/// Delegates all calls directly to the worker. In the future, a
/// `MultiProcExecutor` will use IPC to coordinate multiple workers.
pub struct UniProcExecutor {
    worker: Worker,
    sampler: Sampler,
    eos_token_id: u32,
    #[cfg(feature = "structured-outputs")]
    structured_outputs: Option<StructuredOutputManager>,
}

impl UniProcExecutor {
    pub fn new(mut worker: Worker) -> Self {
        let eos_token_id = worker.model_config().vocab_size as u32; // placeholder
        let sampler = worker.take_sampler().unwrap_or_default();
        Self {
            worker,
            sampler,
            eos_token_id,
            #[cfg(feature = "structured-outputs")]
            structured_outputs: None,
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

    #[cfg(feature = "structured-outputs")]
    pub fn configure_structured_outputs(
        &mut self,
        tokenizer_backend: &str,
        vocab_size: usize,
    ) -> Result<()> {
        self.structured_outputs =
            Some(StructuredOutputManager::new(tokenizer_backend, vocab_size, self.eos_token_id)?);
        Ok(())
    }
}

impl Executor for UniProcExecutor {
    fn initialize(
        &mut self,
        kv_cache_configs: &[KVCacheConfig],
        gpu_memory_utilization: f32,
    ) -> Result<usize> {
        self.worker.initialize_cuda_device()?;
        // Weights must be resident before profiling free memory.
        self.worker.load_model_weights()?;
        let num_blocks = if let Some(config) = kv_cache_configs.first() {
            // Profile GPU memory and shrink the requested (worst-case) block
            // count to what actually fits.
            let fitted = self.worker.fit_kv_blocks(
                &config.spec,
                gpu_memory_utilization,
                config.num_blocks,
            )?;
            let fitted_config = KVCacheConfig { num_blocks: fitted, spec: config.spec.clone() };
            self.worker.initialize_kv_cache(&fitted_config)?;

            // Calibrate INT8 KV cache scales from a forward pass with
            // representative tokens.  No-op when the cache dtype is not INT8.
            if let Err(e) = self.worker.calibrate_kv_cache() {
                tracing::warn!(error = %e, "KV cache calibration failed; using default scales");
            }

            fitted
        } else {
            0
        };
        Ok(num_blocks)
    }

    fn determine_available_memory(&self) -> Result<usize> {
        self.worker.determine_available_memory()
    }

    #[tracing::instrument(skip_all, name = "model_forward")]
    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput> {
        let start = std::time::Instant::now();

        #[cfg(feature = "structured-outputs")]
        if let Some(manager) = &mut self.structured_outputs {
            for request_id in &scheduler_output.finished {
                manager.remove(request_id);
            }
        }

        // 0. Propose speculative draft tokens for decode requests.
        let mut speculative_draft_map: HashMap<RequestId, Vec<u32>> = HashMap::new();
        for rid in &scheduler_output.scheduled_running {
            let spec_cfg = self
                .worker
                .model_runner()
                .get_sampling_params(rid)
                .and_then(|p| p.speculative_decoding.clone());

            if let Some(ref cfg) = spec_cfg {
                if cfg.enabled && cfg.num_speculative_tokens > 0 && cfg.proposer == "ngram" {
                    let (prompt, generated) =
                        self.worker.model_runner().get_context_token_ids(rid).unwrap_or_default();
                    let mut context = prompt;
                    context.extend_from_slice(&generated);

                    let proposer = NGramProposer::new(cfg.min_ngram, cfg.max_ngram);
                    let proposal = proposer.propose(&context, cfg.num_speculative_tokens);

                    if !proposal.token_ids.is_empty() {
                        speculative_draft_map.insert(*rid, proposal.token_ids.clone());
                        self.worker
                            .model_runner_mut()
                            .set_speculative_drafts(*rid, proposal.token_ids);
                    }
                }
            }
        }

        // 1. Build input tensors from scheduler output.
        let batch = self.worker.model_runner_mut().build_tensors(scheduler_output)?;

        if batch.num_seqs == 0 {
            return Ok(ExecutorOutput::empty());
        }

        // 2. Build attention metadata.
        let attn_meta = self.worker.model_runner().build_attention_metadata(&batch);

        // 3. Model forward pass.
        let vocab_size = self.worker.model_runner().vocab_size();

        // Try CUDA Graph replay first (Phase 2 optimization) for decode-only iterations.
        #[cfg(feature = "candle-backend")]
        let mut batched_logits = None;

        // CUDA graph replay path is disabled: graph capture is not yet implemented
        // (see CudaGraphInstance in rllm-worker), so `replay()` always reports
        // "not captured" and `input_ids`/`logits` are never populated. Decode goes
        // straight to the eager forward below. The previous implementation here
        // also depended on a raw device pointer from a candle tensor and the CUDA
        // runtime `cudaMemcpy`, neither of which is available in this crate.

        // Eager batched paged forward (falls back to legacy per-request forward).
        #[cfg(feature = "candle-backend")]
        if batched_logits.is_none()
            && self.worker.has_loaded_model()
            && self.worker.gpu_kv_cache().is_some()
        {
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
                        tracing::debug!("Paged forward unavailable, using legacy forward: {e}");
                    }
                }
            }
        }

        #[cfg(not(feature = "candle-backend"))]
        let batched_logits: Option<()> = None;

        // 4. For each request, extract logits and sample/verify.
        let mut token_offset = 0usize;
        let mut per_request_outputs = Vec::with_capacity(batch.num_seqs);

        for i in 0..batch.num_seqs {
            let request_id = batch.request_ids[i];
            let n_tokens = batch.tokens_per_seq[i];
            let is_prefill = batch.is_prefill[i];
            let has_speculative = !is_prefill && speculative_draft_map.contains_key(&request_id);

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

            if has_speculative {
                // ── Speculative decode path ──
                // We processed (1 + k) tokens in one forward pass. Extract
                // per-position logits, verify drafts, produce multiple tokens.

                // Get the per-position logits for this request.  The batched
                // forward returned logits of shape [1, total_tokens, vocab_size].
                let mut per_pos_logits: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);

                #[cfg(feature = "candle-backend")]
                if let Some(ref all_logits) = batched_logits {
                    // Batched path: extract all positions for this request.
                    let vocab_dim = all_logits.dim(2)?;
                    for j in 0..n_tokens {
                        let pos_idx = token_offset + j;
                        if let Ok(l) = all_logits
                            .narrow(1, pos_idx, 1)
                            .and_then(|t| t.reshape((vocab_dim,)))
                            .and_then(|t| t.to_dtype(candle_core::DType::F32))
                            .and_then(|t| t.to_vec1::<f32>())
                        {
                            per_pos_logits.push(l);
                        } else {
                            per_pos_logits.push(vec![0.0f32; vocab_size]);
                        }
                    }
                }

                #[cfg(not(feature = "candle-backend"))]
                {
                    let _ = vocab_size;
                    // Without Candle, fall through to dummy below.
                }

                // Fallback: if batched path didn't provide per-position logits,
                // try the legacy per-request forward to get them.
                #[cfg(feature = "candle-backend")]
                if per_pos_logits.len() < n_tokens && self.worker.has_loaded_model() {
                    // Run per-request forward with all tokens to get per-position logits.
                    let tokens_to_run = batch
                        .token_ids
                        .get(token_offset..token_offset + n_tokens)
                        .unwrap_or_default()
                        .to_vec();

                    if !tokens_to_run.is_empty() {
                        let start_pos = self.worker.model_runner().num_computed(&request_id);
                        let pos_usize: Vec<usize> =
                            (0..tokens_to_run.len()).map(|j| start_pos + j).collect();

                        if let Ok(Some(logits_tensor)) =
                            self.worker.execute_model_step(&request_id, &tokens_to_run, &pos_usize)
                        {
                            if let Ok(seq_len) = logits_tensor.dim(1) {
                                if let Ok(vocab_dim) = logits_tensor.dim(2) {
                                    for j in 0..seq_len {
                                        if let Ok(l) = logits_tensor
                                            .narrow(1, j, 1)
                                            .and_then(|t| t.reshape((vocab_dim,)))
                                            .and_then(|t| t.to_dtype(candle_core::DType::F32))
                                            .and_then(|t| t.to_vec1::<f32>())
                                        {
                                            per_pos_logits.push(l);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Pad with dummy logits if extraction failed.
                while per_pos_logits.len() < n_tokens {
                    per_pos_logits.push(vec![0.0f32; vocab_size]);
                }

                // Run verification: compare model's greedy prediction with each draft.
                let drafts = speculative_draft_map.get(&request_id).cloned().unwrap_or_default();
                let k = drafts.len();
                let mut accepted_tokens: Vec<u32> = Vec::new();
                let mut accepted_logprobs: Vec<Option<f32>> = Vec::new();

                for j in 0..k {
                    if j >= per_pos_logits.len() {
                        break;
                    }
                    let predicted = argmax_impl(&per_pos_logits[j]) as u32;

                    if j < drafts.len() && predicted == drafts[j] {
                        accepted_tokens.push(drafts[j]);
                        accepted_logprobs.push(None);
                    } else if j < per_pos_logits.len() {
                        // First rejection: sample from the model's distribution.
                        let cur_generated = num_generated as usize + accepted_tokens.len();
                        let input = SamplingInput {
                            logits: per_pos_logits[j].clone(),
                            params: sampling_params.clone(),
                            context_token_ids: context_token_ids.clone(),
                            num_generated: cur_generated as u32,
                            eos_token_id: self.eos_token_id,
                            bad_word_token_ids: vec![],
                        };
                        let output = self.sampler.sample(&input);
                        accepted_tokens.push(output.token_id);
                        accepted_logprobs.push(output.logprob);
                        break;
                    }
                }

                // If all k drafts were accepted, sample the bonus token.
                if accepted_tokens.len() == k && k > 0 && per_pos_logits.len() > k {
                    let bonus_idx = k.min(per_pos_logits.len().saturating_sub(1));
                    let cur_generated = num_generated as usize + accepted_tokens.len();
                    let input = SamplingInput {
                        logits: per_pos_logits[bonus_idx].clone(),
                        params: sampling_params.clone(),
                        context_token_ids: context_token_ids.clone(),
                        num_generated: cur_generated as u32,
                        eos_token_id: self.eos_token_id,
                        bad_word_token_ids: vec![],
                    };
                    let output = self.sampler.sample(&input);
                    accepted_tokens.push(output.token_id);
                    accepted_logprobs.push(output.logprob);
                }

                // Fallback: if verification produced nothing (k == 0), run normal sampling.
                if accepted_tokens.is_empty() {
                    let last_idx =
                        n_tokens.saturating_sub(1).min(per_pos_logits.len().saturating_sub(1));
                    let input = SamplingInput {
                        logits: per_pos_logits[last_idx].clone(),
                        params: sampling_params.clone(),
                        context_token_ids: context_token_ids.clone(),
                        num_generated,
                        eos_token_id: self.eos_token_id,
                        bad_word_token_ids: vec![],
                    };
                    let output = self.sampler.sample(&input);
                    accepted_tokens.push(output.token_id);
                    accepted_logprobs.push(output.logprob);
                }

                // Update model runner state: store accepted tokens and advance computed.
                for &token in &accepted_tokens {
                    self.worker.model_runner_mut().store_generated_token(&request_id, token)?;
                }
                self.worker.model_runner_mut().advance_computed(&request_id, n_tokens)?;

                per_request_outputs.push(PerRequestOutput {
                    request_id,
                    token_ids: accepted_tokens,
                    logprobs: accepted_logprobs,
                });
            } else {
                // ── Normal (non-speculative) path ──
                let (token_id, logprob) = {
                    let mut logits_vec = None;

                    #[cfg(feature = "candle-backend")]
                    {
                        if let Some(ref all_logits) = batched_logits {
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

                        if logits_vec.is_none() && self.worker.has_loaded_model() {
                            let tokens_to_run = if is_prefill {
                                let start = self.worker.model_runner().num_computed(&request_id);
                                let end = start + n_tokens;
                                let prompt_ids = self
                                    .worker
                                    .model_runner()
                                    .get_context_token_ids(&request_id)
                                    .map(|(p, _)| p)
                                    .unwrap_or_default();
                                prompt_ids[start..end].to_vec()
                            } else {
                                let last_token = self
                                    .worker
                                    .model_runner()
                                    .get_context_token_ids(&request_id)
                                    .and_then(|(_, g)| g.last().copied())
                                    .unwrap_or_else(|| {
                                        self.worker
                                            .model_runner()
                                            .get_context_token_ids(&request_id)
                                            .and_then(|(p, _)| p.last().copied())
                                            .unwrap_or(0)
                                    });
                                vec![last_token]
                            };

                            let pos_usize: Vec<usize> = (0..tokens_to_run.len())
                                .map(|j| self.worker.model_runner().num_computed(&request_id) + j)
                                .collect();

                            let logits = self.worker.execute_model_step(
                                &request_id,
                                &tokens_to_run,
                                &pos_usize,
                            )?;
                            if let Some(logits) = logits {
                                let seq_len = logits.dim(1)?;
                                let vocab_dim = logits.dim(2)?;
                                let last_logits = logits
                                    .narrow(1, seq_len - 1, 1)?
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

                    #[cfg(feature = "structured-outputs")]
                    let mut logits = logits;
                    #[cfg(feature = "structured-outputs")]
                    if let Some(manager) = &mut self.structured_outputs {
                        manager.mask_logits(&request_id, &mut logits)?;
                    }

                    let sampling_input = SamplingInput {
                        logits,
                        params: sampling_params.clone(),
                        context_token_ids,
                        num_generated,
                        eos_token_id: self.eos_token_id,
                        bad_word_token_ids: vec![],
                    };
                    let output = self.sampler.sample(&sampling_input);
                    #[cfg(feature = "structured-outputs")]
                    if let Some(manager) = &mut self.structured_outputs {
                        manager.accept_token(&request_id, output.token_id)?;
                    }
                    (output.token_id, output.logprob)
                };

                // Update model runner state.
                if is_prefill {
                    self.worker.model_runner_mut().advance_computed(&request_id, n_tokens)?;
                    self.worker.model_runner_mut().store_generated_token(&request_id, token_id)?;
                } else {
                    self.worker.model_runner_mut().store_generated_token(&request_id, token_id)?;
                    self.worker.model_runner_mut().advance_computed(&request_id, 1)?;
                }

                per_request_outputs.push(PerRequestOutput {
                    request_id,
                    token_ids: vec![token_id],
                    logprobs: vec![logprob],
                });
            }

            token_offset += n_tokens;
        }

        // 5. Async output copy.
        let all_token_ids: Vec<u32> =
            per_request_outputs.iter().flat_map(|o| o.token_ids.iter().copied()).collect();
        let copied_ids = self.worker.model_runner_mut().async_output_copy(&all_token_ids)?;
        self.worker.model_runner_mut().cache_execute_model_state(copied_ids);

        rllm_metrics::histogram!("rllm_model_forward_duration_seconds")
            .record(start.elapsed().as_secs_f64());

        Ok(ExecutorOutput { per_request_outputs })
    }

    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Result<()> {
        if sampling_params.structured_outputs.is_some() {
            #[cfg(feature = "structured-outputs")]
            {
                let params = sampling_params.structured_outputs.as_ref().unwrap();
                let manager = self.structured_outputs.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("structured outputs are not configured for this executor")
                })?;
                manager.register(request_id, params)?;
            }
            #[cfg(not(feature = "structured-outputs"))]
            anyhow::bail!("structured output support requires the structured-outputs feature");
        }
        self.worker.model_runner_mut().add_request(request_id, prompt_token_ids.clone());
        self.worker.model_runner_mut().set_sampling_params(request_id, sampling_params);
        Ok(())
    }

    fn shutdown(&mut self) {
        tracing::info!(worker_id = self.worker.id, "Executor shutting down");
    }
}
