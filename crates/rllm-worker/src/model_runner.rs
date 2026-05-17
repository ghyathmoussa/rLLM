use std::collections::HashMap;
use std::collections::HashSet;

use anyhow::Result;
use rllm_core::config::ModelConfig;
use rllm_core::ids::RequestId;
use rllm_kernels::AttentionMetadata;
use rllm_scheduler::SchedulerOutput;
use rllm_tensor::PinnedBuffer;

use crate::batch::InputBatch;

/// Per-request state tracked by the model runner across steps.
#[derive(Debug, Clone)]
struct RunnerRequestState {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    num_computed_tokens: usize,
}

impl RunnerRequestState {
    fn new(prompt_token_ids: Vec<u32>) -> Self {
        Self {
            prompt_token_ids,
            generated_token_ids: Vec::new(),
            num_computed_tokens: 0,
        }
    }

    fn _total_tokens(&self) -> usize {
        self.prompt_token_ids.len() + self.generated_token_ids.len()
    }
}

/// Builds input tensors and attention metadata from scheduler output.
///
/// The runner maintains per-request token state (prompt + generated tokens)
/// internally because `SchedulerOutput` only carries structural data
/// (which requests, token counts, block tables), not actual token IDs.
pub struct ModelRunner {
    #[allow(dead_code)]
    model_config: ModelConfig,
    #[allow(dead_code)]
    block_size: usize,
    request_states: HashMap<RequestId, RunnerRequestState>,
    /// Pinned host buffer for async CPU copy of output token IDs.
    output_buffer: PinnedBuffer,
    /// Cached output from last model execution for async copy.
    cached_sampled_ids: Vec<u32>,
}

impl ModelRunner {
    pub fn new(model_config: ModelConfig, block_size: usize) -> Self {
        let output_buffer = PinnedBuffer::alloc_typed::<u32>(4096);
        Self {
            model_config,
            block_size,
            request_states: HashMap::new(),
            output_buffer,
            cached_sampled_ids: Vec::new(),
        }
    }

    /// Register a new request with its prompt tokens.
    pub fn add_request(&mut self, request_id: RequestId, prompt_token_ids: Vec<u32>) {
        self.request_states
            .insert(request_id, RunnerRequestState::new(prompt_token_ids));
    }

    /// Remove a finished or preempted request.
    pub fn remove_request(&mut self, request_id: &RequestId) {
        self.request_states.remove(request_id);
    }

    /// Store a generated token for a running request and advance computed count.
    pub fn store_generated_token(&mut self, request_id: &RequestId, token_id: u32) -> Result<()> {
        let state = self
            .request_states
            .get_mut(request_id)
            .ok_or_else(|| anyhow::anyhow!("request {:?} not found", request_id))?;
        state.generated_token_ids.push(token_id);
        state.num_computed_tokens += 1;
        Ok(())
    }

    /// Advance computed tokens after a prefill step (no new generated token yet).
    pub fn advance_computed(&mut self, request_id: &RequestId, n_tokens: usize) -> Result<()> {
        let state = self
            .request_states
            .get_mut(request_id)
            .ok_or_else(|| anyhow::anyhow!("request {:?} not found", request_id))?;
        state.num_computed_tokens += n_tokens;
        Ok(())
    }

    /// Get the number of computed tokens for a request.
    pub fn num_computed(&self, request_id: &RequestId) -> usize {
        self.request_states
            .get(request_id)
            .map(|s| s.num_computed_tokens)
            .unwrap_or(0)
    }

    /// Check if a request is tracked.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.request_states.contains_key(request_id)
    }

    /// Build flat input tensors from a `SchedulerOutput`.
    ///
    /// Processes requests in order: prefill (new + cached) then decode (running).
    /// For prefill requests, selects the prompt token slice starting at
    /// `num_computed_tokens`. For decode requests, uses the last generated token.
    pub fn build_tensors(&mut self, scheduler_output: &SchedulerOutput) -> Result<InputBatch> {
        // Clean up finished and preempted requests.
        for rid in &scheduler_output.finished {
            self.remove_request(rid);
        }
        for rid in &scheduler_output.preempted {
            self.remove_request(rid);
        }

        if scheduler_output.num_scheduled() == 0 {
            return Ok(InputBatch::empty());
        }

        // Collect all scheduled request IDs in order: new, then cached, then running.
        let mut ordered_ids: Vec<RequestId> = Vec::new();
        ordered_ids.extend_from_slice(&scheduler_output.scheduled_new);
        // scheduled_cached requests are fully cached (no tokens to process)
        // but they may transition to decode next step, so skip them here.
        ordered_ids.extend_from_slice(&scheduler_output.scheduled_running);

        let mut request_ids = Vec::new();
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut seq_lens = Vec::new();
        let mut block_tables = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut tokens_per_seq = Vec::new();
        let mut is_prefill = Vec::new();
        let mut num_prefill_tokens = 0usize;
        let mut num_decode_tokens = 0usize;
        let mut max_num_blocks_per_seq = 0usize;

        let new_set: HashSet<RequestId> =
            scheduler_output.scheduled_new.iter().copied().collect();

        for rid in &ordered_ids {
            let state = match self.request_states.get(rid) {
                Some(s) => s,
                None => {
                    // Request state not found — may not have been added yet.
                    // Skip silently; the engine core should call add_request first.
                    continue;
                }
            };

            let n_scheduled = scheduler_output
                .num_scheduled_tokens
                .get(rid)
                .copied()
                .unwrap_or(0);

            if n_scheduled == 0 {
                continue;
            }

            let is_pref = new_set.contains(rid);
            let computed = state.num_computed_tokens;

            // Determine tokens and positions for this request.
            let req_tokens;
            let req_positions;

            if is_pref {
                // Prefill: take prompt tokens from computed offset.
                let start = computed;
                let end = (start + n_scheduled).min(state.prompt_token_ids.len());
                if start >= state.prompt_token_ids.len() {
                    continue;
                }
                req_tokens = state.prompt_token_ids[start..end].to_vec();
                req_positions = (start..end).map(|p| p as u32).collect();
                num_prefill_tokens += req_tokens.len();
            } else {
                // Decode: single generated token at position = total_tokens - 1,
                // but since we haven't stored the token yet, the position is computed
                // (which equals prompt_len + generated_len so far).
                let last_token = state
                    .generated_token_ids
                    .last()
                    .copied()
                    .unwrap_or_else(|| {
                        // First decode step: use last prompt token.
                        *state.prompt_token_ids.last().unwrap_or(&0)
                    });
                req_tokens = vec![last_token];
                req_positions = vec![computed as u32];
                num_decode_tokens += 1;
            }

            let n_tokens = req_tokens.len();
            let seq_len = (computed + n_tokens) as u32;

            // Get block table and compute slot mappings.
            let bt: Vec<u32> = scheduler_output
                .block_tables
                .get(rid)
                .map(|ids| ids.iter().map(|b| b.0).collect())
                .unwrap_or_default();

            max_num_blocks_per_seq = max_num_blocks_per_seq.max(bt.len());

            let slots = compute_slot_mappings(&req_positions, &bt, self.block_size);

            request_ids.push(*rid);
            token_ids.extend_from_slice(&req_tokens);
            positions.extend_from_slice(&req_positions);
            seq_lens.push(seq_len);
            block_tables.push(bt);
            slot_mappings.extend_from_slice(&slots);
            tokens_per_seq.push(n_tokens);
            is_prefill.push(is_pref);
        }

        Ok(InputBatch {
            request_ids,
            token_ids,
            positions,
            seq_lens,
            block_tables,
            slot_mappings,
            num_prefill_tokens,
            num_decode_tokens,
            num_seqs: tokens_per_seq.len(),
            max_num_blocks_per_seq,
            tokens_per_seq,
            is_prefill,
        })
    }

    /// Build `AttentionMetadata` from an `InputBatch`.
    ///
    /// Splits the batch into prefill and decode groups and constructs
    /// the appropriate metadata for each.
    pub fn build_attention_metadata(&self, batch: &InputBatch) -> AttentionMetadata {
        if batch.num_seqs == 0 {
            return AttentionMetadata::new();
        }

        let mut prefill_seq_lens = Vec::new();
        let mut prefill_tokens_per_seq = Vec::new();
        let mut prefill_block_tables: Vec<Vec<i32>> = Vec::new();
        let mut decode_seq_lens = Vec::new();
        let mut decode_block_tables: Vec<Vec<i32>> = Vec::new();

        for i in 0..batch.num_seqs {
            let bt_i32: Vec<i32> = batch.block_tables[i]
                .iter()
                .map(|&b| b as i32)
                .collect();

            if batch.is_prefill[i] {
                prefill_seq_lens.push(batch.seq_lens[i]);
                prefill_tokens_per_seq.push(batch.tokens_per_seq[i] as u32);
                prefill_block_tables.push(bt_i32);
            } else {
                decode_seq_lens.push(batch.seq_lens[i]);
                decode_block_tables.push(bt_i32);
            }
        }

        let max_blocks = batch.max_num_blocks_per_seq;

        // Build combined metadata.
        if prefill_seq_lens.is_empty() && decode_seq_lens.is_empty() {
            return AttentionMetadata::new();
        }

        // If only decode, use for_decode.
        if prefill_seq_lens.is_empty() {
            return AttentionMetadata::for_decode(
                decode_seq_lens,
                decode_block_tables,
                max_blocks,
            );
        }

        // If only prefill, use for_prefill.
        if decode_seq_lens.is_empty() {
            return AttentionMetadata::for_prefill(
                prefill_seq_lens,
                prefill_tokens_per_seq,
                prefill_block_tables,
                max_blocks,
            );
        }

        // Mixed batch: build combined metadata manually.
        let all_seq_lens: Vec<u32> = prefill_seq_lens
            .iter()
            .chain(decode_seq_lens.iter())
            .copied()
            .collect();
        let all_block_tables: Vec<Vec<i32>> = prefill_block_tables
            .into_iter()
            .chain(decode_block_tables)
            .collect();

        let total_prefill = prefill_tokens_per_seq.iter().map(|&t| t as usize).sum::<usize>();
        let num_decode = decode_seq_lens.len();

        // Build query_start_loc from prefill tokens_per_seq + decode (1 each).
        let mut query_start_loc = Vec::with_capacity(all_seq_lens.len() + 1);
        query_start_loc.push(0u32);
        let mut cumulative = 0u32;
        for &t in &prefill_tokens_per_seq {
            cumulative += t;
            query_start_loc.push(cumulative);
        }
        for _ in 0..num_decode {
            cumulative += 1;
            query_start_loc.push(cumulative);
        }

        AttentionMetadata {
            seq_lens: all_seq_lens,
            query_start_loc,
            block_tables: all_block_tables,
            slot_mapping: batch.slot_mappings.clone(),
            num_prefill_tokens: total_prefill,
            num_decode_tokens: num_decode,
            max_num_blocks_per_seq: max_blocks,
        }
    }

    /// Write K/V to cache during forward. (CUDA-gated, stub for now)
    pub fn write_kv_to_cache(&mut self) -> Result<()> {
        Ok(())
    }

    /// Run PagedAttention for decode. (CUDA-gated, stub for now)
    pub fn run_paged_attention(&mut self) -> Result<()> {
        Ok(())
    }

    /// Return logits for sampled positions only. (stub for now)
    pub fn return_logits(&self) -> Result<Vec<f32>> {
        Ok(vec![])
    }

    /// Cache execution model state if sampling is separated from forward.
    pub fn cache_execute_model_state(&mut self, sampled_ids: Vec<u32>) {
        self.cached_sampled_ids = sampled_ids;
    }

    /// Async copy sampled token IDs to a pinned host buffer.
    ///
    /// Returns the copied token IDs from the pinned buffer.
    pub fn async_output_copy(&mut self, sampled_ids: &[u32]) -> Result<Vec<u32>> {
        if sampled_ids.is_empty() {
            return Ok(vec![]);
        }

        // Resize pinned buffer if needed.
        let needed = std::mem::size_of_val(sampled_ids);
        if self.output_buffer.len() < needed {
            self.output_buffer = PinnedBuffer::alloc_typed::<u32>(sampled_ids.len().max(4096));
        }

        // Copy to pinned buffer (simulates async GPU→CPU copy).
        unsafe {
            let dst = self.output_buffer.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(sampled_ids.as_ptr(), dst, sampled_ids.len());
        }

        // Read back from pinned buffer.
        let result = unsafe {
            let src = self.output_buffer.as_ptr() as *const u32;
            std::slice::from_raw_parts(src, sampled_ids.len()).to_vec()
        };

        Ok(result)
    }

    /// CUDA graph capture for fixed decode batch sizes. (stub for now)
    pub fn capture_cuda_graph(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Compute slot mappings from positions, block table, and block size.
///
/// Each slot = block_id * block_size + offset_within_block,
/// where block_id = block_table[position / block_size]
/// and offset = position % block_size.
///
/// Returns -1 for positions that map to an out-of-range block table entry.
fn compute_slot_mappings(positions: &[u32], block_table: &[u32], block_size: usize) -> Vec<i64> {
    let block_size = block_size as u32;
    positions
        .iter()
        .map(|&pos| {
            let block_idx = pos / block_size;
            let offset = pos % block_size;
            if (block_idx as usize) < block_table.len() {
                block_table[block_idx as usize] as i64 * block_size as i64 + offset as i64
            } else {
                -1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_core::config::ModelConfig;
    use rllm_core::dtype::DType;
    use rllm_core::ids::{BlockId, RequestId};
    use rllm_scheduler::SchedulerStats;
    use std::collections::HashMap;

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            model_id: "test-model".to_string(),
            architecture: "LlamaForCausalLM".to_string(),
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_model_len: 4096,
            rope_theta: 10000.0,
            rope_scaling: None,
            dtype: DType::F16,
            quantization: None,
            tokenizer_mode: rllm_core::config::TokenizerMode::Auto,
        }
    }

    fn make_scheduler_output(
        new: Vec<RequestId>,
        running: Vec<RequestId>,
        num_scheduled_tokens: HashMap<RequestId, usize>,
        block_tables: HashMap<RequestId, Vec<BlockId>>,
    ) -> SchedulerOutput {
        SchedulerOutput {
            scheduled_new: new,
            scheduled_cached: Vec::new(),
            scheduled_running: running,
            num_scheduled_tokens,
            block_tables,
            token_budget_used: 0,
            preempted: Vec::new(),
            finished: Vec::new(),
            stats: SchedulerStats::default(),
        }
    }

    // ── Test 1: Model runner builds correct positions ──

    #[test]
    fn test_build_tensors_correct_positions() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        let rid1 = RequestId::new();
        let rid2 = RequestId::new();
        let rid3 = RequestId::new();

        // Add 3 requests with different prompt lengths.
        runner.add_request(rid1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]); // 10 tokens
        runner.add_request(rid2, (1..=20).collect()); // 20 tokens
        runner.add_request(rid3, vec![100, 200, 300, 400, 500]); // 5 tokens

        let block_tables = HashMap::from([
            (rid1, vec![BlockId(0)]),
            (rid2, vec![BlockId(1), BlockId(2)]),
            (rid3, vec![BlockId(3)]),
        ]);

        let num_scheduled = HashMap::from([
            (rid1, 10),
            (rid2, 20),
            (rid3, 5),
        ]);

        let output = make_scheduler_output(
            vec![rid1, rid2, rid3],
            vec![],
            num_scheduled,
            block_tables,
        );

        let batch = runner.build_tensors(&output).unwrap();

        // Positions for rid1: 0..10
        assert_eq!(
            &batch.positions[0..10],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        // Positions for rid2: 0..20
        assert_eq!(
            &batch.positions[10..30],
            &(0..20).collect::<Vec<_>>()
        );
        // Positions for rid3: 0..5
        assert_eq!(
            &batch.positions[30..35],
            &[0, 1, 2, 3, 4]
        );

        // All are prefill.
        assert!(batch.is_prefill.iter().all(|&p| p));
        assert_eq!(batch.num_prefill_tokens, 35);
        assert_eq!(batch.num_decode_tokens, 0);
        assert_eq!(batch.num_seqs, 3);
    }

    // ── Test 2: Slot mapping matches block table ──

    #[test]
    fn test_slot_mapping_matches_block_table() {
        let config = test_model_config();
        let block_size = 16;
        let mut runner = ModelRunner::new(config, block_size);

        let rid = RequestId::new();
        // 32 tokens: first 16 in block 5, next 16 in block 7.
        runner.add_request(rid, (0..32).collect());

        let block_tables = HashMap::from([(rid, vec![BlockId(5), BlockId(7)])]);
        let num_scheduled = HashMap::from([(rid, 32)]);

        let output = make_scheduler_output(
            vec![rid],
            vec![],
            num_scheduled,
            block_tables,
        );

        let batch = runner.build_tensors(&output).unwrap();

        assert_eq!(batch.slot_mappings.len(), 32);

        // First 16 tokens should map into block 5.
        for i in 0..16 {
            let expected = 5 * block_size + i;
            assert_eq!(
                batch.slot_mappings[i], expected as i64,
                "slot_mappings[{}] = {} but expected {} (block 5, offset {})",
                i, batch.slot_mappings[i], expected, i
            );
        }

        // Next 16 tokens should map into block 7.
        for i in 0..16 {
            let expected = 7 * block_size + i;
            assert_eq!(
                batch.slot_mappings[16 + i], expected as i64,
                "slot_mappings[{}] = {} but expected {} (block 7, offset {})",
                16 + i, batch.slot_mappings[16 + i], expected, i
            );
        }
    }

    // ── Test 3: Decode after prefill consistency ──

    #[test]
    fn test_decode_after_prefill_consistency() {
        let config = test_model_config();
        let block_size = 4;
        let mut runner = ModelRunner::new(config, block_size);

        let rid = RequestId::new();
        runner.add_request(rid, vec![10, 20, 30, 40, 50, 60, 70, 80]); // 8 tokens

        // Step 1: Prefill all 8 tokens.
        let block_tables = HashMap::from([(rid, vec![BlockId(0), BlockId(1)])]);
        let num_scheduled = HashMap::from([(rid, 8)]);

        let output = make_scheduler_output(
            vec![rid],
            vec![],
            num_scheduled,
            block_tables.clone(),
        );

        let batch = runner.build_tensors(&output).unwrap();

        // Verify prefill batch.
        assert_eq!(batch.token_ids, vec![10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(batch.positions, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(batch.seq_lens, vec![8]);
        assert_eq!(batch.num_prefill_tokens, 8);
        assert_eq!(batch.num_decode_tokens, 0);
        assert!(batch.is_prefill[0]);

        // Advance computed tokens after prefill.
        runner.advance_computed(&rid, 8).unwrap();
        assert_eq!(runner.num_computed(&rid), 8);

        // Store a generated token (simulating sampling output).
        runner.store_generated_token(&rid, 99).unwrap();
        assert_eq!(runner.num_computed(&rid), 9);

        // Step 2: Decode — schedule running request with 1 token.
        let block_tables2 = HashMap::from([(rid, vec![BlockId(0), BlockId(1), BlockId(2)])]);
        let num_scheduled2 = HashMap::from([(rid, 1)]);

        let output2 = make_scheduler_output(
            vec![],            // no new requests
            vec![rid],         // running
            num_scheduled2,
            block_tables2,
        );

        let batch2 = runner.build_tensors(&output2).unwrap();

        // Verify decode batch.
        assert_eq!(batch2.token_ids, vec![99]); // last generated token
        assert_eq!(batch2.positions, vec![9]); // position = num_computed (now 9)
        // seq_len = computed(9) + n_tokens(1) = 10 (total sequence length including new token)
        assert_eq!(batch2.seq_lens, vec![10]);
        assert_eq!(batch2.num_prefill_tokens, 0);
        assert_eq!(batch2.num_decode_tokens, 1);
        assert!(!batch2.is_prefill[0]); // decode

        // Slot mapping: position 9 → block_idx=9/4=2, offset=9%4=1
        // block_table[2] = BlockId(2) → slot = 2*4 + 1 = 9
        assert_eq!(batch2.slot_mappings.len(), 1);
        assert_eq!(batch2.slot_mappings[0], 9);
    }

    // ── Test 4: Async CPU copy returns valid token IDs ──

    #[test]
    fn test_async_output_copy_roundtrip() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        let tokens: Vec<u32> = vec![42, 1337, 0, 9999, 32000, 1, 2, 3];
        let copied = runner.async_output_copy(&tokens).unwrap();

        assert_eq!(copied, tokens);
    }

    #[test]
    fn test_async_output_copy_empty() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        let copied = runner.async_output_copy(&[]).unwrap();
        assert!(copied.is_empty());
    }

    #[test]
    fn test_async_output_copy_large_batch() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        // More tokens than initial buffer (4096 u32s).
        let tokens: Vec<u32> = (0..8192).collect();
        let copied = runner.async_output_copy(&tokens).unwrap();

        assert_eq!(copied, tokens);
    }

    // ── Additional unit tests ──

    #[test]
    fn test_compute_slot_mappings() {
        let positions = vec![0, 1, 2, 3, 16, 17, 18, 19];
        let block_table = vec![5u32, 7];
        let block_size = 4;

        let slots = compute_slot_mappings(&positions, &block_table, block_size);

        // pos 0-3: block_idx=0, block_table[0]=5 → 5*4+{0,1,2,3}
        assert_eq!(slots[0], 20);
        assert_eq!(slots[1], 21);
        assert_eq!(slots[2], 22);
        assert_eq!(slots[3], 23);
        // pos 16-19: block_idx=4, but block_table only has 2 entries → -1
        assert_eq!(slots[4], -1);
        assert_eq!(slots[5], -1);
    }

    #[test]
    fn test_compute_slot_mappings_with_block_boundary() {
        let positions = vec![3, 4, 7, 8]; // crossing block boundaries at block_size=4
        let block_table = vec![10u32, 20u32, 30u32];
        let block_size = 4;

        let slots = compute_slot_mappings(&positions, &block_table, block_size);

        // pos 3: block_idx=0, offset=3 → 10*4+3=43
        assert_eq!(slots[0], 43);
        // pos 4: block_idx=1, offset=0 → 20*4+0=80
        assert_eq!(slots[1], 80);
        // pos 7: block_idx=1, offset=3 → 20*4+3=83
        assert_eq!(slots[2], 83);
        // pos 8: block_idx=2, offset=0 → 30*4+0=120
        assert_eq!(slots[3], 120);
    }

    #[test]
    fn test_request_state_management() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        let rid = RequestId::new();
        assert!(!runner.has_request(&rid));

        runner.add_request(rid, vec![1, 2, 3]);
        assert!(runner.has_request(&rid));
        assert_eq!(runner.num_computed(&rid), 0);

        runner.advance_computed(&rid, 2).unwrap();
        assert_eq!(runner.num_computed(&rid), 2);

        runner.store_generated_token(&rid, 42).unwrap();
        assert_eq!(runner.num_computed(&rid), 3);

        runner.remove_request(&rid);
        assert!(!runner.has_request(&rid));
    }

    #[test]
    fn test_build_tensors_empty_output() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        let output = SchedulerOutput::empty();
        let batch = runner.build_tensors(&output).unwrap();

        assert_eq!(batch.num_seqs, 0);
        assert_eq!(batch.token_ids.len(), 0);
        assert_eq!(batch.positions.len(), 0);
    }

    #[test]
    fn test_build_tensors_cleans_finished_requests() {
        let config = test_model_config();
        let mut runner = ModelRunner::new(config, 16);

        let rid1 = RequestId::new();
        let rid2 = RequestId::new();

        runner.add_request(rid1, vec![1, 2, 3]);
        runner.add_request(rid2, vec![4, 5, 6]);

        // Output that finishes rid1.
        let mut output = SchedulerOutput::empty();
        output.finished.push(rid1);
        output.scheduled_new.push(rid2);
        output.num_scheduled_tokens.insert(rid2, 3);
        output.block_tables.insert(rid2, vec![BlockId(0)]);

        let _batch = runner.build_tensors(&output).unwrap();

        assert!(!runner.has_request(&rid1));
        assert!(runner.has_request(&rid2));
    }

    #[test]
    fn test_build_attention_metadata_mixed_batch() {
        let config = test_model_config();
        let runner = ModelRunner::new(config, 16);

        let mut batch = InputBatch::empty();
        batch.num_seqs = 3;
        batch.seq_lens = vec![10, 20, 5];
        batch.tokens_per_seq = vec![10, 20, 1];
        batch.is_prefill = vec![true, true, false];
        batch.block_tables = vec![
            vec![0u32],
            vec![1, 2],
            vec![3],
        ];
        batch.max_num_blocks_per_seq = 2;
        batch.slot_mappings = vec![0; 31]; // 10 + 20 + 1

        let meta = runner.build_attention_metadata(&batch);

        assert_eq!(meta.num_prefill_tokens, 30);
        assert_eq!(meta.num_decode_tokens, 1);
        assert_eq!(meta.seq_lens.len(), 3);
        assert_eq!(meta.query_start_loc, vec![0, 10, 30, 31]);
    }
}
