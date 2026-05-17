use rllm_core::ids::RequestId;

/// Flat input batch built from a `SchedulerOutput` for model execution.
///
/// All token arrays are concatenated across sequences in order:
/// prefill sequences first, then decode sequences.
#[derive(Debug, Clone)]
pub struct InputBatch {
    /// Request IDs for each sequence in the batch.
    pub request_ids: Vec<RequestId>,
    /// Flat token IDs: all tokens concatenated across sequences.
    pub token_ids: Vec<u32>,
    /// Flat positions: one per token in `token_ids`.
    pub positions: Vec<u32>,
    /// Per-sequence total context length (prompt + generated tokens).
    pub seq_lens: Vec<u32>,
    /// Per-sequence block tables (physical block IDs).
    pub block_tables: Vec<Vec<u32>>,
    /// Flat slot mapping: maps each token position to a physical KV cache slot.
    /// slot = block_id * block_size + offset_within_block.
    pub slot_mappings: Vec<i64>,
    /// Number of prefill tokens in this batch.
    pub num_prefill_tokens: usize,
    /// Number of decode tokens in this batch.
    pub num_decode_tokens: usize,
    /// Total number of sequences in this batch.
    pub num_seqs: usize,
    /// Maximum block table length across all sequences.
    pub max_num_blocks_per_seq: usize,
    /// Number of tokens each sequence contributes to the flat arrays.
    pub tokens_per_seq: Vec<usize>,
    /// Per-sequence flag: true = prefill, false = decode.
    pub is_prefill: Vec<bool>,
}

impl InputBatch {
    pub fn empty() -> Self {
        Self {
            request_ids: Vec::new(),
            token_ids: Vec::new(),
            positions: Vec::new(),
            seq_lens: Vec::new(),
            block_tables: Vec::new(),
            slot_mappings: Vec::new(),
            num_prefill_tokens: 0,
            num_decode_tokens: 0,
            num_seqs: 0,
            max_num_blocks_per_seq: 0,
            tokens_per_seq: Vec::new(),
            is_prefill: Vec::new(),
        }
    }
}

impl Default for InputBatch {
    fn default() -> Self {
        Self::empty()
    }
}
