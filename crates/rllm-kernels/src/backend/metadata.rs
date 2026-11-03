//! Per-step attention metadata (CSR page lists) shared by all backends.

use crate::attention::AttentionMetadata;

/// Error returned by [`StepMetadata::from_attention_metadata`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MetadataError {
    /// `block_size` must be positive so page lists can be built.
    #[error("block_size must be greater than 0")]
    InvalidBlockSize,
    /// `query_start_loc[0]` must be 0.
    #[error("query_start_loc must start at 0")]
    QueryStartLocNotZero,
    /// `query_start_loc` must be non-decreasing.
    #[error("query_start_loc must be non-decreasing")]
    QueryStartLocNotMonotonic,
    /// `query_start_loc` must have `num_seqs + 1` entries.
    #[error("query_start_loc length {len} != num_seqs + 1 ({expected})")]
    QueryStartLocLength { len: usize, expected: usize },
    /// `block_tables` must have one row per sequence.
    #[error("block_tables length {len} != seq_lens length {expected}")]
    BlockTablesLength { len: usize, expected: usize },
    /// A sequence's query length may not exceed its `seq_len`.
    #[error("seq {seq}: query length {q_len} > seq_len {seq_len}")]
    QueryLongerThanSeq { seq: usize, q_len: usize, seq_len: usize },
    /// Each sequence needs `ceil(seq_len / block_size)` usable block ids.
    #[error("seq {seq}: {len} block ids < {needed} pages for seq_len {seq_len}")]
    NotEnoughBlocks { seq: usize, len: usize, needed: usize, seq_len: usize },
    /// The first `ceil(seq_len / block_size)` block ids must be non-negative.
    #[error("seq {seq}: block id {block} at index {index} is negative")]
    NegativeBlockId { seq: usize, index: usize, block: i32 },
}

/// Per-step attention metadata shared by all layers. Built once per scheduler step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepMetadata {
    /// Original batch metadata (seq lens, block tables, slot mapping, …).
    pub base: AttentionMetadata,
    /// = base.query_start_loc as i32, length num_seqs+1.
    pub qo_indptr: Vec<i32>,
    /// CSR over used pages: kv_indptr[i+1]-kv_indptr[i] = ceil(seq_lens[i]/block_size).
    pub kv_indptr: Vec<i32>,
    /// Concatenated physical block ids (only the used pages, never -1).
    pub kv_indices: Vec<i32>,
    /// Number of valid tokens in each sequence's last page
    /// (1..=block_size; block_size when seq_len % block_size == 0; 0 only when seq_len == 0).
    pub kv_last_page_len: Vec<i32>,
    /// Longest query length in the batch (0 for an empty batch).
    pub max_q_len: usize,
    /// Longest KV (sequence) length in the batch (0 for an empty batch).
    pub max_kv_len: usize,
    /// true when every sequence has exactly one query token
    /// (vacuously true for an empty batch).
    pub is_pure_decode: bool,
}

impl StepMetadata {
    /// Build per-step CSR metadata from an [`AttentionMetadata`] batch.
    ///
    /// `block_size` must match the cache geometry the batch will run against.
    /// Validation rejects non-monotonic / non-zero-starting `query_start_loc`,
    /// mismatched `seq_lens`/`block_tables` lengths, query lengths beyond the
    /// sequence length, and sequences without enough non-negative block ids for
    /// `ceil(seq_len / block_size)` pages.
    pub fn from_attention_metadata(
        base: AttentionMetadata,
        block_size: usize,
    ) -> Result<Self, MetadataError> {
        if block_size == 0 {
            return Err(MetadataError::InvalidBlockSize);
        }
        let num_seqs = base.num_seqs();
        let expected_qo = num_seqs + 1;
        if base.query_start_loc.len() != expected_qo {
            return Err(MetadataError::QueryStartLocLength {
                len: base.query_start_loc.len(),
                expected: expected_qo,
            });
        }
        if base.query_start_loc.first() != Some(&0) {
            return Err(MetadataError::QueryStartLocNotZero);
        }
        if base.block_tables.len() != num_seqs {
            return Err(MetadataError::BlockTablesLength {
                len: base.block_tables.len(),
                expected: num_seqs,
            });
        }
        if base.query_start_loc.windows(2).any(|w| w[1] < w[0]) {
            return Err(MetadataError::QueryStartLocNotMonotonic);
        }

        let mut qo_indptr = Vec::with_capacity(expected_qo);
        let mut kv_indptr = Vec::with_capacity(expected_qo);
        let mut kv_indices = Vec::new();
        let mut kv_last_page_len = Vec::with_capacity(num_seqs);
        let mut max_q_len = 0usize;
        let mut max_kv_len = 0usize;
        let mut is_pure_decode = true;
        let mut kv_pages = 0i32;

        qo_indptr.push(0);
        kv_indptr.push(0);

        for seq in 0..num_seqs {
            let seq_len = base.seq_lens[seq] as usize;
            let q_len = (base.query_start_loc[seq + 1] - base.query_start_loc[seq]) as usize;
            if q_len > seq_len {
                return Err(MetadataError::QueryLongerThanSeq { seq, q_len, seq_len });
            }
            let needed = seq_len.div_ceil(block_size);
            let row = &base.block_tables[seq];
            if row.len() < needed {
                return Err(MetadataError::NotEnoughBlocks {
                    seq,
                    len: row.len(),
                    needed,
                    seq_len,
                });
            }
            for (index, &block) in row[..needed].iter().enumerate() {
                if block < 0 {
                    return Err(MetadataError::NegativeBlockId { seq, index, block });
                }
            }
            kv_indices.extend_from_slice(&row[..needed]);
            kv_last_page_len.push(if seq_len == 0 {
                0
            } else {
                let rem = (seq_len % block_size) as i32;
                if rem == 0 { block_size as i32 } else { rem }
            });
            max_q_len = max_q_len.max(q_len);
            max_kv_len = max_kv_len.max(seq_len);
            is_pure_decode &= q_len == 1;
            qo_indptr.push(base.query_start_loc[seq + 1] as i32);
            kv_pages += needed as i32;
            kv_indptr.push(kv_pages);
        }

        Ok(Self {
            base,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            max_q_len,
            max_kv_len,
            is_pure_decode,
        })
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Assemble a plausible [`AttentionMetadata`] from per-sequence lengths and
    /// query lengths (the same fields a scheduler step would produce).
    fn base_metadata(
        seq_lens: &[u32],
        q_lens: &[u32],
        block_tables: Vec<Vec<i32>>,
    ) -> AttentionMetadata {
        let mut query_start_loc = Vec::with_capacity(q_lens.len() + 1);
        query_start_loc.push(0u32);
        for &q in q_lens {
            let prev = query_start_loc.last().copied().unwrap_or(0);
            query_start_loc.push(prev + q);
        }
        let num_decode = q_lens.iter().filter(|&&q| q == 1).count();
        let num_prefill: usize = q_lens.iter().filter(|&&q| q > 1).map(|&q| q as usize).sum();
        let max_num_blocks_per_seq = block_tables.iter().map(Vec::len).max().unwrap_or(0);
        AttentionMetadata {
            seq_lens: seq_lens.to_vec(),
            query_start_loc,
            block_tables,
            slot_mapping: Vec::new(),
            num_prefill_tokens: num_prefill,
            num_decode_tokens: num_decode,
            max_num_blocks_per_seq,
            common_prefix_blocks: 0,
            sliding_window: None,
        }
    }

    #[test]
    fn builds_csr_for_mixed_batch() {
        // seq0: decode (q_len 1, 3 tokens, 1 page of size 4)
        // seq1: chunked prefill (q_len 2 of 6, 2 pages)
        let base = base_metadata(&[3, 6], &[1, 2], vec![vec![7], vec![3, 5]]);
        let meta = StepMetadata::from_attention_metadata(base, 4).expect("valid batch");
        assert_eq!(meta.qo_indptr, vec![0, 1, 3]);
        assert_eq!(meta.kv_indptr, vec![0, 1, 3]);
        assert_eq!(meta.kv_indices, vec![7, 3, 5]);
        assert_eq!(meta.kv_last_page_len, vec![3, 2]);
        assert_eq!(meta.max_q_len, 2);
        assert_eq!(meta.max_kv_len, 6);
        assert!(!meta.is_pure_decode);
    }

    #[test]
    fn pure_decode_flag_and_page_boundary_lengths() {
        let base = base_metadata(&[4, 8], &[1, 1], vec![vec![0], vec![1, 2]]);
        let meta = StepMetadata::from_attention_metadata(base, 4).expect("valid batch");
        assert!(meta.is_pure_decode);
        // seq_len % block_size == 0 → last page is full.
        assert_eq!(meta.kv_last_page_len, vec![4, 4]);

        let base = base_metadata(&[0], &[0], vec![vec![]]);
        let meta = StepMetadata::from_attention_metadata(base, 4).expect("valid batch");
        assert_eq!(meta.kv_last_page_len, vec![0]);
        assert!(!meta.is_pure_decode);
        assert_eq!(meta.max_q_len, 0);
        assert_eq!(meta.max_kv_len, 0);
    }

    #[test]
    fn rejects_zero_block_size() {
        let base = base_metadata(&[4], &[4], vec![vec![0]]);
        let err = StepMetadata::from_attention_metadata(base, 0).unwrap_err();
        assert_eq!(err, MetadataError::InvalidBlockSize);
    }

    #[test]
    fn rejects_query_start_loc_not_zero() {
        let mut base = base_metadata(&[4], &[4], vec![vec![0]]);
        base.query_start_loc = vec![1, 5];
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::QueryStartLocNotZero);
    }

    #[test]
    fn rejects_query_start_loc_not_monotonic() {
        let mut base = base_metadata(&[5, 5], &[5, 3], vec![vec![0], vec![1]]);
        base.query_start_loc = vec![0, 5, 3];
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::QueryStartLocNotMonotonic);
    }

    #[test]
    fn rejects_query_start_loc_wrong_length() {
        let mut base = base_metadata(&[4, 4], &[4, 4], vec![vec![0], vec![1]]);
        base.query_start_loc = vec![0, 4];
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::QueryStartLocLength { len: 2, expected: 3 });
    }

    #[test]
    fn rejects_block_tables_length_mismatch() {
        // seq_lens.len() == 2 but block_tables.len() == 1
        let base = base_metadata(&[4, 4], &[4, 4], vec![vec![0]]);
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::BlockTablesLength { len: 1, expected: 2 });
    }

    #[test]
    fn rejects_query_longer_than_seq() {
        let base = base_metadata(&[3], &[5], vec![vec![0]]);
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::QueryLongerThanSeq { seq: 0, q_len: 5, seq_len: 3 });
    }

    #[test]
    fn rejects_not_enough_blocks() {
        // seq_len 10, block_size 4 → 3 pages needed, only 2 provided.
        let base = base_metadata(&[10], &[10], vec![vec![0, 1]]);
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::NotEnoughBlocks { seq: 0, len: 2, needed: 3, seq_len: 10 });
    }

    #[test]
    fn rejects_negative_block_in_used_pages() {
        // seq_len 8, block_size 4 → 2 pages needed; second id is -1 padding.
        let base = base_metadata(&[8], &[8], vec![vec![0, -1]]);
        let err = StepMetadata::from_attention_metadata(base, 4).unwrap_err();
        assert_eq!(err, MetadataError::NegativeBlockId { seq: 0, index: 1, block: -1 });
    }

    #[test]
    fn allows_negative_padding_beyond_used_pages() {
        // seq_len 4 → only the first page is used; trailing -1 is ignored.
        let base = base_metadata(&[4], &[4], vec![vec![2, -1, -1]]);
        let meta = StepMetadata::from_attention_metadata(base, 4).expect("valid batch");
        assert_eq!(meta.kv_indices, vec![2]);
        assert_eq!(meta.kv_indptr, vec![0, 1]);
    }

    // ── proptest: CSR invariants over random batches ────────────────────────

    /// Random batch: block_size ∈ {1,4,16}, 1–8 sequences, seq_len 0–100,
    /// q_len 1..=seq_len (0 when seq_len == 0), plus flat non-negative block
    /// ids covering exactly the used pages.
    fn arb_batch() -> impl Strategy<Value = (usize, Vec<u32>, Vec<u32>, Vec<i32>)> {
        (any::<u8>(), proptest::collection::vec((0u32..=100, 0u32..=100), 1..=8)).prop_flat_map(
            |(bs_raw, pairs)| {
                let block_size = [1usize, 4, 16][usize::from(bs_raw % 3)];
                let seq_lens: Vec<u32> = pairs.iter().map(|&(l, _)| l).collect();
                let q_lens: Vec<u32> =
                    pairs.iter().map(|&(l, f)| if l == 0 { 0 } else { f % l + 1 }).collect();
                let total_pages: usize =
                    seq_lens.iter().map(|&l| (l as usize).div_ceil(block_size)).sum();
                (
                    Just(block_size),
                    Just(seq_lens),
                    Just(q_lens),
                    proptest::collection::vec(0i32..32, total_pages..=total_pages),
                )
            },
        )
    }

    proptest::proptest! {
        #[test]
        fn csr_invariants_hold_for_random_batches(
            (block_size, seq_lens, q_lens, flat_blocks) in arb_batch()
        ) {
            let num_seqs = seq_lens.len();
            let needed: Vec<usize> = seq_lens
                .iter()
                .map(|&l| (l as usize).div_ceil(block_size))
                .collect();
            let total_pages: usize = needed.iter().sum();

            // Chunk the flat id pool into per-sequence rows and append -1
            // padding beyond the used pages (ignored by validation).
            let mut tables = Vec::with_capacity(num_seqs);
            let mut cursor = 0usize;
            for &n in &needed {
                let mut row: Vec<i32> = flat_blocks[cursor..cursor + n].to_vec();
                cursor += n;
                row.extend([-1, -1]);
                tables.push(row);
            }

            let base = base_metadata(&seq_lens, &q_lens, tables);
            let meta = StepMetadata::from_attention_metadata(base, block_size)
                .expect("generated batch must be valid");

            // indptr shape / monotonicity / CSR arithmetic
            prop_assert_eq!(meta.qo_indptr.len(), num_seqs + 1);
            prop_assert_eq!(meta.kv_indptr.len(), num_seqs + 1);
            prop_assert_eq!(meta.qo_indptr[0], 0);
            prop_assert_eq!(meta.kv_indptr[0], 0);
            prop_assert!(meta.qo_indptr.windows(2).all(|w| w[0] <= w[1]));
            prop_assert!(meta.kv_indptr.windows(2).all(|w| w[0] <= w[1]));
            prop_assert_eq!(meta.kv_indices.len() as i32, meta.kv_indptr[num_seqs]);
            prop_assert_eq!(meta.kv_indptr[num_seqs] as usize, total_pages);

            // qo_indptr mirrors base.query_start_loc
            let expected_qo: Vec<i32> =
                meta.base.query_start_loc.iter().map(|&x| x as i32).collect();
            prop_assert_eq!(&meta.qo_indptr, &expected_qo);

            // kv_last_page_len range and formula
            for (i, &sl_u32) in seq_lens.iter().enumerate() {
                let sl = sl_u32 as usize;
                let expected_last = if sl == 0 {
                    0
                } else if sl % block_size == 0 {
                    block_size as i32
                } else {
                    (sl % block_size) as i32
                };
                prop_assert_eq!(meta.kv_last_page_len[i], expected_last);
                prop_assert!(meta.kv_last_page_len[i] >= 0);
                prop_assert!(meta.kv_last_page_len[i] as usize <= block_size);
            }

            // kv_indices matches the block_tables prefix of each sequence
            let mut idx = 0usize;
            for (i, &n) in needed.iter().enumerate() {
                prop_assert_eq!(
                    &meta.kv_indices[idx..idx + n],
                    &meta.base.block_tables[i][..n]
                );
                prop_assert!(meta.base.block_tables[i][..n].iter().all(|&b| b >= 0));
                prop_assert_eq!(
                    meta.kv_indptr[i + 1] - meta.kv_indptr[i],
                    n as i32
                );
                idx += n;
            }

            // is_pure_decode / max lengths
            prop_assert_eq!(meta.is_pure_decode, q_lens.iter().all(|&q| q == 1));
            prop_assert_eq!(
                meta.max_q_len,
                q_lens.iter().map(|&q| q as usize).max().unwrap_or(0)
            );
            prop_assert_eq!(
                meta.max_kv_len,
                seq_lens.iter().map(|&l| l as usize).max().unwrap_or(0)
            );
        }
    }
}
