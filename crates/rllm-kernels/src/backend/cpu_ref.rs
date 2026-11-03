//! CPU reference ("oracle") attention backend.
//!
//! [`CpuRefBackend::run_host`] computes exact paged attention on host memory
//! and is the numeric oracle later GPU backends validate against.

use super::{
    AttentionBackend, AttnSpec, BackendKind, HostKvLayer, KvLayout, StepMetadata, Unsupported,
};

/// CPU reference attention backend: slow, exact, layout-agnostic.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuRefBackend;

impl AttentionBackend for CpuRefBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::CpuRef
    }

    fn kv_layout(&self) -> KvLayout {
        KvLayout::Hnd
    }

    fn supports(&self, spec: &AttnSpec) -> Result<(), Unsupported> {
        if spec.head_dim == 0 {
            return Err(Unsupported {
                backend: "cpu_ref",
                reason: "head_dim must be greater than 0".to_string(),
            });
        }
        if spec.num_kv_heads == 0 {
            return Err(Unsupported {
                backend: "cpu_ref",
                reason: "num_kv_heads must be greater than 0".to_string(),
            });
        }
        if spec.num_q_heads % spec.num_kv_heads != 0 {
            return Err(Unsupported {
                backend: "cpu_ref",
                reason: format!(
                    "num_q_heads ({}) must be a multiple of num_kv_heads ({})",
                    spec.num_q_heads, spec.num_kv_heads
                ),
            });
        }
        Ok(())
    }
}

impl CpuRefBackend {
    /// Run paged attention on the host.
    ///
    /// `q` is token-major `[num_tokens, num_q_heads, head_dim]` in the same
    /// token order as `meta.base.query_start_loc`; returns the same shape.
    ///
    /// Semantics:
    /// - For sequence `i`: `q_len = qo_indptr[i+1]-qo_indptr[i]`,
    ///   `kv_len = seq_lens[i]`. Query token `j` (0-based within the sequence)
    ///   sits at absolute position `kv_len - q_len + j` and attends to KV
    ///   positions `p <= pos` (causal, so chunked prefill and decode mix in one
    ///   batch).
    /// - Sliding window `w`: additionally require `pos - p < w`.
    /// - GQA: query head `h` uses kv head `h / (num_q_heads / num_kv_heads)`.
    /// - `logits = scale * q·k`; with `soft_cap = Some(c)` apply
    ///   `c * tanh(logits / c)`; stable softmax; output = Σ p·v.
    /// - K/V are read via [`KvLayout::offset`] through the page table
    ///   (`kv_indices`) and multiplied by `k_scale`/`v_scale`
    ///   (1.0 for float caches). An empty attention set yields zeros.
    ///
    /// # Panics
    /// If `q`/`kv`/`spec` geometries disagree, `q` has the wrong length, or a
    /// page in `kv_indices` lies outside the `kv` buffers.
    pub fn run_host(
        &self,
        q: &[f32],
        kv: &HostKvLayer<'_>,
        meta: &StepMetadata,
        spec: &AttnSpec,
    ) -> Vec<f32> {
        assert_eq!(spec.block_size, kv.block_size, "spec.block_size must match kv.block_size");
        assert_eq!(spec.num_kv_heads, kv.num_kv_heads, "spec.num_kv_heads must match kv");
        assert_eq!(spec.head_dim, kv.head_dim, "spec.head_dim must match kv");
        assert!(kv.num_kv_heads > 0, "num_kv_heads must be greater than 0");
        assert_eq!(
            spec.num_q_heads % kv.num_kv_heads,
            0,
            "num_q_heads must be a multiple of num_kv_heads"
        );

        let head_dim = spec.head_dim;
        let num_q_heads = spec.num_q_heads;
        let num_kv_heads = kv.num_kv_heads;
        let block_size = kv.block_size;
        let num_tokens = meta.qo_indptr.last().copied().unwrap_or(0) as usize;
        assert_eq!(
            q.len(),
            num_tokens * num_q_heads * head_dim,
            "q must be [num_tokens, num_q_heads, head_dim]"
        );

        let mut out = vec![0.0f32; q.len()];
        if num_tokens == 0 {
            return out;
        }
        let gqa = num_q_heads / num_kv_heads;

        for seq in 0..meta.base.num_seqs() {
            let q_begin = meta.qo_indptr[seq] as usize;
            let q_end = meta.qo_indptr[seq + 1] as usize;
            let q_len = q_end - q_begin;
            let kv_len = meta.base.seq_lens[seq] as usize;
            let page_base = meta.kv_indptr[seq] as usize;

            for j in 0..q_len {
                let pos = kv_len - q_len + j;
                // Positions p satisfy p <= pos and pos - p < window.
                let win_start = spec.sliding_window.map_or(0, |w| (pos + 1).saturating_sub(w));
                let token = q_begin + j;
                let q_row = token * num_q_heads * head_dim;

                for q_head in 0..num_q_heads {
                    let kv_head = q_head / gqa;
                    let mut logits: Vec<f64> = Vec::with_capacity(pos + 1 - win_start);
                    for p in win_start..=pos {
                        let block = meta.kv_indices[page_base + p / block_size] as usize;
                        let slot = p % block_size;
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            let qv = q[q_row + q_head * head_dim + d] as f64;
                            let k_idx = kv.layout.offset(
                                block,
                                slot,
                                kv_head,
                                d,
                                block_size,
                                num_kv_heads,
                                head_dim,
                            );
                            // Index (not `get`): an out-of-range page must fail loudly in
                            // the oracle rather than silently read as zero.
                            let kv_val = kv.k[k_idx] as f64 * kv.k_scale as f64;
                            dot += qv * kv_val;
                        }
                        let mut logit = dot * spec.scale as f64;
                        if let Some(cap) = spec.soft_cap {
                            logit = (cap as f64) * (logit / (cap as f64)).tanh();
                        }
                        logits.push(logit);
                    }

                    // Numerically stable softmax (empty logits stay all-zero).
                    let mut max_logit = f64::NEG_INFINITY;
                    for &l in &logits {
                        max_logit = max_logit.max(l);
                    }
                    let mut exp_sum = 0.0f64;
                    for l in logits.iter_mut() {
                        *l = (*l - max_logit).exp();
                        exp_sum += *l;
                    }

                    let out_row = q_row + q_head * head_dim;
                    if exp_sum == 0.0 {
                        continue;
                    }
                    let mut acc = vec![0.0f64; head_dim];
                    for (i, &w) in logits.iter().enumerate() {
                        let p = win_start + i;
                        let block = meta.kv_indices[page_base + p / block_size] as usize;
                        let slot = p % block_size;
                        let weight = w / exp_sum;
                        for (d, acc_d) in acc.iter_mut().enumerate() {
                            let v_idx = kv.layout.offset(
                                block,
                                slot,
                                kv_head,
                                d,
                                block_size,
                                num_kv_heads,
                                head_dim,
                            );
                            let v_val = kv.v[v_idx] as f64 * kv.v_scale as f64;
                            *acc_d += weight * v_val;
                        }
                    }
                    for (d, &a) in acc.iter().enumerate() {
                        out[out_row + d] = a as f32;
                    }
                }
            }
        }
        out
    }

    /// Write new k/v (`[num_tokens, num_kv_heads, head_dim]`) into the paged
    /// host cache at `slot_mapping` (slot = block*block_size + offset;
    /// negative slot = skip/padding, mirroring `cache_write_kernel`).
    ///
    /// # Panics
    /// If `block_size` is 0 or `k`/`v` lengths do not match `slot_mapping`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_host(
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        layout: KvLayout,
        k: &[f32],
        v: &[f32],
        slot_mapping: &[i64],
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        assert!(block_size > 0, "block_size must be greater than 0");
        let row = num_kv_heads * head_dim;
        let num_tokens = slot_mapping.len();
        assert_eq!(k.len(), num_tokens * row, "k must be [num_tokens, kv_heads, head_dim]");
        assert_eq!(v.len(), num_tokens * row, "v must be [num_tokens, kv_heads, head_dim]");

        for (token, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 {
                continue; // padding token
            }
            let slot = slot as usize;
            let block = slot / block_size;
            let offset = slot % block_size;
            for head in 0..num_kv_heads {
                for d in 0..head_dim {
                    let dst =
                        layout.offset(block, offset, head, d, block_size, num_kv_heads, head_dim);
                    if dst >= k_cache.len() || dst >= v_cache.len() {
                        continue; // mirror the CUDA kernel's num_blocks guard
                    }
                    let src = token * row + head * head_dim + d;
                    k_cache[dst] = k[src];
                    v_cache[dst] = v[src];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rllm_core::dtype::DType;

    use super::*;
    use crate::attention::{AttentionMetadata, paged_attention_i8_reference};

    // ── deterministic RNG (no extra deps) ───────────────────────────────────

    /// splitmix64-based RNG so tests are reproducible without `rand`.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// Uniform in [0, 1).
        fn unit(&mut self) -> f32 {
            const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
            ((self.next_u64() >> 40) as u32) as f32 * SCALE
        }

        /// Uniform in [-1, 1).
        fn signed(&mut self) -> f32 {
            self.unit() * 2.0 - 1.0
        }
    }

    /// Per-(stream, position) content stream. `stream == 0` is the shared
    /// prefix, so every sequence writes identical values there.
    fn content_rng(seed: u64, stream: u64, pos: usize) -> Rng {
        let mut h = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ stream.wrapping_mul(0xbf58_476d_1ce4_e5b9)
            ^ (pos as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 29;
        Rng::new(h | 1)
    }

    // ── scenario harness ────────────────────────────────────────────────────

    /// A complete paged-attention scenario used to compare `run_host` against
    /// an independently written dense reference.
    struct Case {
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        /// (seq_len, q_len) per sequence: q_len == seq_len is full prefill,
        /// q_len == 1 is decode, otherwise chunked prefill.
        seqs: Vec<(usize, usize)>,
        /// Pages shared by every sequence (prefix caching).
        shared_pages: usize,
        window: Option<usize>,
        soft_cap: Option<f32>,
        k_scale: f32,
        v_scale: f32,
        scale: f32,
        seed: u64,
    }

    impl Case {
        fn new(
            num_q_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            block_size: usize,
            seqs: Vec<(usize, usize)>,
        ) -> Self {
            Self {
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                seqs,
                shared_pages: 0,
                window: None,
                soft_cap: None,
                k_scale: 1.0,
                v_scale: 1.0,
                scale: 1.0 / (head_dim as f32).sqrt(),
                seed: 0x5EED_CAFE,
            }
        }
    }

    /// Everything derived from a [`Case`] except the layout-specific cache.
    struct Fixture {
        spec: AttnSpec,
        q: Vec<f32>,
        meta: StepMetadata,
        k_dense: Vec<Vec<f32>>,
        v_dense: Vec<Vec<f32>>,
        q_lens: Vec<usize>,
        seq_lens: Vec<usize>,
        /// All history positions of all sequences, token-major.
        all_k: Vec<f32>,
        all_v: Vec<f32>,
        /// Physical slot for each history token.
        slots: Vec<i64>,
        num_blocks: usize,
    }

    impl Fixture {
        fn new(case: &Case) -> Self {
            let row = case.num_kv_heads * case.head_dim;
            let needed: Vec<usize> =
                case.seqs.iter().map(|&(l, _)| l.div_ceil(case.block_size)).collect();
            let shared = case.shared_pages.min(needed.iter().copied().min().unwrap_or(0));
            let unique_total: usize = needed.iter().map(|&n| n - shared).sum();

            // Deterministic shuffle so physical page ids are non-contiguous.
            let mut pool: Vec<i32> = (shared as i32..(shared + unique_total) as i32).collect();
            let mut shuffle = Rng::new(case.seed ^ 0x00DD_BA11);
            for i in (1..pool.len()).rev() {
                let j = (shuffle.next_u64() as usize) % (i + 1);
                pool.swap(i, j);
            }

            let mut tables = Vec::with_capacity(case.seqs.len());
            let mut cursor = 0usize;
            for &n in &needed {
                let mut table = Vec::with_capacity(n + 1);
                for b in 0..shared {
                    table.push(b as i32);
                }
                for _ in shared..n {
                    table.push(pool[cursor]);
                    cursor += 1;
                }
                if n > 0 {
                    table.push(-1); // unused tail, ignored by validation
                }
                tables.push(table);
            }

            // Dense per-sequence history plus flattened write buffers.
            let total_pos: usize = case.seqs.iter().map(|&(l, _)| l).sum();
            let mut all_k = vec![0.0f32; total_pos * row];
            let mut all_v = vec![0.0f32; total_pos * row];
            let mut slots = vec![0i64; total_pos];
            let mut k_dense = Vec::with_capacity(case.seqs.len());
            let mut v_dense = Vec::with_capacity(case.seqs.len());
            let mut token = 0usize;
            for (i, &(seq_len, _)) in case.seqs.iter().enumerate() {
                let start = token;
                for p in 0..seq_len {
                    let phys = tables[i][p / case.block_size] as i64;
                    slots[token] = phys * case.block_size as i64 + (p % case.block_size) as i64;
                    // Shared prefix pages hold identical content everywhere.
                    let stream = if p < shared * case.block_size { 0 } else { i as u64 + 1 };
                    let mut rng = content_rng(case.seed, stream, p);
                    for h in 0..case.num_kv_heads {
                        for d in 0..case.head_dim {
                            let idx = token * row + h * case.head_dim + d;
                            all_k[idx] = rng.signed();
                            all_v[idx] = rng.signed();
                        }
                    }
                    token += 1;
                }
                k_dense.push(all_k[start * row..token * row].to_vec());
                v_dense.push(all_v[start * row..token * row].to_vec());
            }

            // Queries.
            let total_q: usize = case.seqs.iter().map(|&(_, q)| q).sum();
            let mut q_rng = Rng::new(case.seed ^ 0x9A17);
            let mut q = vec![0.0f32; total_q * case.num_q_heads * case.head_dim];
            for x in &mut q {
                *x = q_rng.signed();
            }

            let q_lens: Vec<usize> = case.seqs.iter().map(|&(_, q)| q).collect();
            let seq_lens: Vec<usize> = case.seqs.iter().map(|&(l, _)| l).collect();
            let base = build_base(&seq_lens, &q_lens, tables, case.window);
            let meta = StepMetadata::from_attention_metadata(base, case.block_size)
                .expect("fixture metadata must be valid");
            let spec = AttnSpec {
                num_q_heads: case.num_q_heads,
                num_kv_heads: case.num_kv_heads,
                head_dim: case.head_dim,
                block_size: case.block_size,
                q_dtype: DType::F32,
                kv_dtype: DType::F32,
                sliding_window: case.window,
                scale: case.scale,
                soft_cap: case.soft_cap,
                compute_capability: None,
            };

            Self {
                spec,
                q,
                meta,
                k_dense,
                v_dense,
                q_lens,
                seq_lens,
                all_k,
                all_v,
                slots,
                num_blocks: shared + unique_total,
            }
        }
    }

    /// Assemble an [`AttentionMetadata`] the way a scheduler step would.
    fn build_base(
        seq_lens: &[usize],
        q_lens: &[usize],
        block_tables: Vec<Vec<i32>>,
        window: Option<usize>,
    ) -> AttentionMetadata {
        let mut query_start_loc = Vec::with_capacity(q_lens.len() + 1);
        query_start_loc.push(0u32);
        for &q in q_lens {
            let prev = query_start_loc.last().copied().unwrap_or(0);
            query_start_loc.push(prev + q as u32);
        }
        let num_decode = q_lens.iter().filter(|&&q| q == 1).count();
        let num_prefill: usize = q_lens.iter().filter(|&&q| q > 1).sum();
        let max_num_blocks_per_seq = block_tables.iter().map(Vec::len).max().unwrap_or(0);
        let mut base = AttentionMetadata {
            seq_lens: seq_lens.iter().map(|&l| l as u32).collect(),
            query_start_loc,
            block_tables,
            slot_mapping: Vec::new(),
            num_prefill_tokens: num_prefill,
            num_decode_tokens: num_decode,
            max_num_blocks_per_seq,
            common_prefix_blocks: 0,
            sliding_window: window,
        };
        base.detect_common_prefix_blocks();
        base
    }

    /// Dense, independently written naive attention: contiguous K/V per
    /// sequence, no paging, no `KvLayout` — f64 throughout.
    #[allow(clippy::too_many_arguments)]
    fn dense_reference(
        q: &[f32],
        k_seqs: &[Vec<f32>],
        v_seqs: &[Vec<f32>],
        q_lens: &[usize],
        seq_lens: &[usize],
        spec: &AttnSpec,
        k_scale: f32,
        v_scale: f32,
    ) -> Vec<f32> {
        let head_dim = spec.head_dim;
        let heads_per_kv = spec.num_q_heads / spec.num_kv_heads;
        let mut out = vec![0.0f32; q.len()];
        let mut token_base = 0usize;
        for (i, &q_len) in q_lens.iter().enumerate() {
            let kv_len = seq_lens[i];
            let k_seq = &k_seqs[i];
            let v_seq = &v_seqs[i];
            for j in 0..q_len {
                let pos = kv_len - q_len + j;
                let win_start = spec.sliding_window.map_or(0, |w| (pos + 1).saturating_sub(w));
                let token = token_base + j;
                for q_head in 0..spec.num_q_heads {
                    let kv_head = q_head / heads_per_kv;
                    let mut logits = Vec::with_capacity(pos + 1 - win_start);
                    for p in win_start..=pos {
                        let row = (p * spec.num_kv_heads + kv_head) * head_dim;
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            let qv = q[(token * spec.num_q_heads + q_head) * head_dim + d];
                            let kv = k_seq[row + d] as f64 * k_scale as f64;
                            dot += qv as f64 * kv;
                        }
                        let mut logit = dot * spec.scale as f64;
                        if let Some(cap) = spec.soft_cap {
                            logit = (cap as f64) * (logit / (cap as f64)).tanh();
                        }
                        logits.push(logit);
                    }
                    let mut max_logit = f64::NEG_INFINITY;
                    for &l in &logits {
                        max_logit = max_logit.max(l);
                    }
                    let mut exp_sum = 0.0f64;
                    for l in logits.iter_mut() {
                        *l = (*l - max_logit).exp();
                        exp_sum += *l;
                    }
                    let out_base = (token * spec.num_q_heads + q_head) * head_dim;
                    if exp_sum == 0.0 {
                        continue;
                    }
                    let mut acc = vec![0.0f64; head_dim];
                    for (n, &w) in logits.iter().enumerate() {
                        let p = win_start + n;
                        let row = (p * spec.num_kv_heads + kv_head) * head_dim;
                        let weight = w / exp_sum;
                        for (d, acc_d) in acc.iter_mut().enumerate() {
                            let vv = v_seq[row + d] as f64 * v_scale as f64;
                            *acc_d += weight * vv;
                        }
                    }
                    for (d, &a) in acc.iter().enumerate() {
                        out[out_base + d] = a as f32;
                    }
                }
            }
            token_base += q_len;
        }
        out
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff <= tol,
                "{ctx}: idx {i}: run_host={x} reference={y} (diff {diff} > {tol})"
            );
        }
    }

    /// Write the fixture through `write_kv_host` for every [`KvLayout`], run
    /// `run_host`, and compare each layout against the dense reference and
    /// against each other. Returns the Legacy output.
    fn check_case(case: &Case) -> Vec<f32> {
        let fx = Fixture::new(case);
        let expected = dense_reference(
            &fx.q,
            &fx.k_dense,
            &fx.v_dense,
            &fx.q_lens,
            &fx.seq_lens,
            &fx.spec,
            case.k_scale,
            case.v_scale,
        );

        let mut outputs = Vec::with_capacity(3);
        for layout in [KvLayout::Legacy, KvLayout::Nhd, KvLayout::Hnd] {
            let elems: usize = layout
                .shape(fx.num_blocks, case.block_size, case.num_kv_heads, case.head_dim)
                .iter()
                .product();
            let mut k_cache = vec![0.0f32; elems];
            let mut v_cache = vec![0.0f32; elems];
            CpuRefBackend::write_kv_host(
                &mut k_cache,
                &mut v_cache,
                layout,
                &fx.all_k,
                &fx.all_v,
                &fx.slots,
                case.block_size,
                case.num_kv_heads,
                case.head_dim,
            );
            let kv = HostKvLayer {
                k: &k_cache,
                v: &v_cache,
                layout,
                num_blocks: fx.num_blocks,
                block_size: case.block_size,
                num_kv_heads: case.num_kv_heads,
                head_dim: case.head_dim,
                k_scale: case.k_scale,
                v_scale: case.v_scale,
            };
            let out = CpuRefBackend.run_host(&fx.q, &kv, &fx.meta, &fx.spec);
            assert_close(&out, &expected, 1e-5, &format!("{layout:?} vs dense"));
            outputs.push(out);
        }
        for (i, out) in outputs.iter().enumerate().skip(1) {
            assert_close(out, &outputs[0], 1e-6, &format!("layout {i} vs layout 0"));
        }
        outputs.remove(0)
    }

    // ── scenario tests ──────────────────────────────────────────────────────

    #[test]
    fn pure_prefill_matches_dense() {
        let case = Case::new(4, 4, 16, 4, vec![(16, 16), (9, 9)]);
        check_case(&case);
    }

    #[test]
    fn pure_decode_matches_dense() {
        let case = Case::new(4, 2, 8, 16, vec![(17, 1), (5, 1)]);
        check_case(&case);
    }

    #[test]
    fn mixed_decode_and_chunked_prefill_matches_dense() {
        // seq0 decode, seq1 chunked prefill (5 of 14), seq2 full prefill.
        let case = Case::new(8, 2, 16, 4, vec![(6, 1), (14, 5), (3, 3)]);
        check_case(&case);
    }

    #[test]
    fn gqa_ratios_match_dense() {
        for (num_q, num_kv) in [(4, 4), (8, 2), (8, 1)] {
            let case = Case::new(num_q, num_kv, 8, 4, vec![(11, 1), (13, 6)]);
            check_case(&case);
        }
    }

    #[test]
    fn head_dim_and_block_size_grid_matches_dense() {
        for &head_dim in &[8usize, 64] {
            for &block_size in &[1usize, 4, 16] {
                let case = Case::new(4, 2, head_dim, block_size, vec![(7, 2), (5, 5)]);
                check_case(&case);
            }
        }
    }

    #[test]
    fn shuffled_blocks_and_shared_prefix_match_dense() {
        let mut case = Case::new(4, 2, 16, 4, vec![(20, 1), (14, 4), (9, 9)]);
        case.shared_pages = 2;
        case.seed = 0xABCD_1234;
        check_case(&case);
    }

    #[test]
    fn sliding_window_soft_cap_and_scales_match_dense() {
        let mut case = Case::new(4, 4, 8, 4, vec![(23, 1), (12, 7)]);
        case.window = Some(4);
        case.soft_cap = Some(30.0);
        case.k_scale = 0.5;
        case.v_scale = 2.0;
        check_case(&case);
    }

    #[test]
    fn zero_sliding_window_returns_zeros() {
        // Empty attention set (pos - p < 0 is impossible) → zeros everywhere.
        let mut case = Case::new(2, 2, 8, 4, vec![(5, 2)]);
        case.window = Some(0);
        let out = check_case(&case);
        assert!(out.iter().all(|&x| x == 0.0), "expected zeros, got {out:?}");
    }

    #[test]
    fn layouts_produce_identical_outputs() {
        // check_case asserts cross-layout equality; pin it on a shuffled,
        // shared-prefix batch with a non-trivial feature mix.
        let mut case = Case::new(4, 2, 8, 4, vec![(12, 3), (9, 1)]);
        case.shared_pages = 1;
        let out = check_case(&case);
        assert!(!out.is_empty());
    }

    #[test]
    fn empty_batch_returns_empty_output() {
        let case = Case::new(2, 2, 8, 4, vec![]);
        let out = check_case(&case);
        assert!(out.is_empty());
    }

    // ── agreement with the existing INT8 oracle ─────────────────────────────

    #[test]
    fn i8_reference_agrees_with_run_host() {
        // Single-sequence decode, GQA ratio 2, INT8 KV with non-unit scales.
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let block_size = 4usize;
        let kv_len = 13usize;
        let k_scale = 0.05f32;
        let v_scale = 0.2f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let row = num_kv_heads * head_dim;

        let mut rng = Rng::new(0x188E5);
        let mut key_i8 = vec![0i8; kv_len * row];
        let mut val_i8 = vec![0i8; kv_len * row];
        for x in &mut key_i8 {
            *x = (rng.next_u64() % 128) as i8 - 64;
        }
        for x in &mut val_i8 {
            *x = (rng.next_u64() % 128) as i8 - 64;
        }

        // Paged f32 cache holding raw i8 values; HostKvLayer applies scales.
        let layout = KvLayout::Hnd;
        let num_blocks = kv_len.div_ceil(block_size);
        let elems: usize =
            layout.shape(num_blocks, block_size, num_kv_heads, head_dim).iter().product();
        let mut k_cache = vec![0.0f32; elems];
        let mut v_cache = vec![0.0f32; elems];
        let k_f32: Vec<f32> = key_i8.iter().map(|&x| x as f32).collect();
        let v_f32: Vec<f32> = val_i8.iter().map(|&x| x as f32).collect();
        let slots: Vec<i64> = (0..kv_len as i64).collect();
        CpuRefBackend::write_kv_host(
            &mut k_cache,
            &mut v_cache,
            layout,
            &k_f32,
            &v_f32,
            &slots,
            block_size,
            num_kv_heads,
            head_dim,
        );

        // Single decode token.
        let mut q = vec![0.0f32; num_q_heads * head_dim];
        for x in &mut q {
            *x = rng.signed();
        }
        let pages: Vec<i32> = (0..num_blocks as i32).collect();
        let base = AttentionMetadata::for_decode(vec![kv_len as u32], vec![pages], num_blocks);
        let meta = StepMetadata::from_attention_metadata(base, block_size).expect("valid");
        let spec = AttnSpec {
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            q_dtype: DType::F32,
            kv_dtype: DType::INT8,
            sliding_window: None,
            scale,
            soft_cap: None,
            compute_capability: None,
        };
        let kv = HostKvLayer {
            k: &k_cache,
            v: &v_cache,
            layout,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            k_scale,
            v_scale,
        };
        let out = CpuRefBackend.run_host(&q, &kv, &meta, &spec);

        for q_head in 0..num_q_heads {
            let kv_head = q_head * num_kv_heads / num_q_heads;
            let k_rows: Vec<i8> = (0..kv_len)
                .flat_map(|p| key_i8[p * row + kv_head * head_dim..][..head_dim].iter().copied())
                .collect();
            let v_rows: Vec<i8> = (0..kv_len)
                .flat_map(|p| val_i8[p * row + kv_head * head_dim..][..head_dim].iter().copied())
                .collect();
            let expected = paged_attention_i8_reference(
                &q[q_head * head_dim..][..head_dim],
                &k_rows,
                &v_rows,
                kv_len,
                head_dim,
                k_scale,
                v_scale,
                scale,
            );
            assert_close(
                &out[q_head * head_dim..][..head_dim],
                &expected,
                1e-5,
                &format!("q_head {q_head} vs paged_attention_i8_reference"),
            );
        }
    }
}
