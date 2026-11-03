//! Physical layouts of one layer's paged K/V cache tensor.

/// Physical layout of one layer's K (or V) cache tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvLayout {
    /// Current rLLM kernels: `[num_blocks, num_kv_heads, head_dim, block_size]`.
    Legacy,
    /// `[num_blocks, block_size, num_kv_heads, head_dim]` (FlashAttention).
    Nhd,
    /// `[num_blocks, num_kv_heads, block_size, head_dim]` (FlashInfer, TRTLLM-GEN, Metal).
    Hnd,
}

impl KvLayout {
    /// Element offset of `(block, slot, head, d)` inside one layer tensor.
    ///
    /// Formulas (row-major, outermost dimension first):
    /// - `Legacy`: `((block * num_kv_heads + head) * head_dim + d) * block_size + slot`.
    ///   This matches `cuda/cache_ops.cu::cache_write_kernel` (`cache_idx`) and
    ///   `cuda/paged_attention.cu` (`k_idx` / `v_idx`), which both index
    ///   `[num_blocks, num_kv_heads, head_dim, block_size]` (the CUDA comments
    ///   label this "NHD"; the plan names it `Legacy`).
    /// - `Nhd`: `((block * block_size + slot) * num_kv_heads + head) * head_dim + d`.
    /// - `Hnd`: `((block * num_kv_heads + head) * block_size + slot) * head_dim + d`.
    #[allow(clippy::too_many_arguments)]
    pub fn offset(
        self,
        block: usize,
        slot: usize,
        head: usize,
        d: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> usize {
        match self {
            KvLayout::Legacy => ((block * num_kv_heads + head) * head_dim + d) * block_size + slot,
            KvLayout::Nhd => ((block * block_size + slot) * num_kv_heads + head) * head_dim + d,
            KvLayout::Hnd => ((block * num_kv_heads + head) * block_size + slot) * head_dim + d,
        }
    }

    /// Shape as `[d0, d1, d2, d3]` for a given geometry.
    pub fn shape(
        self,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> [usize; 4] {
        match self {
            KvLayout::Legacy => [num_blocks, num_kv_heads, head_dim, block_size],
            KvLayout::Nhd => [num_blocks, block_size, num_kv_heads, head_dim],
            KvLayout::Hnd => [num_blocks, num_kv_heads, block_size, head_dim],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bijection(
        layout: KvLayout,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let total = num_blocks * block_size * num_kv_heads * head_dim;
        let shape = layout.shape(num_blocks, block_size, num_kv_heads, head_dim);
        assert_eq!(
            shape.iter().product::<usize>(),
            total,
            "{layout:?} shape {shape:?} product != {total}"
        );
        let mut seen = vec![false; total];
        for block in 0..num_blocks {
            for slot in 0..block_size {
                for head in 0..num_kv_heads {
                    for d in 0..head_dim {
                        let off =
                            layout.offset(block, slot, head, d, block_size, num_kv_heads, head_dim);
                        assert!(off < total, "{layout:?} offset {off} out of range {total}");
                        assert!(
                            !seen[off],
                            "{layout:?} duplicate offset {off} for \
                             (block={block}, slot={slot}, head={head}, d={d})"
                        );
                        seen[off] = true;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "{layout:?} not all offsets covered for geometry \
             ({num_blocks}, {block_size}, {num_kv_heads}, {head_dim})"
        );
    }

    #[test]
    fn offsets_are_bijection_for_small_geometries() {
        for layout in [KvLayout::Legacy, KvLayout::Nhd, KvLayout::Hnd] {
            for &(nb, bs, nh, hd) in
                &[(1, 1, 1, 1), (2, 4, 2, 8), (3, 1, 4, 4), (2, 16, 1, 64), (4, 4, 3, 12)]
            {
                assert_bijection(layout, nb, bs, nh, hd);
            }
        }
    }

    #[test]
    fn legacy_offset_matches_cuda_kernel_indexing() {
        // ((block * num_kv_heads + head) * head_dim + d) * block_size + slot
        // as written by cache_ops.cu::cache_write_kernel and paged_attention.cu.
        let layout = KvLayout::Legacy;
        assert_eq!(layout.offset(0, 0, 0, 0, 16, 8, 64), 0);
        // ((1*6 + 3)*7 + 4)*5 + 2 = 67*5 + 2 = 337
        assert_eq!(layout.offset(1, 2, 3, 4, 5, 6, 7), 337);
        // ((2*8 + 1)*64 + 3)*16 + 5 = 1091*16 + 5 = 17461
        assert_eq!(layout.offset(2, 5, 1, 3, 16, 8, 64), 17461);
    }

    #[test]
    fn shape_matches_layout_order() {
        assert_eq!(KvLayout::Legacy.shape(4, 16, 8, 64), [4, 8, 64, 16]);
        assert_eq!(KvLayout::Nhd.shape(4, 16, 8, 64), [4, 16, 8, 64]);
        assert_eq!(KvLayout::Hnd.shape(4, 16, 8, 64), [4, 8, 16, 64]);
    }
}
