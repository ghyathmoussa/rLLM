use rllm_core::dtype::DType;

/// KV cache tensor memory layout.
///
/// Controls how key and value tensors are arranged in GPU memory.
/// - `NHD`: `[num_blocks, num_kv_heads, head_dim, block_size]` — vLLM default
/// - `HND`: `[num_blocks, block_size, num_kv_heads, head_dim]` — transposed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    NHD,
    HND,
}

#[derive(Debug, Clone)]
pub struct KVCacheSpec {
    pub block_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub sliding_window: Option<usize>,
}

impl KVCacheSpec {
    /// Bytes needed for one block (K + V, all layers).
    pub fn bytes_per_block(&self) -> usize {
        2 * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.dtype.bytes_per_scalar()
    }

    /// Bytes needed for one layer's K or V tensor block.
    pub fn kv_bytes_per_block_per_layer(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim * self.dtype.bytes_per_scalar()
    }

    /// Calculate the number of KV blocks that fit in available GPU memory.
    ///
    /// `total_gpu_bytes`: total GPU memory
    /// `model_weight_bytes`: memory used by model weights
    /// `activation_peak_bytes`: peak memory for activations during forward pass
    /// `gpu_memory_utilization`: fraction of GPU memory to use (0.0–1.0)
    pub fn num_blocks_from_available_memory(
        &self,
        total_gpu_bytes: usize,
        model_weight_bytes: usize,
        activation_peak_bytes: usize,
        gpu_memory_utilization: f32,
    ) -> usize {
        let usable = (total_gpu_bytes as f64 * gpu_memory_utilization as f64) as usize;
        let reserved = model_weight_bytes + activation_peak_bytes;
        let available = usable.saturating_sub(reserved);
        let bytes_per_block = self.bytes_per_block();
        if bytes_per_block == 0 {
            return 0;
        }
        available / bytes_per_block
    }

    /// Key tensor shape for a given layout and number of blocks.
    pub fn key_shape(&self, num_blocks: usize, layout: KVLayout) -> Vec<usize> {
        match layout {
            KVLayout::NHD => vec![num_blocks, self.num_kv_heads, self.head_dim, self.block_size],
            KVLayout::HND => vec![num_blocks, self.block_size, self.num_kv_heads, self.head_dim],
        }
    }

    /// Value tensor shape for a given layout and number of blocks.
    pub fn value_shape(&self, num_blocks: usize, layout: KVLayout) -> Vec<usize> {
        match layout {
            KVLayout::NHD => vec![num_blocks, self.num_kv_heads, self.head_dim, self.block_size],
            KVLayout::HND => vec![num_blocks, self.block_size, self.num_kv_heads, self.head_dim],
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    pub num_blocks: usize,
    pub spec: KVCacheSpec,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_block_calculation() {
        let spec = KVCacheSpec {
            block_size: 16,
            num_layers: 32,
            num_kv_heads: 32,
            head_dim: 128,
            dtype: DType::F16,
            sliding_window: None,
        };
        // 2 * 32 * 16 * 32 * 128 * 2 = 2 * 32 * 16 * 32 * 128 * 2
        let expected = 2 * 32 * 16 * 32 * 128 * 2;
        assert_eq!(spec.bytes_per_block(), expected);
    }

    #[test]
    fn num_blocks_from_available_memory() {
        let spec = KVCacheSpec {
            block_size: 16,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F32,
            sliding_window: None,
        };
        let bytes_per_block = spec.bytes_per_block();
        // 2 * 2 * 16 * 4 * 64 * 4 = 65536
        assert_eq!(bytes_per_block, 65536);

        let total = 1_000_000_000usize; // 1 GB
        let model_weights = 200_000_000;
        let activations = 50_000_000;
        let utilization = 0.9;

        let num_blocks =
            spec.num_blocks_from_available_memory(total, model_weights, activations, utilization);
        let usable = (total as f64 * utilization as f64) as usize; // 900M
        let available = usable - model_weights - activations; // 650M
        let expected = available / bytes_per_block;
        assert_eq!(num_blocks, expected);
        assert!(num_blocks > 0);
    }

    #[test]
    fn num_blocks_zero_when_insufficient() {
        let spec = KVCacheSpec {
            block_size: 16,
            num_layers: 32,
            num_kv_heads: 32,
            head_dim: 128,
            dtype: DType::F32,
            sliding_window: None,
        };
        let num_blocks = spec.num_blocks_from_available_memory(1000, 500, 400, 0.9);
        assert_eq!(num_blocks, 0);
    }

    #[test]
    fn key_value_shapes() {
        let spec = KVCacheSpec {
            block_size: 16,
            num_layers: 2,
            num_kv_heads: 8,
            head_dim: 64,
            dtype: DType::F16,
            sliding_window: None,
        };

        let k_nhd = spec.key_shape(100, KVLayout::NHD);
        assert_eq!(k_nhd, vec![100, 8, 64, 16]);

        let v_nhd = spec.value_shape(100, KVLayout::NHD);
        assert_eq!(v_nhd, vec![100, 8, 64, 16]);

        let k_hnd = spec.key_shape(100, KVLayout::HND);
        assert_eq!(k_hnd, vec![100, 16, 8, 64]);

        let v_hnd = spec.value_shape(100, KVLayout::HND);
        assert_eq!(v_hnd, vec![100, 16, 8, 64]);
    }
}
