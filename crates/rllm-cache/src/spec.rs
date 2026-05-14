use rllm_core::dtype::DType;

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
    pub fn bytes_per_block(&self) -> usize {
        2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim
            * self.dtype.bytes_per_scalar()
    }
}

#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    pub num_blocks: usize,
    pub spec: KVCacheSpec,
}
