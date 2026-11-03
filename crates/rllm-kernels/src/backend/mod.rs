//! Attention backend abstraction shared by all kernel families.
//!
//! Later phases add FlashAttention, FlashInfer, TRTLLM-GEN and Metal backends
//! behind [`AttentionBackend`], validating them against the CPU oracle
//! ([`CpuRefBackend`]).

mod cpu_ref;
mod layout;
mod metadata;

use std::str::FromStr;

pub use cpu_ref::CpuRefBackend;
pub use layout::KvLayout;
pub use metadata::{MetadataError, StepMetadata};
use rllm_core::dtype::DType;

/// Identifies which attention implementation executes a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// CPU reference oracle (`cpu_ref`).
    CpuRef,
    /// Simple non-fused paged attention (`paged_naive`).
    PagedNaive,
    /// FlashAttention (`flash_attn`).
    FlashAttn,
    /// FlashInfer (`flashinfer`).
    FlashInfer,
    /// TensorRT-LLM GEN kernels (`trtllm_gen`).
    TrtllmGen,
    /// Apple Metal (`metal`).
    Metal,
}

impl BackendKind {
    /// Canonical lowercase name (snake_case) of this backend.
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::CpuRef => "cpu_ref",
            BackendKind::PagedNaive => "paged_naive",
            BackendKind::FlashAttn => "flash_attn",
            BackendKind::FlashInfer => "flashinfer",
            BackendKind::TrtllmGen => "trtllm_gen",
            BackendKind::Metal => "metal",
        }
    }
}

/// Error returned when parsing an unknown [`BackendKind`] name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown attention backend: {0}")]
pub struct ParseBackendError(pub String);

impl FromStr for BackendKind {
    type Err = ParseBackendError;

    /// Accepts [`BackendKind::as_str`] values and their `-`-separated variants
    /// (e.g. `"flash-attn"`), case-insensitively.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "cpu_ref" => Ok(BackendKind::CpuRef),
            "paged_naive" => Ok(BackendKind::PagedNaive),
            "flash_attn" => Ok(BackendKind::FlashAttn),
            "flashinfer" => Ok(BackendKind::FlashInfer),
            "trtllm_gen" => Ok(BackendKind::TrtllmGen),
            "metal" => Ok(BackendKind::Metal),
            _ => Err(ParseBackendError(s.to_string())),
        }
    }
}

/// Static description of an attention problem, checked by
/// [`AttentionBackend::supports`] before a batch runs.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnSpec {
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key/value heads (GQA/MQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// KV cache page size in tokens.
    pub block_size: usize,
    /// Dtype of the query tensor.
    pub q_dtype: DType,
    /// Dtype of the cached K/V tensors.
    pub kv_dtype: DType,
    /// Optional sliding-window size (tokens) for windowed attention.
    pub sliding_window: Option<usize>,
    /// Softmax scale, typically `1 / sqrt(head_dim)`.
    pub scale: f32,
    /// Gemma-style logits soft cap: logits = cap * tanh(logits / cap).
    pub soft_cap: Option<f32>,
    /// CUDA compute capability as (major, minor); None for CPU/Metal.
    pub compute_capability: Option<(u32, u32)>,
}

/// Returned by [`AttentionBackend::supports`] when a backend rejects a
/// configuration.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("{backend} does not support this configuration: {reason}")]
pub struct Unsupported {
    /// Canonical name of the rejecting backend.
    pub backend: &'static str,
    /// Human-readable explanation of the unsupported configuration.
    pub reason: String,
}

/// Host-visible view of one layer's paged K/V cache, used by the CPU oracle
/// and tests.
#[derive(Debug)]
pub struct HostKvLayer<'a> {
    /// Flat K cache elements, `layout`-ordered.
    pub k: &'a [f32],
    /// Flat V cache elements, `layout`-ordered.
    pub v: &'a [f32],
    /// Physical ordering of `k` / `v`.
    pub layout: KvLayout,
    /// Number of physical blocks in the cache.
    pub num_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Per-tensor dequant scale applied to K on read (1.0 for float caches).
    pub k_scale: f32,
    /// Per-tensor dequant scale applied to V on read (1.0 for float caches).
    pub v_scale: f32,
}

/// Common interface of all attention backends.
///
/// Device-specific `run` entry points (CUDA pointer/stream signature, Metal
/// buffer signature) are added per backend in later phases and are therefore
/// not part of this trait yet; the CPU oracle exposes an inherent `run_host`
/// instead.
pub trait AttentionBackend: Send + Sync {
    /// Which backend this instance is.
    fn kind(&self) -> BackendKind;
    /// Physical K/V layout this backend reads/writes.
    fn kv_layout(&self) -> KvLayout;
    /// Whether this backend can execute `spec`; `Err` explains why not.
    fn supports(&self, spec: &AttnSpec) -> Result<(), Unsupported>;
    /// Called once per scheduler step before the layer loop. Default: no-op.
    fn plan(&mut self, _meta: &StepMetadata, _spec: &AttnSpec) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AttnSpec {
        AttnSpec {
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            block_size: 16,
            q_dtype: DType::F16,
            kv_dtype: DType::F16,
            sliding_window: None,
            scale: 1.0 / 8.0,
            soft_cap: None,
            compute_capability: None,
        }
    }

    #[test]
    fn backend_kind_as_str_values() {
        assert_eq!(BackendKind::CpuRef.as_str(), "cpu_ref");
        assert_eq!(BackendKind::PagedNaive.as_str(), "paged_naive");
        assert_eq!(BackendKind::FlashAttn.as_str(), "flash_attn");
        assert_eq!(BackendKind::FlashInfer.as_str(), "flashinfer");
        assert_eq!(BackendKind::TrtllmGen.as_str(), "trtllm_gen");
        assert_eq!(BackendKind::Metal.as_str(), "metal");
    }

    #[test]
    fn backend_kind_from_str_round_trip() {
        for kind in [
            BackendKind::CpuRef,
            BackendKind::PagedNaive,
            BackendKind::FlashAttn,
            BackendKind::FlashInfer,
            BackendKind::TrtllmGen,
            BackendKind::Metal,
        ] {
            let parsed = BackendKind::from_str(kind.as_str()).expect("as_str must parse");
            assert_eq!(parsed, kind);
        }
    }

    #[test]
    fn backend_kind_from_str_accepts_dash_variants() {
        assert_eq!(BackendKind::from_str("cpu-ref").unwrap(), BackendKind::CpuRef);
        assert_eq!(BackendKind::from_str("paged-naive").unwrap(), BackendKind::PagedNaive);
        assert_eq!(BackendKind::from_str("flash-attn").unwrap(), BackendKind::FlashAttn);
        assert_eq!(BackendKind::from_str("trtllm-gen").unwrap(), BackendKind::TrtllmGen);
        assert_eq!(BackendKind::from_str("FLASH-ATTN").unwrap(), BackendKind::FlashAttn);
        assert_eq!(BackendKind::from_str("flashinfer").unwrap(), BackendKind::FlashInfer);
        assert_eq!(BackendKind::from_str("metal").unwrap(), BackendKind::Metal);
    }

    #[test]
    fn backend_kind_from_str_rejects_unknown() {
        let err = BackendKind::from_str("transformer-engine").unwrap_err();
        assert_eq!(err, ParseBackendError("transformer-engine".to_string()));
        assert!(err.to_string().contains("transformer-engine"));
    }

    #[test]
    fn cpu_ref_supports_accepts_valid_spec() {
        assert_eq!(CpuRefBackend.supports(&spec()), Ok(()));
        // ratio 1 (MHA) is fine too
        let mut mha = spec();
        mha.num_q_heads = 4;
        mha.num_kv_heads = 4;
        assert_eq!(CpuRefBackend.supports(&mha), Ok(()));
    }

    #[test]
    fn cpu_ref_supports_rejects_bad_gqa() {
        let mut bad = spec();
        bad.num_q_heads = 6; // 6 % 4 != 0
        bad.num_kv_heads = 4;
        let err = CpuRefBackend.supports(&bad).unwrap_err();
        assert_eq!(err.backend, "cpu_ref");

        let mut zero_kv = spec();
        zero_kv.num_kv_heads = 0;
        assert!(CpuRefBackend.supports(&zero_kv).is_err());
    }

    #[test]
    fn cpu_ref_supports_rejects_zero_head_dim() {
        let mut bad = spec();
        bad.head_dim = 0;
        let err = CpuRefBackend.supports(&bad).unwrap_err();
        assert_eq!(err.backend, "cpu_ref");
        assert!(err.to_string().contains("head_dim"));
    }
}
