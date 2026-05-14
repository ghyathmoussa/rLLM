use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    FP8E4M3,
    FP8E5M2,
    INT8,
    INT4,
}

impl DType {
    pub fn bytes_per_scalar(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::FP8E4M3 => 1,
            Self::FP8E5M2 => 1,
            Self::INT8 => 1,
            Self::INT4 => 1, // packed 2 per byte in practice
        }
    }
}
