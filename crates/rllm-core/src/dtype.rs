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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_scalar_values() {
        assert_eq!(DType::F32.bytes_per_scalar(), 4);
        assert_eq!(DType::F16.bytes_per_scalar(), 2);
        assert_eq!(DType::BF16.bytes_per_scalar(), 2);
        assert_eq!(DType::FP8E4M3.bytes_per_scalar(), 1);
        assert_eq!(DType::FP8E5M2.bytes_per_scalar(), 1);
        assert_eq!(DType::INT8.bytes_per_scalar(), 1);
        assert_eq!(DType::INT4.bytes_per_scalar(), 1);
    }

    #[test]
    fn dtype_serde_roundtrip() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::FP8E4M3,
            DType::FP8E5M2,
            DType::INT8,
            DType::INT4,
        ] {
            let json = serde_json::to_string(&dtype).unwrap();
            let back: DType = serde_json::from_str(&json).unwrap();
            assert_eq!(dtype, back);
        }
    }
}
