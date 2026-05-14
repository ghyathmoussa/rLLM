use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub Uuid);

impl RequestId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_serde_roundtrip() {
        let id = RequestId::new();
        let json = serde_json::to_string(&id).unwrap();
        let back: RequestId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn request_id_unique() {
        assert_ne!(RequestId::new(), RequestId::new());
    }

    #[test]
    fn sequence_id_serde_roundtrip() {
        let id = SequenceId(42);
        let json = serde_json::to_string(&id).unwrap();
        let back: SequenceId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn block_id_serde_roundtrip() {
        let id = BlockId(7);
        let json = serde_json::to_string(&id).unwrap();
        let back: BlockId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }
}
