#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda { index: usize },
}

impl Device {
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda { .. })
    }
}
