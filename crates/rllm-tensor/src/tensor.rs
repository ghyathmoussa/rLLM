use rllm_core::dtype::DType;

use crate::device::Device;

#[derive(Debug, Clone)]
pub struct TensorView {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub device: Device,
}
