use std::{fmt, ptr::NonNull};

use rllm_core::dtype::DType;

use crate::device::Device;

/// Lightweight view into a tensor's metadata and data pointer.
///
/// Does not own the underlying storage — the caller must ensure
/// the data outlives the view.
#[derive(Clone)]
pub struct TensorView {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub device: Device,
    /// Raw pointer to the tensor data buffer.
    /// For CPU tensors this is a host pointer; for CUDA tensors this is a device pointer.
    pub data_ptr: Option<NonNull<u8>>,
}

impl TensorView {
    pub fn new(dtype: DType, shape: Vec<usize>, strides: Vec<usize>, device: Device) -> Self {
        Self { dtype, shape, strides, device, data_ptr: None }
    }

    pub fn with_data_ptr(mut self, ptr: NonNull<u8>) -> Self {
        self.data_ptr = Some(ptr);
        self
    }

    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total bytes occupied by the tensor data.
    pub fn nbytes(&self) -> usize {
        self.num_elements() * self.dtype.bytes_per_scalar()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Whether the tensor layout is contiguous (C-contiguous, row-major).
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride = 1usize;
        for (dim, stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            if *stride != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }

    /// Compute contiguous strides for the given shape.
    pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Create a TensorView with contiguous strides.
    pub fn contiguous(dtype: DType, shape: Vec<usize>, device: Device) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self { dtype, shape, strides, device, data_ptr: None }
    }
}

impl fmt::Debug for TensorView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("device", &self.device)
            .field("data_ptr", &self.data_ptr.map(|p| p.as_ptr()))
            .finish()
    }
}

// Safety: TensorView is just metadata + a raw pointer.
// The pointer is not dereferenced by TensorView itself.
unsafe impl Send for TensorView {}
unsafe impl Sync for TensorView {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn num_elements() {
        let tv = TensorView::contiguous(DType::F32, vec![2, 3, 4], Device::Cpu);
        assert_eq!(tv.num_elements(), 24);
    }

    #[test]
    fn nbytes() {
        let tv = TensorView::contiguous(DType::F16, vec![8, 16], Device::Cpu);
        assert_eq!(tv.nbytes(), 8 * 16 * 2);
    }

    #[test]
    fn contiguous_strides_correct() {
        let strides = TensorView::contiguous_strides(&[2, 3, 4]);
        assert_eq!(strides, &[12, 4, 1]);

        let tv = TensorView::contiguous(DType::F32, vec![2, 3, 4], Device::Cpu);
        assert_eq!(tv.strides, &[12, 4, 1]);
        assert!(tv.is_contiguous());
    }

    #[test]
    fn non_contiguous_detected() {
        let tv = TensorView::new(DType::F32, vec![2, 3], vec![1, 2], Device::Cpu);
        assert!(!tv.is_contiguous());
    }

    #[test]
    fn scalar_tensor() {
        let tv = TensorView::contiguous(DType::F32, vec![], Device::Cpu);
        assert_eq!(tv.num_elements(), 1);
        assert_eq!(tv.ndim(), 0);
        assert!(tv.is_contiguous());
    }

    #[test]
    fn ndim() {
        let tv = TensorView::contiguous(DType::F32, vec![1, 2, 3, 4], Device::Cpu);
        assert_eq!(tv.ndim(), 4);
    }

    #[test]
    fn with_data_ptr() {
        let mut buf = [1u8, 2, 3, 4];
        let ptr = NonNull::new(buf.as_mut_ptr()).unwrap();
        let tv = TensorView::contiguous(DType::F32, vec![1], Device::Cpu).with_data_ptr(ptr);
        assert!(tv.data_ptr.is_some());
    }
}
