use std::{fmt, ptr::NonNull};

/// CUDA stream handle — opaque pointer to a CUstream.
pub type CudaStreamHandle = NonNull<std::ffi::c_void>;

/// Compute device abstraction.
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda {
        index: usize,
        /// Raw CUDA stream handle. `None` means the default stream (stream 0).
        stream: Option<CudaStreamHandle>,
    },
}

impl Device {
    pub fn cpu() -> Self {
        Self::Cpu
    }

    pub fn cuda(index: usize) -> Self {
        Self::Cuda { index, stream: None }
    }

    pub fn cuda_with_stream(index: usize, stream: CudaStreamHandle) -> Self {
        Self::Cuda { index, stream: Some(stream) }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda { .. })
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns the CUDA device index, or `None` for CPU.
    pub fn cuda_index(&self) -> Option<usize> {
        match self {
            Self::Cuda { index, .. } => Some(*index),
            _ => None,
        }
    }

    /// Returns the CUDA stream handle if present, otherwise `None`.
    pub fn stream(&self) -> Option<CudaStreamHandle> {
        match self {
            Self::Cuda { stream, .. } => *stream,
            _ => None,
        }
    }
}

// Safety: CudaStreamHandle is just a pointer to an opaque CUDA stream.
// It is safe to send across threads because CUDA streams are thread-safe
// as long as synchronization is handled correctly at the call site.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda { index, stream } => {
                write!(f, "cuda:{index}")?;
                if stream.is_some() {
                    write!(f, " (custom stream)")?;
                }
                Ok(())
            }
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda { index: a, stream: sa }, Self::Cuda { index: b, stream: sb }) => {
                a == b && sa == sb
            }
            _ => false,
        }
    }
}

impl Eq for Device {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device() {
        let dev = Device::cpu();
        assert!(dev.is_cpu());
        assert!(!dev.is_cuda());
        assert!(dev.cuda_index().is_none());
        assert_eq!(format!("{dev}"), "cpu");
    }

    #[test]
    fn cuda_device_default_stream() {
        let dev = Device::cuda(0);
        assert!(dev.is_cuda());
        assert!(!dev.is_cpu());
        assert_eq!(dev.cuda_index(), Some(0));
    }

    #[test]
    fn device_equality() {
        assert_eq!(Device::cpu(), Device::cpu());
        assert_eq!(Device::cuda(0), Device::cuda(0));
        assert_ne!(Device::cuda(0), Device::cuda(1));
        assert_ne!(Device::cpu(), Device::cuda(0));
    }
}
