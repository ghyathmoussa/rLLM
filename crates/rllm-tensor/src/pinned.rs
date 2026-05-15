use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// Page-locked ("pinned") host buffer for async CPU/GPU transfers.
///
/// On a CUDA system, the memory is allocated via `cudaMallocHost` which makes
/// it page-locked and enables faster DMA transfers to the GPU. On non-CUDA
/// builds, this falls back to a standard aligned host allocation.
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl PinnedBuffer {
    /// Allocate a pinned buffer of `bytes` bytes with default alignment.
    ///
    /// The memory is zero-initialized.
    pub fn alloc(bytes: usize) -> Self {
        let layout = Layout::from_size_align(bytes.max(1), 64)
            .expect("invalid pinned buffer layout");
        let ptr = Self::alloc_impl(layout);
        // Safety: we own the allocation and it's valid for layout.size() bytes.
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0u8, layout.size());
        }
        Self { ptr, layout }
    }

    /// Allocate with a specific capacity in elements of type T.
    pub fn alloc_typed<T>(count: usize) -> Self {
        let bytes = count * std::mem::size_of::<T>();
        Self::alloc(bytes)
    }

    /// Returns the raw pointer to the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns the mutable raw pointer to the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the buffer size in bytes.
    pub fn len(&self) -> usize {
        self.layout.size()
    }

    /// Returns true if the buffer has zero length.
    pub fn is_empty(&self) -> bool {
        self.layout.size() == 0
    }

    /// Returns a typed slice view of the buffer.
    ///
    /// # Safety
    /// Caller must ensure T's alignment is compatible and the buffer
    /// contains enough initialized bytes for `count` elements.
    /// # Safety
    /// Caller must ensure T's alignment is compatible and the buffer
    /// contains enough initialized bytes for `count` elements.
    pub unsafe fn as_slice<T>(&self, count: usize) -> &[T] {
        // Safety: caller guarantees validity; ptr and layout are valid.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, count) }
    }

    /// Returns a typed mutable slice view of the buffer.
    ///
    /// # Safety
    /// Same as `as_slice`.
    pub unsafe fn as_mut_slice<T>(&mut self, count: usize) -> &mut [T] {
        // Safety: caller guarantees validity; ptr and layout are valid.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count) }
    }

    #[cfg(not(feature = "cuda"))]
    fn alloc_impl(layout: Layout) -> NonNull<u8> {
        // Safety: Layout is valid (checked above), alloc returns null on failure.
        let ptr = unsafe { alloc::alloc(layout) };
        NonNull::new(ptr).expect("pinned buffer allocation failed")
    }

    #[cfg(feature = "cuda")]
    fn alloc_impl(layout: Layout) -> NonNull<u8> {
        // On a CUDA build we would call cudaMallocHost here.
        // For now, fall back to standard alloc until the CUDA runtime
        // FFI is linked (Phase 6+ will provide the real implementation).
        let ptr = unsafe { alloc::alloc(layout) };
        NonNull::new(ptr).expect("pinned buffer allocation failed")
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // Safety: ptr was allocated with this layout and is valid.
        unsafe {
            #[cfg(not(feature = "cuda"))]
            alloc::dealloc(self.ptr.as_ptr(), self.layout);

            #[cfg(feature = "cuda")]
            {
                // When CUDA runtime is linked, this should call cudaFreeHost.
                // Fall back to standard dealloc until the FFI is linked.
                alloc::dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

// Safety: PinnedBuffer owns its allocation and doesn't have interior mutability
// issues beyond the mutable pointer accessor (which requires &mut self).
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_size() {
        let buf = PinnedBuffer::alloc(1024);
        assert_eq!(buf.len(), 1024);
        assert!(!buf.is_empty());
    }

    #[test]
    fn alloc_zero_len() {
        let buf = PinnedBuffer::alloc(0);
        assert_eq!(buf.len(), 1); // max(0, 1) = 1
    }

    #[test]
    fn alloc_typed() {
        let buf = PinnedBuffer::alloc_typed::<f32>(256);
        assert_eq!(buf.len(), 256 * 4);
    }

    #[test]
    fn zero_initialized() {
        let buf = PinnedBuffer::alloc(16);
        let slice = unsafe { buf.as_slice::<u8>(16) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn typed_read_write() {
        let mut buf = PinnedBuffer::alloc_typed::<f32>(4);
        let slice = unsafe { buf.as_mut_slice::<f32>(4) };
        slice.copy_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let read = unsafe { buf.as_slice::<f32>(4) };
        assert_eq!(read, &[1.0f32, 2.0, 3.0, 4.0]);
    }
}
