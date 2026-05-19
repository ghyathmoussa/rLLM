#[cfg(feature = "candle-backend")]
use candle_core::{D, DType, Device, Result, Tensor};

/// Precomputed RoPE (Rotary Position Embedding) cache.
#[cfg(feature = "candle-backend")]
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
}

#[cfg(feature = "candle-backend")]
impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f64> =
            (0..dim).step_by(2).map(|i| 1.0 / (theta as f64).powf(i as f64 / dim as f64)).collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F64)?;

        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F64)?
            .reshape((max_seq_len, 1))?;

        let freqs = positions.matmul(&inv_freq.reshape((1, inv_freq_len))?)?;

        let cos_cache = freqs.cos()?.to_dtype(DType::F32)?;
        let sin_cache = freqs.sin()?.to_dtype(DType::F32)?;

        Ok(Self { cos_cache, sin_cache })
    }

    /// Apply rotary embeddings to query and key tensors.
    pub fn apply(&self, q: &Tensor, k: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let cos = self.gather_cache(&self.cos_cache, positions)?;
        let sin = self.gather_cache(&self.sin_cache, positions)?;

        let q_rot = apply_rotary_emb(q, &cos, &sin)?;
        let k_rot = apply_rotary_emb(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }

    fn gather_cache(&self, cache: &Tensor, positions: &[usize]) -> Result<Tensor> {
        if positions.is_empty() {
            return cache.narrow(0, 0, 0);
        }

        let device = cache.device();
        let indices = Tensor::from_iter(positions.iter().map(|&p| p as u32), device)?;

        cache.index_select(&indices, 0)
    }
}

#[cfg(feature = "candle-backend")]
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [batch, heads, seq_len, head_dim]
    // cos, sin: [seq_len, dim/2]

    let dim = x.dim(D::Minus1)?;
    let half_dim = dim / 2;

    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

    // Reshape cos/sin for broadcasting: [1, 1, seq_len, dim/2]
    // Then repeat along last dim to match head_dim: [1, 1, seq_len, dim]
    let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;

    // rotate_half: [-x2, x1]
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

    // x * cos + rotated * sin
    let out = (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
    Ok(out)
}

#[cfg(all(test, feature = "candle-backend"))]
mod tests {
    use super::*;

    #[test]
    fn rope_shape_preserved() -> Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(128, 4096, 10000.0, &device)?;

        let q = Tensor::randn(0.0f32, 1.0f32, (1, 32, 10, 128), &device)?;
        let k = Tensor::randn(0.0f32, 1.0f32, (1, 8, 10, 128), &device)?;
        let positions: Vec<usize> = (0..10).collect();

        let (q_rot, k_rot) = rope.apply(&q, &k, &positions)?;
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
        Ok(())
    }

    #[test]
    fn rope_position_0_is_identity_like() -> Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 16, 10000.0, &device)?;

        let q = Tensor::new(&[[[[1.0f32, 2.0, 3.0, 4.0]]]], &device)?;
        let k = Tensor::new(&[[[[1.0f32, 2.0, 3.0, 4.0]]]], &device)?;

        let (q_rot, _) = rope.apply(&q, &k, &[0])?;
        let expected = q.flatten_all()?.to_vec1::<f32>()?;
        let actual = q_rot.flatten_all()?.to_vec1::<f32>()?;

        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).abs();
            assert!(diff < 1e-5, "mismatch at index {i}: expected {e}, got {a}");
        }
        Ok(())
    }

    #[test]
    fn rope_different_positions_differ() -> Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(16, 128, 10000.0, &device)?;

        let q = Tensor::ones((1, 1, 1, 16), DType::F32, &device)?;
        let k = Tensor::ones((1, 1, 1, 16), DType::F32, &device)?;

        let (q_pos0, _) = rope.apply(&q, &k, &[0])?;
        let (q_pos1, _) = rope.apply(&q, &k, &[1])?;

        let v0 = q_pos0.flatten_all()?.to_vec1::<f32>()?;
        let v1 = q_pos1.flatten_all()?.to_vec1::<f32>()?;

        assert_ne!(v0, v1, "different positions should produce different embeddings");
        Ok(())
    }
}
