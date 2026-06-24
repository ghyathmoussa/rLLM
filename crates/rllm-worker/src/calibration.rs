//! KV cache scale calibration for INT8 quantization.
//!
//! Runs a calibration forward pass through the model to compute per-layer
//! `k_scale` and `v_scale` values. The scales are derived from the absolute
//! maximum of K/V activations observed during a forward pass with
//! representative input tokens, following the same approach as vLLM:
//!
//! ```text
//! scale[layer] = absmax(K_or_V_activations[layer]) / 127.0
//! ```

use anyhow::Result;

#[cfg(feature = "candle-backend")]
use crate::worker::Worker;

/// Default number of calibration tokens.
const DEFAULT_CALIBRATION_TOKENS: usize = 128;

/// Default token ID used for calibration (BOS-like token).
const DEFAULT_CALIBRATION_TOKEN_ID: u32 = 1;

/// Run a calibration forward pass and set per-layer KV scales on the GPU cache.
///
/// This must be called after `initialize_kv_cache()` and only when the KV
/// cache dtype is INT8.
#[cfg(feature = "candle-backend")]
pub fn calibrate_kv_cache(worker: &mut Worker) -> Result<()> {
    let num_layers = worker.gpu_kv_cache().map(|c| c.num_layers()).unwrap_or(0);

    let is_int8 = worker.gpu_kv_cache().is_some_and(|c| c.dtype() == rllm_core::dtype::DType::INT8);

    if !is_int8 {
        tracing::debug!("skipping KV cache calibration: dtype is not INT8");
        return Ok(());
    }

    tracing::info!(
        num_calibration_tokens = DEFAULT_CALIBRATION_TOKENS,
        num_layers,
        "starting INT8 KV cache scale calibration"
    );

    // Build calibration input: repeated token ID 1 for 128 tokens.
    let calib_tokens = vec![DEFAULT_CALIBRATION_TOKEN_ID; DEFAULT_CALIBRATION_TOKENS];

    // Run the non-paged forward pass to populate local KV cache.
    let model = worker
        .candle_model()
        .ok_or_else(|| anyhow::anyhow!("cannot calibrate KV cache: model not loaded"))?;
    let device = model.device();

    let input_ids =
        candle_core::Tensor::new(calib_tokens.clone(), device)?.reshape((1, calib_tokens.len()))?;
    let positions: Vec<usize> = (0..calib_tokens.len()).collect();
    let mut kv_cache = vec![None; num_layers];

    // Forward pass populates kv_cache with post-RoPE K and raw V tensors.
    let _logits = model.forward(&input_ids, &positions, &mut kv_cache)?;

    // Extract absmax per layer and compute scales.
    let mut k_scales = Vec::with_capacity(num_layers);
    let mut v_scales = Vec::with_capacity(num_layers);

    for (i, kv) in kv_cache.iter().enumerate() {
        let (k, v) = kv.as_ref().ok_or_else(|| {
            anyhow::anyhow!("calibration forward did not populate KV cache for layer {i}")
        })?;

        // K shape: [batch, num_kv_heads, seq_len, head_dim]
        // V shape: [batch, num_kv_heads, seq_len, head_dim]
        let k_f32 = k.to_dtype(candle_core::DType::F32)?;
        let v_f32 = v.to_dtype(candle_core::DType::F32)?;

        let k_absmax = k_f32.flatten_all()?.abs()?.max(0)?.to_scalar::<f32>()?;
        let v_absmax = v_f32.flatten_all()?.abs()?.max(0)?.to_scalar::<f32>()?;

        // scale = absmax / 127.0, with a floor of 1.0 to avoid shrinking
        // activations that are already within the int8 range.
        let k_scale = (k_absmax / 127.0).max(1.0);
        let v_scale = (v_absmax / 127.0).max(1.0);

        k_scales.push(k_scale);
        v_scales.push(v_scale);

        tracing::debug!(layer = i, k_absmax, v_absmax, k_scale, v_scale, "calibrated KV scales");
    }

    if let Some(gpu_cache) = worker.gpu_kv_cache_mut() {
        gpu_cache.set_all_kv_scales(k_scales.clone(), v_scales.clone());
    }

    tracing::info!(
        k_scale_min = k_scales.iter().cloned().fold(f32::INFINITY, f32::min),
        k_scale_max = k_scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        v_scale_min = v_scales.iter().cloned().fold(f32::INFINITY, f32::min),
        v_scale_max = v_scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        "INT8 KV cache calibration complete"
    );

    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
pub fn calibrate_kv_cache(_worker: &mut crate::worker::Worker) -> Result<()> {
    tracing::debug!("skipping KV cache calibration: candle-backend feature not enabled");
    Ok(())
}
