use anyhow::{Result, bail};

pub const GPTQ_SUPPORTED_BITS: usize = 4;
const PACK_FACTOR: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GptqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub damp_percent: f32,
    pub act_order: bool,
}

impl GptqConfig {
    pub fn validate(&self) -> Result<()> {
        if self.bits != GPTQ_SUPPORTED_BITS {
            bail!(
                "only {GPTQ_SUPPORTED_BITS}-bit GPTQ is currently supported, got {}",
                self.bits
            );
        }
        if self.group_size == 0 {
            bail!("group_size must be > 0");
        }
        if !(0.0..1.0).contains(&self.damp_percent) {
            bail!("damp_percent must be in [0.0, 1.0), got {}", self.damp_percent);
        }
        Ok(())
    }
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: GPTQ_SUPPORTED_BITS,
            group_size: 128,
            damp_percent: 0.01,
            act_order: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GptqCalibration {
    in_features: usize,
    hessian: Vec<f64>,
    nsamples: usize,
}

impl GptqCalibration {
    pub fn new(in_features: usize) -> Self {
        Self {
            in_features,
            hessian: vec![0.0; in_features * in_features],
            nsamples: 0,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn nsamples(&self) -> usize {
        self.nsamples
    }

    pub fn observe(&mut self, batch: &[Vec<f32>]) -> Result<()> {
        for sample in batch {
            self.observe_sample(sample)?;
        }
        Ok(())
    }

    pub fn observe_sample(&mut self, sample: &[f32]) -> Result<()> {
        if sample.len() != self.in_features {
            bail!(
                "calibration sample width mismatch: expected {}, got {}",
                self.in_features,
                sample.len()
            );
        }
        for row in 0..self.in_features {
            let row_val = sample[row] as f64;
            let row_offset = row * self.in_features;
            for col in 0..self.in_features {
                self.hessian[row_offset + col] += row_val * sample[col] as f64;
            }
        }
        self.nsamples += 1;
        Ok(())
    }

    pub fn hessian(&self) -> &[f64] {
        &self.hessian
    }

    fn mean_diag(&self) -> f64 {
        if self.in_features == 0 {
            return 0.0;
        }
        (0..self.in_features)
            .map(|i| self.hessian[i * self.in_features + i])
            .sum::<f64>()
            / self.in_features as f64
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedGptqLayer {
    pub qweight: Vec<i32>,
    pub qzeros: Vec<i32>,
    pub scales: Vec<f32>,
    pub g_idx: Vec<u32>,
    pub in_features: usize,
    pub out_features: usize,
    pub num_groups: usize,
    pub bits: usize,
    pub group_size: usize,
}

impl QuantizedGptqLayer {
    pub fn qweight_shape(&self) -> (usize, usize) {
        (self.in_features / PACK_FACTOR, self.out_features)
    }

    pub fn qzeros_shape(&self) -> (usize, usize) {
        (self.num_groups, self.out_features / PACK_FACTOR)
    }

    pub fn scales_shape(&self) -> (usize, usize) {
        (self.num_groups, self.out_features)
    }
}

pub fn quantize_gptq(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    calibration: &GptqCalibration,
    config: GptqConfig,
) -> Result<QuantizedGptqLayer> {
    config.validate()?;
    validate_layer_dims(weights, out_features, in_features)?;
    if calibration.in_features() != in_features {
        bail!(
            "calibration width mismatch: expected {}, got {}",
            in_features,
            calibration.in_features()
        );
    }
    if calibration.nsamples() == 0 {
        bail!("cannot quantize GPTQ layer without calibration samples");
    }

    let order = column_order(calibration, config.act_order);
    let hessian = damped_permuted_hessian(calibration, &order, config.damp_percent)?;
    let hessian_inv = invert_spd(&hessian, in_features)?;
    let g_idx = make_group_index(in_features, config.group_size, &order);
    let num_groups = g_idx
        .iter()
        .copied()
        .map(|v| v as usize)
        .max()
        .map(|v| v + 1)
        .unwrap_or(0);

    let mut q_by_input = vec![0u8; in_features * out_features];
    let mut scales = vec![0.0f32; num_groups * out_features];
    let mut zeros = vec![0u8; num_groups * out_features];
    let qmax = ((1usize << config.bits) - 1) as i32;

    for out_col in 0..out_features {
        let row_start = out_col * in_features;
        let row = &weights[row_start..row_start + in_features];
        let mut work = order.iter().map(|&idx| row[idx] as f64).collect::<Vec<_>>();

        let mut current_group = usize::MAX;
        let mut scale = 1.0f32;
        let mut zero = 0u8;

        for perm_i in 0..in_features {
            let orig_i = order[perm_i];
            let group = g_idx[orig_i] as usize;
            if group != current_group {
                current_group = group;
                let group_end = ((group + 1) * config.group_size).min(in_features);
                let group_slice = &work[perm_i..group_end];
                let params = quant_params(group_slice, qmax)?;
                scale = params.scale;
                zero = params.zero;
                scales[group * out_features + out_col] = scale;
                zeros[group * out_features + out_col] = zero;
            }

            let dequant = quantize_value(work[perm_i], scale, zero, qmax);
            q_by_input[orig_i * out_features + out_col] = dequant.q as u8;
            work[perm_i] = dequant.dequant;

            let err = dequant.error;
            let denom = hessian_inv[perm_i * in_features + perm_i];
            if denom.abs() < 1e-12 {
                continue;
            }
            let coeff = err / denom;
            for perm_j in (perm_i + 1)..in_features {
                work[perm_j] -= coeff * hessian_inv[perm_i * in_features + perm_j];
            }
        }
    }

    Ok(QuantizedGptqLayer {
        qweight: pack_qweight(&q_by_input, in_features, out_features),
        qzeros: pack_qzeros(&zeros, num_groups, out_features)?,
        scales,
        g_idx,
        in_features,
        out_features,
        num_groups,
        bits: config.bits,
        group_size: config.group_size,
    })
}

#[derive(Debug, Clone, Copy)]
struct QuantParams {
    scale: f32,
    zero: u8,
}

#[derive(Debug, Clone, Copy)]
struct QuantizedValue {
    q: i32,
    dequant: f64,
    error: f64,
}

fn validate_layer_dims(weights: &[f32], out_features: usize, in_features: usize) -> Result<()> {
    if in_features == 0 || out_features == 0 {
        bail!("layer dimensions must be > 0");
    }
    if in_features % PACK_FACTOR != 0 {
        bail!(
            "GPTQ packer currently requires in_features divisible by {}, got {}",
            PACK_FACTOR,
            in_features
        );
    }
    if out_features % PACK_FACTOR != 0 {
        bail!(
            "GPTQ zero packing currently requires out_features divisible by {}, got {}",
            PACK_FACTOR,
            out_features
        );
    }
    if weights.len() != out_features * in_features {
        bail!(
            "weight buffer has {} elements, expected {}",
            weights.len(),
            out_features * in_features
        );
    }
    Ok(())
}

fn column_order(calibration: &GptqCalibration, act_order: bool) -> Vec<usize> {
    let mut order = (0..calibration.in_features()).collect::<Vec<_>>();
    if act_order {
        order.sort_by(|&a, &b| {
            let da = calibration.hessian()[a * calibration.in_features() + a];
            let db = calibration.hessian()[b * calibration.in_features() + b];
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    order
}

fn damped_permuted_hessian(
    calibration: &GptqCalibration,
    order: &[usize],
    damp_percent: f32,
) -> Result<Vec<f64>> {
    let n = calibration.in_features();
    let mut h = vec![0.0f64; n * n];
    for perm_row in 0..n {
        let src_row = order[perm_row];
        for perm_col in 0..n {
            let src_col = order[perm_col];
            h[perm_row * n + perm_col] = calibration.hessian()[src_row * n + src_col];
        }
    }
    let damp = calibration.mean_diag() * damp_percent as f64;
    if !damp.is_finite() {
        bail!("invalid Hessian damping value");
    }
    for i in 0..n {
        h[i * n + i] += damp.max(1e-8);
    }
    Ok(h)
}

fn make_group_index(in_features: usize, group_size: usize, order: &[usize]) -> Vec<u32> {
    let mut g_idx = vec![0u32; in_features];
    for (perm_i, &orig_i) in order.iter().enumerate() {
        g_idx[orig_i] = (perm_i / group_size) as u32;
    }
    g_idx
}

fn quant_params(group: &[f64], qmax: i32) -> Result<QuantParams> {
    if group.is_empty() {
        bail!("cannot derive quantization parameters from an empty group");
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in group {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if (max_v - min_v).abs() < 1e-12 {
        return Ok(QuantParams {
            scale: 1.0,
            zero: ((qmax + 2) / 2) as u8,
        });
    }
    let scale = ((max_v - min_v) / qmax as f64).max(1e-8) as f32;
    let zero = (-min_v / scale as f64)
        .round()
        .clamp(1.0, (qmax + 1) as f64) as u8;
    Ok(QuantParams { scale, zero })
}

fn quantize_value(value: f64, scale: f32, zero: u8, qmax: i32) -> QuantizedValue {
    let q = ((value / scale as f64).round() as i32 + zero as i32).clamp(0, qmax);
    let dequant = (q - zero as i32) as f64 * scale as f64;
    QuantizedValue {
        q,
        dequant,
        error: value - dequant,
    }
}

fn pack_qweight(q_by_input: &[u8], in_features: usize, out_features: usize) -> Vec<i32> {
    let packed_rows = in_features / PACK_FACTOR;
    let mut packed = vec![0i32; packed_rows * out_features];
    for packed_row in 0..packed_rows {
        for out_col in 0..out_features {
            let mut acc = 0u32;
            for nibble in 0..PACK_FACTOR {
                let in_idx = packed_row * PACK_FACTOR + nibble;
                let q = q_by_input[in_idx * out_features + out_col] as u32;
                acc |= q << (4 * nibble);
            }
            packed[packed_row * out_features + out_col] = acc as i32;
        }
    }
    packed
}

fn pack_qzeros(zeros: &[u8], num_groups: usize, out_features: usize) -> Result<Vec<i32>> {
    if out_features % PACK_FACTOR != 0 {
        bail!(
            "qzero packer requires out_features divisible by {}, got {}",
            PACK_FACTOR,
            out_features
        );
    }
    let packed_cols = out_features / PACK_FACTOR;
    let mut packed = vec![0i32; num_groups * packed_cols];
    for group in 0..num_groups {
        for packed_col in 0..packed_cols {
            let mut acc = 0u32;
            for nibble in 0..PACK_FACTOR {
                let out_col = packed_col * PACK_FACTOR + nibble;
                let zero = zeros[group * out_features + out_col];
                let encoded = zero.saturating_sub(1) as u32;
                acc |= encoded << (4 * nibble);
            }
            packed[group * packed_cols + packed_col] = acc as i32;
        }
    }
    Ok(packed)
}

fn invert_spd(matrix: &[f64], n: usize) -> Result<Vec<f64>> {
    let chol = cholesky(matrix, n)?;
    let mut inv = vec![0.0f64; n * n];
    for col in 0..n {
        let mut y = vec![0.0f64; n];
        for row in 0..n {
            let mut sum = if row == col { 1.0 } else { 0.0 };
            for k in 0..row {
                sum -= chol[row * n + k] * y[k];
            }
            y[row] = sum / chol[row * n + row];
        }
        let mut x = vec![0.0f64; n];
        for row in (0..n).rev() {
            let mut sum = y[row];
            for k in (row + 1)..n {
                sum -= chol[k * n + row] * x[k];
            }
            x[row] = sum / chol[row * n + row];
        }
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    Ok(inv)
}

fn cholesky(matrix: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    bail!("Hessian is not positive definite after damping");
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "candle-backend")]
    use crate::layers::dequantize_gptq;
    #[cfg(feature = "candle-backend")]
    use candle_core::{Device, Tensor};

    fn quantize_rtn(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        config: GptqConfig,
    ) -> Result<QuantizedGptqLayer> {
        let order = (0..in_features).collect::<Vec<_>>();
        let g_idx = make_group_index(in_features, config.group_size, &order);
        let num_groups = g_idx.iter().copied().max().map(|v| v as usize + 1).unwrap_or(0);
        let qmax = ((1usize << config.bits) - 1) as i32;
        let mut q_by_input = vec![0u8; in_features * out_features];
        let mut scales = vec![0.0f32; num_groups * out_features];
        let mut zeros = vec![0u8; num_groups * out_features];

        for out_col in 0..out_features {
            let row = &weights[out_col * in_features..(out_col + 1) * in_features];
            for group in 0..num_groups {
                let start = group * config.group_size;
                let end = ((group + 1) * config.group_size).min(in_features);
                let params =
                    quant_params(&row[start..end].iter().map(|v| *v as f64).collect::<Vec<_>>(), qmax)?;
                scales[group * out_features + out_col] = params.scale;
                zeros[group * out_features + out_col] = params.zero;
                for in_idx in start..end {
                    let q = quantize_value(row[in_idx] as f64, params.scale, params.zero, qmax);
                    q_by_input[in_idx * out_features + out_col] = q.q as u8;
                }
            }
        }

        Ok(QuantizedGptqLayer {
            qweight: pack_qweight(&q_by_input, in_features, out_features),
            qzeros: pack_qzeros(&zeros, num_groups, out_features)?,
            scales,
            g_idx,
            in_features,
            out_features,
            num_groups,
            bits: config.bits,
            group_size: config.group_size,
        })
    }

    fn dequantize_cpu(layer: &QuantizedGptqLayer) -> Vec<f32> {
        let mut out = vec![0.0f32; layer.out_features * layer.in_features];
        for out_col in 0..layer.out_features {
            for in_idx in 0..layer.in_features {
                let packed = layer.qweight[(in_idx / PACK_FACTOR) * layer.out_features + out_col] as u32;
                let q = ((packed >> (4 * (in_idx % PACK_FACTOR))) & 0xF) as i32;
                let group = layer.g_idx[in_idx] as usize;
                let scale = layer.scales[group * layer.out_features + out_col];
                let packed_zero =
                    layer.qzeros[group * (layer.out_features / PACK_FACTOR) + out_col / PACK_FACTOR] as u32;
                let zero = (((packed_zero >> (4 * (out_col % PACK_FACTOR))) & 0xF) as i32) + 1;
                out[out_col * layer.in_features + in_idx] = (q - zero) as f32 * scale;
            }
        }
        out
    }

    fn reconstruction_error(weights: &[f32], dequant: &[f32], samples: &[Vec<f32>]) -> f64 {
        let in_features = samples[0].len();
        let out_features = weights.len() / in_features;
        let mut total = 0.0f64;
        for sample in samples {
            for out_col in 0..out_features {
                let mut ref_acc = 0.0f64;
                let mut q_acc = 0.0f64;
                for in_idx in 0..in_features {
                    let x = sample[in_idx] as f64;
                    ref_acc += x * weights[out_col * in_features + in_idx] as f64;
                    q_acc += x * dequant[out_col * in_features + in_idx] as f64;
                }
                let diff = ref_acc - q_acc;
                total += diff * diff;
            }
        }
        total / (samples.len() * out_features) as f64
    }

    #[test]
    fn calibration_accumulates_hessian() {
        let mut calib = GptqCalibration::new(3);
        calib.observe(&[
            vec![1.0, 2.0, 3.0],
            vec![0.0, -1.0, 2.0],
        ])
        .unwrap();
        assert_eq!(calib.nsamples(), 2);
        assert_eq!(
            calib.hessian(),
            &[
                1.0, 2.0, 3.0,
                2.0, 5.0, 4.0,
                3.0, 4.0, 13.0,
            ]
        );
    }

    #[test]
    fn gptq_quantization_reduces_calibration_error_vs_rtn() {
        let config = GptqConfig {
            bits: 4,
            group_size: 4,
            damp_percent: 0.05,
            act_order: true,
        };
        let samples = vec![
            vec![4.0, 0.2, -3.8, 0.1, 2.5, -0.4, 0.0, 1.0],
            vec![3.5, 0.1, -3.2, 0.2, 2.1, -0.5, 0.2, 0.8],
            vec![-4.2, -0.1, 4.1, 0.0, -2.4, 0.3, -0.1, -0.9],
            vec![2.9, 0.0, -2.7, 0.3, 1.8, -0.4, 0.1, 0.7],
        ];
        let mut calib = GptqCalibration::new(8);
        calib.observe(&samples).unwrap();

        let weights = vec![
            1.2, 0.02, -1.1, 0.03, 0.8, -0.02, 0.01, 0.4,
            -0.7, 0.01, 0.9, -0.02, -0.6, 0.03, -0.01, -0.3,
            0.5, -0.04, -0.6, 0.02, 0.4, 0.01, 0.02, 0.2,
            -1.0, 0.03, 1.1, -0.01, -0.9, 0.02, 0.0, -0.5,
            0.3, 0.02, -0.2, 0.01, 0.1, -0.01, 0.03, 0.2,
            -0.4, -0.02, 0.5, 0.02, -0.3, 0.01, -0.02, -0.2,
            0.9, 0.01, -0.8, 0.0, 0.7, -0.02, 0.01, 0.3,
            -0.2, -0.01, 0.3, 0.01, -0.1, 0.02, 0.0, -0.1,
        ];

        let gptq = quantize_gptq(&weights, 8, 8, &calib, config).unwrap();
        let rtn = quantize_rtn(&weights, 8, 8, config).unwrap();

        let gptq_err = reconstruction_error(&weights, &dequantize_cpu(&gptq), &samples);
        let rtn_err = reconstruction_error(&weights, &dequantize_cpu(&rtn), &samples);
        assert!(gptq_err <= rtn_err, "gptq_err={gptq_err}, rtn_err={rtn_err}");
    }

    #[cfg(feature = "candle-backend")]
    #[test]
    fn packed_quantized_layer_matches_runtime_dequantizer() -> Result<()> {
        let config = GptqConfig {
            bits: 4,
            group_size: 4,
            damp_percent: 0.01,
            act_order: true,
        };
        let mut calib = GptqCalibration::new(8);
        calib.observe(&[
            vec![1.0, 0.5, -1.0, 0.2, 0.7, -0.3, 0.1, 0.4],
            vec![-1.1, -0.4, 1.2, -0.1, -0.6, 0.2, -0.2, -0.5],
        ])?;

        let weights = vec![
            0.9, 0.1, -0.8, 0.0, 0.7, -0.1, 0.2, 0.3,
            -0.4, 0.0, 0.5, -0.1, -0.3, 0.2, 0.0, -0.2,
            0.6, -0.1, -0.5, 0.1, 0.4, 0.0, 0.1, 0.2,
            -0.7, 0.2, 0.8, -0.2, -0.5, 0.1, -0.1, -0.3,
            0.3, 0.0, -0.2, 0.0, 0.2, -0.1, 0.1, 0.1,
            -0.2, -0.1, 0.3, 0.1, -0.1, 0.1, -0.1, -0.1,
            0.8, 0.1, -0.7, 0.0, 0.6, -0.1, 0.2, 0.2,
            -0.1, 0.0, 0.2, 0.0, -0.1, 0.1, 0.0, -0.1,
        ];

        let quantized = quantize_gptq(&weights, 8, 8, &calib, config)?;
        let device = Device::Cpu;
        let qweight = Tensor::from_vec(quantized.qweight.clone(), quantized.qweight_shape(), &device)?;
        let qzeros = Tensor::from_vec(quantized.qzeros.clone(), quantized.qzeros_shape(), &device)?;
        let scales = Tensor::from_vec(quantized.scales.clone(), quantized.scales_shape(), &device)?;
        let g_idx = Tensor::from_vec(quantized.g_idx.clone(), (quantized.in_features,), &device)?;

        let runtime = dequantize_gptq(
            &qweight,
            &qzeros,
            &scales,
            &g_idx,
            quantized.bits,
            quantized.group_size,
        )?;
        let runtime = runtime.to_vec2::<f32>()?;
        let flat_runtime = runtime.into_iter().flatten().collect::<Vec<_>>();
        let flat_cpu = dequantize_cpu(&quantized);

        assert_eq!(flat_runtime.len(), flat_cpu.len());
        for (lhs, rhs) in flat_runtime.iter().zip(flat_cpu.iter()) {
            assert!((lhs - rhs).abs() < 1e-4, "{lhs} != {rhs}");
        }
        Ok(())
    }
}
