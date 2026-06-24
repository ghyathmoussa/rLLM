#[cfg(feature = "candle-backend")]
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

#[cfg(feature = "candle-backend")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "candle-backend")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle-backend")]
use half::{bf16, f16};
#[cfg(feature = "candle-backend")]
use safetensors::tensor::{Dtype as SafeDtype, TensorView, serialize_to_file};

#[cfg(feature = "candle-backend")]
use crate::{
    gptq::{GptqConfig, QuantizedGptqLayer, quantize_gptq},
    hf_config::parse_hf_config,
    llama::LlamaForCausalLM,
    loader::{WeightMap, load_weights_with_tied_detection, resolve_model_dir},
};
#[cfg(feature = "candle-backend")]
use rllm_tokenizer::Tokenizer;

#[cfg(feature = "candle-backend")]
#[derive(Debug, Clone)]
pub struct GptqExportOptions {
    pub bits: usize,
    pub group_size: usize,
    pub damp_percent: f32,
    pub act_order: bool,
    pub calibration_prompts: Vec<String>,
    pub max_calibration_samples: usize,
    pub max_seq_len: usize,
    pub include_lm_head: bool,
}

#[cfg(feature = "candle-backend")]
impl Default for GptqExportOptions {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            damp_percent: 0.01,
            act_order: false,
            calibration_prompts: Vec::new(),
            max_calibration_samples: 128,
            max_seq_len: 2048,
            include_lm_head: true,
        }
    }
}

#[cfg(feature = "candle-backend")]
pub fn quantize_model_to_gptq(
    model_ref: &str,
    output_dir: &Path,
    opts: &GptqExportOptions,
) -> Result<()> {
    let model_dir = resolve_model_dir(model_ref)?;
    quantize_model_dir_to_gptq(&model_dir, output_dir, opts)
}

#[cfg(feature = "candle-backend")]
pub fn quantize_model_dir_to_gptq(
    model_dir: &Path,
    output_dir: &Path,
    opts: &GptqExportOptions,
) -> Result<()> {
    if opts.calibration_prompts.is_empty() {
        bail!("at least one calibration prompt is required");
    }
    let config_path = model_dir.join("config.json");
    let config = parse_hf_config(&config_path)?;
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path.to_string_lossy())
        .with_context(|| format!("loading tokenizer from {}", tokenizer_path.display()))?;

    let calibration_batches = build_calibration_batches(&tokenizer, opts)?;
    if calibration_batches.is_empty() {
        bail!("calibration prompts produced no tokens");
    }

    let device = Device::Cpu;
    let (weight_map, tied_lm_head) = load_weights_with_tied_detection(model_dir, &device, None)?;
    let model = LlamaForCausalLM::from_weights(config.clone(), weight_map.clone())?;
    let calibrations =
        model.collect_gptq_calibrations(&calibration_batches, opts.include_lm_head)?;

    let quantized = quantize_weight_map(&weight_map, &calibrations, opts, tied_lm_head)?;
    materialize_output_dir(model_dir, output_dir)?;
    write_quantized_checkpoint(output_dir, &quantized)?;
    write_quantized_config(model_dir, output_dir, opts)?;
    Ok(())
}

#[cfg(feature = "candle-backend")]
fn build_calibration_batches(
    tokenizer: &Tokenizer,
    opts: &GptqExportOptions,
) -> Result<Vec<Vec<u32>>> {
    let mut batches = Vec::new();
    for prompt in opts.calibration_prompts.iter().take(opts.max_calibration_samples) {
        let mut ids = tokenizer.encode(prompt, true)?;
        if ids.is_empty() {
            continue;
        }
        if ids.len() > opts.max_seq_len {
            ids.truncate(opts.max_seq_len);
        }
        batches.push(ids);
    }
    Ok(batches)
}

#[cfg(feature = "candle-backend")]
fn quantize_weight_map(
    weight_map: &WeightMap,
    calibrations: &BTreeMap<String, crate::gptq::GptqCalibration>,
    opts: &GptqExportOptions,
    tied_lm_head: bool,
) -> Result<BTreeMap<String, SerializableTensor>> {
    let mut out = BTreeMap::new();
    let gptq_config = GptqConfig {
        bits: opts.bits,
        group_size: opts.group_size,
        damp_percent: opts.damp_percent,
        act_order: opts.act_order,
    };

    for (name, tensor) in &weight_map.weights {
        if should_skip_original_weight(name, calibrations, opts.include_lm_head) {
            continue;
        }
        out.insert(name.clone(), serialize_candle_tensor(tensor)?);
    }

    for (name, calibration) in calibrations {
        let weight = if name == "lm_head"
            && tied_lm_head
            && !weight_map.weights.contains_key("lm_head.weight")
        {
            weight_map
                .weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow::anyhow!("missing tied embedding weight for lm_head"))?
        } else {
            let weight_name = format!("{name}.weight");
            weight_map
                .weights
                .get(&weight_name)
                .ok_or_else(|| anyhow::anyhow!("missing weight tensor {weight_name}"))?
        };

        let weight_f32 = weight.to_dtype(DType::F32)?;
        let dims = weight_f32.dims();
        if dims.len() != 2 {
            bail!("expected 2D linear weight for {name}, got shape {:?}", dims);
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let flat = weight_f32.flatten_all()?.to_vec1::<f32>()?;
        let quantized = quantize_gptq(&flat, out_features, in_features, calibration, gptq_config)?;
        insert_quantized_layer(&mut out, name, quantized)?;
    }

    Ok(out)
}

#[cfg(feature = "candle-backend")]
fn should_skip_original_weight(
    name: &str,
    calibrations: &BTreeMap<String, crate::gptq::GptqCalibration>,
    include_lm_head: bool,
) -> bool {
    if let Some(prefix) = name.strip_suffix(".weight") {
        if calibrations.contains_key(prefix) {
            return true;
        }
        if include_lm_head && prefix == "lm_head" {
            return true;
        }
    }
    false
}

#[cfg(feature = "candle-backend")]
fn insert_quantized_layer(
    out: &mut BTreeMap<String, SerializableTensor>,
    prefix: &str,
    quantized: QuantizedGptqLayer,
) -> Result<()> {
    let qweight_shape = quantized.qweight_shape();
    let qzeros_shape = quantized.qzeros_shape();
    let scales_shape = quantized.scales_shape();
    let in_features = quantized.in_features;

    out.insert(
        format!("{prefix}.qweight"),
        SerializableTensor::from_typed(
            SafeDtype::I32,
            vec![qweight_shape.0, qweight_shape.1],
            quantized.qweight,
        ),
    );
    out.insert(
        format!("{prefix}.qzeros"),
        SerializableTensor::from_typed(
            SafeDtype::I32,
            vec![qzeros_shape.0, qzeros_shape.1],
            quantized.qzeros,
        ),
    );
    out.insert(
        format!("{prefix}.scales"),
        SerializableTensor::from_typed(
            SafeDtype::F32,
            vec![scales_shape.0, scales_shape.1],
            quantized.scales,
        ),
    );
    out.insert(
        format!("{prefix}.g_idx"),
        SerializableTensor::from_typed(SafeDtype::U32, vec![in_features], quantized.g_idx),
    );
    Ok(())
}

#[cfg(feature = "candle-backend")]
fn write_quantized_checkpoint(
    output_dir: &Path,
    tensors: &BTreeMap<String, SerializableTensor>,
) -> Result<()> {
    let model_path = output_dir.join("model.safetensors");
    let metadata: Option<HashMap<String, String>> = None;
    let views = tensors
        .iter()
        .map(|(name, tensor)| {
            let view = TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, metadata, &model_path)
        .with_context(|| format!("writing {}", model_path.display()))
}

#[cfg(feature = "candle-backend")]
fn write_quantized_config(
    model_dir: &Path,
    output_dir: &Path,
    opts: &GptqExportOptions,
) -> Result<()> {
    let config_path = model_dir.join("config.json");
    let content = fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let mut json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    json["quantization_config"] = serde_json::json!({
        "quant_method": "gptq",
        "bits": opts.bits,
        "group_size": opts.group_size,
        "damp_percent": opts.damp_percent,
        "act_order": opts.act_order,
    });
    let output_config = output_dir.join("config.json");
    fs::write(&output_config, serde_json::to_string_pretty(&json)?)
        .with_context(|| format!("writing {}", output_config.display()))
}

#[cfg(feature = "candle-backend")]
fn materialize_output_dir(model_dir: &Path, output_dir: &Path) -> Result<()> {
    if output_dir.exists() {
        let mut entries = fs::read_dir(output_dir)
            .with_context(|| format!("reading {}", output_dir.display()))?;
        if entries.next().is_some() {
            bail!("output directory {} already exists and is not empty", output_dir.display());
        }
    } else {
        fs::create_dir_all(output_dir)
            .with_context(|| format!("creating {}", output_dir.display()))?;
    }
    copy_support_files(model_dir, output_dir)
}

#[cfg(feature = "candle-backend")]
fn copy_support_files(src: &Path, dst: &Path) -> Result<()> {
    for entry in fs::read_dir(src).with_context(|| format!("reading {}", src.display()))? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            fs::create_dir_all(&target)?;
            copy_support_files(&path, &target)?;
            continue;
        }
        if should_skip_copy(&path) {
            continue;
        }
        fs::copy(&path, &target)
            .with_context(|| format!("copying {} to {}", path.display(), target.display()))?;
    }
    Ok(())
}

#[cfg(feature = "candle-backend")]
fn should_skip_copy(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    name == "config.json"
        || name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json")
}

#[cfg(feature = "candle-backend")]
fn serialize_candle_tensor(tensor: &Tensor) -> Result<SerializableTensor> {
    let shape = tensor.dims().to_vec();
    match tensor.dtype() {
        DType::F32 => Ok(SerializableTensor::from_typed(
            SafeDtype::F32,
            shape,
            tensor.flatten_all()?.to_vec1::<f32>()?,
        )),
        DType::F16 => Ok(SerializableTensor::from_typed(
            SafeDtype::F16,
            shape,
            tensor.flatten_all()?.to_vec1::<f16>()?,
        )),
        DType::BF16 => Ok(SerializableTensor::from_typed(
            SafeDtype::BF16,
            shape,
            tensor.flatten_all()?.to_vec1::<bf16>()?,
        )),
        DType::U32 => Ok(SerializableTensor::from_typed(
            SafeDtype::U32,
            shape,
            tensor.flatten_all()?.to_vec1::<u32>()?,
        )),
        DType::I32 => Ok(SerializableTensor::from_typed(
            SafeDtype::I32,
            shape,
            tensor.flatten_all()?.to_vec1::<i32>()?,
        )),
        other => bail!("unsupported tensor dtype for safetensors export: {other:?}"),
    }
}

#[cfg(feature = "candle-backend")]
struct SerializableTensor {
    dtype: SafeDtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[cfg(feature = "candle-backend")]
impl SerializableTensor {
    fn from_typed<T: Copy>(dtype: SafeDtype, shape: Vec<usize>, values: Vec<T>) -> Self {
        Self { dtype, shape, bytes: typed_vec_to_bytes(values) }
    }
}

#[cfg(feature = "candle-backend")]
fn typed_vec_to_bytes<T: Copy>(values: Vec<T>) -> Vec<u8> {
    let len = values.len() * std::mem::size_of::<T>();
    let cap = values.capacity() * std::mem::size_of::<T>();
    let ptr = values.as_ptr() as *mut u8;
    std::mem::forget(values);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

#[cfg(all(test, feature = "candle-backend"))]
mod tests {
    use super::*;

    #[test]
    fn typed_vec_to_bytes_preserves_size() {
        let bytes = typed_vec_to_bytes(vec![1u32, 2u32, 3u32]);
        assert_eq!(bytes.len(), 12);
    }
}
