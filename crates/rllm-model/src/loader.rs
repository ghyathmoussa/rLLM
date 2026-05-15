use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "candle-backend")]
use candle_core::Device;

#[cfg(feature = "candle-backend")]
pub struct WeightMap {
    pub weights: HashMap<String, candle_core::Tensor>,
    pub device: Device,
}

/// Load weights from a local directory containing SafeTensors files.
#[cfg(feature = "candle-backend")]
pub fn load_weights_from_dir(model_dir: &Path, device: &Device) -> Result<WeightMap> {
    let shard_paths = find_safetensor_shards(model_dir)?;
    let mut weights = HashMap::new();

    for shard_path in &shard_paths {
        let shard_weights = candle_core::safetensors::load(shard_path, device)
            .with_context(|| format!("loading shard {}", shard_path.display()))?;
        weights.extend(shard_weights);
    }

    tracing::info!(
        "loaded {} weight tensors from {} shard(s)",
        weights.len(),
        shard_paths.len()
    );

    Ok(WeightMap {
        weights,
        device: device.clone(),
    })
}

/// Load weights from a Hugging Face model ID (downloads via hf-hub).
#[cfg(feature = "candle-backend")]
pub async fn load_weights_from_hub(model_id: &str, device: &Device) -> Result<WeightMap> {
    let model_id_owned = model_id.to_string();
    let device = device.clone();
    tokio::task::spawn_blocking(move || {
        let api = hf_hub::api::sync::Api::new()
            .with_context(|| format!("creating HF API client for {model_id_owned}"))?;
        let repo = api.model(model_id_owned.clone());
        let model_dir = repo.get("config.json")?.parent().unwrap().to_path_buf();
        load_weights_from_dir(&model_dir, &device)
    })
    .await?
}

/// Load weights with auto-detection of tied lm_head.
#[cfg(feature = "candle-backend")]
pub fn load_weights_with_tied_detection(
    model_dir: &Path,
    device: &Device,
) -> Result<(WeightMap, bool)> {
    let weight_map = load_weights_from_dir(model_dir, device)?;
    let has_lm_head = weight_map.weights.contains_key("lm_head.weight");
    let has_embed = weight_map
        .weights
        .contains_key("model.embed_tokens.weight");
    let tied = !has_lm_head && has_embed;
    Ok((weight_map, tied))
}

fn find_safetensor_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("model directory does not exist: {}", dir.display());
    }

    // Check for an index file first (sharded models)
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return load_shard_index(&index_path, dir);
    }

    // Single-file model
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    // Try numbered shards without index
    let mut shards = Vec::new();
    for i in 0..1000usize {
        match find_shard_by_index(dir, i) {
            Some(p) => shards.push(p),
            None => break,
        }
    }

    if shards.is_empty() {
        anyhow::bail!("no SafeTensors files found in {}", dir.display());
    }

    Ok(shards)
}

fn find_shard_by_index(dir: &Path, index: usize) -> Option<PathBuf> {
    let prefix = format!("model-{index:05}-of-");
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(&prefix) && name_str.ends_with(".safetensors") {
            return Some(entry.path());
        }
    }
    None
}

#[derive(serde::Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

fn load_shard_index(index_path: &Path, dir: &Path) -> Result<Vec<PathBuf>> {
    let content = std::fs::read_to_string(index_path)?;
    let index: ShardIndex = serde_json::from_str(&content)?;

    let mut shard_files: Vec<String> = index
        .weight_map
        .values()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    shard_files.sort();

    let paths: Vec<PathBuf> = shard_files.iter().map(|f| dir.join(f)).collect();

    for p in &paths {
        if !p.exists() {
            anyhow::bail!("shard file not found: {}", p.display());
        }
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_shards_in_missing_dir() {
        let result = find_safetensor_shards(Path::new("/nonexistent"));
        assert!(result.is_err());
    }
}
