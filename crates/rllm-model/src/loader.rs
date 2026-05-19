use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "candle-backend")]
use candle_core::Device;
#[cfg(feature = "candle-backend")]
use std::io::{self, IsTerminal, Write};

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

    tracing::debug!(
        model_dir = %model_dir.display(),
        num_shards = shard_paths.len(),
        "loading SafeTensors shards"
    );
    for shard_path in &shard_paths {
        tracing::debug!(shard = %shard_path.display(), "loading SafeTensors shard");
        let shard_weights = candle_core::safetensors::load(shard_path, device)
            .with_context(|| format!("loading shard {}", shard_path.display()))?;
        tracing::debug!(
            shard = %shard_path.display(),
            tensors = shard_weights.len(),
            "SafeTensors shard loaded"
        );
        weights.extend(shard_weights);
    }

    tracing::info!("loaded {} weight tensors from {} shard(s)", weights.len(), shard_paths.len());

    Ok(WeightMap { weights, device: device.clone() })
}

/// Load weights from a Hugging Face model ID (downloads via hf-hub).
#[cfg(feature = "candle-backend")]
pub async fn load_weights_from_hub(model_id: &str, device: &Device) -> Result<WeightMap> {
    let model_id_owned = model_id.to_string();
    let device = device.clone();
    tokio::task::spawn_blocking(move || {
        let model_dir = download_model_from_hub(&model_id_owned)?;
        load_weights_from_dir(&model_dir, &device)
    })
    .await?
}

/// Resolve a local path or Hugging Face model ID to a local model directory.
///
/// Remote models are downloaded into the standard Hugging Face cache. If a
/// gated/private repo rejects unauthenticated access, this prompts for a token
/// on interactive terminals and retries with that token.
pub fn resolve_model_dir(model_ref: &str) -> Result<PathBuf> {
    let path = Path::new(model_ref);
    if path.is_dir() {
        tracing::debug!(model = %model_ref, "resolved model as local directory");
        return Ok(path.to_path_buf());
    }

    #[cfg(feature = "candle-backend")]
    {
        download_model_from_hub(model_ref)
    }
    #[cfg(not(feature = "candle-backend"))]
    {
        anyhow::bail!(
            "remote Hugging Face model resolution requires the candle-backend feature: {model_ref}"
        );
    }
}

#[cfg(feature = "candle-backend")]
pub fn download_model_from_hub(model_id: &str) -> Result<PathBuf> {
    tracing::info!(model = %model_id, "resolving Hugging Face model");
    let token = token_from_env();
    match download_model_from_hub_with_token(model_id, token.clone()) {
        Ok(path) => Ok(path),
        Err(err) if is_auth_error(&err) && token.is_none() => {
            let Some(token) = prompt_for_hf_token(model_id)? else {
                anyhow::bail!(
                    "Hugging Face model '{model_id}' requires authentication. Set HF_TOKEN or run on an interactive terminal to enter a token."
                );
            };
            download_model_from_hub_with_token(model_id, Some(token))
        }
        Err(err) => Err(err),
    }
}

#[cfg(feature = "candle-backend")]
fn download_model_from_hub_with_token(model_id: &str, token: Option<String>) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;

    let mut builder = ApiBuilder::from_env().with_retries(3);
    if token.is_some() {
        builder = builder.with_token(token.clone());
    }
    let api = builder.build().with_context(|| format!("creating HF API client for {model_id}"))?;
    let repo = api.model(model_id.to_string());

    tracing::debug!(model = %model_id, "fetching Hugging Face repo info");
    let info =
        repo.info().with_context(|| format!("fetching Hugging Face repo info for {model_id}"))?;
    tracing::debug!(
        model = %model_id,
        sha = %info.sha,
        files = info.siblings.len(),
        "Hugging Face repo info fetched"
    );

    let config_path = repo
        .get("config.json")
        .with_context(|| format!("downloading config.json for {model_id}"))?;
    let model_dir =
        config_path.parent().context("HF cache path for config.json has no parent")?.to_path_buf();

    let files = files_to_download(&repo, &info)?;
    tracing::info!(
        model = %model_id,
        files = files.len(),
        cache_dir = %model_dir.display(),
        concurrency = hf_download_concurrency(),
        "downloading Hugging Face model files"
    );
    download_hf_files_concurrently(model_id, files, token)?;

    Ok(model_dir)
}

#[cfg(feature = "candle-backend")]
fn download_hf_files_concurrently(
    model_id: &str,
    files: Vec<String>,
    token: Option<String>,
) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }

    let concurrency = hf_download_concurrency().min(files.len()).max(1);
    for (batch_idx, batch) in files.chunks(concurrency).enumerate() {
        tracing::info!(
            model = %model_id,
            batch = batch_idx + 1,
            batch_size = batch.len(),
            concurrency,
            "starting concurrent Hugging Face download batch"
        );

        let handles = batch
            .iter()
            .cloned()
            .map(|file| {
                let model_id = model_id.to_string();
                let token = token.clone();
                std::thread::spawn(move || download_one_hf_file(&model_id, &file, token))
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("Hugging Face download worker panicked"))??;
        }
    }

    Ok(())
}

#[cfg(feature = "candle-backend")]
fn download_one_hf_file(model_id: &str, file: &str, token: Option<String>) -> Result<()> {
    use hf_hub::api::sync::ApiBuilder;

    tracing::debug!(model = %model_id, file = %file, "downloading Hugging Face file");
    let mut builder = ApiBuilder::from_env()
        .with_retries(3)
        .with_progress(false);
    if token.is_some() {
        builder = builder.with_token(token);
    }
    let api = builder
        .build()
        .with_context(|| format!("creating HF API client for {model_id}"))?;
    let repo = api.model(model_id.to_string());
    repo.get(file)
        .with_context(|| format!("downloading {file} for {model_id}"))?;
    tracing::debug!(model = %model_id, file = %file, "downloaded Hugging Face file");
    Ok(())
}

#[cfg(feature = "candle-backend")]
fn hf_download_concurrency() -> usize {
    std::env::var("RLLM_HF_DOWNLOAD_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

#[cfg(feature = "candle-backend")]
fn files_to_download(
    repo: &hf_hub::api::sync::ApiRepo,
    info: &hf_hub::api::RepoInfo,
) -> Result<Vec<String>> {
    let has_file = |name: &str| info.siblings.iter().any(|s| s.rfilename == name);

    if has_file("model.safetensors.index.json") {
        let index_path = repo.get("model.safetensors.index.json")?;
        let shard_paths = load_shard_index(&index_path, Path::new("."))?;
        let files =
            shard_paths.into_iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>();
        tracing::debug!(files = files.len(), "planned sharded SafeTensors download");
        return Ok(files);
    }

    if has_file("model.safetensors") {
        tracing::debug!("planned single SafeTensors download");
        return Ok(vec!["model.safetensors".to_string()]);
    }

    let mut files = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| name.ends_with(".safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    if !files.is_empty() {
        tracing::debug!(files = files.len(), "planned SafeTensors download from repo listing");
        return Ok(files);
    }

    anyhow::bail!("no SafeTensors files found in Hugging Face repo listing");
}

#[cfg(feature = "candle-backend")]
fn token_from_env() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]
        .iter()
        .find_map(|key| std::env::var(key).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

#[cfg(feature = "candle-backend")]
fn prompt_for_hf_token(model_id: &str) -> Result<Option<String>> {
    if !io::stdin().is_terminal() {
        return Ok(None);
    }

    eprintln!("Hugging Face model '{model_id}' requires a token.");
    eprint!("Enter HF token: ");
    io::stderr().flush().ok();

    let mut token = String::new();
    io::stdin().read_line(&mut token)?;
    let token = token.trim().to_string();
    if token.is_empty() { Ok(None) } else { Ok(Some(token)) }
}

#[cfg(feature = "candle-backend")]
fn is_auth_error(err: &anyhow::Error) -> bool {
    let text = format!("{err:#}");
    text.contains("401")
        || text.contains("403")
        || text.contains("Unauthorized")
        || text.contains("Forbidden")
}

/// Load weights with auto-detection of tied lm_head.
#[cfg(feature = "candle-backend")]
pub fn load_weights_with_tied_detection(
    model_dir: &Path,
    device: &Device,
) -> Result<(WeightMap, bool)> {
    let weight_map = load_weights_from_dir(model_dir, device)?;
    let has_lm_head = weight_map.weights.contains_key("lm_head.weight");
    let has_embed = weight_map.weights.contains_key("model.embed_tokens.weight");
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
        tracing::debug!(index = %index_path.display(), "using SafeTensors index");
        return load_shard_index(&index_path, dir);
    }

    // Single-file model
    let single = dir.join("model.safetensors");
    if single.exists() {
        tracing::debug!(file = %single.display(), "using single SafeTensors file");
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
    let content = std::fs::read_to_string(index_path)
        .with_context(|| format!("reading SafeTensors index {}", index_path.display()))?;
    let index: ShardIndex = serde_json::from_str(&content)
        .with_context(|| format!("parsing SafeTensors index {}", index_path.display()))?;

    let mut shard_files: Vec<String> = index
        .weight_map
        .values()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    shard_files.sort();

    let paths: Vec<PathBuf> = shard_files
        .iter()
        .map(|f| {
            // Sanitize: reject filenames with path traversal components
            let p = Path::new(f);
            if f.starts_with('/') || f.starts_with('\\') || f.contains("..") {
                anyhow::bail!(
                    "path traversal detected in shard filename: {}",
                    p.display()
                );
            }
            // Canonicalize to verify the path stays within dir
            let joined = dir.join(f);
            let canonical = joined.canonicalize().with_context(|| {
                format!("resolving shard path: {}", joined.display())
            })?;
            if !canonical.starts_with(&dir.canonicalize()?) {
                anyhow::bail!(
                    "shard path escapes model directory: {}",
                    joined.display()
                );
            }
            Ok(joined)
        })
        .collect::<Result<Vec<_>>>()?;

    for p in &paths {
        if dir != Path::new(".") && !p.exists() {
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

    #[cfg(feature = "candle-backend")]
    #[test]
    #[ignore = "downloads from Hugging Face; set RLLM_REMOTE_HF_TEST_MODEL to override"]
    fn downloads_remote_hf_llama_model_files() {
        let model_id = std::env::var("RLLM_REMOTE_HF_TEST_MODEL")
            .unwrap_or_else(|_| "hf-internal-testing/tiny-random-LlamaForCausalLM".to_string());
        let model_dir = download_model_from_hub(&model_id).unwrap();
        assert!(model_dir.join("config.json").exists());
        assert!(!find_safetensor_shards(&model_dir).unwrap().is_empty());

        let local_runner =
            crate::runner::ModelRunner::from_dir(model_dir.to_str().unwrap()).unwrap();
        assert_eq!(local_runner.config().architecture, "LlamaForCausalLM");
        assert!(local_runner.generate(&[1, 2], 3).unwrap().len() > 2);

        let remote_runner = crate::runner::ModelRunner::from_model_ref(&model_id).unwrap();
        assert_eq!(remote_runner.config().architecture, "LlamaForCausalLM");
    }
}
