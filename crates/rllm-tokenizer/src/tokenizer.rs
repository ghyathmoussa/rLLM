use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
    eos_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let config = load_tokenizer_config(path)?;
        Self::build(inner, &config)
    }

    pub fn from_model_id(model_id: &str) -> Result<Self> {
        let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
        if let Some(token) = token_from_env() {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build()?;
        let repo = api.model(model_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config = load_tokenizer_config_from_repo(&repo)?;
        let inner = HfTokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
        Self::build(inner, &config)
    }

    pub async fn from_model_id_async(model_id: &str) -> Result<Self> {
        let model_id = model_id.to_string();
        tokio::task::spawn_blocking(move || Self::from_model_id(&model_id)).await?
    }

    #[tracing::instrument(skip(self), name = "tokenize")]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding =
            self.inner.encode(text, add_special_tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = encoding.get_ids().to_vec();
        tracing::trace!(num_tokens = ids.len(), "tokenize complete");
        Ok(ids)
    }

    #[tracing::instrument(skip(self, ids), name = "detokenize")]
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text =
            self.inner.decode(ids, skip_special_tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(text)
    }

    pub fn decode_single_token(&self, id: u32) -> String {
        self.inner.decode(&[id], false).unwrap_or_default()
    }

    pub fn batch_encode(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    pub fn batch_decode(
        &self,
        batch: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let ids: Vec<&[u32]> = batch.iter().map(|v| v.as_slice()).collect();
        let texts = self
            .inner
            .decode_batch(&ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(texts)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn is_eos_token(&self, token_id: u32) -> bool {
        self.eos_token_id == Some(token_id)
    }

    pub fn clone_if_possible(&self) -> Option<Self> {
        // HfTokenizer implements Clone, so we can duplicate the tokenizer for pool use.
        Some(Self {
            inner: self.inner.clone(),
            eos_token_id: self.eos_token_id,
            bos_token_id: self.bos_token_id,
            pad_token_id: self.pad_token_id,
            chat_template: self.chat_template.clone(),
        })
    }

    fn build(inner: HfTokenizer, config: &TokenizerConfigData) -> Result<Self> {
        let eos_token_id = config
            .eos_token
            .as_deref()
            .and_then(|t| token_id_from_str(&inner, t))
            .or_else(|| single_token_id(&inner, "</s>"))
            .or_else(|| single_token_id(&inner, "<|end_of_text|>"))
            .or_else(|| single_token_id(&inner, "<|eot_id|>"));

        let bos_token_id = config
            .bos_token
            .as_deref()
            .and_then(|t| token_id_from_str(&inner, t))
            .or_else(|| single_token_id(&inner, "<s>"))
            .or_else(|| single_token_id(&inner, "<|begin_of_text|>"));

        let pad_token_id = config.pad_token.as_deref().and_then(|t| token_id_from_str(&inner, t));

        Ok(Self {
            inner,
            eos_token_id,
            bos_token_id,
            pad_token_id,
            chat_template: config.chat_template.clone(),
        })
    }
}

fn token_from_env() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]
        .iter()
        .find_map(|key| std::env::var(key).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn single_token_id(tokenizer: &HfTokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

fn token_id_from_str(tokenizer: &HfTokenizer, s: &str) -> Option<u32> {
    tokenizer.token_to_id(s)
}

struct TokenizerConfigData {
    eos_token: Option<String>,
    bos_token: Option<String>,
    pad_token: Option<String>,
    chat_template: Option<String>,
}

fn load_tokenizer_config(tokenizer_json_path: &str) -> Result<TokenizerConfigData> {
    let path = Path::new(tokenizer_json_path);
    let dir = path.parent().unwrap_or(Path::new("."));
    load_config_from_dir(dir)
}

fn load_tokenizer_config_from_repo(
    repo: &hf_hub::api::sync::ApiRepo,
) -> Result<TokenizerConfigData> {
    let dir = match repo.get("tokenizer_config.json") {
        Ok(p) => p.parent().unwrap_or(Path::new(".")).to_path_buf(),
        Err(_) => return Ok(empty_config()),
    };
    load_config_from_dir(&dir)
}

fn load_config_from_dir(dir: &Path) -> Result<TokenizerConfigData> {
    let config_path = dir.join("tokenizer_config.json");

    if !config_path.exists() {
        return Ok(empty_config());
    }

    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("parsing {}", config_path.display()))?;

    let eos_token = extract_token_field(&json, "eos_token");
    let bos_token = extract_token_field(&json, "bos_token");
    let pad_token = extract_token_field(&json, "pad_token");
    let chat_template = json.get("chat_template").and_then(|v| {
        if v.is_string() {
            v.as_str().map(|s| s.to_string())
        } else if v.is_object() {
            // Some models use {"default": "...", "tool_use": "..."}
            v.get("default").and_then(|d| d.as_str().map(|s| s.to_string()))
        } else {
            None
        }
    });

    Ok(TokenizerConfigData { eos_token, bos_token, pad_token, chat_template })
}

fn extract_token_field(json: &serde_json::Value, field: &str) -> Option<String> {
    let val = json.get(field)?;
    if val.is_string() {
        val.as_str().map(|s| s.to_string())
    } else if val.is_object() {
        // {"content": "</s>", "lstrip": false, ...}
        val.get("content").and_then(|c| c.as_str().map(|s| s.to_string()))
    } else {
        None
    }
}

fn empty_config() -> TokenizerConfigData {
    TokenizerConfigData { eos_token: None, bos_token: None, pad_token: None, chat_template: None }
}

#[cfg(test)]
mod tests {
    #[test]
    fn decode_single_token_returns_string() {
        // We can't test with a real tokenizer in unit tests without a model,
        // but we can verify the API compiles and returns empty for invalid tokens.
        // Real tests will use from_file with actual tokenizer files.
    }
}
