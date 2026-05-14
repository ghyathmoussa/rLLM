use anyhow::Result;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { inner })
    }

    pub fn from_model_id(model_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        Self::from_file(tokenizer_path.to_str().unwrap_or(""))
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(text)
    }

    pub fn batch_encode(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    pub fn batch_decode(&self, batch: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        let ids: Vec<&[u32]> = batch.iter().map(|v| v.as_slice()).collect();
        let texts = self
            .inner
            .decode_batch(&ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(texts)
    }
}
