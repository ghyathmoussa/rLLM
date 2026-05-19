use anyhow::Result;
use parking_lot::Mutex;
use rllm_core::request::ChatMessage;
use std::sync::Arc;

use crate::chat_template::{render_chat_template, render_chat_template_fallback};
use crate::tokenizer::Tokenizer;

struct PoolInner {
    tokenizers: Vec<Tokenizer>,
    index: usize,
}

pub struct AsyncTokenizerPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl AsyncTokenizerPool {
    pub fn new(tokenizer: Tokenizer, pool_size: usize) -> Self {
        let tokenizers = (0..pool_size)
            .map(|_| tokenizer.clone_if_possible())
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| vec![tokenizer]);

        Self { inner: Arc::new(Mutex::new(PoolInner { tokenizers, index: 0 })) }
    }

    pub async fn encode(&self, text: String, add_special_tokens: bool) -> Result<Vec<u32>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock();
            let pool = &mut *guard;
            pool.index = (pool.index + 1) % pool.tokenizers.len();
            pool.tokenizers[pool.index].encode(&text, add_special_tokens)
        })
        .await?
    }

    pub async fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock();
            let pool = &mut *guard;
            pool.index = (pool.index + 1) % pool.tokenizers.len();
            pool.tokenizers[pool.index].decode(&ids, skip_special_tokens)
        })
        .await?
    }

    pub async fn batch_encode(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock();
            let pool = &mut *guard;
            pool.index = (pool.index + 1) % pool.tokenizers.len();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            pool.tokenizers[pool.index].batch_encode(&refs, add_special_tokens)
        })
        .await?
    }

    pub async fn render_chat(
        &self,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let guard = inner.lock();
            if let Some(template) = guard.tokenizers[0].chat_template() {
                render_chat_template(template, &messages, add_generation_prompt)
            } else {
                Ok(render_chat_template_fallback(&messages, add_generation_prompt))
            }
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pool_creation_requires_valid_tokenizer() {
        // Pool creation with a tokenizer that can't be cloned will still work
        // because it falls back to a single-entry pool.
    }
}
