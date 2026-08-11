use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use dlpark::prelude::{SafeManagedTensorVersioned, TensorView};
use rllm_core::{ids::RequestId, request::StructuredOutputParams};
use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, get_bitmask_size};

/// Tokenizer-aware, per-request XGrammar state for constrained decoding.
pub struct StructuredOutputManager {
    compiler: GrammarCompiler,
    vocab_size: usize,
    bitmask_len: usize,
    matchers: HashMap<RequestId, GrammarMatcher>,
}

pub fn validate_structured_output(params: &StructuredOutputParams) -> Result<()> {
    params.validate().context("invalid structured output parameters")?;
    if let Some(schema) = &params.json_schema {
        let schema = match schema {
            serde_json::Value::String(schema) => schema.clone(),
            schema => schema.to_string(),
        };
        Grammar::from_json_schema(&schema, None, None, None, Some(true), None, None)
            .context("validating JSON schema with XGrammar")?;
    } else if let Some(regex) = &params.regex {
        Grammar::from_regex(regex, None).context("validating regex with XGrammar")?;
    } else if let Some(choices) = &params.choice {
        let regex = choices.iter().map(|choice| regex_escape(choice)).collect::<Vec<_>>().join("|");
        Grammar::from_regex(&format!("(?:{regex})"), None)
            .context("validating choice constraint with XGrammar")?;
    } else if let Some(ebnf) = params.xml.as_ref().or(params.grammar.as_ref()) {
        Grammar::from_ebnf(ebnf, None).context("validating EBNF with XGrammar")?;
    }
    Ok(())
}

impl StructuredOutputManager {
    pub fn new(tokenizer_backend: &str, vocab_size: usize, eos_token_id: u32) -> Result<Self> {
        let vocab_size_i32 =
            i32::try_from(vocab_size).context("tokenizer vocabulary is too large")?;
        if eos_token_id as usize >= vocab_size {
            bail!(
                "structured outputs require a valid EOS token ID, got {eos_token_id} for vocabulary size {vocab_size}"
            );
        }
        let eos_token_id = i32::try_from(eos_token_id).context("EOS token ID is too large")?;
        let tokenizer_info = TokenizerInfo::from_backend_str(
            tokenizer_backend,
            Some(vocab_size),
            vec![eos_token_id],
        )
        .context("building XGrammar tokenizer metadata")?;
        let compiler = GrammarCompiler::new(&tokenizer_info);
        let bitmask_len = get_bitmask_size(vocab_size_i32) as usize;
        Ok(Self { compiler, vocab_size, bitmask_len, matchers: HashMap::new() })
    }

    pub fn register(
        &mut self,
        request_id: RequestId,
        params: &StructuredOutputParams,
    ) -> Result<()> {
        params.validate().context("invalid structured output parameters")?;
        let compiled = if let Some(schema) = &params.json_schema {
            let schema = match schema {
                serde_json::Value::String(schema) => schema.clone(),
                schema => schema.to_string(),
            };
            self.compiler
                .compile_json_schema(&schema, None, None, None, Some(true), None)
                .context("compiling JSON schema with XGrammar")?
        } else if params.json_object == Some(true) {
            self.compiler
                .compile_builtin_json_grammar()
                .context("compiling the built-in JSON grammar")?
        } else if let Some(regex) = &params.regex {
            self.compiler.compile_regex(regex).context("compiling regex with XGrammar")?
        } else if let Some(choices) = &params.choice {
            let regex =
                choices.iter().map(|choice| regex_escape(choice)).collect::<Vec<_>>().join("|");
            self.compiler
                .compile_regex(&format!("(?:{regex})"))
                .context("compiling choice constraint with XGrammar")?
        } else {
            let (kind, ebnf) = if let Some(xml) = &params.xml {
                ("XML", xml)
            } else if let Some(grammar) = &params.grammar {
                ("EBNF", grammar)
            } else {
                bail!("structured output parameters did not contain a constraint");
            };
            let grammar = Grammar::from_ebnf(ebnf, None)
                .with_context(|| format!("parsing {kind} grammar with XGrammar"))?;
            self.compiler
                .compile_grammar(&grammar)
                .with_context(|| format!("compiling {kind} grammar with XGrammar"))?
        };

        self.matchers.insert(request_id, GrammarMatcher::new(&compiled));
        Ok(())
    }

    /// Applies the current grammar bitmask directly to CPU logits.
    pub fn mask_logits(&mut self, request_id: &RequestId, logits: &mut [f32]) -> Result<()> {
        let Some(matcher) = self.matchers.get_mut(request_id) else {
            return Ok(());
        };
        if logits.len() < self.vocab_size {
            bail!(
                "logits vocabulary size {} is smaller than tokenizer vocabulary size {}",
                logits.len(),
                self.vocab_size
            );
        }

        let mut bitmask = SafeManagedTensorVersioned::new(vec![0i32; self.bitmask_len])
            .context("allocating XGrammar token bitmask")?;
        if matcher
            .fill_next_token_bitmask(&mut bitmask, None, None)
            .context("computing XGrammar token bitmask")?
        {
            let words: &[i32] =
                bitmask.as_slice_contiguous().context("reading XGrammar token bitmask")?;
            for token_id in 0..self.vocab_size {
                let word = words[token_id / 32] as u32;
                if word & (1u32 << (token_id % 32)) == 0 {
                    logits[token_id] = f32::NEG_INFINITY;
                }
            }
        }
        if logits[..self.vocab_size].iter().all(|logit| !logit.is_finite()) {
            bail!("structured output constraint rejected every token");
        }
        Ok(())
    }

    pub fn accept_token(&mut self, request_id: &RequestId, token_id: u32) -> Result<()> {
        let Some(matcher) = self.matchers.get_mut(request_id) else {
            return Ok(());
        };
        let token_id = i32::try_from(token_id).context("sampled token ID is too large")?;
        if !matcher.accept_token(token_id, None) {
            bail!("XGrammar rejected sampled token {token_id}");
        }
        Ok(())
    }

    pub fn remove(&mut self, request_id: &RequestId) {
        self.matchers.remove(request_id);
    }
}

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str =
        r#"{"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"<eos>":2},"unk_token":"a"}}"#;

    #[test]
    fn rejects_invalid_xml_grammar() {
        let mut manager = StructuredOutputManager::new(TOKENIZER, 3, 2).unwrap();
        let params = StructuredOutputParams {
            json_schema: None,
            json_object: None,
            xml: Some("root ::= \"<broken>".into()),
            regex: None,
            grammar: None,
            choice: None,
        };
        assert!(manager.register(RequestId::new(), &params).is_err());
    }

    #[test]
    fn unconstrained_request_is_a_noop() {
        let mut manager = StructuredOutputManager::new(TOKENIZER, 3, 2).unwrap();
        let mut logits = vec![1.0, 2.0, 3.0];
        manager.mask_logits(&RequestId::new(), &mut logits).unwrap();
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }
}
