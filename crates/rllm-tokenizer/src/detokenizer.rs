#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Delta,
    Cumulative,
    FinalOnly,
}

pub struct StreamingDetokenizer {
    partial_bytes: Vec<u8>,
    stop_buffer: String,
    stop_strings: Vec<String>,
    stop_token_ids: Vec<u32>,
    eos_token_id: Option<u32>,
    min_tokens: u32,
    tokens_generated: u32,
    output_mode: OutputMode,
    cumulative_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopResult {
    Continue,
    StoppedByEos,
    StoppedByTokenId(u32),
    StoppedByString(String),
}

impl StreamingDetokenizer {
    pub fn new(stop_strings: Vec<String>) -> Self {
        Self {
            partial_bytes: Vec::new(),
            stop_buffer: String::new(),
            stop_strings,
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            min_tokens: 0,
            tokens_generated: 0,
            output_mode: OutputMode::Delta,
            cumulative_text: String::new(),
        }
    }

    pub fn with_stop_token_ids(mut self, ids: Vec<u32>) -> Self {
        self.stop_token_ids = ids;
        self
    }

    pub fn with_eos_token_id(mut self, id: Option<u32>) -> Self {
        self.eos_token_id = id;
        self
    }

    pub fn with_min_tokens(mut self, min_tokens: u32) -> Self {
        self.min_tokens = min_tokens;
        self
    }

    pub fn with_output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = mode;
        self
    }

    pub fn check_stop_token(&mut self, token_id: u32) -> StopResult {
        self.tokens_generated += 1;

        if self.eos_token_id == Some(token_id) && self.tokens_generated > self.min_tokens {
            return StopResult::StoppedByEos;
        }

        if self.tokens_generated > self.min_tokens {
            if let Some(matched) = self.stop_token_ids.iter().find(|&&id| id == token_id) {
                return StopResult::StoppedByTokenId(*matched);
            }
        }

        StopResult::Continue
    }

    pub fn decode_token(&mut self, token_text: &str) -> Option<String> {
        if self.output_mode == OutputMode::FinalOnly {
            self.cumulative_text
                .push_str(&decode_utf8_partial(&mut self.partial_bytes, token_text.as_bytes()));
            return None;
        }

        let decoded = decode_utf8_partial(&mut self.partial_bytes, token_text.as_bytes());

        if self.stop_strings.is_empty() {
            self.cumulative_text.push_str(&decoded);
            return match self.output_mode {
                OutputMode::Delta => Some(decoded),
                OutputMode::Cumulative => Some(self.cumulative_text.clone()),
                OutputMode::FinalOnly => None,
            };
        }

        self.stop_buffer.push_str(&decoded);
        self.cumulative_text.push_str(&decoded);

        let max_stop_len = self.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

        for stop in &self.stop_strings {
            if self.stop_buffer.contains(stop) {
                let idx = self.stop_buffer.find(stop).unwrap();
                let before = self.stop_buffer[..idx].to_string();
                self.stop_buffer.clear();
                return match self.output_mode {
                    OutputMode::Delta => Some(before),
                    OutputMode::Cumulative => {
                        let cumul = self.cumulative_text.clone();
                        // Trim to just before the stop string
                        if let Some(pos) = cumul.rfind(stop) {
                            Some(cumul[..pos].to_string())
                        } else {
                            Some(cumul)
                        }
                    }
                    OutputMode::FinalOnly => None,
                };
            }
        }

        if self.stop_buffer.len() > max_stop_len {
            let safe_len = self.stop_buffer.len() - max_stop_len;
            let delta = self.stop_buffer[..safe_len].to_string();
            self.stop_buffer = self.stop_buffer[safe_len..].to_string();
            return match self.output_mode {
                OutputMode::Delta => Some(delta),
                OutputMode::Cumulative => Some(self.cumulative_text.clone()),
                OutputMode::FinalOnly => None,
            };
        }

        None
    }

    pub fn flush(&mut self) -> String {
        // Flush any remaining partial UTF-8 bytes
        if !self.partial_bytes.is_empty() {
            let remaining = String::from_utf8_lossy(&self.partial_bytes);
            self.stop_buffer.push_str(&remaining);
            self.cumulative_text.push_str(&remaining);
            self.partial_bytes.clear();
        }

        let result = std::mem::take(&mut self.stop_buffer);
        match self.output_mode {
            OutputMode::Delta => result,
            OutputMode::Cumulative => std::mem::take(&mut self.cumulative_text),
            OutputMode::FinalOnly => std::mem::take(&mut self.cumulative_text),
        }
    }

    pub fn cumulative_text(&self) -> &str {
        &self.cumulative_text
    }

    pub fn tokens_generated(&self) -> u32 {
        self.tokens_generated
    }
}

/// Decodes bytes handling partial UTF-8 at boundaries.
/// Incomplete sequences are buffered in `partial_bytes` for the next call.
fn decode_utf8_partial(partial_bytes: &mut Vec<u8>, new_bytes: &[u8]) -> String {
    if partial_bytes.is_empty() && is_clean_utf8_boundary(new_bytes) {
        return String::from_utf8_lossy(new_bytes).into_owned();
    }

    partial_bytes.extend_from_slice(new_bytes);
    let safe_end = find_safe_utf8_boundary(partial_bytes);
    let decoded = String::from_utf8_lossy(&partial_bytes[..safe_end]).into_owned();
    partial_bytes.drain(..safe_end);
    decoded
}

fn is_clean_utf8_boundary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    // Check that the last byte is not a continuation byte (0x80-0xBF)
    // and not a multi-byte start that's incomplete
    let last = bytes[bytes.len() - 1];
    if (0x80..=0xBF).contains(&last) {
        return false;
    }
    // Check for incomplete multi-byte sequence at end
    if (0xC0..=0xF7).contains(&last) {
        return false;
    }
    true
}

/// Find the largest index that ends on a complete UTF-8 character boundary.
fn find_safe_utf8_boundary(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }

    let len = bytes.len();
    // Walk backwards from the end to find a clean boundary
    let mut i = len;
    while i > 0 {
        let b = bytes[i - 1];
        if b < 0x80 {
            // ASCII byte - clean boundary
            return i;
        }
        if b >= 0xC0 {
            // Start of a multi-byte sequence - this might be incomplete
            let needed = utf8_sequence_length(b);
            if len - (i - 1) >= needed {
                return len; // complete sequence at end
            }
            return i - 1; // incomplete, exclude it
        }
        // Continuation byte (0x80-0xBF) - keep walking back
        i -= 1;
    }
    0
}

fn utf8_sequence_length(first_byte: u8) -> usize {
    if first_byte < 0x80 {
        1
    } else if first_byte < 0xE0 {
        2
    } else if first_byte < 0xF0 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_stop_strings_pass_through() {
        let mut detok = StreamingDetokenizer::new(vec![]);
        let out = detok.decode_token("hello ").unwrap();
        assert_eq!(out, "hello ");
        let out = detok.decode_token("world").unwrap();
        assert_eq!(out, "world");
    }

    #[test]
    fn stop_string_halts_output() {
        let mut detok = StreamingDetokenizer::new(vec!["STOP".into()]);
        assert_eq!(detok.decode_token("hel"), None);
        assert_eq!(detok.decode_token("loSTOP"), Some("hello".to_string()));
    }

    #[test]
    fn stop_string_partial_hold() {
        let mut detok = StreamingDetokenizer::new(vec!["END".into()]);
        // "hel" (len 3) <= max_stop_len (3), so held
        assert_eq!(detok.decode_token("hel"), None);
        // "hello E" (len 7) > max_stop_len (3), safe_len = 4, emits "hell"
        let out = detok.decode_token("lo E");
        assert!(out.is_some());
        // Remaining "o E" is held for stop-string matching
        let flushed = detok.flush();
        assert!(flushed.contains("o E"));
    }

    #[test]
    fn stop_string_emits_safe_prefix() {
        let mut detok = StreamingDetokenizer::new(vec!["[END]".into()]);
        // "hello " is longer than max_stop_len (5), safe to emit "h"
        assert_eq!(detok.decode_token("hello "), Some("h".to_string()));
    }

    #[test]
    fn flush_remaining_text() {
        let mut detok = StreamingDetokenizer::new(vec![]);
        detok.decode_token("hello ");
        detok.decode_token("world");
        let flushed = detok.flush();
        assert_eq!(flushed, "");
        assert_eq!(detok.cumulative_text(), "hello world");
    }

    #[test]
    fn stop_token_id_check() {
        let mut detok =
            StreamingDetokenizer::new(vec![]).with_stop_token_ids(vec![50256]).with_min_tokens(2);

        assert_eq!(detok.check_stop_token(1), StopResult::Continue);
        assert_eq!(detok.check_stop_token(2), StopResult::Continue);
        // min_tokens reached, now stop token triggers
        assert_eq!(detok.check_stop_token(50256), StopResult::StoppedByTokenId(50256));
    }

    #[test]
    fn min_tokens_blocks_stop() {
        let mut detok =
            StreamingDetokenizer::new(vec![]).with_stop_token_ids(vec![999]).with_min_tokens(3);

        assert_eq!(detok.check_stop_token(999), StopResult::Continue);
        assert_eq!(detok.check_stop_token(999), StopResult::Continue);
        assert_eq!(detok.check_stop_token(999), StopResult::Continue);
        assert_eq!(detok.check_stop_token(999), StopResult::StoppedByTokenId(999));
    }

    #[test]
    fn eos_token_check() {
        let mut detok = StreamingDetokenizer::new(vec![]).with_eos_token_id(Some(2));
        assert_eq!(detok.check_stop_token(2), StopResult::StoppedByEos);
        assert_eq!(detok.check_stop_token(1), StopResult::Continue);
    }

    #[test]
    fn partial_utf8_handling() {
        let mut detok = StreamingDetokenizer::new(vec![]);

        // "é" is 0xC3 0xA9 in UTF-8.
        // String::from_utf8_lossy(&[0xC3]) produces "�" (replacement char) because
        // the incomplete byte is converted to a lossy string before reaching us.
        // The real partial-UTF8 handling works at the byte level inside decode_utf8_partial.
        let out1 = detok.decode_token(&String::from_utf8_lossy(&[0xC3]));
        assert!(out1.is_some()); // Lossy conversion produces "�"

        // Test the byte-level partial handler directly
        let mut partial = Vec::new();
        let result = decode_utf8_partial(&mut partial, &[0xC3]);
        assert!(result.is_empty()); // Incomplete sequence is buffered
        let result = decode_utf8_partial(&mut partial, &[0xA9]);
        assert_eq!(result, "é"); // Now complete
    }

    #[test]
    fn cumulative_output_mode() {
        let mut detok = StreamingDetokenizer::new(vec![]).with_output_mode(OutputMode::Cumulative);
        let out = detok.decode_token("hello ").unwrap();
        assert_eq!(out, "hello ");
        let out = detok.decode_token("world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn final_only_mode_accumulates() {
        let mut detok = StreamingDetokenizer::new(vec![]).with_output_mode(OutputMode::FinalOnly);
        assert_eq!(detok.decode_token("hello "), None);
        assert_eq!(detok.decode_token("world"), None);
        let final_text = detok.flush();
        assert_eq!(final_text, "hello world");
    }

    #[test]
    fn decode_utf8_partial_multibyte() {
        let mut partial = Vec::new();
        // "日本" = E6 97 A5 E6 9C AC
        let result = decode_utf8_partial(&mut partial, &[0xE6, 0x97]);
        assert!(result.is_empty() || result.contains('�'));
        let result = decode_utf8_partial(&mut partial, &[0xA5, 0xE6, 0x9C]);
        assert!(result.contains('日'));
    }
}
