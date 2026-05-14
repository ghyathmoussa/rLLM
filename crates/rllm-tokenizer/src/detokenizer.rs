/// Streaming detokenizer that handles partial UTF-8 sequences and stop-string buffering.
pub struct StreamingDetokenizer {
    #[allow(dead_code)]
    partial_bytes: Vec<u8>,
    stop_buffer: String,
    stop_strings: Vec<String>,
}

impl StreamingDetokenizer {
    pub fn new(stop_strings: Vec<String>) -> Self {
        Self {
            partial_bytes: Vec::new(),
            stop_buffer: String::new(),
            stop_strings,
        }
    }

    /// Feed decoded text and return the safe-to-emit delta.
    /// Returns None if output is being held back for stop-string matching.
    pub fn step(&mut self, text: &str) -> Option<String> {
        if self.stop_strings.is_empty() {
            return Some(text.to_string());
        }

        self.stop_buffer.push_str(text);

        let max_stop_len = self.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

        for stop in &self.stop_strings {
            if self.stop_buffer.contains(stop) {
                let idx = self.stop_buffer.find(stop).unwrap();
                let before = self.stop_buffer[..idx].to_string();
                self.stop_buffer.clear();
                return Some(before);
            }
        }

        if self.stop_buffer.len() > max_stop_len {
            let safe_len = self.stop_buffer.len() - max_stop_len;
            let delta = self.stop_buffer[..safe_len].to_string();
            self.stop_buffer = self.stop_buffer[safe_len..].to_string();
            return Some(delta);
        }

        None
    }

    /// Flush remaining buffered text (call at end of generation).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.stop_buffer)
    }
}
