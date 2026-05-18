use anyhow::Result;
use rllm_core::ids::RequestId;
use rllm_core::output::RequestOutput;
use rllm_core::request::InferenceRequest;

use crate::engine_core::EngineCore;

/// Synchronous LLM engine for offline generation.
///
/// Wraps `EngineCore` directly without async runtime.
/// Call `step()` in a loop or use `generate()` for blocking generation.
pub struct LLMEngine {
    core: EngineCore,
}

impl LLMEngine {
    pub fn new(core: EngineCore) -> Self {
        Self { core }
    }

    /// Add a new inference request.
    pub fn add_request(&mut self, request: InferenceRequest) -> Result<()> {
        self.core.add_request(request)
    }

    /// Abort a running request.
    pub fn abort_request(&mut self, request_id: RequestId) {
        self.core.abort_request(request_id);
    }

    /// Run one engine step and return outputs.
    pub fn step(&mut self) -> Vec<RequestOutput> {
        self.core.step()
    }

    /// Check if there is any pending work.
    pub fn has_work(&self) -> bool {
        self.core.has_work()
    }

    /// Get the number of active requests.
    pub fn num_active_requests(&self) -> usize {
        self.core.num_active_requests()
    }

    /// Blocking generation loop: run steps until all requests finish.
    ///
    /// Returns all outputs collected during generation.
    pub fn generate(&mut self, requests: Vec<InferenceRequest>) -> Result<Vec<RequestOutput>> {
        for req in requests {
            self.add_request(req)?;
        }

        let mut all_outputs = Vec::new();

        while self.has_work() {
            let outputs = self.step();
            for output in &outputs {
                if output.finished {
                    all_outputs.push(output.clone());
                }
            }
        }

        Ok(all_outputs)
    }

    /// Reset the prefix cache.
    pub fn reset_prefix_cache(&mut self) -> Result<(), String> {
        self.core.reset_prefix_cache()
    }
}
