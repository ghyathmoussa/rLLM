use std::collections::HashMap;

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rllm_cache::spec::KVCacheConfig;
use rllm_core::{ids::RequestId, request::SamplingParams};
use rllm_executor::executor::{Executor, ExecutorOutput};
use rllm_sampling::{Sampler, SamplingInput};
use rllm_scheduler::SchedulerOutput;

/// Behavior mode for the mock executor's logits generation.
#[derive(Debug, Clone)]
pub enum MockMode {
    /// All-zero logits, producing uniform random sampling.
    Zero,
    /// Deterministic: sets `logits[(position + offset) % vocab_size] = 100.0`,
    /// guaranteeing greedy argmax picks a predictable token.
    Deterministic { offset: u32 },
    /// Seeded random logits from ChaCha8Rng.
    SeededRandom { seed: u64 },
    /// Always outputs a specific token ID.
    FixedToken { token_id: u32 },
}

impl Default for MockMode {
    fn default() -> Self {
        Self::Deterministic { offset: 0 }
    }
}

/// Configuration for creating a MockExecutor.
#[derive(Debug, Clone)]
pub struct MockExecutorConfig {
    pub mode: MockMode,
    pub vocab_size: usize,
    pub eos_token_id: u32,
    pub sampler_seed: Option<u64>,
}

impl Default for MockExecutorConfig {
    fn default() -> Self {
        Self {
            mode: MockMode::default(),
            vocab_size: 32000,
            eos_token_id: 2,
            sampler_seed: Some(42),
        }
    }
}

struct MockRequestState {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    sampling_params: SamplingParams,
}

/// Mock executor implementing the `Executor` trait for GPU-free testing.
///
/// Generates synthetic logits according to `MockMode` and runs them through
/// the real `Sampler`, exercising the full scheduling + sampling pipeline.
pub struct MockExecutor {
    config: MockExecutorConfig,
    sampler: Sampler,
    requests: HashMap<RequestId, MockRequestState>,
    rng: Option<ChaCha8Rng>,
}

impl MockExecutor {
    pub fn new(config: MockExecutorConfig) -> Self {
        let sampler = match config.sampler_seed {
            Some(seed) => Sampler::from_seed(seed),
            None => Sampler::new(),
        };
        let rng = match &config.mode {
            MockMode::SeededRandom { seed } => Some(ChaCha8Rng::seed_from_u64(*seed)),
            _ => None,
        };
        Self { config, sampler, requests: HashMap::new(), rng }
    }

    pub fn config(&self) -> &MockExecutorConfig {
        &self.config
    }

    fn generate_logits(&mut self, _request_id: RequestId, position: usize) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];

        match &self.config.mode {
            MockMode::Zero => { /* all zeros */ }
            MockMode::Deterministic { offset } => {
                let idx = (position as u32 + offset) as usize % vocab_size;
                logits[idx] = 100.0;
            }
            MockMode::SeededRandom { .. } => {
                if let Some(ref mut rng) = self.rng {
                    for logit in &mut logits {
                        *logit = rng.random::<f32>() * 2.0 - 1.0;
                    }
                }
            }
            MockMode::FixedToken { token_id } => {
                let idx = *token_id as usize;
                if idx < vocab_size {
                    logits[idx] = 100.0;
                }
            }
        }

        logits
    }
}

impl Executor for MockExecutor {
    fn initialize(
        &mut self,
        kv_cache_configs: &[KVCacheConfig],
        _gpu_memory_utilization: f32,
    ) -> Result<usize> {
        Ok(kv_cache_configs.first().map(|c| c.num_blocks).unwrap_or(0))
    }

    fn determine_available_memory(&self) -> Result<usize> {
        Ok(4 * 1024 * 1024 * 1024)
    }

    fn execute_model(&mut self, scheduler_output: &SchedulerOutput) -> Result<ExecutorOutput> {
        if scheduler_output.num_scheduled() == 0 {
            return Ok(ExecutorOutput { sampled_token_ids: vec![], logprobs: vec![] });
        }

        let scheduled_ids: Vec<RequestId> = scheduler_output
            .scheduled_new
            .iter()
            .chain(scheduler_output.scheduled_cached.iter())
            .chain(scheduler_output.scheduled_running.iter())
            .copied()
            .collect();

        let mut sampled_token_ids = Vec::with_capacity(scheduled_ids.len());
        let mut logprobs = Vec::with_capacity(scheduled_ids.len());

        for request_id in &scheduled_ids {
            // Extract request state first to avoid holding immutable borrow across
            // the mutable call to generate_logits (which touches self.rng).
            let state_snapshot = self.requests.get(request_id).map(|s| {
                let position = s.prompt_token_ids.len() + s.generated_token_ids.len();
                let mut ctx = s.prompt_token_ids.clone();
                ctx.extend_from_slice(&s.generated_token_ids);
                (position, s.sampling_params.clone(), ctx, s.generated_token_ids.len() as u32)
            });

            let (logits, params, context_tokens, num_generated) = match state_snapshot {
                Some((position, params, ctx, num_gen)) => {
                    let logits = self.generate_logits(*request_id, position);
                    (logits, params, ctx, num_gen)
                }
                None => {
                    let logits = vec![0.0f32; self.config.vocab_size];
                    (logits, SamplingParams::default(), vec![], 0)
                }
            };

            let input = SamplingInput {
                logits,
                params,
                context_token_ids: context_tokens,
                num_generated,
                eos_token_id: self.config.eos_token_id,
                bad_word_token_ids: vec![],
            };

            let output = self.sampler.sample(&input);
            let token_id = output.token_id;

            if let Some(state) = self.requests.get_mut(request_id) {
                state.generated_token_ids.push(token_id);
            }

            sampled_token_ids.push(token_id);
            logprobs.push(output.logprob);
        }

        Ok(ExecutorOutput { sampled_token_ids, logprobs })
    }

    fn add_request(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) {
        self.requests.insert(
            request_id,
            MockRequestState { prompt_token_ids, generated_token_ids: Vec::new(), sampling_params },
        );
    }

    fn shutdown(&mut self) {
        self.requests.clear();
    }
}
