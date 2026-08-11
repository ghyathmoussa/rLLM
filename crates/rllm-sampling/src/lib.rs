pub mod logits;
pub mod logprobs;
pub mod sampler;
pub mod speculative;
#[cfg(feature = "structured-outputs")]
pub mod structured;

pub use sampler::{Sampler, SamplingInput, SamplingOutput};
pub use speculative::{
    DraftModelProposer, DraftProposal, EagleProposer, NGramProposer, SpeculativeProposer,
    SpeculativeState, accept_matching_prefix,
};
#[cfg(feature = "structured-outputs")]
pub use structured::{StructuredOutputManager, validate_structured_output};
