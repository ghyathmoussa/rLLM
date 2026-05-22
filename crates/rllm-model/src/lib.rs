pub mod hf_config;
pub mod loader;
pub mod registry;

#[cfg(feature = "candle-backend")]
pub mod layers;
#[cfg(feature = "candle-backend")]
pub mod llama;
#[cfg(feature = "candle-backend")]
pub mod rope;
#[cfg(feature = "candle-backend")]
pub mod runner;

#[cfg(feature = "candle-backend")]
pub use llama::LlamaForCausalLM;
pub use registry::{CausalLM, Model};
#[cfg(feature = "candle-backend")]
pub use runner::ModelRunner;
