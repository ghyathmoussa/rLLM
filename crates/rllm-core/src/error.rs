use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("invalid sampling parameters: {0}")]
    InvalidSamplingParams(String),

    #[error("request {0} not found")]
    RequestNotFound(String),

    #[error("invalid request status transition from {from} to {to}")]
    InvalidStatusTransition { from: String, to: String },
}

pub type Result<T> = std::result::Result<T, CoreError>;
