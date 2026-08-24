use opentelemetry::{global, trace::TracerProvider as _};
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::{Resource, propagation::TraceContextPropagator, trace::SdkTracerProvider};
use thiserror::Error;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Configuration for the OTLP tracing pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtelConfig {
    pub service_name: String,
    pub fallback_filter: String,
}

impl OtelConfig {
    pub fn new(service_name: impl Into<String>, fallback_filter: impl Into<String>) -> Self {
        Self { service_name: service_name.into(), fallback_filter: fallback_filter.into() }
    }

    fn validate(&self) -> Result<(), OtelError> {
        if self.service_name.trim().is_empty() {
            return Err(OtelError::InvalidConfig("service_name must not be empty".into()));
        }
        if self.fallback_filter.trim().is_empty() {
            return Err(OtelError::InvalidConfig("fallback_filter must not be empty".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum OtelError {
    #[error("invalid OpenTelemetry configuration: {0}")]
    InvalidConfig(String),
    #[error("failed to build OTLP trace exporter: {0}")]
    Exporter(String),
    #[error("invalid tracing filter: {0}")]
    Filter(String),
    #[error("failed to install tracing subscriber: {0}")]
    Subscriber(String),
    #[error("failed to shut down OpenTelemetry tracer provider: {0}")]
    Shutdown(String),
}

/// Owns the tracer provider so queued spans can be flushed during shutdown.
pub struct OtelGuard {
    provider: Option<SdkTracerProvider>,
}

impl OtelGuard {
    pub fn force_flush(&self) -> Result<(), OtelError> {
        let Some(provider) = &self.provider else {
            return Ok(());
        };
        provider.force_flush().map_err(|error| OtelError::Shutdown(error.to_string()))
    }

    pub fn shutdown(mut self) -> Result<(), OtelError> {
        let Some(provider) = self.provider.take() else {
            return Ok(());
        };
        provider.shutdown().map_err(|error| OtelError::Shutdown(error.to_string()))
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            let _ = provider.shutdown();
        }
    }
}

/// Install formatted local tracing plus an OTLP/gRPC span exporter.
///
/// The exporter follows the standard `OTEL_EXPORTER_OTLP_*` environment
/// variables. Its default endpoint is `http://localhost:4317`.
pub fn init_otel_tracing(config: OtelConfig) -> Result<OtelGuard, OtelError> {
    config.validate()?;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_new(&config.fallback_filter))
        .map_err(|error| OtelError::Filter(error.to_string()))?;
    let exporter = SpanExporter::builder()
        .with_tonic()
        .build()
        .map_err(|error| OtelError::Exporter(error.to_string()))?;
    let resource = Resource::builder().with_service_name(config.service_name).build();
    let provider =
        SdkTracerProvider::builder().with_resource(resource).with_batch_exporter(exporter).build();
    let tracer = provider.tracer("rllm");

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_opentelemetry::layer().with_tracer(tracer));
    if let Err(error) = subscriber.try_init() {
        let _ = provider.shutdown();
        return Err(OtelError::Subscriber(error.to_string()));
    }

    global::set_text_map_propagator(TraceContextPropagator::new());
    global::set_tracer_provider(provider.clone());
    Ok(OtelGuard { provider: Some(provider) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_configuration_before_exporter_setup() {
        assert!(matches!(OtelConfig::new("", "info").validate(), Err(OtelError::InvalidConfig(_))));
        assert!(matches!(
            OtelConfig::new("rllm", " ").validate(),
            Err(OtelError::InvalidConfig(_))
        ));
        assert!(OtelConfig::new("rllm", "info").validate().is_ok());
    }
}
