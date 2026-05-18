use clap::Parser;
use rllm_server::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match cli {
        Cli::Serve(args) => {
            tracing::info!(
                model = %args.model,
                host = %args.host,
                port = %args.port,
                "Starting rLLM server"
            );
            rllm_server::server::serve(args).await?;
        }
    }

    Ok(())
}
