use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "server", about = "Start the rLLM OpenAI-compatible server")]
struct Args {
    /// Hugging Face model ID or local path
    #[arg(long, short)]
    model: Option<String>,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() {
    let args = Args::parse();

    println!("rLLM server");
    println!("  model: {}", args.model.as_deref().unwrap_or("<not set>"));
    println!("  host: {}", args.host);
    println!("  port: {}", args.port);
    println!("  log_level: {}", args.log_level);
    println!();
    println!("Note: Server will be implemented in Phase 12.");
}
