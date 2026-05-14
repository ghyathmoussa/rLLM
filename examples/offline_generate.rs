use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "offline_generate", about = "Offline text generation with rLLM")]
struct Args {
    /// Hugging Face model ID or local path
    #[arg(long, short)]
    model: Option<String>,

    /// Input prompt text
    #[arg(long, short)]
    prompt: Option<String>,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,

    /// Temperature for sampling (0 = greedy)
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
}

fn main() {
    let args = Args::parse();

    println!("rLLM offline_generate");
    println!("  model: {}", args.model.as_deref().unwrap_or("<not set>"));
    println!("  prompt: {}", args.prompt.as_deref().unwrap_or("<not set>"));
    println!("  max_tokens: {}", args.max_tokens);
    println!("  temperature: {}", args.temperature);
    println!();
    println!("Note: Model loading and generation will be implemented in Phase 4.");
}
