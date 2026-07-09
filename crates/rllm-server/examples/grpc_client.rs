use clap::Parser;
use rllm_server::grpc::pb::{
    self, CompletionRequest, inference_service_client::InferenceServiceClient,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:15051")]
    addr: String,

    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    #[arg(long, default_value = "Hello from gRPC")]
    prompt: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut client = InferenceServiceClient::connect(args.addr.clone()).await?;
    let health = client.health(pb::HealthRequest {}).await?.into_inner();
    println!("health.status={}", health.status);

    let models = client.list_models(pb::ListModelsRequest {}).await?.into_inner();
    let model =
        models.data.first().map(|model| model.id.clone()).unwrap_or_else(|| "unknown".to_string());
    println!("model={model}");

    let mut tasks = Vec::new();
    for index in 0..args.concurrency {
        let addr = args.addr.clone();
        let model = model.clone();
        let prompt = format!("{} #{index}", args.prompt);
        tasks.push(tokio::spawn(async move {
            let mut client = InferenceServiceClient::connect(addr).await?;
            let response =
                client.completion(completion_request(model, prompt, 8)).await?.into_inner();
            let choice = response.choices.first().expect("missing completion choice");
            anyhow::Ok((index, choice.text.clone(), choice.finish_reason.clone()))
        }));
    }

    for task in tasks {
        let (index, content, finish_reason) = task.await??;
        println!("completion[{index}] finish={finish_reason:?} text={content:?}");
    }

    let mut stream = client
        .stream_completion(completion_request(model, format!("{} stream", args.prompt), 8))
        .await?
        .into_inner();
    let mut stream_chunks = 0usize;
    while let Some(chunk) = stream.message().await? {
        stream_chunks += 1;
        let Some(choice) = chunk.choices.first() else {
            continue;
        };
        println!(
            "stream[{stream_chunks}] finished={} finish={:?} text={:?}",
            chunk.finished, choice.finish_reason, choice.text
        );
    }
    println!("stream_chunks={stream_chunks}");

    Ok(())
}

fn completion_request(model: String, prompt: String, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        model,
        prompt,
        temperature: Some(0.0),
        top_p: None,
        max_tokens: Some(max_tokens),
        stop: vec![],
        n: Some(1),
        logprobs: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: Some(0),
    }
}
