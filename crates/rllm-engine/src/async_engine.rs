use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};

use rllm_core::ids::RequestId;
use rllm_core::output::RequestOutput;
use rllm_core::request::InferenceRequest;

use crate::engine_core::EngineCore;

/// Commands sent to the engine background task.
enum EngineCommand {
    AddRequest { request: Box<InferenceRequest> },
    AbortRequest { request_id: RequestId },
    Shutdown,
}

/// Asynchronous LLM engine that runs inference in a background tokio task.
///
/// Requests are submitted via async methods and outputs are consumed
/// through a `Stream<Item = Vec<RequestOutput>>`.
pub struct AsyncLLMEngine {
    cmd_tx: mpsc::UnboundedSender<EngineCommand>,
    /// Watch channel for output batches: each step produces a Vec<RequestOutput>.
    output_rx: watch::Receiver<Vec<RequestOutput>>,
}

impl AsyncLLMEngine {
    /// Create a new async engine from an EngineCore.
    ///
    /// Spawns a background task that runs the engine loop.
    pub fn new(core: EngineCore) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (output_tx, output_rx) = watch::channel(vec![]);

        let core = Arc::new(Mutex::new(core));

        tokio::spawn(engine_loop(core, cmd_rx, output_tx));

        Self { cmd_tx, output_rx }
    }

    /// Add a new inference request.
    pub fn add_request(&self, request: InferenceRequest) -> anyhow::Result<()> {
        self.cmd_tx
            .send(EngineCommand::AddRequest { request: Box::new(request) })
            .map_err(|_| anyhow::anyhow!("Engine task shut down"))?;
        Ok(())
    }

    /// Abort a running request.
    pub fn abort_request(&self, request_id: RequestId) -> anyhow::Result<()> {
        self.cmd_tx
            .send(EngineCommand::AbortRequest { request_id })
            .map_err(|_| anyhow::anyhow!("Engine task shut down"))?;
        Ok(())
    }

    /// Shut down the engine.
    pub fn shutdown(&self) -> anyhow::Result<()> {
        self.cmd_tx
            .send(EngineCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("Engine task shut down"))?;
        Ok(())
    }

    /// Get a clone of the output watch receiver.
    ///
    /// The receiver yields `Vec<RequestOutput>` for each engine step.
    /// Use `output_rx.changed().await` to wait for new outputs.
    pub fn output_receiver(&self) -> watch::Receiver<Vec<RequestOutput>> {
        self.output_rx.clone()
    }
}

/// Background engine loop.
#[tracing::instrument(skip_all, name = "engine_loop")]
async fn engine_loop(
    core: Arc<Mutex<EngineCore>>,
    mut cmd_rx: mpsc::UnboundedReceiver<EngineCommand>,
    output_tx: watch::Sender<Vec<RequestOutput>>,
) {
    loop {
        // Drain all pending commands into a local Vec (no lock needed for try_recv).
        let mut commands = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                EngineCommand::Shutdown => {
                    tracing::info!("Engine loop shutting down");
                    return;
                }
                other => commands.push(other),
            }
        }

        // Acquire mutex once, process all commands + run step, then release.
        {
            let mut core = core.lock();
            for cmd in commands.drain(..) {
                match cmd {
                    EngineCommand::AddRequest { request } => {
                        let request = *request;
                        if let Err(e) = core.add_request(request) {
                            tracing::error!("Failed to add request: {}", e);
                        }
                    }
                    EngineCommand::AbortRequest { request_id } => {
                        core.abort_request(request_id);
                    }
                    EngineCommand::Shutdown => unreachable!(), // handled above
                }
            }

            if core.has_work() {
                let outputs = core.step();
                if !outputs.is_empty() {
                    let _ = output_tx.send(outputs);
                }
            }
        }

        // Yield to the runtime.
        tokio::task::yield_now().await;
    }
}
