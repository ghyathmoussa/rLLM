use std::sync::Arc;

use parking_lot::Mutex;
use rllm_core::{ids::RequestId, output::RequestOutput, request::InferenceRequest};
use tokio::sync::{mpsc, watch};

use crate::engine_core::EngineCore;

/// Commands sent to the engine background task.
enum EngineCommand {
    AddRequest {
        request: Box<InferenceRequest>,
    },
    AddRequestStream {
        request: Box<InferenceRequest>,
        sender: mpsc::UnboundedSender<RequestOutput>,
    },
    AbortRequest {
        request_id: RequestId,
    },
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

    /// Add a new inference request and return a lossless receiver for its outputs.
    pub fn add_request_stream(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<mpsc::UnboundedReceiver<RequestOutput>> {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.cmd_tx
            .send(EngineCommand::AddRequestStream { request: Box::new(request), sender })
            .map_err(|_| anyhow::anyhow!("Engine task shut down"))?;
        Ok(receiver)
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
    let mut senders: std::collections::HashMap<RequestId, mpsc::UnboundedSender<RequestOutput>> =
        std::collections::HashMap::new();

    loop {
        // Check if there is active work in the engine.
        let has_work = {
            let core = core.lock();
            core.has_work()
        };

        if !has_work {
            // Block asynchronously until a command is received to avoid busy-spinning.
            match cmd_rx.recv().await {
                Some(EngineCommand::Shutdown) => {
                    tracing::info!("Engine loop shutting down");
                    return;
                }
                Some(cmd) => {
                    let mut core = core.lock();
                    match cmd {
                        EngineCommand::AddRequest { request } => {
                            let request = *request;
                            if let Err(e) = core.add_request(request) {
                                tracing::error!("Failed to add request: {}", e);
                            }
                        }
                        EngineCommand::AddRequestStream { request, sender } => {
                            let request_id = request.request_id;
                            let request = *request;
                            if let Err(e) = core.add_request(request) {
                                tracing::error!("Failed to add request: {}", e);
                            } else {
                                senders.insert(request_id, sender);
                            }
                        }
                        EngineCommand::AbortRequest { request_id } => {
                            core.abort_request(request_id);
                            senders.remove(&request_id);
                        }
                        EngineCommand::Shutdown => unreachable!(),
                    }
                }
                None => {
                    // Channel closed, terminate loop.
                    return;
                }
            }
        }

        // Drain any other pending commands that arrived.
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
                    EngineCommand::AddRequestStream { request, sender } => {
                        let request_id = request.request_id;
                        let request = *request;
                        if let Err(e) = core.add_request(request) {
                            tracing::error!("Failed to add request: {}", e);
                        } else {
                            senders.insert(request_id, sender);
                        }
                    }
                    EngineCommand::AbortRequest { request_id } => {
                        core.abort_request(request_id);
                        senders.remove(&request_id);
                    }
                    EngineCommand::Shutdown => unreachable!(),
                }
            }

            if core.has_work() {
                let outputs = core.step();
                if !outputs.is_empty() {
                    for output in &outputs {
                        if let Some(sender) = senders.get(&output.request_id) {
                            let _ = sender.send(output.clone());
                        }
                    }
                    for output in &outputs {
                        if output.finished {
                            senders.remove(&output.request_id);
                        }
                    }
                    let _ = output_tx.send(outputs);
                }
            }
        }

        // Yield to the tokio scheduler to let other tasks (like server handlers) execute.
        tokio::task::yield_now().await;
    }
}
