use rllm_core::ids::RequestId;

#[derive(Debug)]
pub struct SchedulerOutput {
    pub scheduled_new: Vec<RequestId>,
    pub scheduled_cached: Vec<RequestId>,
    pub scheduled_running: Vec<RequestId>,
    pub num_scheduled_tokens: std::collections::HashMap<RequestId, usize>,
    pub finished: Vec<RequestId>,
}
