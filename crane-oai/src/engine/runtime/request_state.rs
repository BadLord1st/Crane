use std::collections::VecDeque;

use crate::engine::types::MultimodalInputs;

#[derive(Debug, Clone, Default)]
pub struct RequestState {
    pub id: String,
    pub pending_input_ids: VecDeque<u32>,
    pub processed_tokens: usize,
    pub multimodal_inputs: MultimodalInputs,
}

impl RequestState {
    pub fn pop_incremental(&mut self, max_tokens: usize) -> Vec<u32> {
        let take = max_tokens.min(self.pending_input_ids.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(t) = self.pending_input_ids.pop_front() {
                out.push(t);
            }
        }
        self.processed_tokens += out.len();
        out
    }
}
