use crate::engine::runtime::request_state::RequestState;

#[derive(Debug, Clone)]
pub struct IncrementalInputBuilder {
    pub chunk_size: usize,
}

impl Default for IncrementalInputBuilder {
    fn default() -> Self {
        Self { chunk_size: 1 }
    }
}

impl IncrementalInputBuilder {
    pub fn next_decode_chunk(&self, state: &mut RequestState) -> Vec<u32> {
        state.pop_incremental(self.chunk_size)
    }
}
