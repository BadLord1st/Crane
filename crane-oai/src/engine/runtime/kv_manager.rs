#[derive(Debug, Clone, Default)]
pub struct KvMemoryManager {
    tracked_kv_bytes: u64,
}

impl KvMemoryManager {
    pub fn set_tracked_kv_bytes(&mut self, bytes: u64) {
        self.tracked_kv_bytes = bytes;
    }

    pub fn tracked_kv_bytes(&self) -> u64 {
        self.tracked_kv_bytes
    }
}
