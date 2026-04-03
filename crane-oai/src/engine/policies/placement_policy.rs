#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    KeepOnDevice,
    OffloadReady,
}
