#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeStepKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct RuntimeScheduleItem {
    pub sequence_id: String,
    pub kind: RuntimeStepKind,
}
