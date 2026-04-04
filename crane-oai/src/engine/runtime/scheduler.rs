#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum RuntimeStepKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RuntimeScheduleItem {
    pub sequence_id: String,
    pub kind: RuntimeStepKind,
}
