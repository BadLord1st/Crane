use crate::gemma4_output::OutputMode;

use crate::engine::runtime::OutputStrategy;

pub fn output_mode(strategy: OutputStrategy) -> OutputMode {
    match strategy {
        OutputStrategy::Plain => OutputMode::Plain,
        OutputStrategy::Gemma4 => OutputMode::Gemma4,
    }
}

pub fn effective_include_reasoning(strategy: OutputStrategy, requested: bool) -> bool {
    match strategy {
        OutputStrategy::Gemma4 => true,
        OutputStrategy::Plain => requested,
    }
}
