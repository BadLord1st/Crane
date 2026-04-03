//! Gemma 4 output parsing and sanitization helpers.
//!
//! Keeps Gemma4 control tags out of user-visible responses for both
//! non-streaming and streaming APIs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Plain,
    Gemma4,
}

impl OutputMode {
    pub fn sanitize_text(self, text: &str, include_reasoning: bool) -> String {
        match self {
            Self::Plain => text.to_string(),
            Self::Gemma4 => sanitize_gemma4_output(text, include_reasoning),
        }
    }

    pub fn stream_sanitizer(self, include_reasoning: bool) -> Option<Gemma4StreamSanitizer> {
        match self {
            Self::Plain => None,
            Self::Gemma4 => Some(Gemma4StreamSanitizer::new(include_reasoning)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4ParseState {
    Normal,
    Thinking,
    ToolCall,
}

const GEMMA4_THINK_OPEN: &str = "<|channel>";
const GEMMA4_THINK_CLOSE: &str = "<channel|>";
const GEMMA4_TOOL_OPEN: &str = "<|tool_call>";
const GEMMA4_TOOL_CLOSE: &str = "<tool_call|>";
const GEMMA4_TURN_OPEN: &str = "<|turn>";
const GEMMA4_TURN_CLOSE: &str = "<turn|>";

const GEMMA4_TAGS: &[&str] = &[
    GEMMA4_THINK_OPEN,
    GEMMA4_THINK_CLOSE,
    GEMMA4_TOOL_OPEN,
    GEMMA4_TOOL_CLOSE,
    GEMMA4_TURN_OPEN,
    GEMMA4_TURN_CLOSE,
    "<|think|>",
];

fn sanitize_gemma4_internal(input: &str, done: bool, include_reasoning: bool) -> String {
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    let mut state = Gemma4ParseState::Normal;

    while i < input.len() {
        let rest = &input[i..];

        match state {
            Gemma4ParseState::Normal => {
                if rest.starts_with(GEMMA4_THINK_OPEN) {
                    i += GEMMA4_THINK_OPEN.len();
                    state = Gemma4ParseState::Thinking;
                    continue;
                }
                if rest.starts_with(GEMMA4_TOOL_OPEN) {
                    i += GEMMA4_TOOL_OPEN.len();
                    state = Gemma4ParseState::ToolCall;
                    continue;
                }
                if rest.starts_with(GEMMA4_TURN_CLOSE) {
                    i += GEMMA4_TURN_CLOSE.len();
                    continue;
                }
                if rest.starts_with("<|think|>") {
                    i += "<|think|>".len();
                    continue;
                }
                if rest.starts_with(GEMMA4_TURN_OPEN) {
                    i += GEMMA4_TURN_OPEN.len();
                    if let Some(pos) = input[i..].find('\n') {
                        i += pos + 1;
                    } else if done {
                        i = input.len();
                    } else {
                        break;
                    }
                    continue;
                }

                if !done
                    && rest.starts_with('<')
                    && GEMMA4_TAGS.iter().any(|tag| tag.starts_with(rest))
                {
                    break;
                }

                if let Some(ch) = rest.chars().next() {
                    out.push(ch);
                    i += ch.len_utf8();
                } else {
                    break;
                }
            }
            Gemma4ParseState::Thinking => {
                if let Some(pos) = rest.find(GEMMA4_THINK_CLOSE) {
                    if include_reasoning {
                        out.push_str(&rest[..pos]);
                    }
                    i += pos + GEMMA4_THINK_CLOSE.len();
                    state = Gemma4ParseState::Normal;
                } else if done {
                    if include_reasoning {
                        out.push_str(rest);
                    }
                    i = input.len();
                    state = Gemma4ParseState::Normal;
                } else {
                    break;
                }
            }
            Gemma4ParseState::ToolCall => {
                if let Some(pos) = rest.find(GEMMA4_TOOL_CLOSE) {
                    i += pos + GEMMA4_TOOL_CLOSE.len();
                    state = Gemma4ParseState::Normal;
                } else if done {
                    i = input.len();
                    state = Gemma4ParseState::Normal;
                } else {
                    break;
                }
            }
        }
    }

    out
}

/// Remove Gemma 4 control tags from final model output.
pub fn sanitize_gemma4_output(text: &str, include_reasoning: bool) -> String {
    sanitize_gemma4_internal(text, true, include_reasoning)
        .trim()
        .to_string()
}

/// Streaming sanitizer for Gemma 4 outputs. Accepts raw token deltas and
/// returns clean user-visible text chunks.
#[derive(Default)]
pub struct Gemma4StreamSanitizer {
    raw: String,
    emitted: usize,
    include_reasoning: bool,
}

impl Gemma4StreamSanitizer {
    pub fn new(include_reasoning: bool) -> Self {
        Self {
            raw: String::new(),
            emitted: 0,
            include_reasoning,
        }
    }
}

impl Gemma4StreamSanitizer {
    pub fn push_chunk(&mut self, chunk: &str) -> String {
        self.raw.push_str(chunk);
        let clean = sanitize_gemma4_internal(&self.raw, false, self.include_reasoning);
        if self.emitted > clean.len() {
            self.emitted = clean.len();
        }
        let delta = clean[self.emitted..].to_string();
        self.emitted = clean.len();
        delta
    }

    pub fn finish(&mut self) -> String {
        let clean = sanitize_gemma4_internal(&self.raw, true, self.include_reasoning);
        if self.emitted > clean.len() {
            self.emitted = clean.len();
        }
        let delta = clean[self.emitted..].to_string();
        self.emitted = clean.len();
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::{sanitize_gemma4_output, Gemma4StreamSanitizer};

    #[test]
    fn gemma4_sanitize_strips_control_tags() {
        let raw = "<|turn>model\nHello<turn|>\n<|tool_call>call:x{a:1}<tool_call|>";
        let cleaned = sanitize_gemma4_output(raw, false);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn gemma4_stream_sanitizer_handles_split_tags() {
        let mut s = Gemma4StreamSanitizer::new(false);
        assert_eq!(s.push_chunk("<|turn>model\nHe"), "He");
        assert_eq!(s.push_chunk("llo<turn"), "llo");
        assert_eq!(s.push_chunk("|>"), "");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn gemma4_reasoning_can_be_included() {
        let raw = "<|turn>model\nA<turn|><|channel>hidden-chain<channel|>";
        let cleaned = sanitize_gemma4_output(raw, true);
        assert!(cleaned.contains("hidden-chain"));
    }
}
