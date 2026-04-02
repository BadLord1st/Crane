//! Chat template formatting.
//!
//! Converts OpenAI-style `[{role, content}]` messages into the prompt string
//! expected by each model family.  Two strategies:
//!
//! * **[`AutoChatTemplate`]** — uses the Jinja `chat_template` from
//!   `tokenizer_config.json` (works for Qwen, Llama, Mistral, …).
//! * **[`HunyuanChatTemplate`]** — hardcoded template for Hunyuan models.

use crate::openai_api::{ChatMessage, ChatMessageContent, ContentPart};
use crane_core::autotokenizer::AutoTokenizer;

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Formats chat messages into a model-specific prompt string.
pub trait ChatTemplateProcessor: Send + Sync {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String>;
}

// ─────────────────────────────────────────────────────────────
//  AutoChatTemplate (Jinja-based)
// ─────────────────────────────────────────────────────────────

/// Uses [`AutoTokenizer`]'s Jinja `chat_template` from `tokenizer_config.json`.
pub struct AutoChatTemplate {
    tokenizer: AutoTokenizer,
}

impl AutoChatTemplate {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let tokenizer = AutoTokenizer::from_pretrained(model_path, None)
            .map_err(|e| format!("Failed to load AutoTokenizer: {e}"))?;
        Ok(Self { tokenizer })
    }
}

impl ChatTemplateProcessor for AutoChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        // Build the list of {role, content} values expected by the Jinja template.
        let template_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.text_content(),
                })
            })
            .collect();

        self.tokenizer
            .apply_chat_template(&template_messages, true)
            .map_err(|e| format!("Chat template error: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────
//  HunyuanChatTemplate (hardcoded)
// ─────────────────────────────────────────────────────────────

/// Hardcoded chat template for Hunyuan Dense models.
pub struct HunyuanChatTemplate;

impl ChatTemplateProcessor for HunyuanChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
        const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
        const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
        const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
        const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

        let mut result = String::new();
        result.push_str(BOS);

        let (system_msg, loop_messages) = if !messages.is_empty() && messages[0].role == "system" {
            (Some(messages[0].text_content()), &messages[1..])
        } else {
            (None, &messages[..])
        };

        if let Some(sys) = system_msg {
            result.push_str(&sys);
            result.push_str(SEP);
        }

        loop_messages
            .iter()
            .for_each(|msg| match msg.role.as_str() {
                "user" => {
                    result.push_str(USER);
                    result.push_str(&msg.text_content());
                }
                "assistant" => {
                    result.push_str(ASSISTANT);
                    result.push_str(&msg.text_content());
                    result.push_str(EOS);
                }
                _ => {}
            });

        result.push_str(ASSISTANT);
        Ok(result)
    }
}

/// Text-only chat template for Gemma 4 style chat models.
///
/// Mirrors the core turn format from the official Gemma 4 template:
/// `<bos><|turn>{role}\n{content}<turn|>\n...<|turn>model\n`
pub struct Gemma4ChatTemplate;

fn render_gemma4_content(message: &ChatMessage) -> String {
    const IMAGE_TOKEN: &str = "<|image|>";
    const AUDIO_TOKEN: &str = "<|audio|>";

    match &message.content {
        ChatMessageContent::Text(s) => s.trim().to_string(),
        ChatMessageContent::Parts(parts) => parts
            .iter()
            .fold(String::new(), |mut acc, part| {
                match part {
                    ContentPart::Text { text } => acc.push_str(text),
                    ContentPart::ImageUrl { .. } | ContentPart::Image { .. } => {
                        if !acc.is_empty() && !acc.ends_with('\n') {
                            acc.push('\n');
                        }
                        acc.push_str(IMAGE_TOKEN);
                    }
                    ContentPart::AudioUrl { .. } => {
                        if !acc.is_empty() && !acc.ends_with('\n') {
                            acc.push('\n');
                        }
                        acc.push_str(AUDIO_TOKEN);
                    }
                }
                acc
            })
            .trim()
            .to_string(),
    }
}

impl ChatTemplateProcessor for Gemma4ChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        const BOS: &str = "<bos>";
        const TURN_OPEN: &str = "<|turn>";
        const TURN_CLOSE: &str = "<turn|>";

        let mut out = String::new();
        out.push_str(BOS);

        messages
            .iter()
            .filter_map(|msg| {
                let role = match msg.role.as_str() {
                    "assistant" => "model",
                    "user" => "user",
                    "system" | "developer" => "system",
                    _ => return None,
                };
                Some((role, render_gemma4_content(msg)))
            })
            .for_each(|(role, content)| {
                out.push_str(TURN_OPEN);
                out.push_str(role);
                out.push('\n');
                out.push_str(&content);
                out.push_str(TURN_CLOSE);
                out.push('\n');
            });

        // Add generation prompt for assistant/model continuation.
        out.push_str(TURN_OPEN);
        out.push_str("model\n");

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatTemplateProcessor, HunyuanChatTemplate};
    use crate::openai_api::{AudioUrl, ChatMessage, ChatMessageContent, ContentPart, ImageUrl};

    fn make_messages(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(role, content)| ChatMessage {
                role: role.to_string(),
                content: ChatMessageContent::Text(content.to_string()),
            })
            .collect()
    }

    // ── HunyuanChatTemplate ──

    #[test]
    fn hunyuan_basic_user_message() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[("user", "Hello")]);
        let result = tmpl.apply(&msgs).unwrap();

        // Should start with BOS.
        assert!(result.starts_with("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        // Should contain user tag + content.
        assert!(result.contains("<\u{ff5c}hy_User\u{ff5c}>Hello"));
        // Should end with assistant tag (ready for generation).
        assert!(result.ends_with("<\u{ff5c}hy_Assistant\u{ff5c}>"));
    }

    #[test]
    fn hunyuan_system_message_prepended() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[("system", "You are helpful"), ("user", "Hi")]);
        let result = tmpl.apply(&msgs).unwrap();

        // System content should appear after BOS followed by SEP.
        let sep = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";
        assert!(result.contains(&format!("You are helpful{sep}")));
    }

    #[test]
    fn hunyuan_multi_turn() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "How are you?"),
        ]);
        let result = tmpl.apply(&msgs).unwrap();

        let eos = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
        // Assistant response should end with EOS.
        assert!(result.contains(&format!("Hi!{eos}")));
        // Second user message present.
        assert!(result.contains("How are you?"));
    }

    #[test]
    fn hunyuan_empty_messages() {
        let tmpl = HunyuanChatTemplate;
        let msgs: Vec<ChatMessage> = vec![];
        let result = tmpl.apply(&msgs).unwrap();

        // Should at least have BOS + ASSISTANT.
        assert!(result.starts_with("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        assert!(result.ends_with("<\u{ff5c}hy_Assistant\u{ff5c}>"));
    }

    #[test]
    fn hunyuan_unknown_role_skipped() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[
            ("user", "Hello"),
            ("tool", "some tool output"),
            ("user", "Next"),
        ]);
        let result = tmpl.apply(&msgs).unwrap();
        // "tool" content should not appear with any tag.
        assert!(!result.contains("some tool output"));
    }

    // ── ChatTemplateProcessor trait ──

    #[test]
    fn hunyuan_implements_trait() {
        let proc: Box<dyn ChatTemplateProcessor> = Box::new(HunyuanChatTemplate);
        let msgs = make_messages(&[("user", "test")]);
        assert!(proc.apply(&msgs).is_ok());
    }

    #[test]
    fn gemma4_renders_image_and_audio_placeholders() {
        let tmpl = super::Gemma4ChatTemplate;
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: ChatMessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/a.png".to_string(),
                    },
                },
                ContentPart::AudioUrl {
                    audio_url: AudioUrl {
                        url: "https://example.com/b.wav".to_string(),
                    },
                },
            ]),
        }];

        let out = tmpl.apply(&msgs).unwrap();
        assert!(out.contains("Describe this"));
        assert!(out.contains("<|image|>"));
        assert!(out.contains("<|audio|>"));
        assert!(out.ends_with("<|turn>model\n"));
    }
}
