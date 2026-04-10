#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub text: bool,
    pub multimodal: bool,
    pub tool_call_tokens: bool,
    pub batch_decode: bool,
    pub kv_swap: bool,
    pub accepts_image_inputs: bool,
    pub accepts_audio_inputs: bool,
    pub moe_enabled: bool,
    pub moe_num_experts: Option<usize>,
    pub moe_top_k_experts: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormatStrategy {
    AutoJinja,
    Hunyuan,
    Gemma4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputStrategy {
    Plain,
    Gemma4,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(40),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EngineLimitsProfile {
    pub suggested_max_seq_len: Option<usize>,
    pub kv_memory_hint_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub capabilities: ModelCapabilities,
    pub chat_format_strategy: ChatFormatStrategy,
    pub output_strategy: OutputStrategy,
    pub sampling_defaults: SamplingDefaults,
    pub limits: EngineLimitsProfile,
}

impl ModelSpec {
    pub fn supports_batch_decode(&self) -> bool {
        self.capabilities.batch_decode
    }

    pub fn supports_kv_swap(&self) -> bool {
        self.capabilities.kv_swap
    }

    pub fn effective_include_reasoning(&self, requested: bool) -> bool {
        match self.output_strategy {
            OutputStrategy::Gemma4 => true,
            OutputStrategy::Plain => requested,
        }
    }
}
