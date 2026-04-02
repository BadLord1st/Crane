use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::modeling::{Config, Gemma4TextModel};
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    pub sliding_window: usize,
    pub eos_token_ids: Vec<u32>,
    inner: Gemma4TextModel,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    fn to_vec(&self) -> Vec<u32> {
        match self {
            Self::One(v) => vec![*v],
            Self::Many(v) => v.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4ConfigFile {
    text_config: Config,
    #[serde(default)]
    eos_token_id: Option<EosTokenId>,
}

impl Model {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
        let vb = vb.set_prefix("model.language_model");

        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let top_cfg: Gemma4ConfigFile = serde_json::from_slice(&config_data)?;
        let cfg = top_cfg.text_config;

        let eos_token_ids = if let Some(ids) = top_cfg.eos_token_id.as_ref().map(EosTokenId::to_vec)
        {
            ids
        } else if let Some(id) = cfg.eos_token_id {
            vec![id]
        } else {
            let mut ids = Vec::new();
            if let Some(id) = tokenizer.token_to_id("<end_of_turn>") {
                ids.push(id);
            }
            if let Some(id) = tokenizer.token_to_id("<eos>") {
                ids.push(id);
            }
            if ids.is_empty() {
                ids.push(1);
            }
            ids
        };

        let sliding_window = cfg.sliding_window;
        let inner = Gemma4TextModel::new(&cfg, vb)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            sliding_window,
            eos_token_ids,
            inner,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    pub fn forward_step(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos, self.sliding_window)
    }

    pub fn warmup(&mut self) {
        let token = self.eos_token_ids.first().copied().unwrap_or(1);
        let _ = self.forward_step(&[token], 0);
        self.clear_kv_cache();
    }
}
