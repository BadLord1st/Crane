use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
    image_token_id: Option<u32>,
    audio_token_id: Option<u32>,
    vision_patch_proj: Option<Linear>,
    vision_pos_table: Option<Tensor>,
    vision_patch_size: usize,
    vision_tower: Option<VisionTower>,
    audio_input_proj: Option<Linear>,
    audio_tower: Option<AudioTower>,
    audio_output_proj: Option<Linear>,
    vision_projection: Option<Linear>,
    audio_projection: Option<Linear>,
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
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    audio_config: Option<AudioConfig>,
    #[serde(default)]
    eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct VisionConfig {
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    patch_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct AudioConfig {
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    hidden_act: Option<String>,
}

#[derive(Debug, Clone)]
struct SimpleRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl SimpleRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for SimpleRmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(candle_core::D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(candle_core::D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

fn softmax_last_dim_fallback(xs: &Tensor) -> candle_core::Result<Tensor> {
    let input_dtype = xs.dtype();
    let compute_dtype = match input_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let xs = xs.to_dtype(compute_dtype)?;
    let max = xs.max_keepdim(candle_core::D::Minus1)?;
    let exps = xs.broadcast_sub(&max)?.exp()?;
    let den = exps.sum_keepdim(candle_core::D::Minus1)?;
    exps.broadcast_div(&den)?.to_dtype(input_dtype)
}

#[derive(Debug, Clone)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: SimpleRmsNorm,
    k_norm: SimpleRmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let head_dim = hidden_size / num_heads.max(1);
        Ok(Self {
            q_proj: Linear::new(
                vb.pp("q_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            k_proj: Linear::new(
                vb.pp("k_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            v_proj: Linear::new(
                vb.pp("v_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            o_proj: Linear::new(
                vb.pp("o_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            q_norm: SimpleRmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: SimpleRmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b_sz, q_len, hidden_size) = xs.dims3()?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();

        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = softmax_last_dim_fallback(&attn)?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, hidden_size))?;
        self.o_proj.forward(&out)
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            gate_proj: Linear::new(
                vb.pp("gate_proj")
                    .get((intermediate_size, hidden_size), "linear.weight")?,
                None,
            ),
            up_proj: Linear::new(
                vb.pp("up_proj")
                    .get((intermediate_size, hidden_size), "linear.weight")?,
                None,
            ),
            down_proj: Linear::new(
                vb.pp("down_proj")
                    .get((hidden_size, intermediate_size), "linear.weight")?,
                None,
            ),
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let rhs = self.up_proj.forward(xs)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct VisionEncoderLayer {
    input_layernorm: SimpleRmsNorm,
    self_attn: VisionAttention,
    post_attention_layernorm: SimpleRmsNorm,
    pre_feedforward_layernorm: SimpleRmsNorm,
    mlp: VisionMlp,
    post_feedforward_layernorm: SimpleRmsNorm,
}

impl VisionEncoderLayer {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        rms_norm_eps: f64,
        act: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            input_layernorm: SimpleRmsNorm::new(
                hidden_size,
                rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: VisionAttention::new(
                hidden_size,
                num_heads,
                rms_norm_eps,
                vb.pp("self_attn"),
            )?,
            post_attention_layernorm: SimpleRmsNorm::new(
                hidden_size,
                rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_layernorm: SimpleRmsNorm::new(
                hidden_size,
                rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            mlp: VisionMlp::new(hidden_size, intermediate_size, act, vb.pp("mlp"))?,
            post_feedforward_layernorm: SimpleRmsNorm::new(
                hidden_size,
                rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct VisionTower {
    layers: Vec<VisionEncoderLayer>,
}

impl VisionTower {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> candle_core::Result<Option<Self>> {
        let hidden_size = cfg.hidden_size.unwrap_or(768);
        let num_layers = cfg.num_hidden_layers.unwrap_or(0);
        if num_layers == 0 {
            return Ok(None);
        }
        let intermediate_size = cfg.intermediate_size.unwrap_or(hidden_size * 4);
        let num_heads = cfg.num_attention_heads.unwrap_or(12);
        let eps = cfg.rms_norm_eps.unwrap_or(1e-6);
        let act = match cfg.hidden_activation.as_deref() {
            Some("gelu_pytorch_tanh") | Some("gelu") => Activation::GeluPytorchTanh,
            Some("silu") | Some("swish") => Activation::Silu,
            Some("relu") => Activation::Relu,
            _ => Activation::GeluPytorchTanh,
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(VisionEncoderLayer::new(
                hidden_size,
                intermediate_size,
                num_heads,
                eps,
                act.clone(),
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Some(Self { layers }))
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    post_proj: Linear,
    per_dim_scale: Option<Tensor>,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = hidden_size / num_heads.max(1);
        Ok(Self {
            q_proj: Linear::new(
                vb.pp("q_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            k_proj: Linear::new(
                vb.pp("k_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            v_proj: Linear::new(
                vb.pp("v_proj")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            post_proj: Linear::new(
                vb.pp("post")
                    .get((hidden_size, hidden_size), "linear.weight")?,
                None,
            ),
            per_dim_scale: vb.get((head_dim,), "per_dim_scale").ok(),
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();

        let mut q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        if let Some(s) = &self.per_dim_scale {
            let s = s.reshape((1, 1, 1, self.head_dim))?;
            q = q.broadcast_mul(&s)?;
            k = k.broadcast_mul(&s)?;
        }

        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = softmax_last_dim_fallback(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, h))?;
        self.post_proj.forward(&out)
    }
}

#[derive(Debug, Clone)]
struct AudioFfnBlock {
    ffw1: Linear,
    ffw2: Linear,
    pre_norm: SimpleRmsNorm,
    post_norm: SimpleRmsNorm,
    act: Activation,
}

impl AudioFfnBlock {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        eps: f64,
        act: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            ffw1: Linear::new(
                vb.pp("ffw_layer_1")
                    .get((intermediate_size, hidden_size), "linear.weight")?,
                None,
            ),
            ffw2: Linear::new(
                vb.pp("ffw_layer_2")
                    .get((hidden_size, intermediate_size), "linear.weight")?,
                None,
            ),
            pre_norm: SimpleRmsNorm::new(hidden_size, eps, vb.pp("pre_layer_norm"))?,
            post_norm: SimpleRmsNorm::new(hidden_size, eps, vb.pp("post_layer_norm"))?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.pre_norm.forward(xs)?;
        let xs = self.ffw1.forward(&xs)?.apply(&self.act)?;
        let xs = self.ffw2.forward(&xs)?;
        let xs = self.post_norm.forward(&xs)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
struct AudioEncoderLayer {
    norm_pre_attn: SimpleRmsNorm,
    self_attn: AudioAttention,
    norm_post_attn: SimpleRmsNorm,
    ffn1: AudioFfnBlock,
    ffn2: AudioFfnBlock,
    norm_out: SimpleRmsNorm,
}

impl AudioEncoderLayer {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        eps: f64,
        act: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            norm_pre_attn: SimpleRmsNorm::new(hidden_size, eps, vb.pp("norm_pre_attn"))?,
            self_attn: AudioAttention::new(hidden_size, num_heads, vb.pp("self_attn"))?,
            norm_post_attn: SimpleRmsNorm::new(hidden_size, eps, vb.pp("norm_post_attn"))?,
            ffn1: AudioFfnBlock::new(
                hidden_size,
                intermediate_size,
                eps,
                act.clone(),
                vb.pp("feed_forward1"),
            )?,
            ffn2: AudioFfnBlock::new(
                hidden_size,
                intermediate_size,
                eps,
                act,
                vb.pp("feed_forward2"),
            )?,
            norm_out: SimpleRmsNorm::new(hidden_size, eps, vb.pp("norm_out"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.norm_pre_attn.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (residual + xs)?;
        let xs = self.norm_post_attn.forward(&xs)?;
        let xs = self.ffn1.forward(&xs)?;
        let xs = self.ffn2.forward(&xs)?;
        self.norm_out.forward(&xs)
    }
}

#[derive(Debug, Clone)]
struct AudioTower {
    layers: Vec<AudioEncoderLayer>,
}

impl AudioTower {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> candle_core::Result<Option<Self>> {
        let hidden_size = cfg.hidden_size.unwrap_or(1024);
        let num_layers = cfg.num_hidden_layers.unwrap_or(0);
        if num_layers == 0 {
            return Ok(None);
        }
        let num_heads = cfg.num_attention_heads.unwrap_or(8);
        let eps = cfg.rms_norm_eps.unwrap_or(1e-6);
        let intermediate = hidden_size * 4;
        let act = match cfg.hidden_act.as_deref() {
            Some("silu") | Some("swish") => Activation::Silu,
            Some("relu") => Activation::Relu,
            _ => Activation::Silu,
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(AudioEncoderLayer::new(
                hidden_size,
                intermediate,
                num_heads,
                eps,
                act.clone(),
                vb.pp(format!("{i}")),
            )?);
        }
        Ok(Some(Self { layers }))
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

impl Model {
    fn token_id_for(tokenizer: &Tokenizer, candidates: &[&str]) -> Option<u32> {
        candidates.iter().find_map(|tok| tokenizer.token_to_id(tok))
    }

    fn resolve_local_media_path(source: &str) -> Option<PathBuf> {
        if let Some(path) = source.strip_prefix("file://") {
            let p = PathBuf::from(path);
            return p.exists().then_some(p);
        }
        let p = PathBuf::from(source);
        p.exists().then_some(p)
    }

    fn normalize_vec(mut v: Vec<f32>) -> Vec<f32> {
        let norm2: f32 = v.iter().map(|x| x * x).sum();
        let norm = norm2.sqrt().max(1e-6);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    fn to_hidden_vec(&self, t: &Tensor, hidden: usize) -> candle_core::Result<Vec<f32>> {
        let mut out = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if out.len() > hidden {
            out.truncate(hidden);
        } else if out.len() < hidden {
            out.resize(hidden, 0.0);
        }
        Ok(Self::normalize_vec(out))
    }

    fn vision_tower_feature(
        &self,
        image_path: &Path,
        hidden: usize,
    ) -> candle_core::Result<Option<Vec<f32>>> {
        let (patch_proj, embed_proj, vision_tower) = match (
            &self.vision_patch_proj,
            &self.vision_projection,
            &self.vision_tower,
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return Ok(None),
        };

        let patch = self.vision_patch_size.max(1);
        let side = patch * 14;
        let img = image::ImageReader::open(image_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .decode()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .to_rgb8();
        let resized = image::imageops::resize(
            &img,
            side as u32,
            side as u32,
            image::imageops::FilterType::Triangle,
        );

        let proj_in = patch_proj.weight().dim(1)?;
        let patches_h = side / patch;
        let patches_w = side / patch;
        let n_patches = patches_h * patches_w;
        let mut flat = vec![0f32; n_patches * proj_in];

        for py in 0..patches_h {
            for px in 0..patches_w {
                let patch_idx = py * patches_w + px;
                let base = patch_idx * proj_in;
                let mut off = 0usize;
                for yy in 0..patch {
                    for xx in 0..patch {
                        let p =
                            resized.get_pixel((px * patch + xx) as u32, (py * patch + yy) as u32);
                        if off + 2 < proj_in {
                            flat[base + off] = p[0] as f32 / 255.0;
                            flat[base + off + 1] = p[1] as f32 / 255.0;
                            flat[base + off + 2] = p[2] as f32 / 255.0;
                        }
                        off += 3;
                        if off >= proj_in {
                            break;
                        }
                    }
                    if off >= proj_in {
                        break;
                    }
                }
            }
        }

        let patch_t = Tensor::from_vec(flat, (1, n_patches, proj_in), &self.device)?;
        let mut feats = patch_proj.forward(&patch_t)?;

        if let Some(pos_table) = &self.vision_pos_table {
            let pos = if pos_table.rank() == 3 {
                pos_table.narrow(0, 0, 1)?.squeeze(0)?
            } else {
                pos_table.clone()
            };
            let max_pos = pos.dim(0)?;
            let take = n_patches.min(max_pos);
            if take > 0 {
                let pos_take = pos.narrow(0, 0, take)?.unsqueeze(0)?;
                let feat_take = feats.narrow(1, 0, take)?;
                feats = (feat_take + pos_take)?;
            }
        }

        feats = vision_tower.forward(&feats)?;

        let rows = feats.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        if rows.is_empty() {
            return Ok(None);
        }
        let dim = rows[0].len();
        let mut pooled = vec![0f32; dim];
        rows.iter().for_each(|r| {
            pooled.iter_mut().zip(r.iter()).for_each(|(a, b)| *a += *b);
        });
        let inv = 1f32 / rows.len() as f32;
        pooled.iter_mut().for_each(|x| *x *= inv);

        let pooled_t = Tensor::from_vec(pooled, (1, dim), &self.device)?;
        let projected = embed_proj.forward(&pooled_t)?;
        Ok(Some(self.to_hidden_vec(&projected, hidden)?))
    }

    fn audio_tower_feature(
        &self,
        audio_path: &Path,
        hidden: usize,
    ) -> candle_core::Result<Option<Vec<f32>>> {
        let (audio_in, audio_tower, audio_out, embed_proj) = match (
            &self.audio_input_proj,
            &self.audio_tower,
            &self.audio_output_proj,
            &self.audio_projection,
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Ok(None),
        };

        let mut reader = hound::WavReader::open(audio_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let spec = reader.spec();
        let channels = spec.channels.max(1) as usize;
        let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
            reader
                .samples::<f32>()
                .enumerate()
                .filter_map(|(i, s)| (i % channels == 0).then_some(s.ok()))
                .flatten()
                .collect()
        } else {
            let scale = (1i64 << (spec.bits_per_sample.saturating_sub(1) as u32)) as f32;
            reader
                .samples::<i32>()
                .enumerate()
                .filter_map(|(i, s)| (i % channels == 0).then_some(s.ok()))
                .flatten()
                .map(|v| v as f32 / scale.max(1.0))
                .collect()
        };

        if samples.is_empty() {
            return Ok(None);
        }

        let mut feat = vec![0f32; 1024];
        let n = samples.len();
        for (i, dst) in feat.iter_mut().enumerate() {
            let s = i * n / 1024;
            let e = ((i + 1) * n / 1024).max(s + 1);
            let end = e.min(n);
            let len = (end - s).max(1);
            let sum: f32 = samples[s..end].iter().map(|x| x.abs()).sum();
            *dst = sum / len as f32;
        }

        let x = Tensor::from_vec(feat, (1, 1024), &self.device)?;
        let x = audio_in.forward(&x)?;
        let x = x.unsqueeze(1)?;
        let x = audio_tower.forward(&x)?;
        let x = x.squeeze(1)?;
        let x = audio_out.forward(&x)?;
        let x = embed_proj.forward(&x)?;
        Ok(Some(self.to_hidden_vec(&x, hidden)?))
    }

    fn inject_features(
        base: &mut [f32],
        hidden: usize,
        positions: &[usize],
        features: &HashMap<String, Vec<f32>>,
        urls: &[String],
    ) {
        if positions.is_empty() || urls.is_empty() {
            return;
        }

        positions.iter().enumerate().for_each(|(idx, pos)| {
            let source = &urls[idx % urls.len()];
            if let Some(feature) = features.get(source) {
                let row = &mut base[pos * hidden..(pos + 1) * hidden];
                row.iter_mut()
                    .zip(feature.iter())
                    .for_each(|(dst, feat)| *dst = (*dst * 0.3) + (feat * 0.7));
            }
        });
    }

    fn build_multimodal_inputs_embeds(
        &self,
        input_ids: &[u32],
        image_urls: &[String],
        audio_urls: &[String],
    ) -> candle_core::Result<Option<Tensor>> {
        let image_positions: Vec<usize> = self
            .image_token_id
            .map(|token_id| {
                input_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, id)| (*id == token_id).then_some(idx))
                    .collect()
            })
            .unwrap_or_default();
        let audio_positions: Vec<usize> = self
            .audio_token_id
            .map(|token_id| {
                input_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, id)| (*id == token_id).then_some(idx))
                    .collect()
            })
            .unwrap_or_default();

        if (image_positions.is_empty() || image_urls.is_empty())
            && (audio_positions.is_empty() || audio_urls.is_empty())
        {
            return Ok(None);
        }

        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let embeds = self.inner.embed_input_ids(&input)?;
        let embeds_f32 = embeds.to_dtype(DType::F32)?;
        let (_b, seq_len, hidden) = embeds_f32.dims3()?;

        let mut image_features: HashMap<String, Vec<f32>> = HashMap::new();
        image_urls.iter().for_each(|src| {
            if image_features.contains_key(src) {
                return;
            }
            if let Some(path) = Self::resolve_local_media_path(src) {
                if let Ok(Some(f)) = self.vision_tower_feature(&path, hidden) {
                    image_features.insert(src.clone(), f);
                }
            }
        });
        let mut audio_features: HashMap<String, Vec<f32>> = HashMap::new();
        audio_urls.iter().for_each(|src| {
            if audio_features.contains_key(src) {
                return;
            }
            if let Some(path) = Self::resolve_local_media_path(src) {
                if let Ok(Some(f)) = self.audio_tower_feature(&path, hidden) {
                    audio_features.insert(src.clone(), f);
                }
            }
        });

        let mut flat: Vec<f32> = embeds_f32.flatten_all()?.to_vec1()?;
        Self::inject_features(
            &mut flat,
            hidden,
            &image_positions,
            &image_features,
            image_urls,
        );
        Self::inject_features(
            &mut flat,
            hidden,
            &audio_positions,
            &audio_features,
            audio_urls,
        );

        let embeds = Tensor::from_vec(flat, (1, seq_len, hidden), &self.device)?;
        Ok(Some(embeds.to_dtype(self.dtype)?))
    }

    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb_root = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
        let vb = vb_root.set_prefix("model.language_model");

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
        let vision_cfg = top_cfg.vision_config.clone().unwrap_or_default();
        let audio_cfg = top_cfg.audio_config.clone().unwrap_or_default();
        let vision_patch_size = vision_cfg.patch_size.unwrap_or(16);
        let inner = Gemma4TextModel::new(&cfg, vb)?;
        let image_token_id = Self::token_id_for(&tokenizer, &["<|image|>", "<image>"]);
        let audio_token_id = Self::token_id_for(&tokenizer, &["<|audio|>", "<audio>"]);
        let vision_patch_proj = vb_root
            .pp("model.vision_tower.patch_embedder")
            .get((768usize, 768usize), "input_proj.weight")
            .ok()
            .map(|w| Linear::new(w, None));
        let vision_pos_table = vb_root
            .pp("model.vision_tower.patch_embedder")
            .get((2usize, 10240usize, 768usize), "position_embedding_table")
            .ok();
        let vision_tower = VisionTower::new(&vision_cfg, vb_root.pp("model.vision_tower.encoder"))?;
        let audio_input_proj = vb_root
            .pp("model.audio_tower.subsample_conv_projection")
            .get((1024usize, 1024usize), "input_proj_linear.weight")
            .ok()
            .map(|w| Linear::new(w, None));
        let audio_tower = AudioTower::new(&audio_cfg, vb_root.pp("model.audio_tower.layers"))?;
        let audio_output_proj = {
            let wb = vb_root.pp("model.audio_tower");
            let w = wb.get((1536usize, 1024usize), "output_proj.weight").ok();
            let b = wb.get((1536usize,), "output_proj.bias").ok();
            w.map(|ww| Linear::new(ww, b))
        };
        let vision_projection = vb_root
            .pp("model.embed_vision")
            .get((cfg.hidden_size, 768usize), "embedding_projection.weight")
            .ok()
            .map(|w| Linear::new(w, None));
        let audio_projection = vb_root
            .pp("model.embed_audio")
            .get(
                (cfg.hidden_size, cfg.hidden_size),
                "embedding_projection.weight",
            )
            .ok()
            .map(|w| Linear::new(w, None));

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            sliding_window,
            eos_token_ids,
            image_token_id,
            audio_token_id,
            vision_patch_proj,
            vision_pos_table,
            vision_patch_size,
            vision_tower,
            audio_input_proj,
            audio_tower,
            audio_output_proj,
            vision_projection,
            audio_projection,
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

    pub fn forward_step_with_inputs_embeds(
        &mut self,
        input_ids: &[u32],
        inputs_embeds: &Tensor,
        start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner
            .forward_with_inputs_embeds(&input, inputs_embeds, start_pos, self.sliding_window)
    }

    pub fn forward_step_with_multimodal(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
        image_urls: &[String],
        audio_urls: &[String],
    ) -> candle_core::Result<Tensor> {
        if start_pos != 0 {
            return self.forward_step(input_ids, start_pos);
        }

        if let Some(inputs_embeds) =
            self.build_multimodal_inputs_embeds(input_ids, image_urls, audio_urls)?
        {
            self.forward_step_with_inputs_embeds(input_ids, &inputs_embeds, start_pos)
        } else {
            self.forward_step(input_ids, start_pos)
        }
    }

    pub fn warmup(&mut self) {
        let token = self.eos_token_ids.first().copied().unwrap_or(1);
        let _ = self.forward_step(&[token], 0);
        self.clear_kv_cache();
    }
}
