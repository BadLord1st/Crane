use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};

fn repeat_kv(xs: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, n_kv, s, h) = xs.dims4()?;
    xs.reshape((b, n_kv, 1, s, h))?
        .expand((b, n_kv, n_rep, s, h))?
        .reshape((b, n_kv * n_rep, s, h))
}

fn softmax_last_dim_fallback(xs: &Tensor) -> candle_core::Result<Tensor> {
    // Metal backend does not currently provide softmax-last-dim kernel.
    let input_dtype = xs.dtype();
    let compute_dtype = match input_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let xs = xs.to_dtype(compute_dtype)?;
    let max = xs.max_keepdim(D::Minus1)?;
    let exps = xs.broadcast_sub(&max)?.exp()?;
    let den = exps.sum_keepdim(D::Minus1)?;
    exps.broadcast_div(&den)?.to_dtype(input_dtype)
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub global_head_dim: Option<usize>,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub hidden_size_per_layer_input: Option<usize>,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub final_logit_softcapping: Option<f64>,
    pub rope_parameters: Option<RopeParameters>,
    pub num_kv_shared_layers: Option<usize>,
    pub use_double_wide_mlp: Option<bool>,
    pub eos_token_id: Option<u32>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct RopeParameters {
    pub full_attention: Option<RopeSpec>,
    pub sliding_attention: Option<RopeSpec>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct RopeSpec {
    pub rope_theta: f64,
}

impl Config {
    pub fn hidden_size_per_layer_input(&self) -> usize {
        self.hidden_size_per_layer_input.unwrap_or(0)
    }

    pub fn layer_type(&self, layer_idx: usize) -> &str {
        self.layer_types
            .get(layer_idx)
            .map(String::as_str)
            .unwrap_or("sliding_attention")
    }

    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == "sliding_attention"
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_sliding_layer(layer_idx) {
            self.head_dim
        } else {
            self.global_head_dim.unwrap_or(self.head_dim)
        }
    }

    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f64 {
        match self.layer_type(layer_idx) {
            "full_attention" => self
                .rope_parameters
                .as_ref()
                .and_then(|p| p.full_attention.as_ref())
                .map(|r| r.rope_theta)
                .unwrap_or(1_000_000.0),
            _ => self
                .rope_parameters
                .as_ref()
                .and_then(|p| p.sliding_attention.as_ref())
                .map(|r| r.rope_theta)
                .unwrap_or(10_000.0),
        }
    }

    pub fn mlp_intermediate_for_layer(&self, layer_idx: usize) -> usize {
        let shared = self.num_kv_shared_layers.unwrap_or(0);
        let use_double = self.use_double_wide_mlp.unwrap_or(false);
        if shared > 0 && use_double && layer_idx >= self.num_hidden_layers.saturating_sub(shared) {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    inv_freq: Tensor,
    dtype: DType,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> candle_core::Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        Ok(Self { inv_freq, dtype })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let t = Tensor::arange(
            seqlen_offset as u32,
            (seqlen_offset + seq_len) as u32,
            q.device(),
        )?
        .to_dtype(self.dtype)?
        .reshape((seq_len, 1))?;
        let freqs = t.matmul(&self.inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let target_dtype = q.dtype();
        let cos_full = Tensor::cat(&[&cos, &cos], D::Minus1)?
            .to_dtype(target_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin_full = Tensor::cat(&[&sin, &sin], D::Minus1)?
            .to_dtype(target_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let q_embed = (q.broadcast_mul(&cos_full)? + rotate_half(q)?.broadcast_mul(&sin_full)?)?;
        let k_embed = (k.broadcast_mul(&cos_full)? + rotate_half(k)?.broadcast_mul(&sin_full)?)?;
        Ok((q_embed, k_embed))
    }
}

fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let half_dim = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let gate_proj = linear(hidden_size, intermediate_size, false, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, false, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
}

impl Attention {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let bias = cfg.attention_bias;

        let q_proj = linear(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let is_sliding = cfg.is_sliding_layer(layer_idx);
        let kv_cache = if is_sliding {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.sliding_window,
            ))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))
        };

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            head_dim,
            cfg.rope_theta_for_layer(layer_idx),
            vb.device(),
        )?);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &mut self.kv_cache {
            KvCache::Normal(cache) => cache.append(&key_states, &value_states)?,
            KvCache::Rotating(cache) => cache.append(&key_states, &value_states)?,
        };

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = softmax_last_dim_fallback(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    layer_scalar: Tensor,
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    act_fn: Activation,
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, layer_idx, vb.pp("self_attn"))?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.mlp_intermediate_for_layer(layer_idx),
            cfg.hidden_activation,
            vb.pp("mlp"),
        )?;

        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let layer_scalar = vb.get((1,), "layer_scalar")?;

        let h_per = cfg.hidden_size_per_layer_input();
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if h_per > 0 {
            (
                Some(linear(
                    cfg.hidden_size,
                    h_per,
                    false,
                    vb.pp("per_layer_input_gate"),
                )?),
                Some(linear(
                    h_per,
                    cfg.hidden_size,
                    false,
                    vb.pp("per_layer_projection"),
                )?),
                Some(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_per_layer_input_norm"),
                )?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            layer_scalar,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            act_fn: cfg.hidden_activation,
            is_sliding: cfg.is_sliding_layer(layer_idx),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;

        if let (
            Some(per_layer_input_gate),
            Some(per_layer_projection),
            Some(post_per_layer_input_norm),
            Some(per_layer_input),
        ) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let residual = &xs;
            let x_gate = xs.apply(per_layer_input_gate)?.apply(&self.act_fn)?;
            let x_gate = x_gate.broadcast_mul(per_layer_input)?;
            let x_gate = x_gate.apply(per_layer_projection)?;
            let x_gate = x_gate.apply(post_per_layer_input_norm)?;
            xs = (residual + x_gate)?;
        }

        xs.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub struct Gemma4TextModel {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
}

impl Gemma4TextModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let hidden_size_per_layer_input = cfg.hidden_size_per_layer_input();
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if hidden_size_per_layer_input > 0 {
                let embed_tokens_per_layer = candle_nn::embedding(
                    cfg.vocab_size,
                    cfg.num_hidden_layers * hidden_size_per_layer_input,
                    vb.pp("embed_tokens_per_layer"),
                )?;
                let per_layer_model_projection = linear(
                    cfg.hidden_size,
                    cfg.num_hidden_layers * hidden_size_per_layer_input,
                    false,
                    vb.pp("per_layer_model_projection"),
                )?;
                let per_layer_projection_norm = RmsNorm::new(
                    hidden_size_per_layer_input,
                    cfg.rms_norm_eps,
                    vb.pp("per_layer_projection_norm"),
                )?;
                (
                    Some(embed_tokens_per_layer),
                    Some(per_layer_model_projection),
                    Some(per_layer_projection_norm),
                )
            } else {
                (None, None, None)
            };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            hidden_size_per_layer_input,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
        sliding_window: usize,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }

        let full_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;

        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(sliding_window),
            self.dtype,
            &self.device,
        )?;

        Ok((Some(full_mask), Some(sliding_mask)))
    }

    fn compute_per_layer_inputs(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        if self.hidden_size_per_layer_input == 0 {
            return Ok(None);
        }

        let embed_tokens_per_layer = self.embed_tokens_per_layer.as_ref().expect("checked");
        let per_layer_model_projection = self.per_layer_model_projection.as_ref().expect("checked");
        let per_layer_projection_norm = self.per_layer_projection_norm.as_ref().expect("checked");

        let (b, s) = input_ids.dims2()?;

        let per_layer_from_tokens = embed_tokens_per_layer.forward(input_ids)?;
        let per_layer_from_tokens = (per_layer_from_tokens
            * (self.hidden_size_per_layer_input as f64).sqrt())?
        .reshape((b, s, self.layers.len(), self.hidden_size_per_layer_input))?;

        let per_layer_projection = per_layer_model_projection.forward(inputs_embeds)?;
        let per_layer_projection = (per_layer_projection * (self.hidden_size as f64).powf(-0.5))?
            .reshape((
            b,
            s,
            self.layers.len(),
            self.hidden_size_per_layer_input,
        ))?;
        let per_layer_projection = per_layer_projection_norm.forward(&per_layer_projection)?;

        Ok(Some(
            ((per_layer_from_tokens + per_layer_projection)? * (2f64).powf(-0.5))?,
        ))
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        sliding_window: usize,
    ) -> candle_core::Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;
        let mut xs = (inputs_embeds * (self.hidden_size as f64).sqrt())?;

        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;

        let (full_attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset, sliding_window)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.is_sliding {
                &sliding_attention_mask
            } else {
                &full_attention_mask
            };

            let per_layer_input = if let Some(ref all_inputs) = per_layer_inputs {
                Some(all_inputs.narrow(2, layer_idx, 1)?.squeeze(2)?)
            } else {
                None
            };

            xs = layer.forward(&xs, per_layer_input.as_ref(), mask.as_ref(), seqlen_offset)?;
        }

        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;

        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => ((logits / sc)?.tanh()? * sc),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
