use std::collections::HashMap;
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
    #[serde(default)]
    pub attention_k_eq_v: bool,
    pub head_dim: usize,
    pub global_head_dim: Option<usize>,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub hidden_size_per_layer_input: Option<usize>,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub final_logit_softcapping: Option<f64>,
    pub rope_parameters: Option<RopeParameters>,
    pub num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub expert_intermediate_size: Option<usize>,
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

    pub fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_sliding_layer(layer_idx) {
            self.num_key_value_heads
        } else {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
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
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
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
struct Gemma4TextRouter {
    proj: Linear,
    scale: Tensor,
    per_expert_scale: Tensor,
    top_k_experts: usize,
    eps: f64,
    scalar_root_size: f64,
}

impl Gemma4TextRouter {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(0);
        let top_k_experts = cfg.top_k_experts.unwrap_or(0);
        if num_experts == 0 || top_k_experts == 0 {
            candle_core::bail!("MoE router requires num_experts > 0 and top_k_experts > 0")
        }

        let proj = linear(cfg.hidden_size, num_experts, false, vb.pp("proj"))?;
        let scale = vb.get(cfg.hidden_size, "scale")?;
        let per_expert_scale = vb.get(num_experts, "per_expert_scale")?;

        Ok(Self {
            proj,
            scale,
            per_expert_scale,
            top_k_experts,
            eps: cfg.rms_norm_eps,
            scalar_root_size: (cfg.hidden_size as f64).powf(-0.5),
        })
    }

    fn route(&self, hidden_states: &Tensor) -> candle_core::Result<Vec<Vec<(usize, f32)>>> {
        let x_dtype = hidden_states.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let hidden_size = hidden_states.dim(D::Minus1)?;
        let x = hidden_states.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;

        let x_scaled = (x_normed.broadcast_mul(&self.scale)? * self.scalar_root_size)?;
        let scores = x_scaled.apply(&self.proj)?;
        let probs = softmax_last_dim_fallback(&scores)?;

        let (num_tokens, num_experts) = probs.dims2()?;
        let k = self.top_k_experts.min(num_experts);
        if k == 0 {
            return Ok(vec![Vec::new(); num_tokens]);
        }

        // Device-side top-k indices and probs.
        let sorted_idx = probs.arg_sort_last_dim(false)?;
        let topk_idx = sorted_idx.narrow(1, 0, k)?.contiguous()?;
        let topk_probs = probs.gather(&topk_idx, D::Minus1)?;

        // Normalize top-k weights and apply per-expert scaling.
        let topk_sum = topk_probs.sum_keepdim(D::Minus1)?;
        let topk_norm = topk_probs.broadcast_div(&topk_sum)?;

        let scale_2d = self
            .per_expert_scale
            .unsqueeze(0)?
            .expand((num_tokens, num_experts))?
            .contiguous()?;
        let topk_scales = scale_2d.gather(&topk_idx, D::Minus1)?;
        let topk_weights = topk_norm.broadcast_mul(&topk_scales)?;

        // Only move compact [num_tokens, k] tensors to CPU.
        let idx_cpu = topk_idx
            .to_dtype(DType::U32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<u32>()?;
        let w_cpu = topk_weights
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;

        let mut routes = Vec::with_capacity(num_tokens);
        for (idx_row, w_row) in idx_cpu.into_iter().zip(w_cpu.into_iter()) {
            let mut row = Vec::with_capacity(k);
            for (expert_idx, weight) in idx_row.into_iter().zip(w_row.into_iter()) {
                row.push((expert_idx as usize, weight));
            }
            routes.push(row);
        }
        Ok(routes)
    }
}

#[derive(Debug, Clone)]
struct Gemma4TextExperts {
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act_fn: Activation,
}

impl Gemma4TextExperts {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let num_experts = cfg.num_experts.unwrap_or(0);
        let moe_intermediate_size = cfg
            .moe_intermediate_size
            .or(cfg.expert_intermediate_size)
            .unwrap_or(0);

        if num_experts == 0 || moe_intermediate_size == 0 {
            candle_core::bail!("MoE experts require num_experts > 0 and moe_intermediate_size > 0")
        }

        let gate_up_proj = vb.get(
            (num_experts, 2 * moe_intermediate_size, cfg.hidden_size),
            "gate_up_proj",
        )?;
        let down_proj = vb.get(
            (num_experts, cfg.hidden_size, moe_intermediate_size),
            "down_proj",
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }

    fn offload_to_cpu(&mut self) -> usize {
        if matches!(self.gate_up_proj.device(), Device::Cpu) {
            return 0;
        }

        let mut moved = 0usize;
        if let Ok(t) = self.gate_up_proj.to_device(&Device::Cpu) {
            self.gate_up_proj = t;
            moved += 1;
        }
        if let Ok(t) = self.down_proj.to_device(&Device::Cpu) {
            self.down_proj = t;
            moved += 1;
        }
        moved
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        routes: &[Vec<(usize, f32)>],
    ) -> candle_core::Result<Tensor> {
        let (num_tokens, hidden_size) = hidden_states.dims2()?;
        if routes.len() != num_tokens {
            candle_core::bail!("routes/token mismatch: {} vs {}", routes.len(), num_tokens)
        }

        let mut outputs = Tensor::zeros(
            (num_tokens, hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;

        // Group token contributions by expert to run larger matmuls.
        let mut per_expert: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (token_idx, token_routes) in routes.iter().enumerate() {
            for (expert_idx, weight) in token_routes {
                if *weight == 0.0 {
                    continue;
                }
                per_expert
                    .entry(*expert_idx)
                    .or_default()
                    .push((token_idx, *weight));
            }
        }

        let mut expert_cache: HashMap<usize, (Tensor, Tensor)> = HashMap::new();
        for (expert_idx, assignments) in per_expert {
            if assignments.is_empty() {
                continue;
            }

            if let std::collections::hash_map::Entry::Vacant(e) = expert_cache.entry(expert_idx) {
                let gate_up = self
                    .gate_up_proj
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .to_device(hidden_states.device())?;
                let down = self
                    .down_proj
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .to_device(hidden_states.device())?;
                e.insert((gate_up, down));
            }
            let (gate_up, down) = expert_cache.get(&expert_idx).unwrap();

            let token_ids: Vec<u32> = assignments.iter().map(|(idx, _)| *idx as u32).collect();
            let token_idx_t = Tensor::new(token_ids.as_slice(), hidden_states.device())?;
            let expert_in = hidden_states.index_select(&token_idx_t, 0)?;

            let gate_up_t = gate_up.transpose(0, 1)?;
            let down_t = down.transpose(0, 1)?;

            let gate_up_out = expert_in.matmul(&gate_up_t)?;
            let interm = gate_up_out.dim(1)? / 2;
            let gate = gate_up_out.narrow(1, 0, interm)?;
            let up = gate_up_out.narrow(1, interm, interm)?;
            let hidden = (gate.apply(&self.act_fn)? * up)?;
            let mut expert_out = hidden.matmul(&down_t)?;

            let weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
            let weights_t = Tensor::new(weights.as_slice(), hidden_states.device())?
                .reshape((assignments.len(), 1))?;
            expert_out = expert_out.broadcast_mul(&weights_t)?;

            outputs = outputs.index_add(&token_idx_t, &expert_out, 0)?;
        }

        Ok(outputs)
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
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    attention_k_eq_v: bool,
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
        let num_kv_heads = cfg.num_kv_heads_for_layer(layer_idx);
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            candle_core::bail!(
                "invalid Gemma4 heads config at layer {layer_idx}: num_attention_heads={num_heads}, num_kv_heads={num_kv_heads}"
            );
        }
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let bias = cfg.attention_bias;

        let q_proj = linear(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = if cfg.attention_k_eq_v && !vb.pp("v_proj").contains_tensor("weight") {
            None
        } else {
            Some(linear(
                hidden_sz,
                num_kv_heads * head_dim,
                bias,
                vb.pp("v_proj"),
            )?)
        };
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
            attention_k_eq_v: cfg.attention_k_eq_v,
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
        shared_kv: Option<(&Tensor, &Tensor)>,
    ) -> candle_core::Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = match &self.v_proj {
            Some(v_proj) => v_proj.forward(xs)?,
            None if self.attention_k_eq_v => key_states.clone(),
            None => {
                candle_core::bail!("missing v_proj for Gemma4 attention without attention_k_eq_v")
            }
        };

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

        // Gemma4 text attention applies RMSNorm to V without learnable scale.
        {
            let v_dtype = value_states.dtype();
            let v_internal_dtype = match v_dtype {
                DType::F16 | DType::BF16 => DType::F32,
                d => d,
            };
            let h = value_states.dim(D::Minus1)?;
            let v = value_states.to_dtype(v_internal_dtype)?;
            let norm_v = (v.sqr()?.sum_keepdim(D::Minus1)? / h as f64)?;
            value_states = v
                .broadcast_div(&(norm_v + 1e-6)?.sqrt()?)?
                .to_dtype(v_dtype)?;
        }

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states, produced_kv) = if let Some((k, v)) = shared_kv {
            (k.clone(), v.clone(), None)
        } else {
            let (k, v) = match &mut self.kv_cache {
                KvCache::Normal(cache) => cache.append(&key_states, &value_states)?,
                KvCache::Rotating(cache) => cache.append(&key_states, &value_states)?,
            };
            (k.clone(), v.clone(), Some((k, v)))
        };

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        // Gemma4 text eager-attention path uses scaling=1.0.
        let attn_weights = query_states.matmul(&key_states.transpose(2, 3)?)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = softmax_last_dim_fallback(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        let out = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?;
        Ok((out, produced_kv))
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }

    fn kv_seq_len(&self) -> usize {
        match &self.kv_cache {
            KvCache::Normal(c) => c.current_seq_len(),
            KvCache::Rotating(c) => c.current_seq_len(),
        }
    }

    fn kv_tensors(&self) -> Option<(Tensor, Tensor)> {
        match &self.kv_cache {
            KvCache::Normal(c) => c.k().ok().flatten().zip(c.v().ok().flatten()),
            KvCache::Rotating(c) => c.k().ok().flatten().zip(c.v().ok().flatten()),
        }
    }

    fn restore_kv_cache(&mut self, cache: Option<(Tensor, Tensor)>) {
        self.clear_kv_cache();
        let Some((k, v)) = cache else {
            return;
        };

        match &mut self.kv_cache {
            KvCache::Normal(c) => {
                let _ = c.append(&k, &v);
            }
            KvCache::Rotating(c) => {
                let _ = c.append(&k, &v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_config(attention_k_eq_v: bool) -> Config {
        Config {
            attention_bias: false,
            attention_k_eq_v,
            head_dim: 2,
            global_head_dim: None,
            hidden_activation: Activation::Silu,
            hidden_size: 4,
            hidden_size_per_layer_input: None,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            num_key_value_heads: 1,
            num_global_key_value_heads: None,
            rms_norm_eps: 1e-6,
            vocab_size: 16,
            max_position_embeddings: 16,
            sliding_window: 8,
            layer_types: vec!["sliding_attention".to_string()],
            final_logit_softcapping: None,
            rope_parameters: None,
            num_kv_shared_layers: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            expert_intermediate_size: None,
            use_double_wide_mlp: None,
            eos_token_id: None,
        }
    }

    fn tensor_map(include_v_proj: bool) -> candle_core::Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();

        for name in ["q_proj", "k_proj", "o_proj"] {
            tensors.insert(
                format!("{name}.weight"),
                Tensor::zeros((4, 4), DType::F32, &device)?,
            );
        }
        tensors.insert(
            "k_proj.weight".to_string(),
            Tensor::ones((2, 4), DType::F32, &device)?,
        );
        if include_v_proj {
            tensors.insert(
                "v_proj.weight".to_string(),
                Tensor::ones((2, 4), DType::F32, &device)?,
            );
        }
        tensors.insert(
            "q_norm.weight".to_string(),
            Tensor::ones(2, DType::F32, &device)?,
        );
        tensors.insert(
            "k_norm.weight".to_string(),
            Tensor::ones(2, DType::F32, &device)?,
        );

        Ok(tensors)
    }

    fn moe_test_config(num_experts: usize, top_k_experts: usize) -> Config {
        let mut cfg = test_config(false);
        cfg.enable_moe_block = true;
        cfg.num_experts = Some(num_experts);
        cfg.top_k_experts = Some(top_k_experts);
        cfg
    }

    fn router_tensor_map() -> candle_core::Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "proj.weight".to_string(),
            Tensor::from_vec(
                vec![
                    3f32, 0., 0., 0., // expert 0
                    2., 1., 0., 0., // expert 1
                    1., 2., 0., 0., // expert 2
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("scale".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "per_expert_scale".to_string(),
            Tensor::from_vec(vec![1f32, 2., 3.], 3, &device)?,
        );
        Ok(tensors)
    }

    #[test]
    fn config_deserialization_preserves_attention_k_eq_v_flag() {
        let cfg: Config = serde_json::from_value(json!({
            "attention_bias": false,
            "attention_k_eq_v": true,
            "head_dim": 2,
            "hidden_activation": "silu",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "vocab_size": 16,
            "max_position_embeddings": 16,
            "sliding_window": 8,
            "layer_types": ["sliding_attention"]
        }))
        .expect("config should deserialize");

        assert!(cfg.attention_k_eq_v);
    }

    #[test]
    fn attention_allows_missing_v_proj_when_attention_k_eq_v_is_enabled() {
        let cfg = test_config(true);
        let vb = VarBuilder::from_tensors(
            tensor_map(false).expect("tensor map"),
            DType::F32,
            &Device::Cpu,
        );
        let mut attention =
            Attention::new(&cfg, 0, vb).expect("attention should build without v_proj");
        let xs =
            Tensor::ones((1, 2, cfg.hidden_size), DType::F32, &Device::Cpu).expect("input tensor");

        let (output, produced_kv) = attention
            .forward(&xs, None, 0, None)
            .expect("forward should reuse k projection as v");

        assert_eq!(
            output.dims3().expect("output dims"),
            (1, 2, cfg.hidden_size)
        );
        assert!(produced_kv.is_some());
    }

    #[test]
    fn attention_still_requires_v_proj_when_attention_k_eq_v_is_disabled() {
        let cfg = test_config(false);
        let vb = VarBuilder::from_tensors(
            tensor_map(false).expect("tensor map"),
            DType::F32,
            &Device::Cpu,
        );

        let err = Attention::new(&cfg, 0, vb).expect_err("v_proj should still be required");

        assert!(err.to_string().contains("v_proj.weight"));
    }

    #[test]
    fn router_route_handles_non_contiguous_topk_gather_inputs() {
        let cfg = moe_test_config(3, 2);
        let vb = VarBuilder::from_tensors(
            router_tensor_map().expect("router tensor map"),
            DType::F32,
            &Device::Cpu,
        );
        let router = Gemma4TextRouter::new(&cfg, vb).expect("router should build");
        let hidden_states = Tensor::from_vec(
            vec![
                1f32, 0., 0., 0., // token 0 -> experts 0,1
                0., 1., 0., 0., // token 1 -> experts 2,1
            ],
            (2, cfg.hidden_size),
            &Device::Cpu,
        )
        .expect("hidden states");

        let routes = router.route(&hidden_states).expect("route should succeed");

        assert_eq!(routes.len(), 2);
        assert_eq!(
            routes[0].iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            routes[1].iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(routes
            .iter()
            .flatten()
            .all(|(_, weight)| weight.is_finite() && *weight > 0.0));
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    router: Option<Gemma4TextRouter>,
    experts: Option<Gemma4TextExperts>,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    pre_feedforward_layernorm_2: Option<RmsNorm>,
    post_feedforward_layernorm_1: Option<RmsNorm>,
    post_feedforward_layernorm_2: Option<RmsNorm>,
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

        let enable_moe = cfg.enable_moe_block;
        let (
            router,
            experts,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
        ) = if enable_moe {
            (
                Some(Gemma4TextRouter::new(cfg, vb.pp("router"))?),
                Some(Gemma4TextExperts::new(cfg, vb.pp("experts"))?),
                Some(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("pre_feedforward_layernorm_2"),
                )?),
                Some(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_1"),
                )?),
                Some(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_2"),
                )?),
            )
        } else {
            (None, None, None, None, None)
        };

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
            router,
            experts,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
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
        shared_kv: Option<(&Tensor, &Tensor)>,
    ) -> candle_core::Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, produced_kv) =
            self.self_attn
                .forward(&xs, attention_mask, seqlen_offset, shared_kv)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let mut xs = xs.apply(&self.mlp)?;

        if let (Some(router), Some(experts), Some(pre_ff2), Some(post_ff1), Some(post_ff2)) = (
            self.router.as_ref(),
            self.experts.as_mut(),
            self.pre_feedforward_layernorm_2.as_ref(),
            self.post_feedforward_layernorm_1.as_ref(),
            self.post_feedforward_layernorm_2.as_ref(),
        ) {
            let (b, s, h) = residual.dims3()?;
            let residual_flat = residual.reshape((b * s, h))?;
            let routes = router.route(&residual_flat)?;

            let moe_in = residual_flat.apply(pre_ff2)?;
            let moe_out = experts.forward(&moe_in, &routes)?;
            let moe_out = moe_out.reshape((b, s, h))?.apply(post_ff2)?;

            let mlp_out = xs.apply(post_ff1)?;
            xs = (mlp_out + moe_out)?;
        }

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

        Ok((xs.broadcast_mul(&self.layer_scalar)?, produced_kv))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }

    fn offload_experts_to_cpu(&mut self) -> usize {
        self.experts
            .as_mut()
            .map(|e| e.offload_to_cpu())
            .unwrap_or(0)
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
    shared_kv_source: Vec<Option<usize>>,
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

        let first_shared_layer = cfg
            .num_hidden_layers
            .saturating_sub(cfg.num_kv_shared_layers.unwrap_or(0));
        let mut shared_kv_source = vec![None; cfg.num_hidden_layers];
        if first_shared_layer > 0 {
            for (layer_idx, shared_src) in shared_kv_source
                .iter_mut()
                .enumerate()
                .skip(first_shared_layer)
            {
                let lt = cfg.layer_type(layer_idx);
                let src = (0..first_shared_layer)
                    .rev()
                    .find(|&j| cfg.layer_type(j) == lt);
                *shared_src = src;
            }
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
            shared_kv_source,
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

    fn forward_from_scaled_embeddings(
        &mut self,
        input_ids: &Tensor,
        mut xs: Tensor,
        seqlen_offset: usize,
        sliding_window: usize,
    ) -> candle_core::Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;

        let (full_attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset, sliding_window)?;

        let mut produced_layer_kv: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.is_sliding {
                &sliding_attention_mask
            } else {
                &full_attention_mask
            };

            let shared_kv = self.shared_kv_source[layer_idx]
                .and_then(|src| produced_layer_kv[src].as_ref().map(|(k, v)| (k, v)));

            let per_layer_input = if let Some(ref all_inputs) = per_layer_inputs {
                Some(all_inputs.narrow(2, layer_idx, 1)?.squeeze(2)?)
            } else {
                None
            };

            let (new_xs, produced_kv) = layer.forward(
                &xs,
                per_layer_input.as_ref(),
                mask.as_ref(),
                seqlen_offset,
                shared_kv,
            )?;
            xs = new_xs;
            produced_layer_kv[layer_idx] = produced_kv;
        }

        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;

        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => (logits / sc)?.tanh()? * sc,
        }
    }

    pub fn forward_with_inputs_embeds(
        &mut self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        sliding_window: usize,
    ) -> candle_core::Result<Tensor> {
        let xs = (inputs_embeds * (self.hidden_size as f64).sqrt())?;
        self.forward_from_scaled_embeddings(input_ids, xs, seqlen_offset, sliding_window)
    }

    pub fn embed_input_ids(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        sliding_window: usize,
    ) -> candle_core::Result<Tensor> {
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;
        self.forward_with_inputs_embeds(input_ids, &inputs_embeds, seqlen_offset, sliding_window)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    pub fn offload_experts_to_cpu(&mut self) -> usize {
        let mut moved = 0usize;
        for layer in self.layers.iter_mut() {
            moved += layer.offload_experts_to_cpu();
        }
        moved
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| {
                layer
                    .self_attn
                    .kv_tensors()
                    .map(|(k, v)| {
                        let k_bytes = k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
                        let v_bytes = v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
                        k_bytes + v_bytes
                    })
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Extract per-layer KV caches (valid portion only).
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|layer| {
                layer.self_attn.kv_tensors().map(|(k, v)| {
                    let len = layer.self_attn.kv_seq_len();
                    if len > 0 && len < k.dim(2).unwrap_or(0) {
                        (
                            k.narrow(2, 0, len).unwrap_or_else(|_| k.clone()),
                            v.narrow(2, 0, len).unwrap_or_else(|_| v.clone()),
                        )
                    } else {
                        (k, v)
                    }
                })
            })
            .collect()
    }

    /// Restore per-layer KV caches.
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.into_iter()) {
            layer.self_attn.restore_kv_cache(cache);
        }
    }
}
