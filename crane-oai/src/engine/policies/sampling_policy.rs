use crate::engine::runtime::SamplingDefaults;

#[derive(Debug, Clone)]
pub struct SamplingResolution {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub notes: Vec<String>,
}

pub fn resolve_sampling(
    defaults: SamplingDefaults,
    requested_temperature: Option<f64>,
    requested_top_p: Option<f64>,
    requested_top_k: Option<usize>,
) -> SamplingResolution {
    let mut notes = Vec::new();

    let mut temperature = requested_temperature.or(defaults.temperature);
    if let Some(t) = temperature {
        if !t.is_finite() {
            notes.push(format!("detected temperature={} (fallback to default)", t));
            temperature = defaults.temperature;
        } else if t < 0.0 {
            notes.push(format!(
                "detected temperature={:.4} (clamp to temperature=0.0000)",
                t
            ));
            temperature = Some(0.0);
        }
    }

    let mut top_p = requested_top_p.or(defaults.top_p);
    if let Some(tp) = top_p {
        if !tp.is_finite() {
            notes.push(format!("detected top_p={} (fallback to default)", tp));
            top_p = defaults.top_p;
        } else if tp < 0.0 {
            notes.push(format!("detected top_p={:.4} (clamp to top_p=0.0000)", tp));
            top_p = Some(0.0);
        } else if tp > 1.0 {
            notes.push(format!("detected top_p={:.4} (clamp to top_p=1.0000)", tp));
            top_p = Some(1.0);
        }
    }

    let mut top_k = requested_top_k.or(defaults.top_k);
    if requested_top_k == Some(0) {
        notes.push("detected top_k=0 (disable top_k -> None)".to_string());
        top_k = None;
    }

    SamplingResolution {
        temperature,
        top_p,
        top_k,
        notes,
    }
}

#[allow(dead_code)]
pub fn apply_defaults(
    defaults: SamplingDefaults,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
) -> (Option<f64>, Option<f64>, Option<usize>) {
    let resolved = resolve_sampling(defaults, temperature, top_p, top_k);
    (resolved.temperature, resolved.top_p, resolved.top_k)
}
