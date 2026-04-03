use crate::engine::runtime::SamplingDefaults;

pub fn apply_defaults(
    defaults: SamplingDefaults,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
) -> (Option<f64>, Option<f64>, Option<usize>) {
    (
        temperature.or(defaults.temperature),
        top_p.or(defaults.top_p),
        top_k.or(defaults.top_k),
    )
}
