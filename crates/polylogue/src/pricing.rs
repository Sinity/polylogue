//! One price catalog. API-equivalent dollars are kept apart from
//! subscription credit; an unknown price is unknown, never zero.

use polylogue_model::Usage;

pub struct Price {
    pub model: String,
    pub per_million_input: f64,
    pub per_million_cached_input: f64,
    pub per_million_cache_write: f64,
    pub per_million_output: f64,
}

pub fn api_cost(_model: &str, _usage: &Usage) -> Option<f64> {
    // Architecture 1 (measure): price lookup ported from the current rewrite.
    todo!()
}
