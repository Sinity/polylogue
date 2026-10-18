//! Embeddings, opt-in with a cost preview: Voyage, or a local model through
//! an OpenAI-compatible endpoint. Vectors are keyed by model and NFC text
//! hash and are never destroyed by a reindex.

pub enum Provider {
    Voyage { model: String },
    Local { endpoint: String, model: String },
}

pub fn preview_cost(_provider: &Provider, _texts: usize, _tokens: u64) -> Option<f64> {
    // Architecture 7: cost preview before any purchase.
    todo!()
}
