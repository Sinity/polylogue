//! Reading for models: a session in pages sized to a token budget, with a
//! plain cursor to continue from. Size is a parameter, never a refusal.

pub struct Page {
    pub text: String,
    /// Where the next page starts; `None` at the end.
    pub next: Option<String>,
}

pub fn page(
    _messages: &[polylogue_model::Message],
    _budget_tokens: usize,
    _cursor: Option<&str>,
) -> Page {
    // Architecture 7: pages with a plain cursor and per-block masking.
    todo!()
}
