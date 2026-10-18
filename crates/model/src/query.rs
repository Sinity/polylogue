//! Parsed session queries. `polylogue-query` builds them from text.
use crate::SessionKind;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextTerm {
    Word(String),
    Phrase(String),
    Prefix(String),
}
#[derive(Clone, Debug, Default)]
pub struct SessionQuery {
    pub origin: Option<String>,
    pub kind: Option<SessionKind>,
    pub all_kinds: bool,
    pub id: Option<String>,
    pub text: Vec<TextTerm>,
    pub limit: Option<usize>,
    pub recent: bool,
}

#[derive(Clone, Debug)]
pub struct ToolQuery {
    pub calls_only: bool,
    pub session: String,
    pub text: Option<String>,
    pub limit: usize,
}
