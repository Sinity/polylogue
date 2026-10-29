//! Pure layout and streaming encoders over canonical projected read records.
//! The service selects records and prepares relationships; this crate never
//! reads sources, probes a terminal, or changes archived text.
mod encode;
pub mod html;
mod layout;
pub mod measures;
pub mod pager;
pub mod results;
pub mod schema;
pub mod terminal;
pub mod web;
pub use encode::{ARCHIVE_NOTICE, EncodeOptions, Encoder, encode_document};
pub use layout::{layout_record, visit_record, visit_source_block};
pub use polylogue_model::read;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Terminal,
    Markdown,
    Html,
    Json,
    Jsonl,
}
impl Format {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::Markdown => "markdown",
            Self::Html => "html",
            Self::Json => "json",
            Self::Jsonl => "jsonl",
        }
    }
    pub const NAMES: &[&str] = &["terminal", "markdown", "html", "json", "jsonl"];
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "terminal" => Self::Terminal,
            "markdown" => Self::Markdown,
            "html" => Self::Html,
            "json" => Self::Json,
            "jsonl" => Self::Jsonl,
            _ => return None,
        })
    }
}
/// One argument as a POSIX shell reads it. This only formats stored arguments.
pub fn shell_quote(word: &str) -> String {
    if !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@%+,".contains(c))
    {
        word.to_owned()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}
#[cfg(test)]
mod tests;

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
