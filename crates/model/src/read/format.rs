//! The effective producer format retained by reads and their continuations.
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

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
