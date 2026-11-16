//! The effective producer format kept by reads and their continuations.
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
stored_names!(Format, name, parse {
    Terminal => "terminal",
    Markdown => "markdown",
    Html => "html",
    Json => "json",
    Jsonl => "jsonl",
});

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
