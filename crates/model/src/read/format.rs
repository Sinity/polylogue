//! The effective producer format kept by reads and their continuations.
stored_names! {
    name, parse;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Format {
        Terminal => "terminal",
        Markdown => "markdown",
        Html => "html",
        Json => "json",
        Jsonl => "jsonl",
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
