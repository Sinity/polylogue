//! Logical declarations retained by the kept-result producer.
use query_language::catalog::OutputSchema;
use serde::{Deserialize, Serialize};

/// The declared shape of a kept result's rows and, when it has them, its matches.
/// A part without a declaration reads with the types of its physical columns.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputMetadata {
    pub rows: Option<OutputSchema>,
    pub matches: Option<OutputSchema>,
}
impl OutputMetadata {
    pub fn captured(rows: OutputSchema, matches: Option<OutputSchema>) -> Self {
        Self {
            rows: Some(rows),
            matches,
        }
    }
}

#[derive(Deserialize)]
struct BodyMetadata {
    #[serde(default)]
    output_metadata: OutputMetadata,
}

/// Read a kept body's output declaration; a body without one declares nothing.
pub fn body_metadata(text: &str) -> Result<OutputMetadata, String> {
    if !text.trim_start().starts_with('{') {
        return Err("a kept body must be an object".into());
    }
    let mut parser = serde_json::Deserializer::from_str(text);
    parser.disable_recursion_limit();
    let body = BodyMetadata::deserialize(&mut parser).map_err(|error| error.to_string())?;
    parser.end().map_err(|error| error.to_string())?;
    Ok(body.output_metadata)
}
