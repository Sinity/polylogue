//! Logical declarations retained by the kept-result producer.
use query_language::catalog::OutputSchema;
use serde::{Deserialize, Serialize};

pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchemaProof {
    #[default]
    Unproven,
    Captured {
        schema: OutputSchema,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        unproven_fields: Vec<String>,
    },
}
impl SchemaProof {
    pub fn schema(&self) -> Option<&OutputSchema> {
        match self {
            Self::Captured { schema, .. } => Some(schema),
            Self::Unproven => None,
        }
    }
    pub fn unproven_field(&self, name: &str) -> bool {
        match self {
            Self::Unproven => true,
            Self::Captured {
                unproven_fields, ..
            } => unproven_fields.iter().any(|field| field == name),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct OutputMetadata {
    pub version: u32,
    pub rows: SchemaProof,
    pub matches: SchemaProof,
}
impl Default for OutputMetadata {
    fn default() -> Self {
        Self {
            version: FORMAT_VERSION,
            rows: SchemaProof::Unproven,
            matches: SchemaProof::Unproven,
        }
    }
}
impl OutputMetadata {
    pub fn captured(rows: OutputSchema, matches: Option<OutputSchema>) -> Self {
        Self {
            version: FORMAT_VERSION,
            rows: SchemaProof::Captured {
                schema: rows,
                unproven_fields: vec![],
            },
            matches: matches.map_or(SchemaProof::Unproven, |schema| SchemaProof::Captured {
                schema,
                unproven_fields: vec![],
            }),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported kept output metadata version {}",
                self.version
            ));
        }
        Ok(())
    }
}

/// Version 1 did not retain a logical declaration. Its query text cannot
/// establish the catalog or saved definitions used by the original producer.
#[derive(Deserialize)]
struct BodyMetadata<'a> {
    #[serde(borrow, default, deserialize_with = "raw_metadata")]
    output_metadata: Option<&'a serde_json::value::RawValue>,
}
fn raw_metadata<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<&'de serde_json::value::RawValue>, D::Error> {
    <&serde_json::value::RawValue>::deserialize(deserializer).map(Some)
}

pub fn evolve_body(text: &str) -> Result<String, String> {
    if !text.trim_start().starts_with('{') {
        return Err("a kept body must be an object".into());
    }
    let mut parser = serde_json::Deserializer::from_str(text);
    parser.disable_recursion_limit();
    let body = BodyMetadata::deserialize(&mut parser).map_err(|error| error.to_string())?;
    parser.end().map_err(|error| error.to_string())?;
    if let Some(metadata) = body.output_metadata {
        let metadata: OutputMetadata =
            serde_json::from_str(metadata.get()).map_err(|error| error.to_string())?;
        metadata.validate()?;
        return Ok(text.to_owned());
    }
    let end = text.rfind('}').expect("validated object");
    let start = text.find('{').expect("validated object");
    let metadata =
        serde_json::to_string(&OutputMetadata::default()).expect("kept metadata serializes");
    let mut evolved = String::with_capacity(text.len() + metadata.len() + 20);
    evolved.push_str(&text[..end]);
    if !text[start + 1..end].trim().is_empty() {
        evolved.push(',');
    }
    evolved.push_str("\"output_metadata\":");
    evolved.push_str(&metadata);
    evolved.push_str(&text[end..]);
    Ok(evolved)
}
