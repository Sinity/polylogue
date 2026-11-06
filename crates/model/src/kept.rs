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
pub fn evolve_body(text: &str) -> Result<String, String> {
    let mut body: serde_json::Value =
        serde_json::from_str(text).map_err(|error| error.to_string())?;
    let object = body
        .as_object_mut()
        .ok_or_else(|| "a kept body must be an object".to_owned())?;
    if let Some(metadata) = object.get("output_metadata") {
        let metadata: OutputMetadata =
            serde_json::from_value(metadata.clone()).map_err(|error| error.to_string())?;
        metadata.validate()?;
        return Ok(text.to_owned());
    }
    object.insert(
        "output_metadata".into(),
        serde_json::to_value(OutputMetadata::default()).expect("kept metadata serializes"),
    );
    serde_json::to_string(&body).map_err(|error| error.to_string())
}
