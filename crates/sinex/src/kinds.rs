//! Canonical Polylogue kind declarations and exact native record references.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sinex_model::MaterialId;
pub use sinex_model::event_types::work::{GitIdentity, Objective, TerminalIdentity};

/// A record in a material: another version of a message, or a tool call's
/// output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecordRef {
    pub material: MaterialId,
    /// Byte position of the record in that material.
    pub anchor: u64,
}

/// Kind references are independent of the configured producer identity.
pub(crate) fn kind_reference(name: &str) -> crate::Result<sinex_model::KindRef> {
    static REFERENCES: std::sync::OnceLock<Vec<sinex_model::KindRef>> = std::sync::OnceLock::new();
    REFERENCES
        .get_or_init(|| {
            declarations("polylogue")
                .into_iter()
                .map(|declaration| declaration.kind)
                .collect()
        })
        .iter()
        .find(|kind| kind.name == name)
        .cloned()
        .ok_or_else(|| crate::Error::Invalid(format!("undeclared Polylogue event type {name}")))
}

pub fn definition_hash(producer: &str) -> String {
    let hashes: Vec<_> = declarations(producer)
        .iter()
        .map(|declaration| declaration.definition_hash())
        .collect();
    blake3::hash(&serde_json::to_vec(&hashes).expect("declaration hashes serialize"))
        .to_hex()
        .to_string()
}

pub fn declarations(producer: &str) -> Vec<sinex_model::KindDeclaration> {
    let mut declarations = crate::products::declarations(producer);
    for declaration in &mut declarations {
        declaration.declaring_producer = producer.to_string();
        declaration.normalization = sinex_model::kinds::Normalization::Json;
    }
    declarations
}

#[cfg(test)]
mod tests;
