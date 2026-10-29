//! Renditions of exact source text. A variant is separate from assertions;
//! its parts retain explicit alignment to the original children.

use crate::{Author, BlobHash, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VariantKind {
    Translation,
    Transliteration,
    Simplification,
    Summary,
}

stored_names!(VariantKind {
    Translation => "translation",
    Transliteration => "transliteration",
    Simplification => "simplification",
    Summary => "summary",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VariantRelation {
    Translates,
    Transliterates,
    Simplifies,
    Summarizes,
    Omits,
    Expands,
    Reorders,
}

stored_names!(VariantRelation {
    Translates => "translates",
    Transliterates => "transliterates",
    Simplifies => "simplifies",
    Summarizes => "summarizes",
    Omits => "omits",
    Expands => "expands",
    Reorders => "reorders",
});

/// One revision. Parts stream independently so a rendition of a large
/// session never requires a session-sized collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub id: String,
    pub root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    pub source: String,
    pub source_hash: BlobHash,
    pub kind: VariantKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_lang: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_lang: Option<String>,
    pub author: Author,
    pub via: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    pub machine: String,
    pub created_at: Timestamp,
    pub commit_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantPart {
    pub variant: String,
    pub n: u64,
    pub of: String,
    pub of_hash: BlobHash,
    pub relation: VariantRelation,
    pub body: String,
}

/// Exact authored records carried by preparation and confirmation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantEffect {
    pub variants: Vec<Variant>,
    pub variant_parts: Vec<VariantPart>,
}

/// A rendition submitted through the authored write path. Provenance is
/// computed by the accepting surface before constructing this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewVariant {
    pub base: Option<String>,
    pub source: String,
    pub source_hash: BlobHash,
    pub kind: VariantKind,
    pub from_lang: Option<String>,
    pub to_lang: Option<String>,
    pub author: Author,
    pub via: String,
    pub method: Option<String>,
    pub parts: Vec<NewVariantPart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewVariantPart {
    pub of: String,
    pub of_hash: BlobHash,
    pub relation: VariantRelation,
    pub body: String,
}
