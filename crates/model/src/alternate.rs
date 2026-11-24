//! Alternates of source text. An alternate is separate from annotations; its
//! parts keep explicit alignment to the original children.

use crate::{Author, BlobHash, Timestamp};
use serde::{Deserialize, Serialize};

stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AlternateKind {
        Translation => "translation",
        Transliteration => "transliteration",
        Simplification => "simplification",
        Summary => "summary",
    }
}

stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AlternateRelation {
        Translates => "translates",
        Transliterates => "transliterates",
        Simplifies => "simplifies",
        Summarizes => "summarizes",
        Omits => "omits",
        Expands => "expands",
        Reorders => "reorders",
    }
}

/// One revision. Parts stream independently so an alternate of a large
/// session never requires a session-sized collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Alternate {
    pub id: String,
    pub root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    pub source: String,
    pub source_hash: BlobHash,
    pub kind: AlternateKind,
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
pub struct AlternatePart {
    pub alternate: String,
    pub n: u64,
    pub of: String,
    pub of_hash: BlobHash,
    pub relation: AlternateRelation,
    pub body: String,
}

/// Exact authored records carried by preparation and confirmation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AlternateEffect {
    pub alternates: Vec<Alternate>,
    pub alternate_parts: Vec<AlternatePart>,
}

/// An alternate submitted through the authored write path. Provenance is
/// computed by the accepting surface before constructing this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewAlternate {
    pub base: Option<String>,
    pub source: String,
    pub source_hash: BlobHash,
    pub kind: AlternateKind,
    pub from_lang: Option<String>,
    pub to_lang: Option<String>,
    pub method: Option<String>,
    pub parts: Vec<NewAlternatePart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewAlternatePart {
    pub of: String,
    pub of_hash: BlobHash,
    pub relation: AlternateRelation,
    pub body: String,
}

#[cfg(test)]
mod tests;
