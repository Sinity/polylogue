//! Requested import settlement is narrower than Session/query completeness.
use crate::BlobHash;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRowBasis {
    pub input: String,
    pub original_hash: BlobHash,
    pub chain: i64,
    pub revision: u64,
    pub length: u64,
    pub parser_version: u32,
    pub interpretation: BlobHash,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceRowState {
    Pending {
        obligations: Vec<String>,
    },
    Settled,
    Unavailable {
        usable_result: bool,
        notes: Vec<String>,
        record_errors: u64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRows {
    pub scope: SourceRowScope,
    pub basis: SourceRowBasis,
    #[serde(flatten)]
    pub state: SourceRowState,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceRowScope {
    #[serde(rename = "imported_source_rows")]
    ImportedSourceRows,
}

/// One retained undecodable record, independent of query/candidate paths.
/// Original input identity is context-qualified; offsets never mean Messages.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordFailureId {
    pub input: String,
    pub chain: i64,
    pub revision: u64,
    pub record_start: u64,
}
