//! Whether one stored file has been fully parsed into rows. This says nothing
//! about whether a session or query that uses those rows is complete.
use crate::BlobHash;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// What the rows were parsed from: one revision of one stored file, and the
/// parser that read it.
pub struct ParsedFrom {
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_hash: Option<BlobHash>,
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
        pending_work: Vec<String>,
    },
    Complete,
    Unavailable {
        usable_result: bool,
        notes: Vec<String>,
        record_errors: u64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRows {
    pub scope: SourceRowScope,
    pub parsed_from: ParsedFrom,
    #[serde(flatten)]
    pub state: SourceRowState,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceRowScope {
    #[serde(rename = "imported_source_rows")]
    ImportedSourceRows,
    #[serde(rename = "observed_source_rows")]
    ObservedSourceRows,
}

/// One stored record that could not be decoded. The offset is a byte position
/// in the file, never a message number.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordFailureId {
    pub input: String,
    pub chain: i64,
    pub revision: u64,
    pub record_start: u64,
}
