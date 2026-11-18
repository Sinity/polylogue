//! Recorded links from a shown item to the preserved source bytes.
use crate::{BlobHash, Reference, Route};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TraceCard {
    pub item: Reference,
    pub chain: Vec<ChainStep>,
    pub missing: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChainStep {
    Record {
        reference: Reference,
    },
    InputRecord {
        input: String,
        chain: i64,
        start: u64,
        end: u64,
        route: Route,
        shown: bool,
    },
    StoredBytes {
        blob: BlobHash,
        size: u64,
        offset: u64,
        length: u64,
        blob_offset: u64,
    },
    ReaderVersion {
        reading_version: String,
        status: String,
        parsed_upto: u64,
    },
    Unavailable {
        reason: String,
    },
}
