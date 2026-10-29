//! Transfer units for the same selected, projected reading. Directory and file
//! destinations belong to the client and never cross this boundary.
use super::{ReadDocument, ReadFooter, ReadHeader, ReadOptions, ReadPart};
use crate::bundle::{Counts, ReadSnapshot, Record};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadPage {
    pub snapshot: ReadSnapshot,
    pub records: Vec<Record>,
    pub document: ReadDocument,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadBegin {
    pub snapshot: ReadSnapshot,
    pub header: ReadHeader,
    pub options: ReadOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadRecord {
    pub record: Record,
    pub parts: Vec<ReadPart>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadComplete {
    pub footer: ReadFooter,
    pub counts: Counts,
}
