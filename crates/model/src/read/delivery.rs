//! Transfer units for the same selected, projected reading. Directory and file
//! destinations belong to the client and never cross this boundary.
use super::{ReadDocument, ReadFooter, ReadHeader, ReadOptions, ReadPart};
use crate::bundle::{Counts, ReadSnapshot, Record};
use serde::{Deserialize, Serialize};

/// Paging and streaming share selection and view; only transfer changes.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ReadDelivery {
    #[default]
    Page,
    Stream,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadPage {
    pub snapshot: ReadPageSnapshot,
    pub records: Vec<Record>,
    pub document: ReadDocument,
}

/// The fixed reading identity for a budgeted page. Query text, notes and other
/// unbounded metadata belong in budgeted items or the full stream header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadPageSnapshot {
    pub query: String,
    pub ran_at: crate::Timestamp,
    pub read_at: crate::ChangePosition,
    pub caught_up: bool,
    pub currency: crate::currency::IndexCurrency,
}
impl From<&ReadSnapshot> for ReadPageSnapshot {
    fn from(snapshot: &ReadSnapshot) -> Self {
        Self {
            query: snapshot.query.clone(),
            ran_at: snapshot.ran_at,
            read_at: snapshot.read_at.clone(),
            caught_up: snapshot.caught_up,
            currency: snapshot.currency.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadBegin {
    pub snapshot: ReadSnapshot,
    pub header: ReadHeader,
    pub options: ReadOptions,
}

/// Portable source records and presentation-only evidence are explicit kinds.
/// An omission never stands in for an unavailable canonical record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadRecord {
    Canonical {
        record: Box<Record>,
        parts: Vec<ReadPart>,
    },
    Presentation {
        parts: Vec<ReadPart>,
    },
}

impl ReadRecord {
    pub fn record(&self) -> Option<&Record> {
        match self {
            Self::Canonical { record, .. } => Some(record),
            Self::Presentation { .. } => None,
        }
    }

    pub fn parts(&self) -> &[ReadPart] {
        match self {
            Self::Canonical { parts, .. } | Self::Presentation { parts, .. } => parts,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadComplete {
    pub footer: ReadFooter,
    pub counts: Counts,
}

/// A bounded slice of a declared content-addressed blob. The receiver validates
/// contiguous offsets, declared size and the whole content hash before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadBlobChunk {
    pub hash: crate::BlobHash,
    pub offset: u64,
    /// Standard padded base64, independent of the transport framing.
    pub base64: String,
}
