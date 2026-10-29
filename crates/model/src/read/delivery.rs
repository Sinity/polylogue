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
        source: super::Source,
        reason: PresentationReason,
        parts: Vec<ReadPart>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PresentationReason {
    Omission {
        view: super::View,
        target: super::TargetKind,
        accepted_views: Vec<super::View>,
    },
    QueryResult,
}

impl ReadRecord {
    pub fn canonical(&self) -> Option<&Record> {
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
