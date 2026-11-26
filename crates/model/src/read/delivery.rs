//! Transfer units for the same selected, projected reading. Directory and file
//! destinations belong to the client and never cross this boundary.
use super::{Format, ReadDocument, ReadPart};
use crate::bundle::Record;
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
    /// A stream a directory bundle is written from: it also carries the
    /// bytes of the present attachments its records name.
    Bundle,
}
impl ReadDelivery {
    /// Records go out as they are read, with no pages.
    pub fn streams(self) -> bool {
        self != Self::Page
    }
}

/// One page of a reading: its document (header, items and footer) with the
/// output choice and the compact snapshot the page was read at. A stream
/// sends the same header in its `begin` event, the same items in `record`
/// events, and the footer in its `complete` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadPage {
    /// Effective output choice from the original acknowledged execution.
    pub format: Format,
    pub snapshot: ReadPageSnapshot,
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
    pub freshness: crate::freshness::ReadFreshness,
}
impl From<&crate::bundle::ReadSnapshot> for ReadPageSnapshot {
    fn from(snapshot: &crate::bundle::ReadSnapshot) -> Self {
        Self {
            query: snapshot.query.clone(),
            ran_at: snapshot.ran_at,
            read_at: snapshot.read_at.clone(),
            freshness: snapshot.freshness.clone(),
        }
    }
}

/// Portable source records and presentation-only evidence are explicit kinds.
/// An omission never stands in for an unavailable canonical record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadRecord {
    Original {
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
            Self::Original { record, .. } => Some(record),
            Self::Presentation { .. } => None,
        }
    }

    pub fn parts(&self) -> &[ReadPart] {
        match self {
            Self::Original { parts, .. } | Self::Presentation { parts, .. } => parts,
        }
    }
}

/// A bounded slice of a declared content-addressed blob. The receiver validates
/// contiguous offsets, declared size and the whole content hash before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadBlobChunk {
    pub hash: crate::BlobHash,
    pub offset: u64,
    /// Standard padded base64, independent of the transport framing.
    pub base64: String,
}
