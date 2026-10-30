//! Bounded delivery coordinates under one authenticated prepared read scope.
//! Store resolves metadata ordinals; these are never native archive identities.
use crate::BlobHash;
pub use query_language::identity::QuestionId;
use serde::{Deserialize, Serialize};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ReadRecordKind {
    Session,
    LinkIssue,
    Message,
    Event,
    Link,
    Document,
    Authored,
    Variant,
    VariantPart,
    Blob,
    TextFragment,
    Presentation,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct ReadRecordLocator {
    pub kind: ReadRecordKind,
    /// Position in the selected row's prepared metadata view, before layout.
    /// This is independent of a message's original physical-session order.
    pub ordinal: u64,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct SelectionLocator {
    pub row: u64,
    pub record: ReadRecordLocator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadPosition {
    pub selection: SelectionLocator,
    pub item: u64,
    /// Unicode scalar position in the complete C1-safe encoded ReadItem JSON.
    pub character: u64,
    /// Digest of that complete encoded item, including its original Source.
    pub item_hash: BlobHash,
}

/// Fragments cite an acknowledged query or the actual exact-object run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadFraming {
    Query { question: QuestionId },
    Object { run: crate::RunId },
}
impl ReadFraming {
    pub fn source(&self) -> super::Source {
        super::Source::record(match self {
            Self::Query { question } => format!("query:{question}"),
            Self::Object { run } => format!("run:{run}"),
        })
    }
}

/// Direction is part of the authenticated content-navigation intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReadDirection {
    /// Resume at the first unconsumed character, as captured by the prior page.
    After,
    /// Read content strictly before this page start after validating its digest.
    Before,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadSeek {
    pub direction: ReadDirection,
    pub anchor: ReadPosition,
}

const READ_POSITION_SENTINEL: &str = "read-position-v1";

#[derive(Debug, thiserror::Error)]
pub enum ReadSeekCodecError {
    #[error("read continuation has an unsupported position encoding")]
    Sentinel,
    #[error("read continuation position needs exactly one value, found {found}")]
    Arity { found: usize },
    #[error("read continuation position is invalid: {0}")]
    Value(#[from] serde_json::Error),
}
impl ReadSeek {
    /// Encode bounded direction and anchor. Authentication and root/run bindings
    /// remain with the existing common cursor seal and its caller.
    pub fn to_order_position(
        &self,
    ) -> Result<query_language::cursor::OrderPosition, ReadSeekCodecError> {
        Ok(query_language::cursor::OrderPosition {
            row_identity: READ_POSITION_SENTINEL.into(),
            values: vec![serde_json::to_value(self)?],
        })
    }
    pub fn from_order_position(
        position: &query_language::cursor::OrderPosition,
    ) -> Result<Self, ReadSeekCodecError> {
        if position.row_identity != READ_POSITION_SENTINEL {
            return Err(ReadSeekCodecError::Sentinel);
        }
        let [value] = position.values.as_slice() else {
            return Err(ReadSeekCodecError::Arity {
                found: position.values.len(),
            });
        };
        Ok(serde_json::from_value(value.clone())?)
    }
}
