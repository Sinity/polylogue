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

/// The caller supplies this question only after acknowledged query custody.
/// Per-execution choices and private caller context stay in replay authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFraming {
    pub question: QuestionId,
}
impl ReadFraming {
    pub fn source(self) -> super::Source {
        super::Source::record(format!("query:{}", self.question))
    }
}
