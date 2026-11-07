//! Lineage issues retain their recorded references without choosing a parent.
use crate::{MessageId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema, thiserror::Error,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParentIssue {
    #[error("the parents loop back to {session}")]
    Cycle { session: SessionId },
    #[error("parent {parent} of {child} is not available")]
    MissingParent { child: SessionId, parent: SessionId },
    #[error("parent ID {parent:?} of {child} is invalid")]
    InvalidParent { child: SessionId, parent: String },
    #[error("fork from {parent} to {child} has no available branch anchor ({anchor:?})")]
    MissingAnchor {
        child: SessionId,
        parent: SessionId,
        anchor: Option<String>,
    },
    #[error(
        "{child} records incompatible forks from {first} at {first_anchor} and {second} at {second_anchor}"
    )]
    ConflictingParents {
        child: SessionId,
        first: SessionId,
        first_anchor: String,
        second: SessionId,
        second_anchor: String,
    },
    #[error("copied messages {first} and {second} have no recorded relative order in {child}")]
    UnknownOrder {
        child: SessionId,
        first: MessageId,
        second: MessageId,
    },
    #[error("copied messages {first} and {second} have conflicting recorded order in {child}")]
    ConflictingOrder {
        child: SessionId,
        first: MessageId,
        second: MessageId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Parents {
    pub complete: bool,
    pub issues: Vec<ParentIssue>,
}
