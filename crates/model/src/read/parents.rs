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
    #[error("copied message {message} from {parent} is not available for {child}")]
    MissingCopiedMessage {
        child: SessionId,
        parent: SessionId,
        message: MessageId,
    },
    #[error("parent ID {parent:?} of {child} is invalid")]
    InvalidParent { child: SessionId, parent: String },
    #[error("fork from {parent} to {child} has no available branch point ({branch_point:?})")]
    MissingBranchPoint {
        child: SessionId,
        parent: SessionId,
        branch_point: Option<String>,
    },
    #[error(
        "{child} records incompatible forks from {first} at {first_branch_point} and {second} at {second_branch_point}"
    )]
    ConflictingParents {
        child: SessionId,
        first: SessionId,
        first_branch_point: String,
        second: SessionId,
        second_branch_point: String,
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

/// What a read could not reach of a session's parents; no issue means every
/// parent and copied message was available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Parents {
    pub issues: Vec<ParentIssue>,
}
impl Parents {
    pub fn complete(&self) -> bool {
        self.issues.is_empty()
    }
}
