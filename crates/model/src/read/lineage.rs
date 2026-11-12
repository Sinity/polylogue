//! Structural ancestry comes only from explicit provider links.
use crate::SessionId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SessionLineage {
    pub root: Option<SessionId>,
    pub depth: Option<u64>,
    pub root_pending: bool,
    pub issue: Option<LineageIssue>,
}

#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema, thiserror::Error,
)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LineageIssue {
    #[error("parent {parent} of {child} is not available")]
    MissingParent { child: SessionId, parent: SessionId },
    #[error("{child} records incompatible parent sessions")]
    ConflictingParents {
        child: SessionId,
        parents: Vec<SessionId>,
    },
    #[error("the lineage loops back to {session}")]
    Cycle { session: SessionId },
}
