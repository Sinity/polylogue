//! Structural ancestry comes only from explicit provider links.
use crate::SessionId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionLineage {
    pub root: Option<SessionId>,
    pub depth: Option<u64>,
    pub root_pending: bool,
    pub issue: Option<LineageIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LineageIssue {
    MissingParent {
        child: SessionId,
        parent: SessionId,
    },
    ConflictingParents {
        child: SessionId,
        parents: Vec<SessionId>,
    },
    Cycle {
        session: SessionId,
    },
}
