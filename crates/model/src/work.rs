use crate::{SessionId, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkNodeKind {
    Goal,
    Attempt,
    Verification,
    Artifact,
    Decision,
}

/// A node of the work graph: what sessions were trying to do, what they
/// tried, how it was checked, what it produced and what was decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkNode {
    pub id: i64,
    pub kind: WorkNodeKind,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Timestamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkEdgeKind {
    /// An attempt toward a goal.
    Pursues,
    /// A verification of an attempt or artifact.
    Checks,
    Produces,
    Supersedes,
    DependsOn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkEdge {
    pub from: i64,
    pub to: i64,
    pub kind: WorkEdgeKind,
}
