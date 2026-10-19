use serde::{Deserialize, Serialize};

/// What a work-graph node is (spec model.md, The work graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkNodeKind {
    /// Something a session declared it would do: a claimed Bead, a todo or
    /// plan item.
    Goal,
    /// One session pursuing one goal.
    Attempt,
    /// A check command and its outcome.
    Verification,
    /// A commit or pull request.
    Artifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkEdgeKind {
    /// An attempt toward a goal.
    Pursues,
    /// A verification run while an attempt was open.
    Checks,
    /// An artifact made while an attempt was open.
    Produces,
}

impl WorkNodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkNodeKind::Goal => "goal",
            WorkNodeKind::Attempt => "attempt",
            WorkNodeKind::Verification => "verification",
            WorkNodeKind::Artifact => "artifact",
        }
    }
}

impl WorkEdgeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkEdgeKind::Pursues => "pursues",
            WorkEdgeKind::Checks => "checks",
            WorkEdgeKind::Produces => "produces",
        }
    }
}
