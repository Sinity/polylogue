use serde::{Deserialize, Serialize};

/// What a work-graph node is.
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

stored_names!(WorkNodeKind {
    Goal => "goal",
    Attempt => "attempt",
    Verification => "verification",
    Artifact => "artifact",
});

stored_names!(WorkEdgeKind {
    Pursues => "pursues",
    Checks => "checks",
    Produces => "produces",
});
