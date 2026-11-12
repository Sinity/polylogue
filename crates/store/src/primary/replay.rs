//! The local index's place in Sinex's confirmed stream. The durable consumer
//! is read from the applied position (0 for a rebuild); there is no other way
//! into the index, except an owner-certified current projection at a new
//! stream's origin.
use serde::{Deserialize, Serialize};

/// The last stream message staged in the index, so its acknowledgement can be
/// sent to the durable consumer only after the facts it carried are saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeDeliveryCursor {
    pub stream_instance: String,
    pub stream_sequence: u64,
}

/// Saved in the index in the same transaction as the facts it describes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReplayState {
    /// The last complete commit position applied; none before the first.
    pub applied_position: Option<i64>,
    /// The read scope Sinex provisioned the consumer under.
    pub scope: Option<String>,
    /// Sinex's commit position when the stream was last read.
    #[serde(default)]
    pub head: Option<i64>,
    pub staged_delivery: Option<NativeDeliveryCursor>,
    /// A signed current projection, saved only with its visible application.
    #[serde(default)]
    pub basis: Option<ProjectionBasis>,
    /// The next bounded snapshot page; its predecessors are durably staged.
    #[serde(default)]
    pub projection: Option<ProjectionReplay>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProjectionBasis {
    pub stamp: String,
    pub position: i64,
    pub scope: String,
    pub catalog: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProjectionReplay {
    pub origin: i64,
    pub stream_instance: String,
    pub position: Option<i64>,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "control", rename_all = "snake_case")]
pub enum ReplayControl {
    /// The consumer's read scope differs from the one the index was built
    /// under: the index is cleared and read again from position 0.
    Reset { scope: String },
    /// Replace the reconstructible cache before staging a current projection.
    ProjectionStart {
        scope: String,
        origin: i64,
        stream_instance: String,
        reason: Option<String>,
    },
    /// This page and the owner's next cursor share the staging transaction.
    ProjectionPage { position: i64, cursor: String },
    /// Every snapshot page is staged; install it and its certificate together.
    ProjectionComplete { basis: ProjectionBasis },
    /// Sinex's commit position when this page was read.
    Head { position: i64 },
    /// Every change committed at `position` has been delivered.
    Completion { position: i64 },
    /// The stream messages up to this cursor are staged in the index.
    Delivered { delivery: NativeDeliveryCursor },
}

pub use polylogue_model::native::{
    ConfirmedOrigin, NativeFactContext, NativeKind, NativeSource, ProjectionIdentity,
};
