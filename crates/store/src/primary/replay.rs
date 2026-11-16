//! The local index's place in Sinex's confirmed stream. The durable consumer
//! is read from the applied position (0 for a rebuild); there is no other way
//! into the index, except a complete snapshot of Sinex's current state at a
//! new stream's origin.
use serde::{Deserialize, Serialize};

/// The last stream message staged in the index, so its acknowledgment can be
/// sent to the durable consumer only after the events it carried are saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StreamCursor {
    pub stream_instance: String,
    pub stream_sequence: u64,
}

/// The completion marker saved with its fully applied local position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletedDelivery {
    pub stream_instance: String,
    pub stream_sequence: u64,
    pub fingerprint: String,
}

/// Saved in the index in the same transaction as the events it describes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReplayState {
    /// The last complete commit position applied; none before the first.
    pub applied_position: Option<i64>,
    /// The read scope Sinex provisioned the consumer under.
    pub scope: Option<String>,
    /// Sinex's commit position when the stream was last read.
    #[serde(default)]
    pub latest_position: Option<i64>,
    pub staged_delivery: Option<StreamCursor>,
    /// A completed prefix, distinct from a newer partially staged page.
    #[serde(default)]
    pub completed_delivery: Option<CompletedDelivery>,
    /// The applied snapshot, saved only with its visible application; the
    /// first consumer request after it names where following starts.
    #[serde(default)]
    pub snapshot: Option<AppliedSnapshot>,
    /// The next bounded snapshot page; the pages before it are staged.
    #[serde(default)]
    pub snapshot_progress: Option<SnapshotProgress>,
}

/// A complete snapshot: its position and scope, the stream instance it was
/// read for, and the last stream message before its read (none when the
/// stream was empty). Following starts after that message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AppliedSnapshot {
    pub position: i64,
    pub scope: String,
    pub stream_instance: String,
    pub last: Option<LastStreamMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LastStreamMessage {
    pub sequence: u64,
    pub fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SnapshotProgress {
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
    /// Replace the reconstructible cache before staging a current snapshot.
    SnapshotStart {
        scope: String,
        origin: i64,
        stream_instance: String,
        reason: Option<String>,
    },
    /// This page and Sinex's next cursor share the staging transaction.
    SnapshotPage { position: i64, cursor: String },
    /// Every snapshot page is staged; install it and its completion together.
    SnapshotComplete { snapshot: AppliedSnapshot },
    /// Sinex's commit position when this page was read.
    LatestPosition { position: i64 },
    /// Every change committed at `position` has been delivered.
    Completion {
        position: i64,
        delivery: Option<CompletedDelivery>,
    },
    /// The stream messages up to this cursor are staged in the index.
    Delivered { delivery: StreamCursor },
}

pub use polylogue_model::sinex_event::{
    ConfirmedOrigin, PayloadIdentity, SinexEventContext, SinexKind, SinexSource,
};
