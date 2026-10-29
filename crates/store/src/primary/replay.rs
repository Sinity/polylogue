//! Owner evidence for the confirmed cache. These declarations do not turn a
//! stream checkpoint or a raw-input certificate into full projection coverage.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FullProjectionComplete {
    pub stamp: String,
    pub position: i64,
    pub scope: String,
    pub catalog: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RawInputProof {
    pub stamp: String,
    pub position: i64,
    pub scope: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RawInputCoverage {
    Complete {
        proof: RawInputProof,
    },
    Incomplete {
        reason: String,
    },
    #[default]
    NotRequested,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProjectionCoverage {
    pub policy_epoch: i64,
    pub projection: FullProjectionComplete,
    pub raw_inputs: RawInputCoverage,
}

/// A native delivered marker is delivery eligibility, not local application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeliveredCompletion {
    pub stream_instance: String,
    pub stream_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReplayCursor {
    pub position: i64,
    pub policy_epoch: i64,
    pub scope: String,
    pub cursor: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProjectionAvailability {
    #[default]
    Unknown,
    /// Requires applied full coverage and the complete local projector catalog.
    /// Raw-input eligibility remains independent.
    Certified,
    Unverified,
    /// Optional owner identities do not assert an epoch transition.
    Invalidated {
        policy_epoch: Option<i64>,
        scope: Option<String>,
    },
}

/// Persisted with bounded page facts in the same SQLite transaction. A fresh
/// cache distinguishes unknown application from a certified empty position0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReplayState {
    pub applied_position: Option<i64>,
    pub policy_epoch: Option<i64>,
    pub scope: Option<String>,
    pub cursor: Option<ReplayCursor>,
    pub coverage: Option<ProjectionCoverage>,
    pub completion: Option<DeliveredCompletion>,
    /// Actual owner erasure Q. Only a real completed prefix at or beyond Q can
    /// clear this fence; it never advances the applied position itself.
    pub erasure_fence: Option<i64>,
    pub availability: ProjectionAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ReplayContinuation {
    Next {
        cursor: String,
    },
    Complete {
        projection: FullProjectionComplete,
        raw_inputs: RawInputCoverage,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReplayPage {
    pub position: i64,
    pub policy_epoch: i64,
    pub scope: String,
    pub continuation: ReplayContinuation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ReplayBasisStatus {
    Invalidated {
        policy_epoch: Option<i64>,
        scope: Option<String>,
    },
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "control", rename_all = "snake_case")]
pub enum ReplayControl {
    Page {
        page: ReplayPage,
    },
    BasisStatus {
        status: ReplayBasisStatus,
    },
    Completion {
        position: i64,
        delivered: DeliveredCompletion,
    },
    ErasedThrough {
        position: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProjectionIdentity {
    pub policy_epoch: i64,
    pub payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeKind {
    pub name: String,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeSource {
    pub name: String,
    pub version: u32,
}

/// Only owner fetch metadata supplies origin. Product payloads, names and
/// acquisition-route declarations cannot authenticate it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConfirmedOrigin {
    pub source: Option<NativeSource>,
    pub source_fingerprint: Option<String>,
    pub producer_family: Option<String>,
    pub producer_identity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeFactContext {
    pub kind: NativeKind,
    pub kind_definition_hash: String,
    pub projection: ProjectionIdentity,
    pub origin: ConfirmedOrigin,
}
