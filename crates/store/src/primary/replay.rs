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
    pub policy_epoch: i64,
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
        basis: ReplayBasisStatus,
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

/// A staged native delivery can be acknowledged without retiring producer work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeDeliveryCursor {
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
    #[serde(default)]
    pub staged_delivery: Option<NativeDeliveryCursor>,
    /// Only an authenticated PublicContext write-purpose reply supplies this.
    /// The snapshot raw-input stamp above remains the request basis.
    #[serde(default)]
    pub publication: Option<RawInputProof>,
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
    Reset {
        scope: String,
        policy_epoch: i64,
    },
    Delivered {
        delivery: NativeDeliveryCursor,
    },
    Coverage {
        coverage: RawInputCoverage,
    },
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

pub use polylogue_model::native::{
    ConfirmedOrigin, NativeFactContext, NativeKind, NativeSource, ProjectionIdentity,
};

impl ReplayState {
    /// The signed full basis remains at P while actual native completion can
    /// extend its applied prefix to Q. No copied stamp is relabeled as Q.
    pub fn certified_position(&self) -> Option<i64> {
        let applied = self.applied_position?;
        let coverage = self.coverage.as_ref()?;
        (self.availability == ProjectionAvailability::Certified
            && self.erasure_fence.is_none()
            && applied >= coverage.projection.position
            && self.policy_epoch == Some(coverage.policy_epoch)
            && self.scope.as_deref() == Some(coverage.projection.scope.as_str()))
        .then_some(applied)
    }
}

impl RawInputProof {
    /// Identity comparison for already authenticated proof envelopes. It does
    /// not validate a stamp or establish its purpose or admission authority.
    pub fn same_basis(&self, other: &Self) -> bool {
        self.position == other.position
            && self.policy_epoch == other.policy_epoch
            && self.scope == other.scope
    }
}
