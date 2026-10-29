//! Routing status for a physical component whose logical root is unavailable.
//! Receiver provenance and evidence do not enter its stable component key.
use super::{VERSION as REDUCER_VERSION, membership};
use crate::SessionId;
use serde::{Deserialize, Serialize};

pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    LogicalWork,
}

/// Each case names one registered raw Product path, checked by the byte owner.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalField {
    SessionId,
    MessageSession,
    EventSession,
    BackgroundSession,
    LinkParent,
    LinkChild,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct PhysicalEvidence {
    pub event: String,
    pub field: PhysicalField,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRef {
    pub session: SessionId,
    pub evidence: Vec<PhysicalEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RootRef {
    pub session: SessionId,
    pub reducer_version: u32,
}

/// Public provenance of the reduction. The owner certificate stays in the
/// authenticated publication envelope, rather than this domain record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MembershipBasis {
    pub membership_version: u32,
    pub selection_version: super::SelectionVersion,
    pub completed_position: i64,
    pub scope: String,
    pub membership_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComponentState {
    Unavailable {
        reasons: Vec<String>,
    },
    Resolved {
        superseding_root_refs: Vec<RootRef>,
        basis: MembershipBasis,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentStatus {
    pub version: u32,
    pub reducer_version: u32,
    pub position: i64,
    pub component_kind: ComponentKind,
    /// Canonical UTF-8 order of the typed session's string representation.
    pub physical: Vec<PhysicalRef>,
    pub links: Vec<membership::ConfirmedLink>,
    pub status: ComponentState,
}

impl ComponentStatus {
    pub fn sessions(&self) -> impl Iterator<Item = &SessionId> {
        self.physical.iter().map(|physical| &physical.session)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != VERSION || self.reducer_version != REDUCER_VERSION {
            return Err("unsupported component reducer version");
        }
        if self.position < 0 || self.physical.is_empty() {
            return Err("component needs physical evidence at a completed position");
        }
        super::snapshot::validate_physical(&self.physical)?;
        match &self.status {
            ComponentState::Unavailable { reasons } => {
                if reasons.is_empty() || reasons.iter().any(String::is_empty) {
                    return Err("unavailable component needs a reason");
                }
            }
            ComponentState::Resolved {
                superseding_root_refs,
                basis,
            } => {
                if superseding_root_refs.is_empty()
                    || superseding_root_refs
                        .iter()
                        .any(|root| root.reducer_version != REDUCER_VERSION)
                    || superseding_root_refs
                        .windows(2)
                        .any(|pair| pair[0].session.to_string() >= pair[1].session.to_string())
                {
                    return Err("resolved component needs distinct supported root references");
                }
                basis.validate(self.position)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
