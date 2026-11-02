//! Immutable authored effects prepared against one canonical view. Resolution
//! belongs to the store; transport, replay and bundles carry these same values.
use crate::user::{
    BackfillEvent, BackfillJob, ChatAction, ChatEvent, Delivery, UserOutcome, UserRequest,
};
use crate::{Author, DeliveryOutcome, Timestamp};
use serde::{Deserialize, Serialize};

/// Whether an action may run here. Actions restored from another archive keep
/// their state but never run: the receiving host computes this, never a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEligibility {
    Live,
    Historical,
}
stored_names!(ExecutionEligibility { Live => "live", Historical => "historical" });

/// Computed from the local bundle ingress and authenticated importing channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImportAttribution {
    /// The receiving machine, computed by the import boundary.
    pub machine: String,
    pub input: String,
    pub chain: i64,
    pub actor: Author,
    pub at: Timestamp,
}

/// Locally observed ingress is distinct from claims carried by a bundle.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImportHistory {
    pub observed: Vec<ImportAttribution>,
    pub declared: Vec<ImportAttribution>,
}

impl ImportHistory {
    /// An observation's identity excludes wall time: the first admitted tuple
    /// survives a later clock rollback. Sender history remains declared.
    pub fn merge(&mut self, claims: &[ImportAttribution], current: &ImportAttribution) {
        if !self.observed.iter().any(|known| {
            known.input == current.input
                && known.chain == current.chain
                && known.machine == current.machine
                && known.actor == current.actor
        }) {
            self.observed.push(current.clone());
        }
        for claim in claims {
            if !self.declared.contains(claim) {
                self.declared.push(claim.clone());
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedRequest {
    pub request: UserRequest,
    pub outcome: UserOutcome,
    pub effect: AuthoredEffect,
}

/// A durable authored operation waiting for confirmation keeps its authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum PendingAuthored {
    Live {
        prepared: PreparedRequest,
    },
    Restore {
        prepared: PreparedRequest,
    },
    RestoreAssertion {
        fact: crate::assertion::AssertionFact,
    },
}
impl PendingAuthored {
    pub fn prepared(&self) -> Option<&PreparedRequest> {
        match self {
            Self::Live { prepared } | Self::Restore { prepared, .. } => Some(prepared),
            Self::RestoreAssertion { .. } => None,
        }
    }
}

/// A request may insert several linked records; empty effects still preserve
/// an entity-deduplicated request's identity and original outcome.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AuthoredEffect {
    pub assertions: Vec<crate::assertion::AssertionFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signal_checkpoints: Vec<crate::signal::Checkpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signal_events: Vec<crate::signal::EventRecord>,
    pub variants: crate::VariantEffect,
    pub kept_results: Vec<KeptResult>,
    pub deliveries: Vec<DeliveryChange>,
    pub chats: Vec<ChatChange>,
    pub backfills: Vec<BackfillChange>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct KeptResult {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub query: String,
    pub ran_at: Timestamp,
    pub read_at: String,
    pub rows_json: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeliveryEvent {
    pub request: String,
    pub what: DeliveryOutcome,
    pub at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeliveryRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<crate::signal::Metadata>,
    pub id: String,
    pub prepared: Delivery,
    pub at: Timestamp,
    pub events: Vec<DeliveryEvent>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeliveryChange {
    pub delivery: DeliveryRecord,
    pub added_events: Vec<DeliveryEvent>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChatChange {
    pub action: ChatAction,
    pub added_events: Vec<ChatEvent>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BackfillChange {
    pub job: BackfillJob,
    pub added_events: Vec<BackfillEvent>,
}
