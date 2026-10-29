//! Signal runtime facts use the common evaluator and delivery schemas. The
//! daemon computes these values; they are not an authored rule or a bypass of
//! the final host policy. Bundles retain them as historical state.
use crate::{BlobHash, Timestamp, authored::ExecutionEligibility, user};
use serde::{Deserialize, Serialize};
use shared_signals::notifications::{Notice, NotificationReceipt, NotificationState};
use shared_signals::{
    ActiveRule, Delivery, DeliveryOutcome, DeliveryReceipt, Evaluator, FiredIntent, HostPolicy,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// The accepting machine's configured identity, within this archive.
    pub owner: String,
    pub evaluator: Evaluator,
    pub rules: Vec<ActiveRule>,
    /// Present only for the standalone host deliverer. Primary uses Forward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<HostPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notifications: Option<NotificationState>,
    pub outlet_attempts: BTreeMap<String, DeliveryOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Outlet {
    pub name: String,
    /// Identity of the loaded deployment definition, not a rule-supplied path.
    pub configuration: BlobHash,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub delivery: Delivery,
    pub notice: Notice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<FiredIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outlet: Option<Outlet>,
    pub execution: ExecutionEligibility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Outbox {
    pub record: user::Delivery,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    HostDecision { delivery: Delivery },
    Desktop { receipt: NotificationReceipt },
    Outlet { receipt: DeliveryReceipt },
}
impl Event {
    pub fn delivery_id(&self) -> &str {
        match self {
            Self::HostDecision { delivery } => &delivery.id,
            Self::Desktop { receipt } => &receipt.delivery_id,
            Self::Outlet { receipt } => &receipt.delivery_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventRecord {
    /// Original order within this delivery, retained through historical restore.
    pub position: u64,
    pub request: String,
    pub ordinal: u64,
    pub at: Timestamp,
    pub event: Event,
}

/// Checkpoint, immutable outbox entries and returned receipts commit together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Commit {
    /// Digest of the canonical checkpoint resolved before this update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior: Option<BlobHash>,
    pub checkpoint: Checkpoint,
    pub outbox: Vec<Outbox>,
    pub events: Vec<Event>,
}

impl Checkpoint {
    pub fn identity(&self) -> Result<BlobHash, serde_json::Error> {
        Ok(BlobHash(
            *blake3::hash(&serde_json::to_vec(self)?).as_bytes(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StoredOutbox {
    pub id: String,
    pub at: Timestamp,
    pub outbox: Outbox,
    pub events: Vec<EventRecord>,
}
