//! Signal runtime facts use the common evaluator and delivery schemas. The
//! daemon computes these values; they are not an authored rule or a bypass of
//! the final host policy. Bundles retain them as historical state.
use crate::{BlobHash, Timestamp, authored::ExecutionEligibility, user};
use serde::{Deserialize, Serialize};
use shared_signals::notifications::{Notice, NotificationReceipt, NotificationState};
use shared_signals::{ActiveRule, Delivery, DeliveryReceipt, Evaluator, HostPolicy};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// The accepting machine's configured identity, within this archive.
    pub owner: String,
    pub evaluator: Evaluator,
    pub rules: Vec<ActiveRule>,
    /// Present only for the standalone host deliverer. In primary mode the
    /// evaluator's reports go to Sinex, whose deliverer decides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<HostPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notifications: Option<NotificationState>,
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
    /// The host policy's handoff record: the firing, its decision and the
    /// attention it was decided under.
    pub delivery: Delivery,
    pub notice: Notice,
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
    HostDecision {
        #[serde(deserialize_with = "buffered_delivery")]
        delivery: Box<Delivery>,
    },
    Desktop {
        receipt: NotificationReceipt,
    },
    Outlet {
        receipt: DeliveryReceipt,
    },
}
impl Event {
    pub fn delivery_id(&self) -> &str {
        match self {
            Self::HostDecision { delivery } => delivery.id(),
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
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredOutbox {
    pub id: String,
    pub at: Timestamp,
    pub outbox: Outbox,
    pub events: Vec<EventRecord>,
}

// Internally tagged authored writes buffer JSON before this field is decoded.
// Normalize serde_json's arbitrary-precision Number representation once at
// that boundary so nested shared f64 fields retain their ordinary wire form.
impl<'de> Deserialize<'de> for Commit {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            #[serde(default)]
            prior: Option<BlobHash>,
            checkpoint: Checkpoint,
            outbox: Vec<Outbox>,
            events: Vec<Event>,
        }
        let value = serde_json::Value::deserialize(deserializer)?;
        let fields: Fields = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            prior: fields.prior,
            checkpoint: fields.checkpoint,
            outbox: fields.outbox,
            events: fields.events,
        })
    }
}

// An internally tagged event receives serde's buffered content. With JSON
// arbitrary precision enabled, normalize numbers before decoding shared f64s.
fn buffered_delivery<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Box<Delivery>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}
