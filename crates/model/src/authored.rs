//! Immutable authored effects prepared against one canonical view. Resolution
//! belongs to the store; transport, replay and bundles carry these same values.
use crate::user::{
    BackfillEvent, BackfillJob, ChatAction, ChatEvent, Delivery, UserOutcome, UserRequest,
};
use crate::{Author, DeliveryOutcome, Note, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEligibility {
    Live,
    Historical,
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PreparedRequest {
    pub request: UserRequest,
    pub outcome: UserOutcome,
    pub effect: AuthoredEffect,
}

/// A request may insert several linked records; empty effects still preserve
/// an entity-deduplicated request's identity and original outcome.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AuthoredEffect {
    pub variants: crate::VariantEffect,
    pub notes: Vec<NoteRevision>,
    pub withdrawals: Vec<Withdrawal>,
    pub settings: Vec<Setting>,
    pub kept_results: Vec<KeptResult>,
    pub deliveries: Vec<DeliveryChange>,
    pub chats: Vec<ChatChange>,
    pub backfills: Vec<BackfillChange>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NoteRevision {
    pub note: Note,
    pub machine: String,
    pub revises: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Withdrawal {
    pub id: String,
    pub first_note: String,
    pub machine: String,
    pub author: Author,
    pub created_at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub withdraws: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Setting {
    pub id: String,
    pub machine: String,
    pub key: String,
    pub value: String,
    pub author: Author,
    pub set_at: Timestamp,
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
