//! What Sinex recorded for a confirmed event. Copies carried in a payload are stated
//! values, not values Sinex vouches for.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PayloadIdentity {
    pub payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SinexKind {
    pub name: String,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SinexSource {
    pub name: String,
    pub version: u32,
}

/// Only Sinex fetch metadata supplies origin. Product payloads, names and
/// acquisition-route declarations cannot authenticate it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConfirmedOrigin {
    pub source: Option<SinexSource>,
    pub source_fingerprint: Option<String>,
    pub producer_family: Option<String>,
    pub producer_identity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SinexLocation {
    pub material: String,
    pub anchor: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SinexEventContext {
    pub kind: SinexKind,
    pub kind_definition_hash: String,
    pub payload_identity: PayloadIdentity,
    /// Immutable Sinex record identity, separate from projected payload bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<SinexLocation>,
    pub origin: ConfirmedOrigin,
}
