//! Native owner metadata. Payload-carried copies are claims, not authenticated owner evidence.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ProjectionIdentity {
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
