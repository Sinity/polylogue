//! Source field evidence stays separate from canonical messages and their hashes.
use crate::{BlobHash, MessageId};
use serde::{Deserialize, Serialize};

/// A parser claim about a particular native field, before receiving admission
/// supplies and validates the owning input and chain. Imported claims confer no
/// native authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HookFieldClaim {
    pub block: u64,
    pub start: u64,
    pub end: u64,
    pub pointer: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeHookField {
    pub input: String,
    pub chain: i64,
    pub start: u64,
    pub end: u64,
    pub pointer: String,
}

/// Missing observed evidence means unknown injection attribution. Declared
/// historical claims remain available without becoming local observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InjectionEvidence {
    pub message: MessageId,
    pub source_version: BlobHash,
    pub block: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<NativeHookField>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub declared: Vec<NativeHookField>,
}

/// A producer's post-success emission claim. The invocation receipt is distinct
/// from a native transcript reference; this does not attribute any Message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HookContextEmission {
    pub invocation: String,
    pub session: crate::SessionId,
    pub hook_event: String,
    pub deliveries: Vec<String>,
}
