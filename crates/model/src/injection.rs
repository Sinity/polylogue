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

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
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

impl NativeHookField {
    pub fn validate(&self) -> Result<(), crate::bundle::BundleError> {
        if self.input.is_empty()
            || self.chain < 1
            || self.start >= self.end
            || !valid_pointer(&self.pointer)
        {
            return Err(crate::bundle::BundleError::InvalidInjectionEvidence);
        }
        Ok(())
    }
}

impl InjectionEvidence {
    /// Imported observations remain historical declarations at the receiver.
    pub fn into_declared(mut self) -> Self {
        if let Some(observed) = self.observed.take() {
            self.declared.push(observed);
        }
        self.declared.sort();
        self.declared.dedup();
        self
    }

    pub fn validate_for(
        &self,
        message: &MessageId,
        projection: &crate::bundle::MessageProjection,
    ) -> Result<(), crate::bundle::BundleError> {
        if &self.message != message
            || self.source_version != projection.source_version
            || projection
                .block_ordinals
                .binary_search(&self.block)
                .is_err()
        {
            return Err(crate::bundle::BundleError::InvalidInjectionEvidence);
        }
        for claim in self.observed.iter().chain(&self.declared) {
            claim.validate()?;
        }
        Ok(())
    }
}

fn valid_pointer(pointer: &str) -> bool {
    if !pointer.starts_with('/') {
        return false;
    }
    let mut bytes = pointer.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'~' && !matches!(bytes.next(), Some(b'0' | b'1')) {
            return false;
        }
    }
    true
}
