//! Attachment claims transported separately from a message's source value.
//!
//! A candidate describes claimed bytes and their original sources. Receiving
//! archives establish availability through their own validated import members.

use crate::BlobHash;
use serde::{Deserialize, Serialize};

/// Original import namespace qualified by its actual source input and chain.
/// This is portable provenance, not receiving-archive membership authority.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct AttachmentNamespace {
    pub input: String,
    pub chain: i64,
    pub import_id: String,
}
impl AttachmentNamespace {
    fn valid(&self) -> bool {
        !self.input.is_empty() && self.chain > 0 && !self.import_id.is_empty()
    }
}

/// One attachment slot in the complete source message. Sparse projections keep
/// the original block ordinal instead of renumbering the blocks they carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentBinding {
    pub source_version: BlobHash,
    /// Retained even when candidate bytes are unknown. Several namespaces may
    /// describe the same source slot without merging their claims.
    pub namespace: AttachmentNamespace,
    pub block: u64,
    pub attachment: u64,
    pub source_id: String,
    /// No candidates means unknown bytes. Multiple candidates remain ambiguous.
    pub candidates: Vec<AttachmentCandidate>,
}

/// A complete content hash can span several stored extents. Original source
/// claims remain distinct even when candidate hashes happen to agree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentCandidate {
    pub hash: BlobHash,
    pub size: u64,
    pub sources: Vec<AttachmentSourceClaim>,
}

/// Historical provenance supplied by the exporting archive. These fields never
/// grant access to a receiving archive's inputs, chains, or import namespaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentSourceClaim {
    /// Unknown historical provenance stays unknown; ingress never substitutes
    /// the receiving namespace for an original source claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<AttachmentNamespace>,
    pub input: String,
    pub chain: i64,
    pub source_version: String,
    pub members: Vec<String>,
    pub names: Vec<String>,
    pub media_types: Vec<String>,
    pub declared_sizes: Vec<u64>,
}

pub(crate) fn validate_record(
    record: &crate::bundle::MessageRecord,
) -> Result<(), crate::bundle::BundleError> {
    use crate::{Block, bundle::BundleError};
    let mut slots = std::collections::BTreeSet::new();
    for binding in &record.bindings {
        if binding.source_version != record.source_version
            || binding.source_id.is_empty()
            || !binding.namespace.valid()
            || !slots.insert((binding.block, binding.attachment, &binding.namespace))
        {
            return Err(BundleError::InvalidAttachmentBinding);
        }
        let position = record
            .block_ordinals
            .binary_search(&binding.block)
            .map_err(|_| BundleError::InvalidAttachmentBinding)?;
        let index = usize::try_from(binding.attachment)
            .map_err(|_| BundleError::InvalidAttachmentBinding)?;
        let attachment = match &record.message.blocks[position] {
            Block::Image(attachment) | Block::File(attachment) if index == 0 => Some(attachment),
            Block::ToolResult { attachments, .. } => attachments.get(index),
            _ => None,
        }
        .ok_or(BundleError::InvalidAttachmentBinding)?;
        if attachment.blob.is_some()
            || attachment.source_id.as_deref() != Some(binding.source_id.as_str())
        {
            return Err(BundleError::InvalidAttachmentBinding);
        }
        for candidate in &binding.candidates {
            // Hash and size are typed claims. Only ingress with the actual bytes
            // can establish their truth; declared source sizes stay untouched.
            for source in &candidate.sources {
                if source.input.is_empty()
                    || source.chain <= 0
                    || source.source_version.is_empty()
                    || source
                        .namespace
                        .as_ref()
                        .is_some_and(|namespace| !namespace.valid())
                {
                    return Err(BundleError::InvalidAttachmentBinding);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
