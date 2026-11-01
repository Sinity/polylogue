//! Attachment claims transported separately from a message's source value.
//!
//! A candidate describes claimed bytes and their original sources. Receiving
//! archives establish availability through their own validated import members.

use crate::BlobHash;
use serde::{Deserialize, Serialize};

/// Original import namespace qualified by its actual source input and chain.
/// This is portable provenance, not receiving-archive membership authority.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentBinding {
    pub source_version: BlobHash,
    /// Retained even when candidate bytes are unknown. Several namespaces may
    /// describe the same source slot without merging their claims.
    pub namespace: AttachmentNamespace,
    /// Actual context of this export. A receiver retains it as a declaration,
    /// independently of its own validated receiving members.
    pub exported_from: AttachmentNamespace,
    /// Supplied prior export declarations, preserved in their original order.
    /// Export never manufactures or appends a context history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prior_exports: Vec<AttachmentNamespace>,
    pub block: u64,
    pub attachment: u64,
    pub source_id: String,
    /// No candidates means unknown bytes. Multiple candidates remain ambiguous.
    pub candidates: Vec<AttachmentCandidate>,
}

/// A complete content hash can span several stored extents. Original source
/// claims remain distinct even when candidate hashes happen to agree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentCandidate {
    pub hash: BlobHash,
    pub size: u64,
    pub sources: Vec<AttachmentSourceClaim>,
}

/// Historical provenance supplied by the exporting archive. These fields never
/// grant access to a receiving archive's inputs, chains, or import namespaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
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

impl AttachmentBinding {
    /// Checks claim structure. Only ingress with actual bytes and a receiving
    /// member can establish candidate truth or local availability.
    pub fn validate(&self) -> Result<(), crate::bundle::BundleError> {
        use crate::bundle::BundleError;
        if self.source_id.is_empty()
            || !self.namespace.valid()
            || !self.exported_from.valid()
            || self
                .prior_exports
                .iter()
                .any(|namespace| !namespace.valid())
        {
            return Err(BundleError::InvalidAttachmentBinding);
        }
        for candidate in &self.candidates {
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
        Ok(())
    }
}

/// One attachment of a source message that names bytes it does not carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentSlot {
    /// Original block ordinal in the complete source message.
    pub block: u64,
    pub attachment: u64,
    pub source_id: String,
}

/// The unresolved attachments of a message's blocks. `ordinals[i]` is the
/// original ordinal of `blocks[i]`.
pub fn attachment_slots(blocks: &[crate::Block], ordinals: &[u64]) -> Vec<AttachmentSlot> {
    use crate::Block;
    let mut slots = Vec::new();
    for (block, ordinal) in blocks.iter().zip(ordinals) {
        let attachments: &[crate::Attachment] = match block {
            Block::Image(attachment) | Block::File(attachment) => std::slice::from_ref(attachment),
            Block::ToolResult { attachments, .. } => attachments,
            _ => &[],
        };
        for (index, attachment) in attachments.iter().enumerate() {
            if attachment.blob.is_none()
                && let Some(source_id) = attachment.source_id.as_deref().filter(|id| !id.is_empty())
            {
                slots.push(AttachmentSlot {
                    block: *ordinal,
                    attachment: index as u64,
                    source_id: source_id.to_owned(),
                });
            }
        }
    }
    slots
}

pub(crate) fn validate_record(
    record: &crate::bundle::MessageRecord,
) -> Result<(), crate::bundle::BundleError> {
    use crate::{Block, bundle::BundleError};
    let mut claims = std::collections::HashSet::new();
    for binding in &record.bindings {
        binding.validate()?;
        if binding.source_version != record.source_version || !claims.insert(binding) {
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
    }
    Ok(())
}

#[cfg(test)]
mod tests;
