//! Attachment claims transported separately from a message's source value.
//!
//! A candidate describes claimed bytes and their original sources. Receiving
//! archives establish availability through their own validated import members.

use crate::BlobHash;
use serde::{Deserialize, Serialize};

/// One attachment slot in the complete source message. Sparse projections keep
/// the original block ordinal instead of renumbering the blocks they carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentBinding {
    pub source_version: BlobHash,
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
    pub input: String,
    pub chain: i64,
    pub source_version: String,
    pub members: Vec<String>,
    pub names: Vec<String>,
    pub media_types: Vec<String>,
    pub declared_sizes: Vec<u64>,
}
