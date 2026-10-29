//! Portable readings over the canonical domain records. Provider bytes stay
//! in their archive; imported source claims never become local byte extents.

use crate::{Author, BlobHash, ChangePosition, Document, Event, Link, Message, Session, Timestamp};
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "polylogue-bundle";
pub const VERSION: u32 = 1;

/// The primary manifest stays small. A directory package names a streamed
/// inventory, rather than collecting all members in the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub query: String,
    pub query_text: String,
    pub ran_at: Timestamp,
    pub read_at: ChangePosition,
    pub caught_up: bool,
    pub counts: Counts,
    pub flags: Flags,
    pub enabled_variants: Vec<String>,
    pub author: Author,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inventory: Option<FileDescriptor>,
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub sessions: u64,
    pub messages: u64,
    pub documents: u64,
    pub events: u64,
    pub links: u64,
    pub authored: u64,
    pub variants: u64,
    pub variant_parts: u64,
    pub blobs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Flags {
    pub format: Format,
    pub view: crate::read::View,
    pub with_blobs: bool,
    pub inherited: bool,
    pub assertions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Json,
    Jsonl,
    Markdown,
    Html,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileDescriptor {
    pub path: String,
    pub hash: BlobHash,
    pub size: u64,
}

/// The primary manifest-bearing file and inventory itself are not inventory
/// members: including either would make the pinned hashes recursive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InventoryRecord {
    pub kind: MemberKind,
    #[serde(rename = "ref", default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    pub file: FileDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MemberKind {
    Records,
    Rendered,
    Blob,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub session: Session,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MessageRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    /// Reading position within the physical session, independent of commit
    /// order and of the receiving bundle's actual local byte extents.
    pub order: u64,
    /// Identity of the complete source value, unchanged by view projection.
    pub source_version: BlobHash,
    /// Original source cardinality; absent positions are unavailable or omitted.
    pub block_count: u64,
    /// Original source positions aligned to `message.blocks`.
    pub block_ordinals: Vec<u64>,
    pub message: Message,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DocumentRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub document: Document,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LinkRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub link: Link,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BlobRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub file: FileDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub variant: crate::Variant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantPartRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub part: crate::VariantPart,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub prepared: crate::authored::PreparedRequest,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<crate::authored::ImportAttribution>,
}

/// A stream starts with exactly one manifest. All remaining values are
/// independently staged and validated before the receiving archive writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "record", content = "data", rename_all = "snake_case")]
pub enum Record {
    Manifest(Manifest),
    Session(SessionRecord),
    Message(Box<MessageRecord>),
    Event(InputEventRecord),
    Link(LinkRecord),
    Document(DocumentRecord),
    Authored(Box<AuthoredRecord>),
    Variant(VariantRecord),
    VariantPart(VariantPartRecord),
    Blob(BlobRecord),
}

impl Record {
    pub fn reference(&self) -> Option<&str> {
        Some(match self {
            Self::Manifest(_) => return None,
            Self::Session(r) => &r.reference,
            Self::Message(r) => &r.reference,
            Self::Event(r) => &r.reference,
            Self::Link(r) => &r.reference,
            Self::Document(r) => &r.reference,
            Self::Authored(r) => &r.reference,
            Self::Variant(r) => &r.reference,
            Self::VariantPart(r) => &r.reference,
            Self::Blob(r) => &r.reference,
        })
    }

    pub fn commit(&self) -> Option<u64> {
        Some(match self {
            Self::Manifest(_) | Self::Blob(_) => return None,
            Self::Session(r) => r.commit,
            Self::Message(r) => r.commit,
            Self::Event(r) => r.commit,
            Self::Link(r) => r.commit,
            Self::Document(r) => r.commit,
            Self::Authored(r) => r.commit,
            Self::Variant(r) => r.commit,
            Self::VariantPart(r) => r.commit,
        })
    }

    pub fn validate(&self) -> Result<(), BundleError> {
        match self {
            Self::Manifest(r) => r.validate(),
            Self::Session(r) => r.validate(),
            Self::Message(r) => r.validate(),
            Self::Event(r) => r.validate(),
            Self::Link(r) => r.validate(),
            Self::Document(r) => r.validate(),
            Self::Authored(r) => {
                if r.reference != format!("request:{}", r.prepared.request.id) {
                    return Err(BundleError::ReferenceMismatch);
                }
                Ok(())
            }
            Self::Variant(r) => {
                if r.reference != format!("variant:{}@{}", r.variant.root, r.variant.id) {
                    return Err(BundleError::ReferenceMismatch);
                }
                Ok(())
            }
            Self::VariantPart(r) => {
                // Matching root/revision and part ordinals is a staged package
                // relation: a part need not follow its header in the stream.
                if r.part.variant.is_empty() || r.reference.is_empty() {
                    return Err(BundleError::ReferenceMismatch);
                }
                Ok(())
            }
            Self::Blob(r) => r.validate(),
        }
    }
}

/// Projection metadata transported through parse and primary projection. It
/// describes selected content without changing the canonical source identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MessageProjection {
    pub source_version: BlobHash,
    pub block_count: u64,
    pub block_ordinals: Vec<u64>,
}

/// One original source claim carried through a bundle. The receiving
/// archive keeps its actual local bundle input and chain separately.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct EventSourceClaim {
    pub input: String,
    pub source: String,
    pub source_chain: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end: Option<u64>,
}

/// A stable event identity and all original claims for the selected
/// source reading. Equal imports union claims rather than dropping one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportedEventIdentity {
    pub id: String,
    pub sources: Vec<EventSourceClaim>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputEventRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub identity: ImportedEventIdentity,
    pub event: Event,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BundleError {
    #[error("not a current Polylogue bundle (format {0:?})")]
    WrongFormat(String),
    #[error(
        "Polylogue bundle version {found} is unsupported; this build reads only version {VERSION}"
    )]
    UnsupportedVersion { found: u32 },
    #[error("bundle query reference must be query: followed by its full BLAKE3 hash")]
    InvalidQuery,
    #[error("bundle read position is invalid")]
    InvalidPosition,
    #[error("bundle member path is not a canonical relative path")]
    InvalidPath,
    #[error("bundle record reference does not match its canonical identity")]
    ReferenceMismatch,
    #[error("message projection has invalid original block positions")]
    InvalidBlockSelection,
    #[error("complete message content does not match its source version")]
    SourceVersionMismatch,
    #[error("partial message observations conflict at the same source version")]
    ProjectionConflict,
    #[error("cannot encode canonical message content: {0}")]
    CanonicalMessage(String),
    #[error("invalid source time evidence: {0}")]
    Time(#[from] crate::TimeEvidenceError),
    #[error("invalid event identity or original source claim")]
    InvalidEventIdentity,
    #[error("event reference does not name its imported identity")]
    EventReferenceMismatch,
    #[error("event after a message has no session")]
    InvalidEventAssociation,
}

impl EventSourceClaim {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.input.is_empty()
            || self.source.is_empty()
            || self.source_chain < 1
            || self.source_start.is_some() != self.source_end.is_some()
            || self
                .source_start
                .zip(self.source_end)
                .is_some_and(|(a, b)| a > b)
        {
            return Err(BundleError::InvalidEventIdentity);
        }
        Ok(())
    }
}

impl ImportedEventIdentity {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.id.is_empty() || self.sources.is_empty() {
            return Err(BundleError::InvalidEventIdentity);
        }
        self.sources.iter().try_for_each(EventSourceClaim::validate)
    }

    pub fn canonicalize(&mut self) {
        self.sources.sort();
        self.sources.dedup();
    }
}

impl InputEventRecord {
    pub fn validate(&self) -> Result<(), BundleError> {
        self.identity.validate()?;
        self.event.time_evidence().validate(self.event.at)?;
        if self.reference != format!("event:{}", self.identity.id) {
            return Err(BundleError::EventReferenceMismatch);
        }
        if self.event.session.is_none() && self.event.after.is_some() {
            return Err(BundleError::InvalidEventAssociation);
        }
        Ok(())
    }
}

impl Manifest {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.format != FORMAT {
            return Err(BundleError::WrongFormat(self.format.clone()));
        }
        if self.version != VERSION {
            return Err(BundleError::UnsupportedVersion {
                found: self.version,
            });
        }
        if !self.query.strip_prefix("query:").is_some_and(|hash| {
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        }) {
            return Err(BundleError::InvalidQuery);
        }
        if self.read_at.generation.is_empty() || self.read_at.index < 0 || self.read_at.archive < 0
        {
            return Err(BundleError::InvalidPosition);
        }
        if let Some(inventory) = &self.inventory {
            inventory.validate()?;
        }
        Ok(())
    }
}

impl FileDescriptor {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.path.is_empty()
            || self.path.contains(['\\', '\0'])
            || self
                .path
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(BundleError::InvalidPath);
        }
        Ok(())
    }
}

impl SessionRecord {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.reference != format!("session:{}", self.session.id) {
            return Err(BundleError::ReferenceMismatch);
        }
        self.session
            .time_evidence()
            .validate(self.session.started)?;
        Ok(())
    }
}

impl MessageRecord {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.reference != format!("message:{}", self.message.id) {
            return Err(BundleError::ReferenceMismatch);
        }
        self.message.time_evidence().validate(self.message.at)?;
        self.projection().validate(self.message.blocks.len())?;
        if self.is_complete()
            && self
                .message
                .content_hash()
                .map_err(|e| BundleError::CanonicalMessage(e.to_string()))?
                != self.source_version
        {
            return Err(BundleError::SourceVersionMismatch);
        }
        Ok(())
    }

    pub fn projection(&self) -> MessageProjection {
        MessageProjection {
            source_version: self.source_version,
            block_count: self.block_count,
            block_ordinals: self.block_ordinals.clone(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.block_count == self.block_ordinals.len() as u64
    }
}

impl MessageProjection {
    pub fn validate(&self, selected_blocks: usize) -> Result<(), BundleError> {
        if self.block_ordinals.len() != selected_blocks
            || self
                .block_ordinals
                .iter()
                .any(|&index| index >= self.block_count)
            || self
                .block_ordinals
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(BundleError::InvalidBlockSelection);
        }
        Ok(())
    }
}

impl MessageProjection {
    /// Distinct source versions stay distinct. Compatible views of one
    /// version union original positions; an overlap must contain equal facts.
    pub fn merge(
        &self,
        message: &Message,
        other: &Self,
        other_message: &Message,
    ) -> Result<Option<(Message, Self)>, BundleError> {
        self.validate(message.blocks.len())?;
        other.validate(other_message.blocks.len())?;
        if self.source_version != other.source_version {
            return Ok(None);
        }
        let mut header = message.clone();
        header.blocks.clear();
        let mut other_header = other_message.clone();
        other_header.blocks.clear();
        if header != other_header || self.block_count != other.block_count {
            return Err(BundleError::ProjectionConflict);
        }
        let mut blocks = std::collections::BTreeMap::new();
        for (index, block) in self
            .block_ordinals
            .iter()
            .zip(&message.blocks)
            .chain(other.block_ordinals.iter().zip(&other_message.blocks))
        {
            if let Some(previous) = blocks.insert(*index, block.clone())
                && previous != *block
            {
                return Err(BundleError::ProjectionConflict);
            }
        }
        let ordinals = blocks.keys().copied().collect();
        header.blocks = blocks.into_values().collect();
        let projection = Self {
            source_version: self.source_version,
            block_count: self.block_count,
            block_ordinals: ordinals,
        };
        if projection.block_ordinals.len() as u64 == projection.block_count
            && header
                .content_hash()
                .map_err(|e| BundleError::CanonicalMessage(e.to_string()))?
                != projection.source_version
        {
            return Err(BundleError::SourceVersionMismatch);
        }
        Ok(Some((header, projection)))
    }
}

impl DocumentRecord {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.reference != format!("document:{}", self.document.id) {
            return Err(BundleError::ReferenceMismatch);
        }
        self.document
            .time_evidence()
            .validate(self.document.source_time)?;
        Ok(())
    }
}

impl LinkRecord {
    pub fn reference(link: &Link) -> String {
        // Only natural identity participates; later evidence can fill fields
        // without changing the link's reference.
        let key = serde_json::to_vec(&(link.kind, &link.parent, &link.child))
            .expect("link keys serialize");
        format!("link:{}", blake3::hash(&key).to_hex())
    }
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.reference != Self::reference(&self.link) {
            return Err(BundleError::ReferenceMismatch);
        }
        Ok(())
    }
}

impl BlobRecord {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.reference != format!("blob:{}", self.file.hash) {
            return Err(BundleError::ReferenceMismatch);
        }
        self.file.validate()
    }
}

#[cfg(test)]
mod tests;
