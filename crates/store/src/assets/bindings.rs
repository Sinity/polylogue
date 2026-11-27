//! Attachment bindings as each source states them, and receiving members
//! checked when their bytes arrive.
//!
//! Stored records keep byte matches. Reads require the same member observation
//! and pinned source versions; a binding stated by an older reading cannot
//! create one.
use super::{AssetSource, AttachmentState, ObservedMember};
use crate::ConnExt as _;
use crate::row::{Rows as _, row};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    BlobHash, MessageId, attachment_binding::AttachmentBinding, bundle::MessageRecord,
};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    io::Read,
    ops::{ControlFlow, Range},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BindingContext {
    pub message: MessageId,
    /// Complete source message identity, independent of attachment metadata.
    pub source_version: BlobHash,
    /// The receiving source input whose reading contains the message.
    pub input: String,
    pub chain: i64,
    pub import_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingSelection {
    pub context: BindingContext,
    /// Complete declared identity, including original slot and namespace. Equal
    /// candidate bytes never merge different stated bindings or export contexts.
    pub claim: AttachmentBinding,
}

/// Portable identity of actual receiving bytes, separate from original claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReceivingMember {
    pub import_id: String,
    pub input: String,
    pub member: String,
    pub chain: i64,
    /// Pinned source-extent fingerprint, not the complete content hash.
    pub source_version: String,
    pub hash: BlobHash,
    pub size: u64,
}

/// Minted by bounded byte validation. Canonical products may preserve it;
/// portable bundle records carry only historical claims, never this witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidatedMember {
    version: u8,
    member: ReceivingMember,
    // The source observation ID, not a local SQLite rowid.
    observation: String,
}
impl ValidatedMember {
    pub fn identity(&self) -> &ReceivingMember {
        &self.member
    }
    pub fn observation(&self) -> &str {
        &self.observation
    }
    #[cfg(any(test, feature = "fixture"))]
    pub fn for_candidate(self, candidate: u64) -> CandidateMember {
        CandidateMember {
            candidate,
            witness: self,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CandidateMember {
    candidate: u64,
    witness: ValidatedMember,
}
impl CandidateMember {
    pub fn candidate(&self) -> u64 {
        self.candidate
    }
    pub fn byte_match(&self) -> &ValidatedMember {
        &self.witness
    }
}

/// Canonical sidecar fact. Construction validates the message slot before it
/// reaches the writer; projection never decodes or hashes a large message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentBindingRecord {
    context: BindingContext,
    claim: AttachmentBinding,
    members: Vec<CandidateMember>,
}
impl AttachmentBindingRecord {
    pub fn new(
        context: BindingContext,
        record: &MessageRecord,
        binding: usize,
        members: Vec<CandidateMember>,
    ) -> Result<Self> {
        record
            .validate()
            .map_err(|_| crate::StoreError::stored_data("invalid attachment binding record"))?;
        if context.message != record.message.id || context.source_version != record.source_version {
            return Err(crate::StoreError::internal(
                "binding context differs from its message source",
            ));
        }
        let claim = record
            .bindings
            .get(binding)
            .cloned()
            .ok_or_else(|| crate::StoreError::stored_data("missing attachment binding slot"))?;
        let value = Self {
            context,
            claim,
            members,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn context(&self) -> &BindingContext {
        &self.context
    }
    pub fn claim(&self) -> &AttachmentBinding {
        &self.claim
    }
    pub fn members(&self) -> &[CandidateMember] {
        &self.members
    }
    pub fn selection(&self) -> BindingSelection {
        BindingSelection {
            context: self.context.clone(),
            claim: self.claim.clone(),
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        self.claim
            .validate()
            .map_err(|_| crate::StoreError::stored_data("invalid stated attachment binding"))?;
        if self.context.input.is_empty()
            || self.context.chain <= 0
            || self.context.import_id.is_empty()
            || self.context.source_version != self.claim.source_version
            || self.claim.source_id.is_empty()
        {
            return Err(crate::StoreError::stored_data(
                "invalid attachment binding context",
            ));
        }
        for association in &self.members {
            let byte_match = &association.witness;
            let candidate = usize::try_from(association.candidate)
                .ok()
                .and_then(|index| self.claim.candidates.get(index))
                .ok_or_else(|| {
                    crate::StoreError::internal("attachment byte match has no candidate")
                })?;
            if byte_match.version != 1
                || byte_match.observation.is_empty()
                || byte_match.member.import_id != self.context.import_id
                || byte_match.member.input.is_empty()
                || byte_match.member.member.is_empty()
                || byte_match.member.chain <= 0
                || byte_match.member.source_version.is_empty()
                || byte_match.member.hash != candidate.hash
                || byte_match.member.size != candidate.size
            {
                return Err(crate::StoreError::internal(
                    "attachment byte match differs from its receiving candidate",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingResolution {
    pub selection: BindingSelection,
    pub claim: AttachmentBinding,
    pub state: AttachmentState,
    pub members: Vec<ReceivingMember>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingContexts {
    pub state: AttachmentState,
    /// Stored binding contexts, including distinct conflicting stated bindings.
    pub namespace_count: u64,
    pub contexts: Vec<BindingResolution>,
}

/// The reader borrow pins both membership and extents to this read snapshot.
/// Exposing its natural identity does not expose a byte-match constructor.
pub struct PinnedMember<'reader, 'pool> {
    reader: &'reader Reader<'pool>,
    member: ReceivingMember,
}
impl PinnedMember<'_, '_> {
    pub fn identity(&self) -> &ReceivingMember {
        &self.member
    }
    pub fn read_range(&self, range: Range<u64>) -> Result<Vec<u8>> {
        if range.start > range.end || range.end > self.member.size {
            return Err(crate::StoreError::internal(
                "attachment range is outside its pinned bytes",
            ));
        }
        if self.member.size == 0 {
            let pin = pinned_source(
                (self.reader.conn()?, self.reader.conn()?),
                "source_",
                &self.member.input,
                self.member.chain,
                0,
                None,
                Some(&self.member.source_version),
            )?
            .ok_or_else(|| {
                crate::StoreError::internal("pinned empty attachment source is unavailable")
            })?;
            return super::empty::read(self.reader, &pin);
        }
        let blobs = self.reader.blob_reader()?;
        self.reader.stored_range(
            &self.member.input,
            self.member.chain,
            range.start,
            range.end,
            &blobs,
        )
    }
    pub fn content_hash(&self) -> Result<BlobHash> {
        let pin = pinned_source(
            (self.reader.conn()?, self.reader.conn()?),
            "source_",
            &self.member.input,
            self.member.chain,
            self.member.size,
            None,
            Some(&self.member.source_version),
        )?
        .ok_or_else(|| crate::StoreError::internal("pinned attachment source is unavailable"))?;
        let hash = hash_member(self.reader, &self.member.input, self.member.chain, &pin)?;
        if hash != self.member.hash {
            return Err(crate::StoreError::stored_data(
                "pinned attachment hash differs from its byte match",
            ));
        }
        Ok(hash)
    }
}

pub enum AttachmentPin<'reader, 'pool> {
    Available(PinnedMember<'reader, 'pool>),
    Unavailable,
    Ambiguous,
    Unknown,
}
impl AttachmentPin<'_, '_> {
    pub fn state(&self) -> AttachmentState {
        match self {
            Self::Available(_) => AttachmentState::Available,
            Self::Unavailable => AttachmentState::Unavailable,
            Self::Ambiguous => AttachmentState::Ambiguous,
            Self::Unknown => AttachmentState::Unknown,
        }
    }
}

fn source_prefix(conn: &Connection) -> Result<&'static str> {
    Ok(match crate::source::Source::of(conn)? {
        crate::source::Source::Standalone => "",
        crate::source::Source::Cache => "source_",
    })
}

macro_rules! pinned_extents {
    ($prefix:literal) => {
        concat!("SELECT e.offset,e.length,e.blob,e.blob_offset,b.size
            FROM ", $prefix, "extent e LEFT JOIN ", $prefix, "blob b ON b.hash=e.blob
            WHERE e.input=?1 AND e.chain=?2 AND (e.offset<?3 OR (?3=0 AND e.offset=0)) ORDER BY e.offset")
    };
}

row! {
    struct PinnedExtent {
        offset: i64,
        length: i64,
        blob: Vec<u8>,
        blob_offset: i64,
        size: Option<i64>,
    }
    statements {
        const ARCHIVE = pinned_extents!("");
        const INDEX = pinned_extents!("source_");
    }
}

macro_rules! member_observation {
    ($prefix:literal, $columns:literal) => {
        concat!(
            "SELECT ",
            $columns,
            " FROM ",
            $prefix,
            "observation o
            LEFT JOIN ",
            $prefix,
            "revision r ON r.input=o.input AND r.number=o.revision
            WHERE o.id=?1 AND o.input=?2"
        )
    };
}

row! {
    struct MemberObservation {
        chain: Option<i64>,
        length: Option<i64>,
        details_json: Option<String>,
    }
    statements {
        const ARCHIVE = member_observation!("", "coalesce(o.chain,r.chain) AS chain,coalesce(o.length,r.length) AS length,o.details_json");
        const INDEX = member_observation!("source_", "coalesce(o.chain,r.chain) AS chain,coalesce(o.length,r.length) AS length,o.details_json");
    }
}

row! {
    struct MemberDetails {
        chain: Option<i64>,
        details_json: Option<String>,
    }
    statements {
        const ARCHIVE = member_observation!("", "coalesce(o.chain,r.chain) AS chain,o.details_json");
        const INDEX = member_observation!("source_", "coalesce(o.chain,r.chain) AS chain,o.details_json");
    }
}

row! {
    struct ImportMemberKey {
        import_id: String,
        input: String,
        member: String,
    }
    statements {
        const OF_OBSERVATION = "SELECT import_id,input,member FROM import_member WHERE observation=?1";
    }
}

row! {
    struct MemberObservationId {
        observation: String,
    }
    statements {
        const OF_IMPORT_INPUT = "SELECT observation FROM import_member WHERE input=?1 AND import_id=?2 ORDER BY observation";
    }
}

/// The recorded endpoint is required; a later revision never selects a
/// different chain or silently extends the validated member's bytes.
fn pinned_source(
    (index, conn): (&Connection, &Connection),
    prefix: &str,
    input: &str,
    chain: i64,
    size: u64,
    observation: Option<&str>,
    version: Option<&str>,
) -> Result<Option<AssetSource>> {
    if size == 0 {
        return super::empty::source(index, conn, prefix, input, chain, observation, version);
    }
    let size_sql = i64::try_from(size)
        .map_err(|_| crate::StoreError::internal("attachment exceeds archive address space"))?;
    let exists = conn.scalar::<bool>(&format!("SELECT EXISTS(SELECT 1 FROM {prefix}revision WHERE input=?1 AND chain=?2 AND length=?3)"), params![input,chain,size_sql])?;
    if !exists {
        return Ok(None);
    }
    let extents = if prefix.is_empty() {
        PinnedExtent::ARCHIVE
    } else {
        PinnedExtent::INDEX
    };
    let mut end = 0i64;
    let mut count = 0usize;
    let mut blob = None;
    let mut digest = blake3::Hasher::new();
    digest.update(&chain.to_le_bytes());
    digest.update(&size.to_le_bytes());
    let read = conn.each(&extents, params![input, chain, size_sql], |row| {
        let PinnedExtent {
            offset,
            length,
            blob: hash,
            blob_offset,
            size: stored,
        } = row;
        if offset != end
            || length < 0
            || blob_offset < 0
            || stored.is_none_or(|stored| {
                stored < 0 || blob_offset.checked_add(length).is_none_or(|at| at > stored)
            })
            || offset.checked_add(length).is_none_or(|at| at > size_sql)
        {
            return Ok(ControlFlow::Break(()));
        }
        let hash = BlobHash(
            hash.try_into()
                .map_err(|_| crate::StoreError::stored_data("invalid pinned attachment hash"))?,
        );
        end = offset
            .checked_add(length)
            .ok_or_else(|| crate::StoreError::stored_data("attachment extent overflow"))?;
        count += 1;
        digest.update(&offset.to_le_bytes());
        digest.update(&length.to_le_bytes());
        digest.update(&hash.0);
        digest.update(&blob_offset.to_le_bytes());
        blob = if count == 1
            && offset == 0
            && blob_offset == 0
            && length == size_sql
            && stored == Some(length)
        {
            Some(hash)
        } else {
            None
        };
        Ok(ControlFlow::Continue(()))
    })?;
    if read.is_break() || end != size_sql || (count == 0 && size_sql != 0) {
        return Ok(None);
    }
    Ok(Some(AssetSource {
        chain,
        length: size,
        version: digest.finalize().to_hex().to_string(),
        blob,
    }))
}

fn observed_pin(
    conn: &Connection,
    prefix: &str,
    source: &ObservedMember,
) -> Result<Option<(i64, u64)>> {
    let statement = if prefix.is_empty() {
        MemberObservation::ARCHIVE
    } else {
        MemberObservation::INDEX
    };
    let Some(MemberObservation {
        chain: Some(chain),
        length: Some(length),
        details_json: Some(details),
    }) = conn.maybe(&statement, params![source.observation, source.input])?
    else {
        return Ok(None);
    };
    let details: serde_json::Value = serde_json::from_str(&details)?;
    if details.get("import") != Some(&serde_json::to_value(&source.membership)?)
        || chain <= 0
        || length < 0
    {
        return Ok(None);
    }
    Ok(Some((chain, length as u64)))
}

pub(super) fn hash_member(
    reader: &Reader<'_>,
    input: &str,
    chain: i64,
    pin: &AssetSource,
) -> Result<BlobHash> {
    if pin.length == 0 {
        return Ok(BlobHash::of(&super::empty::read(reader, pin)?));
    }
    let size = pin.length;
    let mut source = reader
        .stored_chain(input, chain, size)?
        .reader(Arc::new(reader.blob_reader()?));
    let mut buffer = [0u8; 64 * 1024];
    let mut digest = blake3::Hasher::new();
    let mut length = 0u64;
    loop {
        reader.conn()?;
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        length = length
            .checked_add(read as u64)
            .ok_or_else(|| crate::StoreError::stored_data("attachment size overflow"))?;
    }
    if length != size {
        return Err(crate::StoreError::stored_data(
            "pinned attachment length differs",
        ));
    }
    Ok(BlobHash(*digest.finalize().as_bytes()))
}

fn check_member_is_current(
    index: &Connection,
    catalog: &Connection,
    prefix: &str,
    byte_match: &ValidatedMember,
) -> Result<bool> {
    let Some(ImportMemberKey {
        import_id,
        input,
        member,
    }) = index.maybe(&ImportMemberKey::OF_OBSERVATION, [&byte_match.observation])?
    else {
        return Ok(false);
    };
    if import_id != byte_match.member.import_id
        || input != byte_match.member.input
        || member != byte_match.member.member
    {
        return Ok(false);
    }
    // The source observation already stores authorship. Match the
    // membership keys here without replacing its recorded attribution.
    let statement = if prefix.is_empty() {
        MemberObservation::ARCHIVE
    } else {
        MemberObservation::INDEX
    };
    let Some(MemberObservation {
        chain: Some(chain),
        length: Some(length),
        details_json: Some(details),
    }) = catalog.maybe(
        &statement,
        params![byte_match.observation, byte_match.member.input],
    )?
    else {
        return Ok(false);
    };
    let details: serde_json::Value = serde_json::from_str(&details)?;
    if details.pointer("/import/version").and_then(|v| v.as_u64()) != Some(1)
        || details
            .pointer("/import/import_id")
            .and_then(|v| v.as_str())
            != Some(byte_match.member.import_id.as_str())
        || details.pointer("/import/member").and_then(|v| v.as_str())
            != Some(byte_match.member.member.as_str())
        || chain != byte_match.member.chain
        || length < 0
        || length as u64 != byte_match.member.size
    {
        return Ok(false);
    }
    Ok(pinned_source(
        (index, catalog),
        prefix,
        &byte_match.member.input,
        chain,
        byte_match.member.size,
        Some(&byte_match.observation),
        Some(&byte_match.member.source_version),
    )?
    .is_some_and(|pin| pin.version == byte_match.member.source_version))
}

fn source_context(
    index: &Connection,
    catalog: &Connection,
    prefix: &str,
    context: &BindingContext,
) -> Result<bool> {
    let exists: bool = index.exists("SELECT EXISTS(SELECT 1 FROM message_parse s JOIN input_ref i ON i.pk=s.input_pk WHERE s.message_id=?1 AND s.hash=?2 AND i.id=?3 AND s.chain=?4)", params![context.message.to_string(),context.source_version.0.as_slice(),context.input,context.chain])?;
    if !exists {
        return Ok(false);
    }
    let statement = if prefix.is_empty() {
        MemberDetails::ARCHIVE
    } else {
        MemberDetails::INDEX
    };
    let found = index.each(
        &MemberObservationId::OF_IMPORT_INPUT,
        params![context.input, context.import_id],
        |row| {
            let observed = catalog.maybe(&statement, params![row.observation, context.input])?;
            if let Some(MemberDetails {
                chain: Some(chain),
                details_json: Some(details),
            }) = observed
            {
                let details: serde_json::Value = serde_json::from_str(&details)?;
                if chain == context.chain
                    && details
                        .pointer("/import/version")
                        .and_then(|value| value.as_u64())
                        == Some(1)
                    && details
                        .pointer("/import/import_id")
                        .and_then(|value| value.as_str())
                        == Some(context.import_id.as_str())
                {
                    return Ok(ControlFlow::Break(()));
                }
            }
            Ok(ControlFlow::Continue(()))
        },
    )?;
    Ok(found.is_break())
}

pub(super) fn asset_match(
    conn: &Connection,
    member: &ReceivingMember,
) -> Result<super::AssetMatch> {
    let pin = pinned_source(
        (conn, conn),
        "source_",
        &member.input,
        member.chain,
        member.size,
        None,
        Some(&member.source_version),
    )?
    .filter(|pin| pin.version == member.source_version)
    .ok_or_else(|| crate::StoreError::internal("attachment member pin is unavailable"))?;
    Ok(super::AssetMatch {
        input: member.input.clone(),
        chain: member.chain,
        blob: pin.blob,
        source_version: member.source_version.clone(),
        size: member.size,
        members: vec![member.member.clone()],
        names: Vec::new(),
        media_types: Vec::new(),
        declared_sizes: Vec::new(),
    })
}

fn binding_body(conn: &Connection, selection: &BindingSelection) -> Result<Option<Vec<u8>>> {
    // The key keeps the complete typed stated-binding bytes, rather than a
    // candidate hash or a digest that could silently merge different sources.
    let claim = serde_json::to_vec(&selection.claim)?;
    conn.scalar_optional(
        "SELECT body FROM attachment_binding WHERE message=?1 AND source_version=?2
         AND input=?3 AND chain=?4 AND import_id=?5 AND block=?6 AND attachment=?7
         AND declared_input=?8 AND declared_chain=?9 AND declared_import_id=?10 AND claim=?11",
        params![
            selection.context.message.to_string(),
            selection.context.source_version.0.as_slice(),
            selection.context.input,
            selection.context.chain,
            selection.context.import_id,
            selection.claim.block.to_string(),
            selection.claim.attachment.to_string(),
            selection.claim.namespace.input,
            selection.claim.namespace.chain,
            selection.claim.namespace.import_id,
            claim
        ],
    )
}

row! {
    struct StoredBinding {
        source_version: Vec<u8>,
        input: String,
        chain: i64,
        import_id: String,
        block: String,
        attachment: String,
        declared_input: String,
        declared_chain: i64,
        declared_import_id: String,
        claim: Vec<u8>,
        body: Vec<u8>,
    }
    statements {
        const OF_MESSAGE = "SELECT b.source_version,b.input,b.chain,b.import_id,b.block,b.attachment,
                b.declared_input,b.declared_chain,b.declared_import_id,b.claim,b.body
         FROM attachment_binding b
         WHERE b.message=?1 AND (?2 IS NULL OR b.source_version=?2)
           AND (?3 IS NULL OR (b.block=?3 AND b.attachment=?4))
           AND EXISTS(SELECT 1 FROM message m
             JOIN message_parse s ON s.message_id=m.id AND s.hash=b.source_version AND s.chain=b.chain
             JOIN input_ref i ON i.pk=s.input_pk AND i.id=b.input
             WHERE m.id=b.message AND (?5=0 OR (s.input_pk=m.input_pk AND s.chain=m.source_chain)))
         ORDER BY b.input,b.chain,b.import_id,b.source_version,b.declared_input,b.declared_chain,b.declared_import_id,b.claim";
    }
}

fn stored_bindings(
    conn: &Connection,
    message: &MessageId,
    source_version: Option<BlobHash>,
    slot: Option<(u64, u64)>,
    shown_only: bool,
) -> Result<Vec<AttachmentBindingRecord>> {
    let block = slot.map(|(block, _)| block.to_string());
    let attachment = slot.map(|(_, attachment)| attachment.to_string());
    let version = source_version.as_ref().map(|version| version.0.as_slice());
    let mut records = Vec::new();
    let _ = conn.each(
        &StoredBinding::OF_MESSAGE,
        params![message.to_string(), version, block, attachment, shown_only],
        |row| {
            let encoded_claim = row.claim;
            let claim: AttachmentBinding = serde_json::from_slice(&encoded_claim)?;
            if serde_json::to_vec(&claim)? != encoded_claim {
                return Err(crate::StoreError::stored_data(
                    "attachment binding key is not in canonical JSON form",
                ));
            }
            if claim.block.to_string() != row.block
                || claim.attachment.to_string() != row.attachment
                || claim.namespace.input != row.declared_input
                || claim.namespace.chain != row.declared_chain
                || claim.namespace.import_id != row.declared_import_id
            {
                return Err(crate::StoreError::stored_data(
                    "attachment binding key differs from its declared namespace or slot",
                ));
            }
            let selection = BindingSelection {
                context: BindingContext {
                    message: message.clone(),
                    source_version: BlobHash(row.source_version.try_into().map_err(|_| {
                        crate::StoreError::stored_data("invalid binding source version")
                    })?),
                    input: row.input,
                    chain: row.chain,
                    import_id: row.import_id,
                },
                claim,
            };
            let record: AttachmentBindingRecord = crate::encoding::unpack_json(&row.body)?;
            record.validate()?;
            if record.selection() != selection {
                return Err(crate::StoreError::stored_data(
                    "attachment binding key differs from its stored record",
                ));
            }
            records.push(record);
            Ok(ControlFlow::Continue(()))
        },
    )?;
    Ok(records)
}

pub(crate) fn known(conn: &Connection, record: &AttachmentBindingRecord) -> Result<bool> {
    record.validate()?;
    let Some(body) = binding_body(conn, &record.selection())? else {
        return Ok(false);
    };
    let existing: AttachmentBindingRecord = crate::encoding::unpack_json(&body)?;
    if existing != *record {
        return Err(StoreError::internal(
            "different attachment binding records under one source context",
        ));
    }
    Ok(true)
}

pub(crate) fn project(
    index: &Connection,
    catalog: &Connection,
    record: &AttachmentBindingRecord,
) -> Result<()> {
    if known(index, record)? {
        return Ok(());
    }
    let selection = record.selection();
    let body = serde_json::to_vec(record)?;
    // Source/member absence is an unavailable association, not a reason to
    // discard stated bindings or accept a different member by its hash.
    let (catalog, prefix) = if matches!(
        crate::source::Source::of(index)?,
        crate::source::Source::Cache
    ) {
        (index, "source_")
    } else {
        (catalog, source_prefix(catalog)?)
    };
    let _present = source_context(index, catalog, prefix, &record.context)?;
    for association in &record.members {
        let _present = check_member_is_current(index, catalog, prefix, &association.witness)?;
    }
    let packed = crate::encoding::pack_bytes(&body)?;
    index.execute(
        "INSERT INTO attachment_binding VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
        params![
            selection.context.message.to_string(),
            selection.context.source_version.0.as_slice(),
            selection.context.input,
            selection.context.chain,
            selection.context.import_id,
            selection.claim.block.to_string(),
            selection.claim.attachment.to_string(),
            selection.claim.namespace.input,
            selection.claim.namespace.chain,
            selection.claim.namespace.import_id,
            serde_json::to_vec(&selection.claim)?,
            packed
        ],
    )?;
    Ok(())
}

impl Reader<'_> {
    /// Hashes the stored bytes in fixed-size chunks before recording a byte match.
    #[cfg(any(test, feature = "fixture"))]
    pub fn validate_attachment_member(
        &self,
        source: &ObservedMember,
        hash: BlobHash,
        size: u64,
    ) -> Result<ValidatedMember> {
        if !source.membership.is_valid() {
            return Err(crate::StoreError::internal(
                "invalid receiving import member",
            ));
        }
        let Some((chain, length)) = observed_pin(self.conn()?, "source_", source)? else {
            return Err(crate::StoreError::internal(
                "receiving member has no source observation",
            ));
        };
        if length != size {
            return Err(crate::StoreError::internal(
                "receiving member size differs from its candidate",
            ));
        }
        let pin = pinned_source(
            (self.conn()?, self.conn()?),
            "source_",
            &source.input,
            chain,
            size,
            Some(&source.observation),
            None,
        )?
        .ok_or_else(|| {
            crate::StoreError::internal("receiving member has no pinned stored endpoint")
        })?;
        if hash_member(self, &source.input, chain, &pin)? != hash {
            return Err(crate::StoreError::internal(
                "receiving member hash differs from its candidate",
            ));
        }
        Ok(ValidatedMember {
            version: 1,
            observation: source.observation.clone(),
            member: ReceivingMember {
                import_id: source.membership.import_id.clone(),
                input: source.input.clone(),
                member: source.membership.member.clone(),
                chain,
                source_version: pin.version,
                hash,
                size,
            },
        })
    }

    pub fn attachment_binding(
        &self,
        selection: &BindingSelection,
    ) -> Result<Option<BindingResolution>> {
        let conn = self.conn()?;
        let body = binding_body(conn, selection)?;
        let Some(body) = body else {
            return Ok(None);
        };
        let record: AttachmentBindingRecord = crate::encoding::unpack_json(&body)?;
        record.validate()?;
        if record.selection() != *selection {
            return Err(crate::StoreError::stored_data(
                "attachment binding key differs from its stored record",
            ));
        }
        if matches!(
            crate::source::Source::of(conn)?,
            crate::source::Source::Cache
        ) {
            // The stored record keeps its byte match, but only the original
            // commit context can show which member received the bytes.
            return Ok(Some(BindingResolution {
                selection: selection.clone(),
                claim: record.claim,
                state: AttachmentState::Unknown,
                members: Vec::new(),
            }));
        }
        let mut members = Vec::new();
        let mut seen = BTreeSet::new();
        let context_present = source_context(conn, conn, "source_", &record.context)?;
        if context_present {
            for association in &record.members {
                if check_member_is_current(conn, conn, "source_", &association.witness)?
                    && seen.insert(serde_json::to_vec(&(
                        association.candidate,
                        &association.witness.member,
                    ))?)
                {
                    members.push(association.witness.member.clone());
                }
            }
        }
        let state = match (record.claim.candidates.len(), members.len()) {
            (0, _) => AttachmentState::Unknown,
            (2.., _) | (_, 2..) => AttachmentState::Ambiguous,
            (1, 1) => AttachmentState::Available,
            _ => AttachmentState::Unavailable,
        };
        Ok(Some(BindingResolution {
            selection: selection.clone(),
            claim: record.claim,
            state,
            members,
        }))
    }

    /// Stated bindings from every stored matching reading, including sources
    /// not selected for display. This metadata never grants member availability.
    pub fn stated_attachment_bindings(
        &self,
        message: &MessageId,
        source_version: BlobHash,
    ) -> Result<Vec<AttachmentBindingRecord>> {
        stored_bindings(self.conn()?, message, Some(source_version), None, false)
    }

    pub fn attachment_binding_contexts(
        &self,
        message: &MessageId,
        block: u64,
        attachment: u64,
    ) -> Result<BindingContexts> {
        let records =
            stored_bindings(self.conn()?, message, None, Some((block, attachment)), true)?;
        let mut contexts = Vec::new();
        for record in records {
            if let Some(resolution) = self.attachment_binding(&record.selection())? {
                contexts.push(resolution);
            }
        }
        let state = match contexts.as_slice() {
            [] => AttachmentState::Unknown,
            [one] => one.state,
            _ => AttachmentState::Ambiguous,
        };
        Ok(BindingContexts {
            state,
            namespace_count: contexts.len() as u64,
            contexts,
        })
    }
}

impl<'pool> Reader<'pool> {
    #[cfg(any(test, feature = "fixture"))]
    pub fn pin_attachment<'reader>(
        &'reader self,
        selection: &BindingSelection,
    ) -> Result<AttachmentPin<'reader, 'pool>> {
        let Some(resolution) = self.attachment_binding(selection)? else {
            return Ok(AttachmentPin::Unknown);
        };
        Ok(match resolution.state {
            AttachmentState::Available => AttachmentPin::Available(PinnedMember {
                reader: self,
                member: resolution
                    .members
                    .into_iter()
                    .next()
                    .expect("available association has one member"),
            }),
            AttachmentState::Unavailable => AttachmentPin::Unavailable,
            AttachmentState::Ambiguous => AttachmentPin::Ambiguous,
            AttachmentState::Unknown => AttachmentPin::Unknown,
        })
    }
}

#[cfg(test)]
pub(super) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<PinnedExtent>();
    check.row::<MemberObservation>();
    check.row::<MemberDetails>();
    check.row::<ImportMemberKey>();
    check.row::<MemberObservationId>();
    check.row::<StoredBinding>();
}

#[cfg(test)]
mod tests;
