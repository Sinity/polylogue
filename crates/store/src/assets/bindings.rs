//! Source-scoped attachment claims and ingress-validated receiving members.
//!
//! Canonical products preserve opaque witnesses. Reads require the same actual
//! member observation and exact source pins; historical claims cannot mint one.
use super::{AssetSource, AttachmentState, ObservedMember};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    BlobHash, MessageId, attachment_binding::AttachmentBinding, bundle::MessageRecord,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, io::Read, ops::Range, sync::Arc};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BindingSelection {
    pub context: BindingContext,
    pub block: u64,
    pub attachment: u64,
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
    // Canonical source fact identity, not a local SQLite rowid.
    observation: String,
}
impl ValidatedMember {
    pub fn identity(&self) -> &ReceivingMember {
        &self.member
    }
    pub fn observation(&self) -> &str {
        &self.observation
    }
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
    pub fn witness(&self) -> &ValidatedMember {
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
            .map_err(|_| invalid("invalid attachment binding record"))?;
        if context.message != record.message.id || context.source_version != record.source_version {
            return Err(invalid("binding context differs from its message source"));
        }
        let claim = record
            .bindings
            .get(binding)
            .cloned()
            .ok_or_else(|| invalid("missing attachment binding slot"))?;
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
            block: self.claim.block,
            attachment: self.claim.attachment,
        }
    }
    fn validate(&self) -> Result<()> {
        if self.context.input.is_empty()
            || self.context.chain <= 0
            || self.context.import_id.is_empty()
            || self.context.source_version != self.claim.source_version
            || self.claim.source_id.is_empty()
        {
            return Err(invalid("invalid attachment binding context"));
        }
        for association in &self.members {
            let witness = &association.witness;
            let candidate = usize::try_from(association.candidate)
                .ok()
                .and_then(|index| self.claim.candidates.get(index))
                .ok_or_else(|| invalid("attachment witness has no candidate"))?;
            if witness.version != 1
                || witness.observation.is_empty()
                || witness.member.import_id != self.context.import_id
                || witness.member.input.is_empty()
                || witness.member.member.is_empty()
                || witness.member.chain <= 0
                || witness.member.source_version.is_empty()
                || witness.member.hash != candidate.hash
                || witness.member.size != candidate.size
            {
                return Err(invalid(
                    "attachment witness differs from its receiving candidate",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BindingResolution {
    pub selection: BindingSelection,
    pub claim: AttachmentBinding,
    pub state: AttachmentState,
    pub members: Vec<ReceivingMember>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BindingContexts {
    pub state: AttachmentState,
    pub namespace_count: u64,
    pub contexts: Vec<BindingResolution>,
}

/// The reader borrow pins both membership and extents to this read snapshot.
/// Exposing its natural identity does not expose a proof constructor.
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
            return Err(invalid("attachment range is outside its pinned bytes"));
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
        let hash = hash_member(
            self.reader,
            &self.member.input,
            self.member.chain,
            self.member.size,
        )?;
        if hash != self.member.hash {
            return Err(invalid("pinned attachment hash differs from its witness"));
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

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
fn source_prefix(conn: &Connection) -> Result<&'static str> {
    Ok(match crate::source::Source::of(conn)? {
        crate::source::Source::Ledger => "",
        crate::source::Source::Cache => "source_",
    })
}

/// The exact recorded endpoint is required; a later revision never selects a
/// different chain or silently extends the validated member's bytes.
fn pinned_source(
    conn: &Connection,
    prefix: &str,
    input: &str,
    chain: i64,
    size: u64,
) -> Result<Option<AssetSource>> {
    let size_sql =
        i64::try_from(size).map_err(|_| invalid("attachment exceeds archive address space"))?;
    let exists = conn.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {prefix}revision WHERE input=?1 AND chain=?2 AND length=?3)"), params![input,chain,size_sql], |row| row.get::<_,bool>(0))?;
    if !exists {
        return Ok(None);
    }
    let mut statement = conn.prepare(&format!("SELECT e.offset,e.length,e.blob,e.blob_offset,b.size FROM {prefix}extent e LEFT JOIN {prefix}blob b ON b.hash=e.blob WHERE e.input=?1 AND e.chain=?2 AND (e.offset<?3 OR (?3=0 AND e.offset=0)) ORDER BY e.offset"))?;
    let mut rows = statement.query(params![input, chain, size_sql])?;
    let mut end = 0i64;
    let mut count = 0usize;
    let mut blob = None;
    let mut digest = blake3::Hasher::new();
    digest.update(&chain.to_le_bytes());
    digest.update(&size.to_le_bytes());
    while let Some(row) = rows.next()? {
        let offset: i64 = row.get(0)?;
        let length: i64 = row.get(1)?;
        let hash: Vec<u8> = row.get(2)?;
        let blob_offset: i64 = row.get(3)?;
        let stored: Option<i64> = row.get(4)?;
        if offset != end
            || length < 0
            || blob_offset < 0
            || stored.is_none_or(|stored| {
                stored < 0 || blob_offset.checked_add(length).is_none_or(|at| at > stored)
            })
            || offset.checked_add(length).is_none_or(|at| at > size_sql)
        {
            return Ok(None);
        }
        let hash = BlobHash(
            hash.try_into()
                .map_err(|_| invalid("invalid pinned attachment hash"))?,
        );
        end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("attachment extent overflow"))?;
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
    }
    if end != size_sql || (count == 0 && size_sql != 0) {
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
    let row: Option<(Option<i64>,Option<i64>,Option<String>)> = conn.query_row(&format!("SELECT coalesce(o.chain,r.chain),coalesce(o.length,r.length),o.details_json FROM {prefix}observation o LEFT JOIN {prefix}revision r ON r.input=o.input AND r.number=o.revision WHERE o.id=?1 AND o.input=?2"), params![source.observation,source.input], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?;
    let Some((Some(chain), Some(length), Some(details))) = row else {
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

fn hash_member(reader: &Reader<'_>, input: &str, chain: i64, size: u64) -> Result<BlobHash> {
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
            .ok_or_else(|| invalid("attachment size overflow"))?;
    }
    if length != size {
        return Err(invalid("pinned attachment length differs"));
    }
    Ok(BlobHash(*digest.finalize().as_bytes()))
}

fn current_witness(
    index: &Connection,
    catalog: &Connection,
    prefix: &str,
    witness: &ValidatedMember,
) -> Result<bool> {
    let actual: Option<(String, String, String)> = index
        .query_row(
            "SELECT import_id,input,member FROM import_member WHERE observation=?1",
            [&witness.observation],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((namespace, input, member)) = actual else {
        return Ok(false);
    };
    if namespace != witness.member.import_id
        || input != witness.member.input
        || member != witness.member.member
    {
        return Ok(false);
    }
    // Authorship is already retained by the source observation. Match the exact
    // membership keys here without replacing its recorded attribution.
    let row: Option<(Option<i64>,Option<i64>,Option<String>)> = catalog.query_row(&format!("SELECT coalesce(o.chain,r.chain),coalesce(o.length,r.length),o.details_json FROM {prefix}observation o LEFT JOIN {prefix}revision r ON r.input=o.input AND r.number=o.revision WHERE o.id=?1 AND o.input=?2"), params![witness.observation,witness.member.input], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?;
    let Some((Some(chain), Some(length), Some(details))) = row else {
        return Ok(false);
    };
    let details: serde_json::Value = serde_json::from_str(&details)?;
    if details.pointer("/import/version").and_then(|v| v.as_u64()) != Some(1)
        || details
            .pointer("/import/import_id")
            .and_then(|v| v.as_str())
            != Some(witness.member.import_id.as_str())
        || details.pointer("/import/member").and_then(|v| v.as_str())
            != Some(witness.member.member.as_str())
        || chain != witness.member.chain
        || length < 0
        || length as u64 != witness.member.size
    {
        return Ok(false);
    }
    Ok(pinned_source(
        catalog,
        prefix,
        &witness.member.input,
        chain,
        witness.member.size,
    )?
    .is_some_and(|pin| pin.version == witness.member.source_version))
}

fn source_context(
    index: &Connection,
    catalog: &Connection,
    prefix: &str,
    context: &BindingContext,
) -> Result<bool> {
    let exists: bool = index.query_row("SELECT EXISTS(SELECT 1 FROM message_source s JOIN message m ON m.pk=s.message_pk JOIN input_ref i ON i.pk=s.input_pk WHERE m.id=?1 AND s.version_hash=?2 AND i.id=?3 AND s.chain=?4)", params![context.message.to_string(),context.source_version.0.as_slice(),context.input,context.chain], |row| row.get(0))?;
    if !exists {
        return Ok(false);
    }
    let mut statement = index.prepare("SELECT observation FROM import_member WHERE input=?1 AND import_id=?2 ORDER BY observation")?;
    let ids = statement.query_map(params![context.input, context.import_id], |row| {
        row.get::<_, String>(0)
    })?;
    for id in ids {
        let id = id?;
        let observed: Option<(Option<i64>,Option<String>)> = catalog.query_row(&format!("SELECT coalesce(o.chain,r.chain),o.details_json FROM {prefix}observation o LEFT JOIN {prefix}revision r ON r.input=o.input AND r.number=o.revision WHERE o.id=?1 AND o.input=?2"), params![id,context.input], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
        if let Some((Some(chain), Some(details))) = observed {
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
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(super) fn asset_match(
    conn: &Connection,
    member: &ReceivingMember,
) -> Result<super::AssetMatch> {
    let pin = pinned_source(conn, "source_", &member.input, member.chain, member.size)?
        .filter(|pin| pin.version == member.source_version)
        .ok_or_else(|| invalid("attachment member pin is unavailable"))?;
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

pub(crate) fn project(
    index: &Connection,
    catalog: &Connection,
    record: &AttachmentBindingRecord,
) -> Result<()> {
    record.validate()?;
    let selection = record.selection();
    let body = serde_json::to_vec(record)?;
    let old: Option<Vec<u8>> = index.query_row("SELECT body FROM attachment_binding WHERE message=?1 AND source_version=?2 AND input=?3 AND chain=?4 AND import_id=?5 AND block=?6 AND attachment=?7", params![selection.context.message.to_string(),selection.context.source_version.0.as_slice(),selection.context.input,selection.context.chain,selection.context.import_id,selection.block.to_string(),selection.attachment.to_string()], |row| row.get(0)).optional()?;
    if let Some(old) = old {
        let existing: AttachmentBindingRecord = crate::encoding::unpack_json(&old)?;
        if existing != *record {
            return Err(StoreError::Conflict(
                "different attachment binding facts under one source context".into(),
            ));
        }
        return Ok(());
    }
    // Source/member absence is an unavailable association, not a reason to
    // discard canonical claims or certify a different member by its hash.
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
        let _present = current_witness(index, catalog, prefix, &association.witness)?;
    }
    let packed = crate::encoding::pack_bytes(&body)?;
    index.execute(
        "INSERT INTO attachment_binding VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        params![
            selection.context.message.to_string(),
            selection.context.source_version.0.as_slice(),
            selection.context.input,
            selection.context.chain,
            selection.context.import_id,
            selection.block.to_string(),
            selection.attachment.to_string(),
            packed
        ],
    )?;
    Ok(())
}

impl Reader<'_> {
    /// Hashes actual stored bytes in fixed-size chunks before minting a witness.
    pub fn validate_attachment_member(
        &self,
        source: &ObservedMember,
        hash: BlobHash,
        size: u64,
    ) -> Result<ValidatedMember> {
        if source.membership.version != 1
            || source.membership.import_id.is_empty()
            || source.membership.member.is_empty()
        {
            return Err(invalid("invalid receiving import member"));
        }
        let Some((chain, length)) = observed_pin(self.conn()?, "source_", source)? else {
            return Err(invalid("receiving member has no exact source observation"));
        };
        if length != size {
            return Err(invalid("receiving member size differs from its candidate"));
        }
        let pin = pinned_source(self.conn()?, "source_", &source.input, chain, size)?
            .ok_or_else(|| invalid("receiving member has no pinned stored endpoint"))?;
        if hash_member(self, &source.input, chain, size)? != hash {
            return Err(invalid("receiving member hash differs from its candidate"));
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
        let body: Option<Vec<u8>> = conn.query_row("SELECT body FROM attachment_binding WHERE message=?1 AND source_version=?2 AND input=?3 AND chain=?4 AND import_id=?5 AND block=?6 AND attachment=?7", params![selection.context.message.to_string(),selection.context.source_version.0.as_slice(),selection.context.input,selection.context.chain,selection.context.import_id,selection.block.to_string(),selection.attachment.to_string()], |row| row.get(0)).optional()?;
        let Some(body) = body else {
            return Ok(None);
        };
        let record: AttachmentBindingRecord = crate::encoding::unpack_json(&body)?;
        record.validate()?;
        if record.selection() != *selection {
            return Err(invalid("attachment projection key differs from its fact"));
        }
        let mut members = Vec::new();
        let mut seen = BTreeSet::new();
        let context_present = source_context(conn, conn, "source_", &record.context)?;
        if context_present {
            for association in &record.members {
                if current_witness(conn, conn, "source_", &association.witness)?
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

    pub fn attachment_binding_contexts(
        &self,
        message: &MessageId,
        block: u64,
        attachment: u64,
    ) -> Result<BindingContexts> {
        let conn = self.conn()?;
        let mut statement = conn.prepare("SELECT b.source_version,b.input,b.chain,b.import_id FROM attachment_binding b JOIN message m ON m.id=b.message JOIN message_source s ON s.message_pk=m.pk AND s.version_hash=b.source_version AND s.chain=b.chain AND s.shown=1 JOIN input_ref i ON i.pk=s.input_pk AND i.id=b.input WHERE b.message=?1 AND b.block=?2 AND b.attachment=?3 ORDER BY b.input,b.chain,b.import_id,b.source_version")?;
        let rows = statement.query_map(
            params![
                message.to_string(),
                block.to_string(),
                attachment.to_string()
            ],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )?;
        let mut contexts = Vec::new();
        for row in rows {
            let (version, input, chain, import_id) = row?;
            let selection = BindingSelection {
                context: BindingContext {
                    message: message.clone(),
                    source_version: BlobHash(
                        version
                            .try_into()
                            .map_err(|_| invalid("invalid binding source version"))?,
                    ),
                    input,
                    chain,
                    import_id,
                },
                block,
                attachment,
            };
            if let Some(resolution) = self.attachment_binding(&selection)? {
                contexts.push(resolution);
            }
        }
        let state = match contexts.as_slice() {
            [] => AttachmentState::Unknown,
            [one] => one.state.clone(),
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
mod tests;
