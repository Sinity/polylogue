//! Derived import associations. The engine parses the sources; reads check
//! the archive's current state before returning any bytes.
use crate::ConnExt as _;
use crate::row::{Rows as _, row};
use crate::{ImportMember, Reader, Result, StoreError, export_items::ExportItems};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

pub mod bindings;
pub(crate) mod empty;
mod read;
pub use read::{AssetMatch, AttachmentResolution, AttachmentState};

pub const RULES: u32 = 1;

/// The meta key holding the archive position through which import
/// observations are projected. Observations of one commit become visible
/// together, so a position is a complete cursor.
const OBSERVATION_ROW_KEY: &str = "import_observation_position";

#[derive(Debug, Clone)]
pub struct AssetSource {
    pub chain: i64,
    pub length: u64,
    pub version: String,
    pub blob: Option<polylogue_model::BlobHash>,
}

row! {
    struct AssetRevision {
        chain: i64,
        length: i64,
    }
    statements {
        const ARCHIVE = "SELECT chain,length FROM revision WHERE input=?1 ORDER BY number DESC LIMIT 1";
        const INDEX = "SELECT chain,length FROM source_revision WHERE input=?1 ORDER BY number DESC LIMIT 1";
    }
}

macro_rules! asset_extents {
    ($prefix:literal) => {
        concat!(
            "SELECT e.offset,e.length,e.blob,e.blob_offset,b.size
            FROM ",
            $prefix,
            "extent e LEFT JOIN ",
            $prefix,
            "blob b ON b.hash=e.blob
            WHERE e.input=?1 AND e.chain=?2 ORDER BY e.offset"
        )
    };
}

row! {
    struct AssetExtent {
        offset: i64,
        length: i64,
        blob: Vec<u8>,
        blob_offset: i64,
        size: Option<i64>,
    }
    statements {
        const ARCHIVE = asset_extents!("");
        const INDEX = asset_extents!("source_");
    }
}

row! {
    struct AssetSidecar {
        input: String,
        sidecar: String,
        version: Option<u32>,
        source_version: Option<String>,
    }
    statements {
        const WITH_REVISION = "SELECT DISTINCT m.input,m.sidecar,s.version,s.source_version FROM import_member m
            LEFT JOIN import_asset_sidecar s ON s.input=m.input AND s.sidecar=m.sidecar
            WHERE m.sidecar IS NOT NULL
              AND EXISTS(SELECT 1 FROM source_revision r WHERE r.input=m.input)
            ORDER BY m.input,m.sidecar";
    }
}

row! {
    struct ImportMembersExist {
        present: bool,
    }
    statements {
        const CHECK = "SELECT EXISTS(SELECT 1 FROM import_member) AS present";
    }
}

row! {
    struct ImportChange {
        import_id: String,
        changed: i64,
    }
    statements {
        const OF_SESSION = "SELECT DISTINCT im.import_id, coalesce(c.changed,0) AS changed
             FROM session s JOIN message m ON m.session_pk=s.pk
             JOIN input_ref i ON i.pk=m.input_pk
             JOIN import_member im ON im.input=i.id
             LEFT JOIN import_change c ON c.import_id=im.import_id
             WHERE s.id=?1 ORDER BY im.import_id";
    }
}

macro_rules! import_containers {
    ($prefix:literal) => {
        concat!("SELECT DISTINCT c.container_input,json_extract(o.details_json,'$.import.import_id') AS import_id
        FROM ", $prefix, "observation o JOIN ", $prefix, "observation_container c ON c.observation_id=o.id
        WHERE o.commit_seq>?1 AND o.commit_seq<=?2 AND (?3 IS NULL OR c.container_input=?3)
          AND json_type(o.details_json,'$.import.import_id')='text'")
    };
}

row! {
    struct ImportContainer {
        container_input: String,
        import_id: String,
    }
    statements {
        const INDEX = import_containers!("source_");
        const ARCHIVE = import_containers!("");
    }
}

macro_rules! import_observations {
    ($prefix:literal) => {
        concat!(
            "SELECT o.id,o.input,o.details_json FROM ",
            $prefix,
            "observation o
        WHERE o.commit_seq>?1 AND o.commit_seq<=?2 AND (?3 IS NULL OR o.input=?3)
          AND o.details_json IS NOT NULL ORDER BY o.commit_seq, o.id"
        )
    };
}

row! {
    struct ImportObservation {
        id: String,
        input: String,
        details_json: String,
    }
    statements {
        const INDEX = import_observations!("source_");
        const ARCHIVE = import_observations!("");
    }
}

fn asset_source(
    index: &Connection,
    conn: &Connection,
    prefix: &str,
    input: &str,
) -> Result<Option<AssetSource>> {
    let (revision, extents) = if prefix.is_empty() {
        (AssetRevision::ARCHIVE, AssetExtent::ARCHIVE)
    } else {
        (AssetRevision::INDEX, AssetExtent::INDEX)
    };
    let Some(AssetRevision { chain, length }) = conn.maybe(&revision, [input])? else {
        return Ok(None);
    };
    let length =
        u64::try_from(length).map_err(|_| crate::StoreError::stored(0, "negative asset length"))?;
    if length == 0 {
        return empty::source(index, conn, prefix, input, chain, None, None);
    }
    let mut hash = blake3::Hasher::new();
    hash.update(&chain.to_le_bytes());
    hash.update(&length.to_le_bytes());
    let mut end = 0u64;
    let mut count = 0;
    let mut blob = None;
    let _ = conn.each(&extents, params![input, chain], |row| {
        let AssetExtent {
            offset,
            length: size,
            blob: bytes,
            blob_offset,
            size: blob_size,
        } = row;
        if offset < 0
            || size < 0
            || blob_offset < 0
            || offset as u64 != end
            || blob_size.is_none_or(|blob_size| {
                blob_size < 0
                    || blob_offset
                        .checked_add(size)
                        .is_none_or(|end| end > blob_size)
            })
        {
            return Err(StoreError::stored_data(
                "the asset's stored extents are incomplete",
            ));
        }
        let current = polylogue_model::BlobHash(
            bytes
                .try_into()
                .map_err(|_| crate::StoreError::stored(2, "invalid asset extent hash"))?,
        );
        count += 1;
        end = end
            .checked_add(size as u64)
            .ok_or_else(|| crate::StoreError::stored(1, "asset extent overflow"))?;
        hash.update(&offset.to_le_bytes());
        hash.update(&size.to_le_bytes());
        hash.update(&current.0);
        hash.update(&blob_offset.to_le_bytes());
        blob = if count == 1
            && offset == 0
            && blob_offset == 0
            && size as u64 == length
            && blob_size == Some(size)
        {
            Some(current)
        } else {
            None
        };
        Ok(ControlFlow::Continue(()))
    })?;
    if end != length || count == 0 {
        return Err(StoreError::stored_data(
            "the asset's stored extents are incomplete",
        ));
    }
    Ok(Some(AssetSource {
        chain,
        length,
        version: hash.finalize().to_hex().to_string(),
        blob,
    }))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedMember {
    pub observation: String,
    pub input: String,
    pub membership: ImportMember,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssetRow {
    Member {
        source: ObservedMember,
        source_id: Option<String>,
        sidecar: Option<String>,
    },
    Binding {
        record: bindings::AttachmentBindingRecord,
    },
    Metadata {
        input: String,
        sidecar: String,
        ordinal: u64,
        source_id: String,
        name: Option<String>,
        media_type: Option<String>,
        size: Option<u64>,
    },
    Sidecar {
        input: String,
        sidecar: String,
        source_version: String,
        error: Option<String>,
    },
    ResetMetadata {
        input: String,
        sidecar: String,
    },
    Scan {
        through: i64,
    },
}

impl Reader<'_> {
    pub fn asset_source(&self, input: &str) -> Result<Option<AssetSource>> {
        asset_source(self.conn()?, self.conn()?, "source_", input)
    }
    /// Only unseen import metadata is decoded. Content and observations
    /// remain separate: equal bytes can belong to several import members.
    pub fn unindexed_import_members(&self) -> Result<(Vec<ObservedMember>, Option<i64>)> {
        let conn = self.conn()?;
        let after: i64 = conn
            .scalar_optional(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key=?1",
                [OBSERVATION_ROW_KEY],
            )?
            .unwrap_or(0);
        let mut out = Vec::new();
        let through = visit_import_members(conn, "source_", after, None, |member| {
            out.push(member);
            Ok(())
        })?;
        Ok((out, through))
    }

    pub fn pending_asset_sidecars(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut pending = Vec::new();
        let _ = conn.each(&AssetSidecar::WITH_REVISION, [], |row| {
            let AssetSidecar {
                input,
                sidecar,
                version,
                source_version: recorded,
            } = row;
            let current = asset_source(conn, conn, "source_", &input)?.map(|source| source.version);
            if version != Some(RULES) || recorded != current {
                pending.push((input, sidecar));
            }
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(pending)
    }
}

impl Reader<'_> {
    /// Attachment presentation depends on every receiving namespace of the
    /// session's shown source messages, including newly ambiguous membership.
    pub fn session_attachment_version(
        &self,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::BlobHash>> {
        let conn = self.conn()?;
        // The join below reads every message row of the session; an archive
        // that holds no import has nothing for it to find.
        if !conn.one(&ImportMembersExist::CHECK, [])?.present {
            self.check()?;
            return Ok(None);
        }
        let mut digest = blake3::Hasher::new();
        let mut any = false;
        let _ = conn.each(&ImportChange::OF_SESSION, [session.to_string()], |row| {
            any = true;
            digest.update(&(row.import_id.len() as u64).to_le_bytes());
            digest.update(row.import_id.as_bytes());
            digest.update(&row.changed.to_le_bytes());
            Ok(ControlFlow::Continue(()))
        })?;
        self.check()?;
        Ok(any.then(|| polylogue_model::BlobHash(*digest.finalize().as_bytes())))
    }
}

/// Visits import metadata inside the caller's source snapshot. Rowids are only
/// a discovery cursor; replay selects each changed input from zero after the source
/// rows are rebuilt, whose local rowids need not preserve their old order.
pub(crate) fn visit_import_members(
    conn: &Connection,
    prefix: &str,
    after: i64,
    input: Option<&str>,
    mut visit: impl FnMut(ObservedMember) -> Result<()>,
) -> Result<Option<i64>> {
    let through: i64 = conn.scalar(
        &format!("SELECT COALESCE(max(commit_seq),0) FROM {prefix}observation"),
        [],
    )?;
    // A ZIP is the source of its members, not one of the attachments among them.
    let mut originals =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    let (containers, observations) = if prefix.is_empty() {
        (ImportContainer::ARCHIVE, ImportObservation::ARCHIVE)
    } else {
        (ImportContainer::INDEX, ImportObservation::INDEX)
    };
    let _ = conn.each(&containers, params![after, through, input], |row| {
        originals
            .entry(row.container_input)
            .or_default()
            .insert(row.import_id);
        Ok(ControlFlow::Continue(()))
    })?;
    let _ = conn.each(&observations, params![after, through, input], |row| {
        let ImportObservation {
            id: observation,
            input,
            details_json: details,
        } = row;
        let details: serde_json::Value = serde_json::from_str(&details)?;
        if let Some(member) = details.get("import") {
            let membership: ImportMember = serde_json::from_value(member.clone())?;
            if originals
                .get(&input)
                .is_some_and(|imports| imports.contains(&membership.import_id))
            {
                return Ok(ControlFlow::Continue(()));
            }
            if !membership.is_valid() {
                return Err(StoreError::internal(
                    "unsupported or invalid import membership",
                ));
            }
            visit(ObservedMember {
                observation,
                input,
                membership,
            })?;
        }
        Ok(ControlFlow::Continue(()))
    })?;
    Ok((through != after).then_some(through))
}

row! {
    struct StoredMember {
        import_id: String,
        input: String,
        member: String,
        source_id: Option<String>,
        sidecar: Option<String>,
    }
    statements {
        const OF_OBSERVATION = "SELECT import_id,input,member,source_id,sidecar FROM import_member WHERE observation=?1";
    }
}

/// The same member indexing is used by import refresh and confirmed replay.
pub(crate) fn project_member(
    conn: &Connection,
    source: &ObservedMember,
    source_id: Option<&str>,
    sidecar: Option<&str>,
) -> Result<()> {
    let previous = conn.maybe(&StoredMember::OF_OBSERVATION, [&source.observation])?;
    if let Some(StoredMember {
        import_id,
        input,
        member,
        source_id: known_id,
        sidecar: known_sidecar,
    }) = previous
    {
        if import_id != source.membership.import_id
            || input != source.input
            || member != source.membership.member
            || known_id
                .as_deref()
                .zip(source_id)
                .is_some_and(|(old, new)| old != new)
            || known_sidecar
                .as_deref()
                .zip(sidecar)
                .is_some_and(|(old, new)| old != new)
        {
            return Err(StoreError::internal(
                "conflicting identity or classification for one import member",
            ));
        }
        // A member seen again only changes when it learns an identity or a
        // sidecar it did not have; an identical re-import changes nothing.
        let learned = (known_id.is_none() && source_id.is_some())
            || (known_sidecar.is_none() && sidecar.is_some());
        if !learned {
            return Ok(());
        }
        conn.execute(
            "UPDATE import_member SET source_id=?2,sidecar=?3 WHERE observation=?1",
            params![
                source.observation,
                known_id.as_deref().or(source_id),
                known_sidecar.as_deref().or(sidecar)
            ],
        )?;
    } else {
        conn.execute(
            "INSERT INTO import_member VALUES (?1,?2,?3,?4,?5,?6)",
            params![
                source.observation,
                source.membership.import_id,
                source.input,
                source.membership.member,
                source_id,
                sidecar
            ],
        )?;
    }
    count_import_change(conn, &source.membership.import_id)
}

fn count_import_change(conn: &Connection, import_id: &str) -> Result<()> {
    conn.prepare_cached(
        "INSERT INTO import_change(import_id,changed) VALUES (?1,1)
         ON CONFLICT(import_id) DO UPDATE SET changed=changed+1",
    )?
    .execute([import_id])?;
    Ok(())
}

/// Asset metadata and sidecars are keyed by input; every import holding the
/// input sees the change.
fn count_input_change(conn: &Connection, input: &str) -> Result<()> {
    conn.prepare_cached(
        "INSERT INTO import_change(import_id,changed)
         SELECT DISTINCT import_id,1 FROM import_member WHERE input=?1
         ON CONFLICT(import_id) DO UPDATE SET changed=changed+1",
    )?
    .execute([input])?;
    Ok(())
}

struct MetadataRow<'a> {
    input: &'a str,
    sidecar: &'a str,
    ordinal: i64,
    source_id: &'a str,
    name: Option<&'a str>,
    media_type: Option<&'a str>,
    size: Option<String>,
}

/// Stores one asset metadata row; true only when the stored row changed.
fn put_metadata(conn: &Connection, row: &MetadataRow<'_>) -> Result<bool> {
    let changed = conn.execute(
        "INSERT INTO import_asset_metadata VALUES (?1,?2,?3,?4,?5,?6,?7)
         ON CONFLICT(input,sidecar,ordinal) DO UPDATE SET
           source_id=excluded.source_id,name=excluded.name,
           media_type=excluded.media_type,size=excluded.size
         WHERE (source_id,name,media_type,size)
           IS NOT (excluded.source_id,excluded.name,excluded.media_type,excluded.size)",
        params![
            row.input,
            row.sidecar,
            row.ordinal,
            row.source_id,
            row.name,
            row.media_type,
            row.size
        ],
    )?;
    Ok(changed > 0)
}

/// Stores one sidecar's parse outcome; true only when it changed.
fn put_sidecar(
    conn: &Connection,
    input: &str,
    sidecar: &str,
    recorded: &str,
    error: Option<&str>,
) -> Result<bool> {
    let changed = conn.execute(
        "INSERT INTO import_asset_sidecar VALUES (?1,?2,?3,?4,?5)
         ON CONFLICT(input,sidecar) DO UPDATE SET
           version=excluded.version,source_version=excluded.source_version,
           error=excluded.error
         WHERE (version,source_version,error)
           IS NOT (excluded.version,excluded.source_version,excluded.error)",
        params![input, sidecar, RULES, recorded, error],
    )?;
    Ok(changed > 0)
}

/// Removes a sidecar's metadata rows; true only when there were any.
fn reset_metadata(conn: &Connection, input: &str, sidecar: &str) -> Result<bool> {
    let removed = conn.execute(
        "DELETE FROM import_asset_metadata WHERE input=?1 AND sidecar=?2",
        params![input, sidecar],
    )?;
    Ok(removed > 0)
}

pub(crate) fn install(
    conn: &mut Connection,
    archive: &Connection,
    mut staged: ExportItems,
) -> Result<()> {
    crate::fault::check("assets.install", "")?;
    let catalog_prefix = crate::source::Source::of(archive)?.prefix();
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    while let Some(item) = staged.next_item()? {
        match serde_json::from_reader(item.reader)? {
            AssetRow::Member {
                source,
                source_id,
                sidecar,
            } => {
                project_member(&tx, &source, source_id.as_deref(), sidecar.as_deref())?;
            }
            AssetRow::Binding { record } => bindings::project(&tx, archive, &record)?,
            AssetRow::Metadata {
                input,
                sidecar,
                ordinal,
                source_id,
                name,
                media_type,
                size,
            } => {
                let ordinal = crate::to_i64(ordinal, "too many asset metadata entries")?;
                let row = MetadataRow {
                    input: &input,
                    sidecar: &sidecar,
                    ordinal,
                    source_id: &source_id,
                    name: name.as_deref(),
                    media_type: media_type.as_deref(),
                    size: size.map(|n| n.to_string()),
                };
                let changed = put_metadata(&tx, &row)?;
                if changed {
                    count_input_change(&tx, &input)?;
                }
            }
            AssetRow::Sidecar {
                input,
                sidecar,
                source_version: recorded,
                error,
            } => {
                if asset_source(&tx, archive, catalog_prefix, &input)?
                    .as_ref()
                    .map(|source| source.version.as_str())
                    != Some(recorded.as_str())
                {
                    return Err(StoreError::internal(
                        "asset sidecar changed while it was parsed",
                    ));
                }
                let changed = put_sidecar(&tx, &input, &sidecar, &recorded, error.as_deref())?;
                if changed {
                    count_input_change(&tx, &input)?;
                }
            }
            AssetRow::ResetMetadata { input, sidecar } => {
                if reset_metadata(&tx, &input, &sidecar)? {
                    count_input_change(&tx, &input)?;
                }
            }
            AssetRow::Scan { through } => {
                tx.execute(
                    "INSERT OR REPLACE INTO meta VALUES (?1,?2)",
                    params![OBSERVATION_ROW_KEY, through.to_string()],
                )?;
            }
        }
    }
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<AssetRevision>();
    check.row::<AssetExtent>();
    check.row::<AssetSidecar>();
    check.row::<ImportMembersExist>();
    check.row::<ImportChange>();
    check.row::<ImportContainer>();
    check.row::<ImportObservation>();
    check.row::<StoredMember>();
    bindings::declared_rows(check);
}

#[cfg(test)]
mod tests;
