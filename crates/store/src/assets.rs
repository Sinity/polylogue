//! Derived import associations. The engine parses the sources; reads check
//! the archive's current state before returning any bytes.
use crate::{ImportMember, Reader, Result, StoreError, export_items::ExportItems};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

pub mod bindings;
pub(crate) mod empty;
mod read;
pub use read::{AssetMatch, AttachmentResolution, AttachmentState};

pub const RULES: u32 = 1;

#[derive(Debug, Clone)]
pub struct AssetSource {
    pub chain: i64,
    pub length: u64,
    pub version: String,
    pub blob: Option<polylogue_model::BlobHash>,
}

fn asset_source(
    index: &Connection,
    conn: &Connection,
    schema: &str,
    input: &str,
) -> Result<Option<AssetSource>> {
    let revision: Option<(i64, i64)> = conn
        .query_row(
            &format!("SELECT chain,length FROM {schema}revision WHERE input=?1 ORDER BY number DESC LIMIT 1"),
            [input],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((chain, length)) = revision else {
        return Ok(None);
    };
    let length =
        u64::try_from(length).map_err(|_| StoreError::Invalid("negative asset length".into()))?;
    if length == 0 {
        return empty::source(index, conn, schema, input, chain, None, None);
    }
    let mut hash = blake3::Hasher::new();
    hash.update(&chain.to_le_bytes());
    hash.update(&length.to_le_bytes());
    let mut stmt=conn.prepare_cached(&format!("SELECT e.offset,e.length,e.blob,e.blob_offset,b.size
        FROM {schema}extent e LEFT JOIN {schema}blob b ON b.hash=e.blob WHERE e.input=?1 AND e.chain=?2 ORDER BY e.offset"))?;
    let mut rows = stmt.query(params![input, chain])?;
    let mut end = 0u64;
    let mut count = 0;
    let mut blob = None;
    while let Some(row) = rows.next()? {
        let offset = row.get::<_, i64>(0)?;
        let size = row.get::<_, i64>(1)?;
        let stored = row.get::<_, Option<i64>>(4)?;
        let at = row.get::<_, i64>(3)?;
        if offset < 0
            || size < 0
            || at < 0
            || offset as u64 != end
            || stored.is_none_or(|stored| {
                stored < 0 || at.checked_add(size).is_none_or(|end| end > stored)
            })
        {
            return Err(StoreError::Invalid(
                "the asset's stored extents are incomplete".into(),
            ));
        }
        let bytes: Vec<u8> = row.get(2)?;
        let current = polylogue_model::BlobHash(
            bytes
                .try_into()
                .map_err(|_| StoreError::Invalid("invalid asset extent hash".into()))?,
        );
        count += 1;
        end = end
            .checked_add(size as u64)
            .ok_or_else(|| StoreError::Invalid("asset extent overflow".into()))?;
        hash.update(&offset.to_le_bytes());
        hash.update(&size.to_le_bytes());
        hash.update(&current.0);
        hash.update(&at.to_le_bytes());
        blob = if count == 1
            && offset == 0
            && at == 0
            && size as u64 == length
            && stored == Some(size)
        {
            Some(current)
        } else {
            None
        };
    }
    if end != length || count == 0 {
        return Err(StoreError::Invalid(
            "the asset's stored extents are incomplete".into(),
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
    /// remain separate: equal bytes can have several physical memberships.
    pub fn unindexed_import_members(&self) -> Result<(Vec<ObservedMember>, Option<i64>)> {
        let conn = self.conn()?;
        let after: i64 = conn
            .query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='import_observation_row'",
                [],
                |r| r.get(0),
            )
            .optional()?
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
        let mut stmt = conn.prepare(
            "SELECT DISTINCT m.input,m.sidecar,s.version,s.source_version FROM import_member m
            LEFT JOIN import_asset_sidecar s ON s.input=m.input AND s.sidecar=m.sidecar
            WHERE m.sidecar IS NOT NULL
              AND EXISTS(SELECT 1 FROM source_revision r WHERE r.input=m.input)
            ORDER BY m.input,m.sidecar",
        )?;
        let mut pending = Vec::new();
        for row in stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<u32>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })? {
            let (input, sidecar, version, recorded) = row?;
            let current = asset_source(conn, conn, "source_", &input)?.map(|source| source.version);
            if version != Some(RULES) || recorded != current {
                pending.push((input, sidecar));
            }
        }
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
        let mut statement = conn.prepare_cached(
            "SELECT DISTINCT im.import_id, coalesce(c.changed,0)
             FROM session s JOIN message m ON m.session_pk=s.pk
             JOIN message_source ms ON ms.message_pk=m.pk AND ms.shown=1
             JOIN input_ref i ON i.pk=ms.input_pk
             JOIN import_member im ON im.input=i.id
             LEFT JOIN import_change c ON c.import_id=im.import_id
             WHERE s.id=?1 ORDER BY im.import_id",
        )?;
        let mut rows = statement.query([session.to_string()])?;
        let mut digest = blake3::Hasher::new();
        let mut any = false;
        while let Some(row) = rows.next()? {
            any = true;
            let import: String = row.get(0)?;
            let changed: i64 = row.get(1)?;
            digest.update(&(import.len() as u64).to_le_bytes());
            digest.update(import.as_bytes());
            digest.update(&changed.to_le_bytes());
        }
        self.check()?;
        Ok(any.then(|| polylogue_model::BlobHash(*digest.finalize().as_bytes())))
    }
}

/// Visits import metadata inside the caller's source snapshot. Rowids are only
/// a discovery cursor; replay selects each changed input from zero after source
/// rematerialization, whose local rowids need not preserve their old order.
pub(crate) fn visit_import_members(
    conn: &Connection,
    prefix: &str,
    after: i64,
    input: Option<&str>,
    mut visit: impl FnMut(ObservedMember) -> Result<()>,
) -> Result<Option<i64>> {
    let through: i64 = conn.query_row(
        &format!("SELECT COALESCE(max(rowid),0) FROM {prefix}observation"),
        [],
        |row| row.get(0),
    )?;
    // A ZIP is the source of its members, not one of the attachments among them.
    let mut originals =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    let mut roots = conn.prepare(&format!(
        "SELECT DISTINCT c.container_input,json_extract(o.details_json,'$.import.import_id')
        FROM {prefix}observation o JOIN {prefix}observation_container c ON c.observation_id=o.id
        WHERE o.rowid>?1 AND o.rowid<=?2 AND (?3 IS NULL OR c.container_input=?3)
          AND json_type(o.details_json,'$.import.import_id')='text'"
    ))?;
    for row in roots.query_map(params![after, through, input], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })? {
        let (input, import_id) = row?;
        originals.entry(input).or_default().insert(import_id);
    }
    let mut observations = conn.prepare(&format!(
        "SELECT o.id,o.input,o.details_json FROM {prefix}observation o
        WHERE o.rowid>?1 AND o.rowid<=?2 AND (?3 IS NULL OR o.input=?3)
          AND o.details_json IS NOT NULL ORDER BY o.rowid"
    ))?;
    for row in observations.query_map(params![after, through, input], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })? {
        let (observation, input, details) = row?;
        let details: serde_json::Value = serde_json::from_str(&details)?;
        if let Some(member) = details.get("import") {
            let membership: ImportMember = serde_json::from_value(member.clone())?;
            if originals
                .get(&input)
                .is_some_and(|imports| imports.contains(&membership.import_id))
            {
                continue;
            }
            if membership.version != 1
                || membership.import_id.is_empty()
                || membership.member.is_empty()
            {
                return Err(StoreError::Invalid(
                    "unsupported or invalid import membership".into(),
                ));
            }
            visit(ObservedMember {
                observation,
                input,
                membership,
            })?;
        }
    }
    Ok((through != after).then_some(through))
}

struct StoredMember {
    namespace: String,
    input: String,
    member: String,
    source_id: Option<String>,
    sidecar: Option<String>,
}

/// The same actual-member projection is used by import refresh and confirmed replay.
pub(crate) fn project_member(
    conn: &Connection,
    source: &ObservedMember,
    source_id: Option<&str>,
    sidecar: Option<&str>,
) -> Result<()> {
    let previous: Option<StoredMember> = conn.query_row(
        "SELECT import_id,input,member,source_id,sidecar FROM import_member WHERE observation=?1",
        [&source.observation], |row| Ok(StoredMember { namespace: row.get(0)?, input: row.get(1)?, member: row.get(2)?, source_id: row.get(3)?, sidecar: row.get(4)? })).optional()?;
    if let Some(StoredMember {
        namespace,
        input,
        member,
        source_id: known_id,
        sidecar: known_sidecar,
    }) = previous
    {
        if namespace != source.membership.import_id
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
            return Err(StoreError::Conflict(
                "conflicting identity or classification for one import member".into(),
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
         SELECT DISTINCT import_id,1 FROM import_member WHERE input=?1 AND true
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
    let catalog_prefix = match crate::source::Source::of(archive)? {
        crate::source::Source::Cache => "source_",
        crate::source::Source::Ledger => "",
    };
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
                let ordinal = i64::try_from(ordinal)
                    .map_err(|_| StoreError::Invalid("too many asset metadata entries".into()))?;
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
                    return Err(StoreError::Conflict(
                        "asset sidecar changed while it was parsed".into(),
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
                    "INSERT OR REPLACE INTO meta VALUES ('import_observation_row',?1)",
                    [through.to_string()],
                )?;
            }
        }
    }
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests;
