//! Derived import associations. Source interpretation is supplied by the
//! engine; reads join the admitted archive state before returning any bytes.
use crate::{ImportMember, Reader, Result, StoreError, export_items::ExportItems};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

mod read;
pub use read::{AssetCandidate, AttachmentResolution, AttachmentState};

pub const RULES: u32 = 1;

pub(crate) fn invalidate_metadata(conn: &Connection, input: &str) -> Result<()> {
    conn.execute("DELETE FROM import_asset_metadata WHERE input=?1", [input])?;
    conn.execute("DELETE FROM import_asset_sidecar WHERE input=?1", [input])?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct AssetBasis {
    pub chain: i64,
    pub length: u64,
    pub hash: String,
    pub blob: Option<polylogue_model::BlobHash>,
}

fn basis(conn: &Connection, schema: &str, input: &str) -> Result<Option<AssetBasis>> {
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
    let mut hash = blake3::Hasher::new();
    hash.update(&chain.to_le_bytes());
    hash.update(&length.to_le_bytes());
    let mut stmt=conn.prepare_cached(&format!("SELECT e.offset,e.length,e.blob,e.blob_offset,b.size,b.state
        FROM {schema}extent e LEFT JOIN {schema}blob b ON b.hash=e.blob WHERE e.input=?1 AND e.chain=?2 ORDER BY e.offset"))?;
    let mut rows = stmt.query(params![input, chain])?;
    let mut end = 0u64;
    let mut count = 0;
    let mut blob = None;
    while let Some(row) = rows.next()? {
        let offset = row.get::<_, i64>(0)?;
        let size = row.get::<_, i64>(1)?;
        let stored = row.get::<_, Option<i64>>(4)?;
        let state = row.get::<_, Option<String>>(5)?;
        let at = row.get::<_, i64>(3)?;
        if offset < 0
            || size < 0
            || at < 0
            || offset as u64 != end
            || state.as_deref() != Some("present")
            || stored.is_none_or(|stored| {
                stored < 0 || at.checked_add(size).is_none_or(|end| end > stored)
            })
        {
            return Err(StoreError::Invalid(
                "retained asset extent catalog is incomplete".into(),
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
            "retained asset extent catalog is incomplete".into(),
        ));
    }
    Ok(Some(AssetBasis {
        chain,
        length,
        hash: hash.finalize().to_hex().to_string(),
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
        basis: String,
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
    pub fn asset_basis(&self, input: &str) -> Result<Option<AssetBasis>> {
        basis(self.conn()?, "a.", input)
    }
    /// Only unseen acquisition metadata is decoded. Content and observations
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
        let through: i64 = conn.query_row(
            "SELECT COALESCE(max(rowid),0) FROM a.observation",
            [],
            |r| r.get(0),
        )?;
        // A ZIP is the source of its members, not one of the attachments among them.
        let mut originals =
            std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
        let mut roots=conn.prepare("SELECT DISTINCT c.container_input,json_extract(o.details_json,'$.import.acquisition') FROM a.observation o
            JOIN a.observation_container c ON c.observation_id=o.id WHERE o.rowid>?1 AND o.rowid<=?2")?;
        for row in roots.query_map([after, through], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (input, acquisition) = row?;
            originals.entry(input).or_default().insert(acquisition);
        }
        let mut stmt = conn.prepare(
            "SELECT o.id,o.input,o.details_json FROM a.observation o
            WHERE o.rowid>?1 AND o.rowid<=?2 AND o.details_json IS NOT NULL AND o.class<>'excised'",
        )?;
        let mut out = Vec::new();
        for row in stmt.query_map([after, through], |row| {
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
                    .is_some_and(|acquisitions| acquisitions.contains(&membership.acquisition))
                {
                    continue;
                }
                if membership.version != 1
                    || membership.acquisition.is_empty()
                    || membership.member.is_empty()
                {
                    return Err(StoreError::Invalid(
                        "unsupported or invalid import membership".into(),
                    ));
                }
                out.push(ObservedMember {
                    observation,
                    input,
                    membership,
                });
            }
        }
        Ok((out, (through != after).then_some(through)))
    }

    pub fn pending_asset_sidecars(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT m.input,m.sidecar,s.version,s.basis FROM import_member m
            LEFT JOIN import_asset_sidecar s ON s.input=m.input AND s.sidecar=m.sidecar
            WHERE m.sidecar IS NOT NULL
              AND EXISTS(SELECT 1 FROM a.revision r WHERE r.input=m.input)
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
            let current = basis(conn, "a.", &input)?.map(|basis| basis.hash);
            if version != Some(RULES) || recorded != current {
                pending.push((input, sidecar));
            }
        }
        Ok(pending)
    }
}

pub(crate) fn install(
    conn: &mut Connection,
    archive: &Connection,
    mut staged: ExportItems,
) -> Result<()> {
    let tx = crate::publication::transaction(conn, rusqlite::TransactionBehavior::Immediate)?;
    while let Some(item) = staged.next_item()? {
        match serde_json::from_reader(item.reader)? {
            AssetRow::Member {
                source,
                source_id,
                sidecar,
            } => {
                tx.execute(
                    "INSERT OR IGNORE INTO import_member VALUES (?1,?2,?3,?4,?5,?6)",
                    params![
                        source.observation,
                        source.membership.acquisition,
                        source.input,
                        source.membership.member,
                        source_id,
                        sidecar
                    ],
                )?;
            }
            AssetRow::Metadata {
                input,
                sidecar,
                ordinal,
                source_id,
                name,
                media_type,
                size,
            } => {
                tx.execute(
                    "INSERT OR REPLACE INTO import_asset_metadata VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        input,
                        sidecar,
                        i64::try_from(ordinal).map_err(|_| StoreError::Invalid(
                            "too many asset metadata entries".into()
                        ))?,
                        source_id,
                        name,
                        media_type,
                        size.map(|n| n.to_string())
                    ],
                )?;
            }
            AssetRow::Sidecar {
                input,
                sidecar,
                basis: recorded,
                error,
            } => {
                if basis(archive, "", &input)?
                    .as_ref()
                    .map(|basis| basis.hash.as_str())
                    != Some(recorded.as_str())
                {
                    return Err(StoreError::Conflict(
                        "asset sidecar changed while it was interpreted".into(),
                    ));
                }
                tx.execute(
                    "INSERT OR REPLACE INTO import_asset_sidecar VALUES (?1,?2,?3,?4,?5)",
                    params![input, sidecar, RULES, recorded, error],
                )?;
            }
            AssetRow::ResetMetadata { input, sidecar } => {
                tx.execute(
                    "DELETE FROM import_asset_metadata WHERE input=?1 AND sidecar=?2",
                    params![input, sidecar],
                )?;
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
