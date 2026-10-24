//! Excision owns the identities and current replacement batches of the units
//! it changes. The manifest holds references, never before-images or user text.

use super::{Unit, read};
use crate::{Result, StoreError};
use rusqlite::{Connection, Transaction};
use serde::{Deserialize, Serialize};
mod decode;
mod digest;
mod retirement;
pub(crate) use retirement::CurrentBatches;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replacement {
    pub unit: Unit,
    /// Content-addressed replacement batch, or a tombstone for a removed unit.
    pub batch: Option<String>,
    pub bytes: u64,
    pub rows: u64,
    pub blobs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcisionPublication {
    pub version: u32,
    pub replacements: Vec<Replacement>,
    /// Earlier local pending batches no longer selected for their units.
    pub retired_batches: Vec<String>,
    pub digest: String,
}

pub(crate) struct Capture;

impl Capture {
    pub(crate) fn start(tx: &Transaction<'_>, request: &str) -> Result<Self> {
        tx.execute_batch(
            "CREATE TEMP TABLE excision_units (
            commit_seq INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
            PRIMARY KEY(commit_seq, kind, key));
            CREATE TEMP TABLE excision_request(id TEXT NOT NULL);",
        )?;
        tx.execute("INSERT INTO excision_request VALUES (?1)", [request])?;
        for (table, event, body) in triggers() {
            let name = event.replace(' ', "_");
            tx.execute_batch(&format!(
                "CREATE TEMP TRIGGER excision_{table}_{name} {event} ON main.{table}
                 BEGIN {body} END;"
            ))?;
        }
        Ok(Self)
    }

    pub(crate) fn finish(
        self,
        tx: &Transaction<'_>,
        store: &hashed_blobs::Store,
    ) -> Result<ExcisionPublication> {
        let mut statement = tx.prepare(
            "SELECT commit_seq,kind,key FROM excision_units ORDER BY commit_seq,kind,key",
        )?;
        let units = statement.query_map([], |r| {
            Ok(Unit {
                commit_seq: r.get(0)?,
                kind: r.get(1)?,
                key: r.get(2)?,
            })
        })?;
        let local = store.primary_only();
        let mut replacements = Vec::new();
        for unit in units {
            let unit = unit?;
            let mut batch = read(tx, &unit)?;
            let hash = if batch.rows == 0 {
                None
            } else {
                Some(format!(
                    "blake3:{}",
                    local
                        .put_plain_checked::<StoreError>(
                            &mut batch.file,
                            crate::blobs::LEVEL,
                            || Ok(())
                        )?
                        .hex()
                ))
            };
            replacements.push(Replacement {
                unit,
                batch: hash,
                bytes: batch.bytes,
                rows: batch.rows,
                blobs: batch
                    .blobs
                    .iter()
                    .map(|h| format!("blake3:{}", super::hex(h)))
                    .collect(),
            });
        }
        drop(statement);
        let retired = retirement::batches(tx, &replacements)?;
        for (table, event, _) in triggers() {
            let name = event.replace(' ', "_");
            tx.execute_batch(&format!("DROP TRIGGER excision_{table}_{name};"))?;
        }
        tx.execute_batch("DROP TABLE excision_units; DROP TABLE excision_request;")?;
        let mut publication = ExcisionPublication {
            version: 1,
            replacements,
            retired_batches: retired,
            digest: String::new(),
        };
        publication.digest = publication.computed_digest()?;
        Ok(publication)
    }
}

/// Decode and check one historical manifest at a time. Callers use its owner's
/// identity to choose the latest replacement, independent of scan order.
pub(super) fn visit_publications(
    conn: &Connection,
    mut visit: impl FnMut(&Unit, &decode::View<'_>) -> Result<()>,
) -> Result<()> {
    let mut stmt = conn.prepare("SELECT commit_seq,id,refused_json FROM excision")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let json = row.get_ref(2)?;
        let json = json
            .as_str()
            .map_err(|e| StoreError::Invalid(e.to_string()))?;
        if let Some(publication) = decode::View::decode(json)? {
            let unit = Unit {
                commit_seq: row.get(0)?,
                kind: "excision".into(),
                key: row.get(1)?,
            };
            visit(&unit, &publication)?;
        }
    }
    Ok(())
}

fn triggers() -> Vec<(&'static str, &'static str, String)> {
    let mut out = Vec::new();
    for (table, kind) in [
        ("note", "note"),
        ("delivery", "delivery"),
        ("excision", "excision"),
    ] {
        for event in ["AFTER UPDATE", "BEFORE DELETE"] {
            let filter = if table == "excision" {
                " WHERE OLD.id <> (SELECT id FROM excision_request)"
            } else {
                ""
            };
            out.push((table, event, format!(
                "INSERT OR IGNORE INTO excision_units SELECT OLD.commit_seq, '{kind}', OLD.id{filter};"
            )));
        }
    }
    for event in ["AFTER UPDATE", "BEFORE DELETE"] {
        out.push((
            "extracted_blob",
            event,
            format!(
                "INSERT OR IGNORE INTO excision_units SELECT OLD.commit_seq,'extracted_blob',{};",
                crate::retention::extracted::key_sql("OLD")
            ),
        ));
    }
    for table in ["note_target", "note_relation"] {
        for event in ["AFTER UPDATE", "BEFORE DELETE", "AFTER INSERT"] {
            let row = if event == "AFTER INSERT" {
                "NEW"
            } else {
                "OLD"
            };
            out.push((table, event, format!(
                "INSERT OR IGNORE INTO excision_units SELECT commit_seq, 'note', id FROM note WHERE id = {row}.note_id;"
            )));
        }
    }
    for event in ["AFTER UPDATE", "BEFORE DELETE", "AFTER INSERT"] {
        let row = if event == "AFTER INSERT" {
            "NEW"
        } else {
            "OLD"
        };
        out.push(("delivery_event", event, format!(
            "INSERT OR IGNORE INTO excision_units SELECT commit_seq, 'delivery', id FROM delivery WHERE id = {row}.delivery;
 INSERT OR IGNORE INTO excision_units SELECT {row}.commit_seq,'delivery_event',json_array({row}.delivery,{row}.what,{row}.at) WHERE {row}.commit_seq>0;"
        )));
        out.push(("extent", event, format!(
            "INSERT OR IGNORE INTO excision_units SELECT commit_seq, 'revision', input || '/' || number
             FROM revision WHERE input = {row}.input AND chain = {row}.chain AND length > {row}.offset;"
        )));
    }
    for event in ["AFTER UPDATE", "BEFORE DELETE"] {
        out.push(("observation",event,
            "INSERT OR IGNORE INTO excision_units SELECT OLD.commit_seq,'observation',OLD.id WHERE OLD.commit_seq>0;
             INSERT OR IGNORE INTO excision_units SELECT commit_seq,'revision',input||'/'||number FROM revision WHERE input=OLD.input AND number=OLD.revision;
 INSERT OR IGNORE INTO excision_units SELECT commit_seq,'excision',id FROM excision WHERE id=OLD.excision AND id<>(SELECT id FROM excision_request);".into()));
    }
    out.push(("observation_container", "BEFORE DELETE",
        "INSERT OR IGNORE INTO excision_units SELECT commit_seq,'observation',id FROM observation WHERE id=OLD.observation_id AND commit_seq>0;".into()));
    out.push(("revision", "BEFORE DELETE", "INSERT OR IGNORE INTO excision_units SELECT OLD.commit_seq, 'revision', OLD.input || '/' || OLD.number;".into()));
    out
}

impl ExcisionPublication {
    pub(super) fn computed_digest(&self) -> Result<String> {
        digest::compute(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != 1 || self.digest != self.computed_digest()? {
            return Err(StoreError::Invalid(
                "excision replacement manifest failed its version/hash check".into(),
            ));
        }
        let mut previous = None;
        for replacement in &self.replacements {
            replacement.validate(previous)?;
            previous = Some(&replacement.unit);
        }
        for hash in &self.retired_batches {
            batch_hash(hash)?;
        }
        Ok(())
    }
}

impl Replacement {
    fn validate(&self, previous: Option<&Unit>) -> Result<()> {
        let unit = &self.unit;
        if unit.commit_seq < 1
            || unit.key.is_empty()
            || ![
                "revision",
                "note",
                "excision",
                "setting",
                "delivery",
                "observation",
                "delivery_event",
                "extracted_blob",
            ]
            .contains(&unit.kind.as_str())
            || previous.is_some_and(|previous| previous >= unit)
        {
            return Err(StoreError::Invalid(
                "replacement units are not unique known identities in archive order".into(),
            ));
        }
        match &self.batch {
            Some(hash) => {
                batch_hash(hash)?;
                if self.rows == 0 || self.bytes == 0 {
                    return Err(StoreError::Invalid("a replacement batch is empty".into()));
                }
            }
            None if self.bytes != 0 || self.rows != 0 || !self.blobs.is_empty() => {
                return Err(StoreError::Invalid(
                    "a tombstone cites retained bytes".into(),
                ));
            }
            None => {}
        }
        for blob in &self.blobs {
            batch_hash(blob)?;
        }
        Ok(())
    }
}

pub(super) fn batch_hash(name: &str) -> Result<hashed_blobs::Hash> {
    let hex = name
        .strip_prefix("blake3:")
        .ok_or_else(|| StoreError::Invalid("a batch has no BLAKE3 name".into()))?;
    Ok(hashed_blobs::Hash(super::unhex(hex)?.try_into().map_err(
        |_| StoreError::Invalid("a batch name is not 32 bytes".into()),
    )?))
}

pub(crate) fn visit_retired_local_batches(
    conn: &Connection,
    publication: &ExcisionPublication,
    mut visit: impl FnMut(polylogue_model::BlobHash) -> Result<()>,
) -> Result<()> {
    for batch in &publication.retired_batches {
        let hash = batch_hash(batch)?;
        // The catalog owns source bytes, including attachments and delta bases
        // which need not have an extent. Only uncataloged pending batches retire.
        let retained: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM blob WHERE hash = ?1)",
            [&hash.0[..]],
            |r| r.get(0),
        )?;
        if !retained {
            visit(polylogue_model::BlobHash(hash.0))?;
        }
    }
    Ok(())
}

pub(super) fn publication(conn: &Connection, unit: &Unit) -> Result<Option<ExcisionPublication>> {
    if unit.kind != "excision" {
        return Ok(None);
    }
    let json: String = conn.query_row(
        "SELECT refused_json FROM excision WHERE id = ?1",
        [&unit.key],
        |r| r.get(0),
    )?;
    let publication = decode::publication(&json)?
        .ok_or_else(|| StoreError::Invalid("legacy excision has no replacement manifest".into()))?;
    Ok(Some(publication))
}
