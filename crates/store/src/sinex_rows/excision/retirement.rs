//! Select local pending batches that no current replacement still references.
use super::{Replacement, visit_publications};
use crate::Result;
use rusqlite::{Connection, params};

pub(crate) struct CurrentBatches {
    db: Connection,
    _file: tempfile::NamedTempFile,
}

impl CurrentBatches {
    pub(crate) fn new(source: &Connection) -> Result<Self> {
        let current = Self::scan(source, |_, _| Ok(()))?;
        current.db.execute_batch(
            "CREATE TABLE held(hash TEXT PRIMARY KEY) WITHOUT ROWID;
             INSERT OR IGNORE INTO held SELECT batch FROM current WHERE batch IS NOT NULL;
             INSERT OR IGNORE INTO held SELECT j.value FROM current c, json_each(c.blobs) j;
             COMMIT;",
        )?;
        Ok(current)
    }

    pub(crate) fn contains(&self, hash: &polylogue_model::BlobHash) -> Result<bool> {
        Ok(self.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM held WHERE hash=?1)",
            [format!("blake3:{hash}")],
            |r| r.get(0),
        )?)
    }

    fn scan(
        source: &Connection,
        mut visit: impl FnMut(&Connection, &Replacement) -> Result<()>,
    ) -> Result<Self> {
        let file = tempfile::NamedTempFile::new()?;
        let db = Connection::open(file.path())?;
        db.execute_batch(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
             PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
             CREATE TABLE current(seq INTEGER NOT NULL,kind TEXT NOT NULL,key TEXT NOT NULL,
                owner_seq INTEGER NOT NULL,owner_key TEXT NOT NULL,batch TEXT,blobs TEXT NOT NULL,
                PRIMARY KEY(seq,kind,key)) WITHOUT ROWID;
             CREATE TABLE retired(batch TEXT PRIMARY KEY) WITHOUT ROWID;
             BEGIN;",
        )?;
        {
            let mut insert = db.prepare(
                "INSERT INTO current VALUES (?1,?2,?3,?4,?5,?6,?7)
                 ON CONFLICT(seq,kind,key) DO UPDATE SET owner_seq=excluded.owner_seq,
                    owner_key=excluded.owner_key,batch=excluded.batch,blobs=excluded.blobs
                 WHERE (excluded.owner_seq,excluded.owner_key)>(current.owner_seq,current.owner_key)",
            )?;
            visit_publications(source, |owner, publication| {
                for replacement in &publication.replacements {
                    let unit = &replacement.unit;
                    insert.execute(params![
                        unit.commit_seq,
                        unit.kind,
                        unit.key,
                        owner.commit_seq,
                        owner.key,
                        replacement.batch,
                        serde_json::to_string(&replacement.blobs)?
                    ])?;
                    visit(&db, replacement)?;
                }
                Ok(())
            })?;
        }
        Ok(Self { db, _file: file })
    }
}

pub(super) fn batches(source: &Connection, replacements: &[Replacement]) -> Result<Vec<String>> {
    let mut changed = source.prepare(
        "SELECT EXISTS(SELECT 1 FROM excision_units WHERE commit_seq=?1 AND kind=?2 AND key=?3)",
    )?;
    let current = CurrentBatches::scan(source, |db, replacement| {
        let unit = &replacement.unit;
        let affected: bool =
            changed.query_row(params![unit.commit_seq, unit.kind, unit.key], |r| r.get(0))?;
        if affected && let Some(batch) = &replacement.batch {
            db.execute("INSERT OR IGNORE INTO retired VALUES (?1)", [batch])?;
        }
        Ok(())
    })?;
    let db = &current.db;
    // These replacements are the operation being committed, after all history.
    let mut update = db.prepare(
        "INSERT INTO current VALUES (?1,?2,?3,0,'',?4,?5)
             ON CONFLICT(seq,kind,key) DO UPDATE SET batch=excluded.batch,blobs=excluded.blobs",
    )?;
    for replacement in replacements {
        let unit = &replacement.unit;
        update.execute(params![
            unit.commit_seq,
            unit.kind,
            unit.key,
            replacement.batch,
            serde_json::to_string(&replacement.blobs)?
        ])?;
    }
    db.execute_batch(
        "DELETE FROM retired WHERE batch IN (SELECT batch FROM current WHERE batch IS NOT NULL);
        COMMIT;",
    )?;
    // The manifest itself contains this list; historical selection stays on disk.
    Ok(db
        .prepare("SELECT batch FROM retired ORDER BY batch")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests;
