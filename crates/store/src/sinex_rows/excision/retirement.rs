//! Select local pending batches that no current replacement still references.
use super::{Replacement, visit_publications};
use crate::Result;
use rusqlite::{Connection, params};

pub(super) fn batches(source: &Connection, replacements: &[Replacement]) -> Result<Vec<String>> {
    let file = tempfile::NamedTempFile::new()?;
    let db = Connection::open(file.path())?;
    db.execute_batch(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
        CREATE TABLE current(seq INTEGER NOT NULL,kind TEXT NOT NULL,key TEXT NOT NULL,
            owner_seq INTEGER NOT NULL,owner_key TEXT NOT NULL,batch TEXT,
            PRIMARY KEY(seq,kind,key)) WITHOUT ROWID;
        CREATE TABLE retired(batch TEXT PRIMARY KEY) WITHOUT ROWID;
        BEGIN;",
    )?;
    {
        let mut changed = source.prepare(
            "SELECT EXISTS(SELECT 1 FROM excision_units WHERE commit_seq=?1 AND kind=?2 AND key=?3)",
        )?;
        let mut current = db.prepare(
            "INSERT INTO current VALUES (?1,?2,?3,?4,?5,?6)
             ON CONFLICT(seq,kind,key) DO UPDATE SET owner_seq=excluded.owner_seq,
                owner_key=excluded.owner_key,batch=excluded.batch
             WHERE (excluded.owner_seq,excluded.owner_key)>(current.owner_seq,current.owner_key)",
        )?;
        let mut retired = db.prepare("INSERT OR IGNORE INTO retired VALUES (?1)")?;
        visit_publications(source, |owner, publication| {
            for replacement in &publication.replacements {
                let unit = &replacement.unit;
                current.execute(params![
                    unit.commit_seq,
                    unit.kind,
                    unit.key,
                    owner.commit_seq,
                    owner.key,
                    replacement.batch
                ])?;
                let affected: bool = changed
                    .query_row(params![unit.commit_seq, unit.kind, unit.key], |r| r.get(0))?;
                if affected && let Some(batch) = &replacement.batch {
                    retired.execute([batch])?;
                }
            }
            Ok(())
        })?;
        // These replacements are the operation being committed, after all history.
        let mut update = db.prepare(
            "INSERT INTO current VALUES (?1,?2,?3,0,'',?4)
             ON CONFLICT(seq,kind,key) DO UPDATE SET batch=excluded.batch",
        )?;
        for replacement in replacements {
            let unit = &replacement.unit;
            update.execute(params![
                unit.commit_seq,
                unit.kind,
                unit.key,
                replacement.batch
            ])?;
        }
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
