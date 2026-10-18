//! Read-only physical Codex history snapshots. Every SQLite operation stays here.
use crate::{BlobCapture, Error, PreparedBlob, ReadHandle, Result, StoredSource, db};
use polylogue_model::tool_item::HistoryRow;
use rusqlite::{Connection, OpenFlags};
use std::{
    io::Write,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub const SCHEMA: &str = "codex-rust-v0.162.1/thread-history-migrations-1-through-8";
const ITEM_COLUMNS: &[(&str, &str, i64)] = &[
    ("thread_id", "TEXT", 1),
    ("turn_id", "TEXT", 2),
    ("item_id", "TEXT", 3),
    ("rollout_ordinal", "INTEGER", 0),
    ("created_at_ms", "INTEGER", 0),
    ("item_json", "TEXT", 0),
    ("item_type", "TEXT", 0),
    ("updated_at_ordinal", "INTEGER", 0),
    ("started_at_ms", "INTEGER", 0),
    ("completed_at_ms", "INTEGER", 0),
];
fn validate(c: &Connection, p: &Path) -> Result<()> {
    let columns = db(p, || {
        let mut s = c.prepare("PRAGMA table_info(thread_items)")?;
        s.query_map([], |r| {
            Ok((
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(5)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
    })?;
    if columns.len() != ITEM_COLUMNS.len()
        || columns
            .iter()
            .zip(ITEM_COLUMNS)
            .any(|(a, b)| a.0 != b.0 || a.1 != b.1 || a.2 != b.2)
    {
        return Err(Error::Invariant(format!(
            "no_parser_yet: unsupported Codex history schema; expected {SCHEMA}"
        )));
    }
    let required: i64 = db(p, || {
        c.query_row("SELECT count(*) FROM pragma_table_info('thread_turns') WHERE name='root_turn_id' AND type='TEXT'",[],|r|r.get(0))
    })?;
    let ledger: bool = db(p, || {
        c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type='table' AND name='_sqlx_migrations')", [], |r|r.get(0))
    })?;
    if !ledger {
        return Err(Error::Invariant(
            "no_parser_yet: missing Codex history migration ledger".into(),
        ));
    }
    let migrations: Vec<(i64, bool)> = db(p, || {
        let mut s = c.prepare("SELECT version,success FROM _sqlx_migrations ORDER BY version")?;
        s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect()
    })?;
    if required != 1 || migrations != (1..=8).map(|n| (n, true)).collect::<Vec<_>>() {
        return Err(Error::Invariant(
            "no_parser_yet: unsupported Codex history migration ledger".into(),
        ));
    }
    Ok(())
}
impl BlobCapture {
    /// Explicitly supplied database only. A read transaction is established
    /// before backup, so even a writer cannot mix page generations. The source
    /// is opened read-only; no native state or projection is repaired.
    pub fn capture_history_snapshot(
        &self,
        path: &Path,
        cancelled: &AtomicBool,
    ) -> Result<PreparedBlob> {
        Ok(self
            .capture_history_pages(path, None, cancelled)?
            .into_plain())
    }
}
impl ReadHandle {
    /// Decode only the exact retained physical bytes. A current source path is
    /// never consulted. This first adapter admits a bounded snapshot per turn.
    pub fn codex_history_rows(
        &self,
        input: &StoredSource,
        cancelled: &AtomicBool,
    ) -> Result<Vec<HistoryRow>> {
        const MAX: u64 = 16 * 1024 * 1024;
        if input.input.length > MAX {
            return Err(Error::ParserAdmission {
                resource: "physical history snapshot bytes",
                required: input.input.length,
            });
        }
        let mut temp = tempfile::NamedTempFile::new()?;
        let mut offset = 0;
        while offset < input.input.length {
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let bytes = self.read_source_range(
                &input.input.id,
                input.input.chain,
                input.binding,
                offset,
                (input.input.length - offset).min(65536),
            )?;
            if bytes.is_empty() {
                return Err(Error::Invariant(
                    "history retained snapshot is incomplete".into(),
                ));
            }
            temp.write_all(&bytes)?;
            offset += bytes.len() as u64;
        }
        temp.flush()?;
        let p = temp.path();
        let c = db(p, || {
            Connection::open_with_flags(
                p,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
        })?;
        db(p, || c.execute_batch("PRAGMA query_only=ON; BEGIN"))?;
        validate(&c, p)?;
        let mut statement = db(p, || {
            c.prepare("SELECT thread_id,turn_id,item_id,rollout_ordinal,updated_at_ordinal,created_at_ms,started_at_ms,completed_at_ms,item_type,item_json FROM thread_items ORDER BY thread_id,turn_id,item_id")
        })?;
        let mut query = db(p, || statement.query([]))?;
        let mut rows = Vec::new();
        let mut bytes = 0usize;
        while let Some(r) = db(p, || query.next())? {
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let row = db(p, || {
                Ok(HistoryRow {
                    thread_id: r.get(0)?,
                    turn_id: r.get(1)?,
                    item_id: r.get(2)?,
                    rollout_ordinal: r.get(3)?,
                    updated_at_ordinal: r.get(4)?,
                    created_at_ms: r.get(5)?,
                    started_at_ms: r.get(6)?,
                    completed_at_ms: r.get(7)?,
                    item_type: r.get(8)?,
                    item_json: r.get(9)?,
                })
            })?;
            bytes = bytes.saturating_add(row.item_json.len());
            if bytes > MAX as usize || rows.len() >= 4096 {
                return Err(Error::ParserAdmission {
                    resource: "history typed rows",
                    required: bytes as u64,
                });
            }
            rows.push(row);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn schema_pin_refuses_a_missing_or_newer_ledger_and_changed_row_shape() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(include_str!("../tests/fixtures/codex-history-v0162.sql"))
            .unwrap();
        let path = Path::new("synthetic-history");
        assert!(
            validate(&c, path)
                .unwrap_err()
                .to_string()
                .contains("no_parser_yet")
        );
        c.execute_batch(
            "CREATE TABLE _sqlx_migrations(version INTEGER PRIMARY KEY,success INTEGER NOT NULL)",
        )
        .unwrap();
        for n in 1..=8 {
            c.execute("INSERT INTO _sqlx_migrations VALUES(?1,1)", [n])
                .unwrap();
        }
        validate(&c, path).unwrap();
        c.execute("INSERT INTO _sqlx_migrations VALUES(9,0)", [])
            .unwrap();
        assert!(
            validate(&c, path)
                .unwrap_err()
                .to_string()
                .contains("no_parser_yet")
        );
        c.execute("DELETE FROM _sqlx_migrations WHERE version=9", [])
            .unwrap();
        c.execute("ALTER TABLE thread_items ADD COLUMN unexpected TEXT", [])
            .unwrap();
        assert!(
            validate(&c, path)
                .unwrap_err()
                .to_string()
                .contains("no_parser_yet")
        );
    }

    #[test]
    fn physical_snapshot_includes_committed_wal_and_excludes_an_open_write() {
        let root = tempfile::tempdir().unwrap();
        let source_path = root.path().join("source.sqlite");
        let source = Connection::open(&source_path).unwrap();
        source.execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE evidence(value INTEGER); INSERT INTO evidence VALUES(1); BEGIN IMMEDIATE; UPDATE evidence SET value=2").unwrap();
        let archive = root.path().join("archive");
        let writer = crate::Writer::open(&archive).unwrap();
        let captured = writer
            .blob_capture()
            .unwrap()
            .capture_history_snapshot(&source_path, &AtomicBool::new(false))
            .unwrap();
        let reader = hashed_blobs::Reader::new(&archive, vec![]);
        let mut raw = Vec::new();
        reader
            .read(&hashed_blobs::Hash(*captured.hash().bytes()))
            .unwrap()
            .read_to_end(&mut raw)
            .unwrap();
        assert_eq!(raw.len() as u64, captured.length());
        let snapshot_path = root.path().join("snapshot.sqlite");
        std::fs::write(&snapshot_path, raw).unwrap();
        let snapshot =
            Connection::open_with_flags(&snapshot_path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        assert_eq!(
            snapshot
                .query_row("SELECT value FROM evidence", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            source
                .query_row("SELECT value FROM evidence", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2
        );
        source.execute_batch("ROLLBACK").unwrap();
    }
}
