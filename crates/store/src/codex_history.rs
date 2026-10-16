//! Read-only physical Codex history snapshots. Every SQLite operation stays here.
use crate::{BlobCapture, Error, PreparedBlob, ReadHandle, Result, RetainedSource, db};
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
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.is_file() {
            return Err(Error::Invariant(
                "history snapshot requires a regular non-symlink file".into(),
            ));
        }
        let source = db(path, || {
            Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
        })?;
        db(path, || source.execute_batch("PRAGMA query_only=ON; BEGIN"))?;
        let _: i64 = db(path, || {
            source.query_row("PRAGMA schema_version", [], |r| r.get(0))
        })?;
        let temp = tempfile::NamedTempFile::new()?;
        let mut dest = db(temp.path(), || Connection::open(temp.path()))?;
        {
            let backup = db(path, || rusqlite::backup::Backup::new(&source, &mut dest))?;
            loop {
                if cancelled.load(Ordering::Acquire) {
                    return Err(Error::Cancelled);
                }
                match db(path, || backup.step(128))? {
                    rusqlite::backup::StepResult::Done => break,
                    rusqlite::backup::StepResult::More => {}
                    _ => {
                        return Err(Error::Invariant(
                            "history snapshot backup busy; retry explicit import".into(),
                        ));
                    }
                }
            }
        }
        dest.close().map_err(|(_, e)| Error::Sqlite {
            file: temp.path().into(),
            source: e,
        })?;
        db(path, || source.execute_batch("ROLLBACK"))?;
        self.capture_regular_file(temp.path(), cancelled)
    }
}
impl ReadHandle {
    /// Decode only the exact retained physical bytes. A current source path is
    /// never consulted. This first adapter admits a bounded snapshot per turn.
    pub fn codex_history_rows(
        &self,
        input: &RetainedSource,
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
