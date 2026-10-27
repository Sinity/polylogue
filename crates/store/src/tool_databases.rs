//! Tool databases as typed row dumps: a database
//! that disappears must not take archived rows with it, but every version
//! of its pages is not needed.
//!
//! A tool's SQLite database (Codex `state`, `goals`, `memories` and
//! thread history; Hermes `state.db`) is never stored as page images.
//! Each export reads it inside one read transaction, which sees a
//! consistent state including the WAL, and appends to an append-only
//! JSON Lines file in the archive:
//!
//! - `{"db", "at_ms", "schema": {table: sql}}` when the schema changes;
//! - `{"t": table, "k": key, "r": row, "h": digest}` for a row that is new
//!   or changed since the last export;
//! - `{"t": table, "k": key, "kh": key digest, "gone": true}` for a row no
//!   longer there; `kh` is the hex digest of its table and key.
//!
//! The file is an ordinary input: the loop stores its appended bytes and
//! the origin's parser reads the lines. A row the tool deletes stays in
//! the export. Values keep their SQLite types; blobs are base64.
//!
//! Exports wait for quiet files, with an hourly capture during continuous
//! writes. Comparison state persists in a SQLite catalog beside the export.
//! Declared table cursors read changes every three minutes and verify every
//! row daily; undeclared tables compare complete rows. Changed rows are staged
//! privately, then a synced marker naming the
//! completed prefix commits them, every 64 MB of rows and at the end, so a
//! large database's capture becomes input while it is read. Bytes beyond
//! that marker are removed on the next open.

use crate::FileFacts;
use base64::Engine as _;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// While a database keeps changing, it is still exported this often.
const BUSY_EXPORT: Duration = Duration::from_secs(3600);
const CURSOR_EXPORT: Duration = Duration::from_secs(180);
const VERIFY_EVERY_MS: i64 = 24 * 3600 * 1000;

/// An origin declares a column to read after, or comparison for the table.
pub type TableCursor = fn(&Path, &str) -> Option<&'static str>;

mod binding;
mod catalog;
mod prefix_hash;
mod synced_prefix;
mod turns;
use catalog::Catalog;
use synced_prefix::SyncedPrefix;
pub use turns::{CaptureTurn, Captures};

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}: {1}")]
    Sqlite(PathBuf, rusqlite::Error),
    #[error("{0}")]
    Invalid(String),
}

/// The database file and its WAL: either changing is a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DbFacts {
    db: Option<FileFacts>,
    wal: Option<FileFacts>,
}

impl DbFacts {
    fn of(db: &Path) -> Self {
        let facts = |p: &Path| fs::metadata(p).ok().map(|m| FileFacts::of(&m));
        let mut wal = db.as_os_str().to_owned();
        wal.push("-wal");
        Self {
            db: facts(db),
            wal: facts(Path::new(&wal)),
        }
    }
}

#[derive(Default)]
struct DbState {
    /// Facts seen at the previous pass, and at the last export.
    last_seen: Option<DbFacts>,
    exported: Option<DbFacts>,
    exported_at: Option<Instant>,
    catalog: Option<Catalog>,
    synced: Option<SyncedPrefix>,
    cursor_at: Option<Instant>,
}

#[derive(Default)]
pub struct Exporter {
    dbs: HashMap<PathBuf, DbState>,
    cursor: Option<TableCursor>,
}

/// What one export call did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exported {
    pub rows: u64,
    pub gone: u64,
}

impl Exporter {
    pub fn with_cursor(cursor: TableCursor) -> Self {
        Self {
            cursor: Some(cursor),
            ..Self::default()
        }
    }
    /// Exports `db` into `out` when due. Returns `None` when nothing was
    /// due.
    pub fn export(
        &mut self,
        db: &Path,
        out: &Path,
        observation: Option<&crate::Observation>,
    ) -> Result<Option<Exported>, ExportError> {
        self.export_with(db, out, observation, &mut || Ok(()))
    }

    fn export_with(
        &mut self,
        db: &Path,
        out: &Path,
        observation: Option<&crate::Observation>,
        checkpoint: &mut dyn FnMut() -> Result<(), ExportError>,
    ) -> Result<Option<Exported>, ExportError> {
        if let Some(dir) = out.parent() {
            fs::create_dir_all(dir)?;
        }
        let requested = observation.map(binding::Binding::observed).transpose()?;
        let st = self.dbs.entry(db.to_path_buf()).or_default();
        if st.synced.as_ref().is_none_or(|p| !p.current(out)) {
            st.catalog = None;
            st.synced = None;
            st.exported = None;
            let (mut catalog, synced) = Catalog::load(out, checkpoint)?;
            catalog.bind(requested.clone())?;
            catalog.save(&synced)?;
            st.synced = Some(synced);
            st.catalog = Some(catalog);
        }
        st.catalog
            .as_mut()
            .expect("loaded catalog")
            .bind(requested)?;
        let now = DbFacts::of(db);
        let quiet = st.last_seen == Some(now);
        st.last_seen = Some(now);
        let changed = st.exported != Some(now);
        let first = st.exported.is_none();
        let overdue = st.exported_at.is_none_or(|at| at.elapsed() >= BUSY_EXPORT);
        let cursor_due = st.cursor_at.is_none_or(|at| at.elapsed() >= CURSOR_EXPORT);
        let compare = changed && (first || quiet || overdue);
        if !compare && !cursor_due {
            return Ok(None);
        }
        let replaced = first
            || st
                .exported
                .and_then(|f| f.db)
                .zip(now.db)
                .is_some_and(|(old, new)| old.device != new.device || old.inode != new.inode);
        let done = match write_export(
            db,
            out,
            st,
            self.cursor,
            Due {
                compare,
                cursor: cursor_due,
                replaced,
            },
            checkpoint,
        ) {
            Ok(done) => done,
            Err(e) => {
                // The marker may have been written before a local cache failure.
                // Rebuild both from the verified synced prefix next time.
                st.catalog = None;
                st.synced = None;
                return Err(e);
            }
        };
        if compare {
            st.exported = Some(now);
            st.exported_at = Some(Instant::now());
        }
        st.cursor_at = Some(Instant::now());
        Ok(Some(done))
    }
}

/// Rows staged before they are appended to the capture, so a large
/// database's capture becomes input in parts while it is read.
const APPEND_BYTES: u64 = 64 << 20;

/// Appends the staged rows to the capture's synced prefix and empties the
/// staging file.
fn append_staged(
    w: &mut BufWriter<fs::File>,
    synced: Option<&mut SyncedPrefix>,
    out: &Path,
) -> Result<(), ExportError> {
    w.flush()?;
    let staged = w.get_mut();
    synced
        .expect("verified synced prefix")
        .append(out, staged)?;
    staged.set_len(0)?;
    staged.seek(std::io::SeekFrom::Start(0))?;
    Ok(())
}

/// The digest naming a row: its table and its key.
fn row_key(table: &str, key: &Value) -> u64 {
    digest(&format!("{table}\0{key}"))
}

fn digest(s: &str) -> u64 {
    let h = blake3::hash(s.as_bytes());
    u64::from_le_bytes(h.as_bytes()[..8].try_into().expect("8 bytes"))
}

fn value(v: ValueRef<'_>) -> Result<Value, ExportError> {
    Ok(match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => i.into(),
        ValueRef::Real(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| {
                ExportError::Invalid(
                    "non-finite SQLite REAL cannot be represented in a row dump".into(),
                )
            })?,
        ValueRef::Text(t) => std::str::from_utf8(t)
            .map_err(|_| {
                ExportError::Invalid(
                    "non-UTF-8 SQLite TEXT cannot be represented in a row dump".into(),
                )
            })?
            .into(),
        ValueRef::Blob(b) => {
            json!({ "base64": base64::engine::general_purpose::STANDARD.encode(b) })
        }
    })
}

struct Table {
    name: String,
    sql: String,
    columns: Vec<String>,
    /// Primary-key column positions; empty when the rowid is the key.
    key: Vec<usize>,
}

fn tables(conn: &Connection) -> rusqlite::Result<Vec<Table>> {
    let mut out = Vec::new();
    let mut stmt = conn.prepare(
        "SELECT name, sql FROM sqlite_master
         WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    )?;
    let names: Vec<(String, String)> = stmt
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get::<_, Option<String>>(1)?.unwrap_or_default(),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    for (name, sql) in names {
        let mut info = conn.prepare(&format!(
            "PRAGMA table_xinfo(\"{}\")",
            name.replace('"', "\"\"")
        ))?;
        let cols: Vec<(String, i64)> = info
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, i64>(6)?,
                ))
            })?
            .filter_map(|row| match row {
                Ok((_, _, 1)) => None,
                Ok((name, key, _)) => Some(Ok((name, key))),
                Err(error) => Some(Err(error)),
            })
            .collect::<rusqlite::Result<_>>()?;
        let mut key: Vec<(i64, usize)> = cols
            .iter()
            .enumerate()
            .filter(|(_, (_, pk))| *pk > 0)
            .map(|(i, (_, pk))| (*pk, i))
            .collect();
        key.sort_unstable();
        out.push(Table {
            name,
            sql,
            columns: cols.into_iter().map(|(c, _)| c).collect(),
            key: key.into_iter().map(|(_, i)| i).collect(),
        });
    }
    Ok(out)
}

struct Due {
    compare: bool,
    cursor: bool,
    replaced: bool,
}

fn write_export(
    db: &Path,
    out: &Path,
    st: &mut DbState,
    cursor: Option<TableCursor>,
    due: Due,
    checkpoint: &mut dyn FnMut() -> Result<(), ExportError>,
) -> Result<Exported, ExportError> {
    let sql_err = |e| ExportError::Sqlite(db.to_path_buf(), e);
    let conn = Connection::open_with_flags(
        db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(sql_err)?;
    conn.busy_timeout(Duration::from_secs(10))
        .map_err(sql_err)?;
    // One read transaction: every table from the same moment, WAL
    // included, without blocking the tool's writer.
    conn.execute_batch("BEGIN").map_err(sql_err)?;
    let tables = tables(&conn).map_err(sql_err)?;
    let catalog = st.catalog.as_mut().expect("loaded catalog");
    let transaction = catalog.conn.transaction().map_err(sql_err)?;
    transaction.execute_batch("CREATE TEMP TABLE IF NOT EXISTS present_table(t TEXT PRIMARY KEY); DELETE FROM present_table;
        CREATE TEMP TABLE IF NOT EXISTS compared_table(t TEXT PRIMARY KEY); DELETE FROM compared_table;").map_err(sql_err)?;
    let at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64);
    let mut w = BufWriter::new(tempfile::tempfile()?);
    let schema: Map<String, Value> = tables
        .iter()
        .map(|t| (t.name.clone(), Value::String(t.sql.clone())))
        .collect();
    let schema = Value::Object(schema);
    let schema_digest = digest(&schema.to_string());
    if catalog.schema != Some(schema_digest) {
        let at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as i64);
        let name = db.file_name().map(|n| n.to_string_lossy().into_owned());
        let mut header = json!({ "db": name, "at_ms": at_ms, "schema": schema });
        if let Some(binding) = &catalog.binding {
            header["source"] = serde_json::to_value(binding).map_err(std::io::Error::from)?;
        }
        serde_json::to_writer(&mut w, &header).map_err(std::io::Error::from)?;
        w.write_all(b"\n")?;
    }
    let mut done = Exported::default();
    let mut staged_bytes = 0;
    for t in &tables {
        transaction
            .execute("INSERT INTO present_table VALUES(?1)", [&t.name])
            .map_err(sql_err)?;
        let declared = cursor.and_then(|cursor| cursor(db, &t.name));
        let previous: Option<(String, Option<i64>, i64)> = transaction
            .query_row(
                "SELECT column_name,value,verified_at FROM table_cursor WHERE t=?1",
                [&t.name],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(sql_err)?;
        let full = due.replaced
            || declared.is_none()
            || previous.as_ref().is_none_or(|(column, value, verified)| {
                Some(column.as_str()) != declared
                    || value.is_none()
                    || at_ms.saturating_sub(*verified) >= VERIFY_EVERY_MS
            })
            || catalog.schema != Some(schema_digest);
        if (declared.is_none() && !due.compare)
            || (declared.is_some()
                && !due.cursor
                && !due.replaced
                && catalog.schema == Some(schema_digest))
        {
            continue;
        }
        if full {
            transaction
                .execute("UPDATE row_state SET seen=0 WHERE t=?1", [&t.name])
                .map_err(sql_err)?;
            transaction
                .execute("INSERT INTO compared_table VALUES(?1)", [&t.name])
                .map_err(sql_err)?;
        }
        let quoted = t.name.replace('"', "\"\"");
        let select = if t.key.is_empty() {
            let rowid = ["rowid", "_rowid_", "oid"]
                .into_iter()
                .find(|name| !t.columns.iter().any(|c| c.eq_ignore_ascii_case(name)))
                .ok_or_else(|| {
                    ExportError::Invalid(format!("table {} shadows all SQLite rowid names", t.name))
                })?;
            format!("SELECT {rowid}, * FROM \"{quoted}\"")
        } else {
            format!("SELECT NULL, * FROM \"{quoted}\"")
        };
        if let Some(column) = declared
            && !t.columns.iter().any(|c| c == column)
        {
            return Err(ExportError::Invalid(format!(
                "declared cursor {column} is missing in {}",
                t.name
            )));
        }
        let select = if let Some(column) = declared.filter(|_| !full) {
            format!(
                "{select} WHERE \"{}\">{}",
                column.replace('"', "\"\""),
                previous
                    .as_ref()
                    .expect("stored cursor")
                    .1
                    .expect("populated cursor")
            )
        } else {
            select
        };
        let mut high = if full {
            None
        } else {
            previous.as_ref().and_then(|(_, value, _)| *value)
        };
        let mut stmt = conn.prepare(&select).map_err(sql_err)?;
        let mut q = stmt.query([]).map_err(sql_err)?;
        while let Some(r) = q.next().map_err(sql_err)? {
            let mut row = Map::new();
            for (i, c) in t.columns.iter().enumerate() {
                row.insert(c.clone(), value(r.get_ref(i + 1).map_err(sql_err)?)?);
            }
            let k = if t.key.is_empty() {
                json!([value(r.get_ref(0).map_err(sql_err)?)?])
            } else {
                Value::Array(t.key.iter().map(|&i| row[&t.columns[i]].clone()).collect())
            };
            let row = Value::Object(row);
            if let Some(column) = declared {
                let value = row.get(column).and_then(Value::as_i64).ok_or_else(|| {
                    ExportError::Invalid(format!(
                        "declared cursor {column} in {} is not an integer",
                        t.name
                    ))
                })?;
                high = Some(high.map_or(value, |old: i64| old.max(value)));
            }
            let encoded = row.to_string();
            let hash = blake3::hash(encoded.as_bytes());
            let h = digest(&encoded);
            if catalog::changed(&transaction, &t.name, &k, &hash, full).map_err(sql_err)? {
                serde_json::to_writer(
                    &mut w,
                    &json!({ "t": t.name, "k": k, "r": row, "h": format!("{h:016x}") }),
                )
                .map_err(std::io::Error::from)?;
                w.write_all(b"\n")?;
                crate::fault::check("tool_export.after_row", &t.name)
                    .map_err(std::io::Error::other)?;
                done.rows += 1;
                staged_bytes += encoded.len() as u64;
                if staged_bytes >= APPEND_BYTES {
                    // The rows so far become input now, so the loop can
                    // store and parse a large database's capture while the
                    // rest is still read.
                    append_staged(&mut w, st.synced.as_mut(), out)?;
                    staged_bytes = 0;
                }
            }
            checkpoint()?;
        }
        if declared.is_some() {
            let verified = if full {
                at_ms
            } else {
                previous.as_ref().expect("stored cursor").2
            };
            transaction.execute("INSERT INTO table_cursor VALUES(?1,?2,?3,?4) ON CONFLICT(t) DO UPDATE SET column_name=excluded.column_name,value=excluded.value,verified_at=excluded.verified_at", rusqlite::params![t.name,declared.expect("declared cursor"),high,verified]).map_err(sql_err)?;
        }
    }
    // Only fully compared tables and dropped tables can have removals.
    {
        let mut query = transaction
            .prepare("SELECT t,k,kh FROM row_state WHERE (seen=0 AND t IN compared_table) OR t NOT IN present_table ORDER BY t,k")
            .map_err(sql_err)?;
        let mut gone = query.query([]).map_err(sql_err)?;
        while let Some(row) = gone.next().map_err(sql_err)? {
            let table: String = row.get(0).map_err(sql_err)?;
            let key: String = row.get(1).map_err(sql_err)?;
            let kh: String = row.get(2).map_err(sql_err)?;
            let key: Value = serde_json::from_str(&key).map_err(std::io::Error::from)?;
            serde_json::to_writer(&mut w, &json!({"t":table,"k":key,"kh":kh,"gone":true}))
                .map_err(std::io::Error::from)?;
            w.write_all(b"\n")?;
            crate::fault::check("tool_export.after_gone", &table).map_err(std::io::Error::other)?;
            done.gone += 1;
            checkpoint()?;
        }
    }
    transaction
        .execute("DELETE FROM row_state WHERE (seen=0 AND t IN compared_table) OR t NOT IN present_table", [])
        .map_err(sql_err)?;
    conn.execute_batch("COMMIT").map_err(sql_err)?;
    append_staged(&mut w, st.synced.as_mut(), out)?;
    crate::fault::check("tool_export.after_complete", "").map_err(std::io::Error::other)?;
    catalog::save(
        &transaction,
        Some(schema_digest),
        &catalog.binding,
        catalog.binding.is_none(),
        st.synced.as_ref().expect("verified prefix"),
    )?;
    transaction.commit().map_err(sql_err)?;
    catalog.schema = Some(schema_digest);
    Ok(done)
}

#[cfg(test)]
mod tests;
