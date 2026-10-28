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
//! writes. Comparison state is a SQLite catalog beside each capture,
//! stamped with the capture prefix it describes; a restart reuses it, and the
//! marker's facts let it trust an unchanged capture without reading it.
//! Changed rows are staged privately, then a synced marker naming the
//! completed prefix commits them, every 64 MB of rows and at the end, so a
//! large database's capture becomes input while it is read. Bytes beyond
//! that marker are removed on the next open.

use crate::FileFacts;
use base64::Engine as _;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// While a database keeps changing, it is still exported this often.
const BUSY_EXPORT: Duration = Duration::from_secs(3600);

mod binding;
mod catalog;
mod synced_prefix;
mod turns;
use catalog::Catalog;
use synced_prefix::SyncedPrefix;
pub use synced_prefix::completed_facts;
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
    /// As the catalog's stamp keeps them.
    fn encode(self) -> String {
        let one = |f: Option<FileFacts>| {
            f.map(|f| {
                format!(
                    "{},{},{},{},{}",
                    f.device, f.inode, f.size, f.mtime_ns, f.ctime_ns
                )
            })
            .unwrap_or_default()
        };
        format!("{};{}", one(self.db), one(self.wal))
    }

    fn decode(text: &str) -> Option<Self> {
        let one = |part: &str| -> Option<Option<FileFacts>> {
            if part.is_empty() {
                return Some(None);
            }
            let n: Vec<&str> = part.split(',').collect();
            let [device, inode, size, mtime_ns, ctime_ns] = n.as_slice() else {
                return None;
            };
            Some(Some(FileFacts {
                device: device.parse().ok()?,
                inode: inode.parse().ok()?,
                size: size.parse().ok()?,
                mtime_ns: mtime_ns.parse().ok()?,
                ctime_ns: ctime_ns.parse().ok()?,
            }))
        };
        let (db, wal) = text.split_once(';')?;
        Some(Self {
            db: one(db)?,
            wal: one(wal)?,
        })
    }

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
}

#[derive(Default)]
pub struct Exporter {
    dbs: HashMap<PathBuf, DbState>,
}

/// What one export call did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exported {
    pub rows: u64,
    pub gone: u64,
}

impl Exporter {
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
            let mut synced = SyncedPrefix::open(out, checkpoint)?;
            let mut catalog = Catalog::open(out, synced.stamp(), checkpoint)?;
            catalog.bind(requested.clone())?;
            synced.establish(out)?;
            // A source unchanged since the export the catalog was stamped
            // at, before this process started, is not exported again.
            st.exported = catalog.exported;
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
        if !(changed && (first || quiet || overdue)) {
            return Ok(None);
        }
        let done = match write_export(db, out, st, checkpoint, now) {
            Ok(done) => done,
            Err(e) => {
                // The marker may have been written before a local cache failure.
                // Rebuild both from the verified synced prefix next time.
                st.catalog = None;
                st.synced = None;
                return Err(e);
            }
        };
        st.exported = Some(now);
        st.exported_at = Some(Instant::now());
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
            "PRAGMA table_info(\"{}\")",
            name.replace('"', "\"\"")
        ))?;
        let cols: Vec<(String, i64)> = info
            .query_map([], |r| Ok((r.get(1)?, r.get(5)?)))?
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

fn write_export(
    db: &Path,
    out: &Path,
    st: &mut DbState,
    checkpoint: &mut dyn FnMut() -> Result<(), ExportError>,
    now: DbFacts,
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
    catalog::start(&transaction).map_err(sql_err)?;
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
    let header_written = catalog.schema != Some(schema_digest);
    let mut done = Exported::default();
    let mut staged_bytes = 0;
    for t in &tables {
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
            let encoded = row.to_string();
            let hash = blake3::hash(encoded.as_bytes());
            let h = digest(&encoded);
            if catalog::changed(&transaction, &t.name, &k, &hash).map_err(sql_err)? {
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
    }
    // All tables were scanned successfully. This also finds dropped tables.
    {
        let mut query = transaction.prepare(catalog::GONE).map_err(sql_err)?;
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
        .execute(
            "DELETE FROM row_state WHERE NOT EXISTS
               (SELECT 1 FROM temp.seen s WHERE s.t = row_state.t AND s.k = row_state.k)",
            [],
        )
        .map_err(sql_err)?;
    conn.execute_batch("COMMIT").map_err(sql_err)?;
    append_staged(&mut w, st.synced.as_mut(), out)?;
    crate::fault::check("tool_export.after_complete", "").map_err(std::io::Error::other)?;
    // As replaying these lines would record: written without a source
    // binding, the capture stays unbound.
    if header_written || done.rows > 0 || done.gone > 0 {
        catalog.unbound |= catalog.binding.is_none();
    }
    let stamp = catalog::StampValues {
        schema: Some(schema_digest),
        binding: catalog.binding.clone(),
        unbound: catalog.unbound,
        exported: Some(now),
    };
    let prefix = st.synced.as_ref().expect("verified synced prefix").stamp();
    catalog::write_stamp(&transaction, prefix, &stamp).map_err(sql_err)?;
    transaction.commit().map_err(sql_err)?;
    catalog.schema = Some(schema_digest);
    catalog.exported = Some(now);
    Ok(done)
}

#[cfg(test)]
mod tests;
