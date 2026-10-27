//! Tool databases read in one SQLite transaction and emitted as row-dump
//! records. The writer stores these bytes as an ordinary input. Comparison
//! hashes live in the index, beside the parsed rows.
use crate::{Archive, FileFacts, InputKey, Observation};
use base64::Engine as _;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde_json::{Map, Value, json};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

mod catalog;
pub use catalog::Comparison;

const BUSY_READ: Duration = Duration::from_secs(3600);
/// A batching threshold, never a record-size limit.
pub const APPEND_BYTES: usize = 8 << 20;

#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Store(#[from] crate::StoreError),
    #[error("{0}: {1}")]
    Sqlite(PathBuf, rusqlite::Error),
    #[error("database read stopped")]
    Stopped,
}

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
    last_seen: Option<DbFacts>,
    read: Option<DbFacts>,
    read_at: Option<Instant>,
}
#[derive(Default)]
pub struct DatabaseReader {
    dbs: HashMap<PathBuf, DbState>,
    stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReadReport {
    pub rows: u64,
    pub gone: u64,
    pub visited: u64,
}
impl DatabaseReader {
    pub fn forget(&mut self, path: &Path) {
        self.dbs.remove(path);
    }
    pub fn stop_when(&mut self, stop: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.stop = Some(stop);
    }
    /// File facts decide whether any source or archive read is needed.
    pub fn observe(&mut self, db: &Path) -> Option<DatabaseRead> {
        let facts = DbFacts::of(db);
        let st = self.dbs.entry(db.to_owned()).or_default();
        if facts.db.is_none() {
            *st = DbState::default();
            return None;
        }
        let quiet = st.last_seen == Some(facts);
        st.last_seen = Some(facts);
        let due = st.read.is_none()
            || (st.read != Some(facts)
                && (quiet || st.read_at.is_some_and(|at| at.elapsed() >= BUSY_READ)));
        due.then_some(DatabaseRead { facts })
    }

    pub fn capture(
        &mut self,
        db: &Path,
        archive: &Archive,
        input: &InputKey,
        observation: &Observation,
        send: impl FnMut(Vec<u8>) -> Result<(), DatabaseError>,
    ) -> Result<Option<ReadReport>, DatabaseError> {
        let Some(due) = self.observe(db) else {
            return Ok(None);
        };
        self.read(db, archive, input, observation, due, send)
            .map(Some)
    }
    /// Queue complete groups while their predecessors commit. The caller
    /// waits for acknowledgment before reading the same input again.
    pub fn read(
        &mut self,
        db: &Path,
        archive: &Archive,
        input: &InputKey,
        observation: &Observation,
        due: DatabaseRead,
        mut send: impl FnMut(Vec<u8>) -> Result<(), DatabaseError>,
    ) -> Result<ReadReport, DatabaseError> {
        let done = read_rows(
            db,
            archive,
            input,
            observation,
            self.stop.as_deref(),
            &mut send,
        )?;
        let st = self.dbs.entry(db.to_owned()).or_default();
        st.read = Some(due.facts);
        st.read_at = Some(Instant::now());
        Ok(done)
    }
}

pub struct DatabaseRead {
    facts: DbFacts,
}

fn row_hash(row: &Value) -> blake3::Hash {
    // SQLite values are scalars or one-key blob objects. Column order is
    // schema metadata, not a change to the row's named values.
    let fields: std::collections::BTreeMap<_, _> =
        row.as_object().expect("row object").iter().collect();
    blake3::hash(&serde_json::to_vec(&fields).expect("JSON row values serialize"))
}

fn value(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => i.into(),
        ValueRef::Real(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or_else(|| json!({"real_bits": format!("{:016x}", f.to_bits())})),
        ValueRef::Text(t) => match std::str::from_utf8(t) {
            Ok(text) => text.into(),
            Err(_) => json!({"text_base64": base64::engine::general_purpose::STANDARD.encode(t)}),
        },
        ValueRef::Blob(b) => {
            json!({ "base64": base64::engine::general_purpose::STANDARD.encode(b) })
        }
    }
}

struct Table {
    name: String,
    sql: String,
    columns: Vec<String>,
    /// Primary-key column positions; empty when the rowid is the key.
    key: Vec<usize>,
    rowid: Option<String>,
}

fn tables(conn: &Connection) -> rusqlite::Result<Vec<Table>> {
    let mut out = Vec::new();
    let mut stmt = conn.prepare(
        "SELECT name, sql FROM sqlite_master
         WHERE type = 'table' AND name NOT GLOB 'sqlite_*' ORDER BY name",
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
        let without_rowid: bool = conn.query_row(
            "SELECT wr FROM pragma_table_list WHERE schema='main' AND name=?1",
            [&name],
            |r| r.get(0),
        )?;
        let rowid = (!without_rowid)
            .then(|| {
                ["rowid", "_rowid_", "oid"]
                    .into_iter()
                    .find(|name| !cols.iter().any(|c| c.0.eq_ignore_ascii_case(name)))
                    .map(str::to_owned)
            })
            .flatten();
        out.push(Table {
            name,
            sql,
            columns: cols.into_iter().map(|(c, _)| c).collect(),
            key: key.into_iter().map(|(_, i)| i).collect(),
            rowid,
        });
    }
    Ok(out)
}

fn read_rows(
    db: &Path,
    archive: &Archive,
    input: &InputKey,
    observation: &Observation,
    stop: Option<&std::sync::atomic::AtomicBool>,
    send: &mut dyn FnMut(Vec<u8>) -> Result<(), DatabaseError>,
) -> Result<ReadReport, DatabaseError> {
    let stopped = || {
        if stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed)) {
            Err(DatabaseError::Stopped)
        } else {
            Ok(())
        }
    };
    stopped()?;
    let _busy = crate::timing::enter(crate::timing::Stage::Database);
    let sql_err = |e| DatabaseError::Sqlite(db.to_owned(), e);
    let source = Connection::open_with_flags(
        db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(sql_err)?;
    source
        .busy_timeout(Duration::from_secs(10))
        .map_err(sql_err)?;
    source.execute_batch("BEGIN").map_err(sql_err)?;
    let tables = tables(&source).map_err(sql_err)?;
    let catalog = catalog::Catalog::open(archive, &input.id)?;
    let schema: Map<String, Value> = tables
        .iter()
        .map(|t| (t.name.clone(), t.sql.clone().into()))
        .collect();
    let schema = Value::Object(schema);
    let changed_schema = catalog.schema.as_ref() != Some(&schema);
    let at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64);
    let mut bytes = Vec::new();
    if changed_schema {
        serde_json::to_writer(&mut bytes, &json!({"db":db.file_name().map(|n| n.to_string_lossy()), "at_ms":at_ms, "schema":schema, "source":{"machine":observation.machine,"path":observation.path}})).map_err(std::io::Error::from)?;
        bytes.push(b'\n');
    }
    let mut done = ReadReport::default();
    for table in &tables {
        let quoted = table.name.replace('"', "\"\"");
        let select = if let Some(rowid) = &table.rowid {
            format!(r#"SELECT {rowid}, * FROM "{quoted}""#)
        } else {
            format!(r#"SELECT NULL, * FROM "{quoted}""#)
        };
        let mut statement = source.prepare(&select).map_err(sql_err)?;
        let mut rows = statement.query([]).map_err(sql_err)?;
        let mut position = 0_u64;
        while let Some(row) = rows.next().map_err(sql_err)? {
            position += 1;
            stopped()?;
            done.visited += 1;
            let mut values = Map::new();
            for (i, column) in table.columns.iter().enumerate() {
                values.insert(column.clone(), value(row.get_ref(i + 1).map_err(sql_err)?));
            }
            // Non-null primary keys keep their declared identities. A NULL
            // key is not unique in SQLite, so that row names its rowid.
            let key = if !table.key.is_empty()
                && table
                    .key
                    .iter()
                    .all(|&i| !values[&table.columns[i]].is_null())
            {
                Value::Array(
                    table
                        .key
                        .iter()
                        .map(|&i| values[&table.columns[i]].clone())
                        .collect(),
                )
            } else if table.rowid.is_some() {
                let rowid = value(row.get_ref(0).map_err(sql_err)?);
                if table.key.is_empty() {
                    json!([rowid])
                } else {
                    json!({"rowid":rowid})
                }
            } else {
                json!({"position":position})
            };
            let encoded = Value::Object(values);
            if catalog.changed(&table.name, &key, &encoded)? {
                serde_json::to_writer(
                    &mut bytes,
                    &json!({"t":table.name,"k":key,"r":encoded,"at_ms":at_ms,"encoding":"sqlite"}),
                )
                .map_err(std::io::Error::from)?;
                bytes.push(b'\n');
                done.rows += 1;
                crate::fault::check("database.after_row", &table.name)?;
            }
            if bytes.len() >= APPEND_BYTES {
                send(std::mem::take(&mut bytes))?;
            }
        }
        catalog.gone(&table.name, &mut |key| {
            stopped()?;
            serde_json::to_writer(
                &mut bytes,
                &json!({"t":table.name,"k":key,"gone":true,"at_ms":at_ms}),
            )
            .map_err(std::io::Error::from)?;
            bytes.push(b'\n');
            done.gone += 1;
            crate::fault::check("database.after_gone", &table.name)?;
            if bytes.len() >= APPEND_BYTES {
                send(std::mem::take(&mut bytes))?;
            }
            Ok(())
        })?;
    }
    catalog.dropped(
        &tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        &mut |table, key| {
            stopped()?;
            serde_json::to_writer(
                &mut bytes,
                &json!({"t":table,"k":key,"gone":true,"at_ms":at_ms}),
            )
            .map_err(std::io::Error::from)?;
            bytes.push(b'\n');
            done.gone += 1;
            if bytes.len() >= APPEND_BYTES {
                send(std::mem::take(&mut bytes))?;
            }
            Ok(())
        },
    )?;
    stopped()?;
    if !bytes.is_empty() {
        send(bytes)?;
    }
    Ok(done)
}

#[cfg(test)]
mod tests;
