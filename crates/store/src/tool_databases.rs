//! Tool databases read in one SQLite transaction and emitted as row-dump
//! records. The writer stores these bytes as an ordinary input. Comparison
//! hashes live in the index, beside the parsed rows. Every completed read
//! records an observation with the database's and its log's file facts, so
//! a database left as it was when the process restarts is not read again.
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
    /// The same database: both files unchanged, and the log present in both
    /// or in neither.
    fn unchanged_as(&self, other: &Self) -> bool {
        let same = |a: &Option<FileFacts>, b: &Option<FileFacts>| match (a, b) {
            (Some(a), Some(b)) => a.unchanged_as(b),
            (None, None) => true,
            _ => false,
        };
        same(&self.db, &other.db) && same(&self.wal, &other.wal)
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
    /// Created only inside the actual consistent provider read. It is custody
    /// provenance, never a resumable database revision or data_version.
    pub snapshot: Option<crate::primary::DatabaseSnapshotToken>,
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
        let read = st.read.is_some_and(|read| read.unchanged_as(&facts));
        let due = st.read.is_none()
            || (!read && (quiet || st.read_at.is_some_and(|at| at.elapsed() >= BUSY_READ)));
        due.then_some(DatabaseRead { facts })
    }

    /// Takes, for a database this reader has not read, the facts of the
    /// last read whose observation the archive holds for this input at this
    /// machine and exact path. An unchanged database is then not read again
    /// after a restart; any other evidence leaves it to be read.
    pub fn recall(
        &mut self,
        db: &Path,
        archive: &Archive,
        input: &InputKey,
        observation: &Observation,
    ) -> Result<(), DatabaseError> {
        if self
            .stop
            .as_deref()
            .is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(DatabaseError::Stopped);
        }
        if self.dbs.get(db).is_some_and(|st| st.read.is_some()) || DbFacts::of(db).db.is_none() {
            return Ok(());
        }
        let Some(facts) = recorded(archive, input, observation)? else {
            return Ok(());
        };
        // A header this build states differently is written by a full read.
        if !header_current(db, archive, input)? {
            return Ok(());
        }
        let st = self.dbs.entry(db.to_owned()).or_default();
        st.read = Some(facts);
        st.read_at = Some(Instant::now());
        // A database changed since that read is read at once, as by a
        // reader that never read it, not after it settles.
        st.last_seen = Some(DbFacts::of(db));
        Ok(())
    }

    pub fn capture(
        &mut self,
        db: &Path,
        archive: &Archive,
        input: &InputKey,
        observation: &Observation,
        send: impl FnMut(Vec<u8>, &Observation) -> Result<(), DatabaseError>,
    ) -> Result<Option<ReadReport>, DatabaseError> {
        self.recall(db, archive, input, observation)?;
        let Some(due) = self.observe(db) else {
            return Ok(None);
        };
        self.read(db, archive, input, observation, due, send)
            .map(Some)
    }
    /// Queue complete groups while their predecessors commit. The caller
    /// waits for acknowledgment before reading the same input again. Each
    /// group is sent with the observation it records: `observation` with
    /// the file facts this read was decided on. A read that finds no changed
    /// row sends one empty group, so its observation is recorded too.
    pub fn read(
        &mut self,
        db: &Path,
        archive: &Archive,
        input: &InputKey,
        observation: &Observation,
        due: DatabaseRead,
        mut send: impl FnMut(Vec<u8>, &Observation) -> Result<(), DatabaseError>,
    ) -> Result<ReadReport, DatabaseError> {
        let observation = Observation {
            facts: due.facts.db,
            database: Some(crate::DatabaseLog { wal: due.facts.wal }),
            ..observation.clone()
        };
        let mut sent = false;
        let done = read_rows(
            db,
            archive,
            input,
            &observation,
            self.stop.as_deref(),
            &mut |bytes| {
                sent = true;
                send(bytes, &observation)
            },
        )?;
        if !sent {
            send(Vec::new(), &observation)?;
        }
        let st = self.dbs.entry(db.to_owned()).or_default();
        st.read = Some(due.facts);
        st.read_at = Some(Instant::now());
        Ok(done)
    }
}

pub struct DatabaseRead {
    facts: DbFacts,
}

/// File facts as an observation's details record them.
pub(crate) fn facts_json(facts: FileFacts) -> Value {
    json!({
        "device": facts.device,
        "inode": facts.inode,
        "size": facts.size,
        "mtime_ns": i64::try_from(facts.mtime_ns).ok(),
        "ctime_ns": i64::try_from(facts.ctime_ns).ok(),
    })
}

fn facts_of(value: &Value) -> Option<FileFacts> {
    Some(FileFacts {
        device: value.get("device")?.as_u64()?,
        inode: value.get("inode")?.as_u64()?,
        size: value.get("size")?.as_u64()?,
        mtime_ns: value.get("mtime_ns")?.as_i64()?.into(),
        ctime_ns: value.get("ctime_ns")?.as_i64()?.into(),
    })
}

/// The database and log facts of the last completed read of `input` at this
/// observation's machine and exact path: the newest such observation the
/// archive holds, committed with that read's rows. Primary mode holds it
/// once Sinex confirmed it.
fn recorded(
    archive: &Archive,
    input: &InputKey,
    observation: &Observation,
) -> crate::Result<Option<DbFacts>> {
    let conn = crate::archive::open_read(&archive.index_db())?;
    if !archive.is_primary() {
        let uri = format!("file:{}?mode=ro", archive.archive_db().display());
        conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        crate::source::views(&conn)?;
    }
    let row = conn
        .query_row(
            "SELECT device,inode,size,mtime_ns,ctime_ns,details_json FROM source_observation
             WHERE input=?1 AND machine=?2 AND path_bytes=?3
             ORDER BY commit_seq DESC, id DESC LIMIT 1",
            params![input.id, observation.machine, observation.path_bytes],
            |r| {
                let column = |n| r.get::<_, Option<i64>>(n);
                let facts = match (column(0)?, column(1)?, column(2)?, column(3)?, column(4)?) {
                    (Some(device), Some(inode), Some(size), Some(mtime), Some(ctime)) => {
                        Some(FileFacts {
                            device: device as u64,
                            inode: inode as u64,
                            size: size as u64,
                            mtime_ns: mtime.into(),
                            ctime_ns: ctime.into(),
                        })
                    }
                    _ => None,
                };
                Ok((facts, r.get::<_, Option<String>>(5)?))
            },
        )
        .optional()?;
    let Some((Some(db), Some(details))) = row else {
        return Ok(None);
    };
    let details: Value = serde_json::from_str(&details)?;
    let Some(wal) = details.get("database_wal") else {
        return Ok(None);
    };
    let wal = match wal {
        Value::Null => None,
        wal => match facts_of(wal) {
            Some(facts) => Some(facts),
            None => return Ok(None),
        },
    };
    Ok(Some(DbFacts { db: Some(db), wal }))
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
    metadata: Value,
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
        let metadata_columns: Vec<Value> = info
            .query_map([], |r| {
                Ok(
                    json!({"cid":r.get::<_,i64>(0)?,"name":r.get::<_,String>(1)?,
                "declared_type":r.get::<_,String>(2)?,"not_null":r.get::<_,bool>(3)?,
                "primary_key":r.get::<_,i64>(5)?,"hidden":r.get::<_,i64>(6)?}),
                )
            })?
            .collect::<rusqlite::Result<_>>()?;
        let cols: Vec<(String, i64)> = metadata_columns
            .iter()
            .filter(|column| column["hidden"] != 1)
            .map(|column| {
                (
                    column["name"].as_str().expect("column name").to_owned(),
                    column["primary_key"].as_i64().expect("primary key ordinal"),
                )
            })
            .collect();
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
            metadata: json!({"columns":metadata_columns,"without_rowid":without_rowid}),
            columns: cols.into_iter().map(|(c, _)| c).collect(),
            key: key.into_iter().map(|(_, i)| i).collect(),
            rowid,
        });
    }
    Ok(out)
}

/// The tables' SQL and native metadata, as a row-dump header states them.
/// Comparison includes native metadata so a pre-metadata capture emits the
/// additive header even when its SQL declaration has not changed.
fn signature(tables: &[Table]) -> Value {
    let schema: Map<String, Value> = tables
        .iter()
        .map(|t| (t.name.clone(), t.sql.clone().into()))
        .collect();
    let table_metadata: Map<String, Value> = tables
        .iter()
        .map(|table| (table.name.clone(), table.metadata.clone()))
        .collect();
    json!({"schema":schema,"table_metadata":table_metadata})
}

/// Whether the archive's comparison state holds the header this database's
/// tables give now; reading only its schema, no rows.
fn header_current(db: &Path, archive: &Archive, input: &InputKey) -> Result<bool, DatabaseError> {
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
    Ok(catalog.schema.as_ref() == Some(&signature(&tables)))
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
    let signature = signature(&tables);
    let (schema, table_metadata) = (&signature["schema"], &signature["table_metadata"]);
    let changed_schema = catalog.schema.as_ref() != Some(&signature);
    // The accepting engine supplied this observation from its injected clock.
    let at_ms = observation.seen_at_ms;
    let mut bytes = Vec::new();
    if changed_schema {
        serde_json::to_writer(&mut bytes, &json!({"db":db.file_name().map(|n| n.to_string_lossy()), "at_ms":at_ms, "schema":schema, "table_metadata":table_metadata, "source":{"machine":observation.machine,"path":observation.path}})).map_err(std::io::Error::from)?;
        bytes.push(b'\n');
    }
    let mut done = ReadReport {
        snapshot: Some(crate::primary::DatabaseSnapshotToken {
            nonce: uuid::Uuid::now_v7().to_string(),
        }),
        ..Default::default()
    };
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
