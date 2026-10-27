//! Tool databases read in one SQLite transaction and emitted as row-dump
//! records. The writer stores these bytes as an ordinary input. Comparison
//! hashes and declared cursors live in the index, beside the parsed rows.
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
pub(crate) use catalog::save_when_parsed;
pub use catalog::{Comparison, SavedDatabase};

const BUSY_EXPORT: Duration = Duration::from_secs(3600);
const CURSOR_EXPORT: Duration = Duration::from_secs(180);
const VERIFY_EVERY_MS: i64 = 24 * 3600 * 1000;
/// A batching threshold, never a record-size limit.
pub const APPEND_BYTES: usize = 8 << 20;
pub type TableCursor = fn(&Path, &str) -> Option<&'static str>;

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Store(#[from] crate::StoreError),
    #[error("{0}: {1}")]
    Sqlite(PathBuf, rusqlite::Error),
    #[error("{0}")]
    Invalid(String),
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
    exported: Option<DbFacts>,
    exported_at: Option<Instant>,
    cursor_at: Option<Instant>,
    has_cursor: bool,
    missing: bool,
}
#[derive(Default)]
pub struct Exporter {
    dbs: HashMap<PathBuf, DbState>,
    cursor: Option<TableCursor>,
    stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exported {
    pub rows: u64,
    pub gone: u64,
    pub visited: u64,
    has_cursor: bool,
}
impl Exporter {
    pub fn with_cursor(cursor: TableCursor) -> Self {
        Self {
            cursor: Some(cursor),
            ..Self::default()
        }
    }
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
            *st = DbState {
                missing: true,
                ..DbState::default()
            };
            return None;
        }
        let quiet = st.last_seen == Some(facts);
        st.last_seen = Some(facts);
        let compare = st.exported.is_none()
            || (st.exported != Some(facts)
                && (quiet || st.exported_at.is_some_and(|at| at.elapsed() >= BUSY_EXPORT)));
        let cursor = st.exported.is_none()
            || (st.has_cursor && st.cursor_at.is_none_or(|at| at.elapsed() >= CURSOR_EXPORT));
        (compare || cursor).then_some(DatabaseRead {
            facts,
            compare,
            cursor,
            verify: st.missing,
        })
    }
    pub fn export(
        &mut self,
        db: &Path,
        archive: &Archive,
        input: &InputKey,
        observation: &Observation,
        send: impl FnMut(Vec<u8>, Option<SavedDatabase>) -> Result<(), ExportError>,
    ) -> Result<Option<Exported>, ExportError> {
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
        mut send: impl FnMut(Vec<u8>, Option<SavedDatabase>) -> Result<(), ExportError>,
    ) -> Result<Exported, ExportError> {
        let done = write_export(
            db,
            archive,
            input,
            observation,
            self.cursor,
            due.compare,
            due.cursor,
            due.verify,
            self.stop.as_deref(),
            &mut send,
        )?;
        let st = self.dbs.entry(db.to_owned()).or_default();
        st.missing = false;
        st.has_cursor = done.has_cursor;
        if due.compare {
            st.exported = Some(due.facts);
            st.exported_at = Some(Instant::now());
        }
        if due.cursor {
            st.cursor_at = Some(Instant::now());
        }
        Ok(done)
    }
}

pub struct DatabaseRead {
    facts: DbFacts,
    compare: bool,
    cursor: bool,
    verify: bool,
}

fn file_identity(path: &Path) -> Option<(u64, u64, i128)> {
    let metadata = fs::metadata(path).ok()?;
    let facts = FileFacts::of(&metadata);
    let created = match metadata
        .created()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
    {
        Ok(value) => i128::try_from(value.as_nanos()).ok()?,
        Err(error) => -i128::try_from(error.duration().as_nanos()).ok()?,
    };
    Some((facts.device, facts.inode, created))
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
        let cols: Vec<(String, i64, bool, String)> = info
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, i64>(6)?,
                    r.get::<_, i64>(3)? != 0,
                    r.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|row| match row {
                Ok((_, _, 1, _, _)) => None,
                Ok((name, key, _, not_null, kind)) => Some(Ok((name, key, not_null, kind))),
                Err(error) => Some(Err(error)),
            })
            .collect::<rusqlite::Result<_>>()?;
        let mut key: Vec<(i64, usize)> = cols
            .iter()
            .enumerate()
            .filter(|(_, (_, pk, _, _))| *pk > 0)
            .map(|(i, (_, pk, _, _))| (*pk, i))
            .collect();
        key.sort_unstable();
        let without_rowid: bool = conn.query_row(
            "SELECT wr FROM pragma_table_list WHERE schema='main' AND name=?1", [&name], |r| r.get(0)
        )?;
        // SQLite permits duplicate NULL primary keys in ordinary rowid
        // tables. Only enforced non-null keys identify every row uniquely.
        if !without_rowid && key.iter().any(|(_, i)| {
            !cols[*i].2 && !(key.len() == 1 && cols[*i].3.eq_ignore_ascii_case("INTEGER"))
        }) {
            key.clear();
        }
        let rowid = (!without_rowid).then(|| ["rowid", "_rowid_", "oid"].into_iter().find(|name| !cols.iter().any(|c| c.0.eq_ignore_ascii_case(name))).map(str::to_owned)).flatten();
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

#[allow(clippy::too_many_arguments)]
fn write_export(
    db: &Path,
    archive: &Archive,
    input: &InputKey,
    observation: &Observation,
    cursor: Option<TableCursor>,
    compare: bool,
    cursor_due: bool,
    verify: bool,
    stop: Option<&std::sync::atomic::AtomicBool>,
    send: &mut dyn FnMut(Vec<u8>, Option<SavedDatabase>) -> Result<(), ExportError>,
) -> Result<Exported, ExportError> {
    let stopped = || {
        if stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed)) {
            Err(ExportError::Stopped)
        } else {
            Ok(())
        }
    };
    stopped()?;
    let _busy = crate::timing::enter(crate::timing::Stage::Database);
    let sql_err = |e| ExportError::Sqlite(db.to_owned(), e);
    let facts = DbFacts::of(db).db.ok_or_else(|| {
        ExportError::Invalid("source database disappeared before its read".into())
    })?;
    let source_file = file_identity(db);
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
    let mut saved = catalog.saved.clone();
    let replaced = verify || source_file.is_none() || saved.source_file != source_file;
    let schema: Map<String, Value> = tables
        .iter()
        .map(|t| (t.name.clone(), t.sql.clone().into()))
        .collect();
    let schema = Value::Object(schema);
    let changed_schema = saved.schema.as_ref() != Some(&schema);
    let at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64);
    let mut bytes = Vec::new();
    if changed_schema {
        serde_json::to_writer(&mut bytes, &json!({"db":db.file_name().map(|n| n.to_string_lossy()), "at_ms":at_ms, "schema":schema, "source":{"machine":observation.machine,"path":observation.path}})).map_err(std::io::Error::from)?;
        bytes.push(b'\n');
    }
    let mut done = Exported::default();
    for table in &tables {
        let declared = cursor.and_then(|c| c(db, &table.name));
        let previous = saved.cursors.get(&table.name).cloned();
        let full = replaced
            || changed_schema
            || declared.is_none()
            || previous.as_ref().is_none_or(|p| {
                Some(p.column.as_str()) != declared
                    || p.value.is_none()
                    || at_ms.saturating_sub(p.verified_at) >= VERIFY_EVERY_MS
            });
        if !replaced
            && !changed_schema
            && ((declared.is_none() && !compare) || (declared.is_some() && !cursor_due))
        {
            continue;
        }
        let quoted = table.name.replace('"', "\"\"");
        let select = if table.key.is_empty() && let Some(rowid) = &table.rowid {
            format!(r#"SELECT {rowid}, * FROM "{quoted}""#)
        } else {
            format!(r#"SELECT NULL, * FROM "{quoted}""#)
        };
        if let Some(column) = declared
            && !table.columns.iter().any(|c| c == column)
        {
            return Err(ExportError::Invalid(format!(
                "declared cursor {column} is missing in {}",
                table.name
            )));
        }
        let select = if let Some(column) = declared.filter(|_| !full) {
            format!(
                r#"{select} WHERE "{}">{}"#,
                column.replace('"', "\"\""),
                previous
                    .as_ref()
                    .and_then(|p| p.value)
                    .expect("stored cursor")
            )
        } else {
            select
        };
        let mut high = if full {
            None
        } else {
            previous.as_ref().and_then(|p| p.value)
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
            let key = if table.key.is_empty() {
                if table.rowid.is_some() {
                    json!([value(row.get_ref(0).map_err(sql_err)?)])
                } else {
                    json!({"position":position})
                }
            } else {
                Value::Array(
                    table
                        .key
                        .iter()
                        .map(|&i| values[&table.columns[i]].clone())
                        .collect(),
                )
            };
            if let Some(column) = declared {
                let value = values.get(column).and_then(Value::as_i64).ok_or_else(|| {
                    ExportError::Invalid(format!(
                        "declared cursor {column} in {} is not an integer",
                        table.name
                    ))
                })?;
                high = Some(high.map_or(value, |old: i64| old.max(value)));
            }
            let encoded = Value::Object(values);
            if catalog.changed(&table.name, &key, &encoded, full)? {
                serde_json::to_writer(&mut bytes, &json!({"t":table.name,"k":key,"r":encoded,"at_ms":at_ms,"encoding":"sqlite"}))
                    .map_err(std::io::Error::from)?;
                bytes.push(b'\n');
                done.rows += 1;
                crate::fault::check("tool_export.after_row", &table.name)?;
            }
            if bytes.len() >= APPEND_BYTES {
                send(std::mem::take(&mut bytes), None)?;
            }
        }
        if full {
            catalog.gone(&table.name, &mut |key| {
                stopped()?;
                serde_json::to_writer(&mut bytes, &json!({"t":table.name,"k":key,"gone":true,"at_ms":at_ms}))
                    .map_err(std::io::Error::from)?;
                bytes.push(b'\n');
                done.gone += 1;
                crate::fault::check("tool_export.after_gone", &table.name)?;
                if bytes.len() >= APPEND_BYTES {
                    send(std::mem::take(&mut bytes), None)?;
                }
                Ok(())
            })?;
        }
        if let Some(column) = declared {
            saved.cursors.insert(
                table.name.clone(),
                catalog::Cursor {
                    column: column.into(),
                    value: high,
                    verified_at: if full {
                        at_ms
                    } else {
                        previous.expect("stored cursor").verified_at
                    },
                },
            );
        } else {
            saved.cursors.remove(&table.name);
        }
    }
    catalog.dropped(
        &tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        &mut |table, key| {
            stopped()?;
            serde_json::to_writer(&mut bytes, &json!({"t":table,"k":key,"gone":true,"at_ms":at_ms}))
                .map_err(std::io::Error::from)?;
            bytes.push(b'\n');
            done.gone += 1;
            if bytes.len() >= APPEND_BYTES {
                send(std::mem::take(&mut bytes), None)?;
            }
            Ok(())
        },
    )?;
    saved.schema = Some(schema);
    stopped()?;
    let unchanged_file = source_file.is_some()
        && file_identity(db) == source_file
        && DbFacts::of(db)
            .db
            .is_some_and(|now| now.device == facts.device && now.inode == facts.inode);
    saved.source_file = if unchanged_file { source_file } else { None };
    if !unchanged_file {
        saved.cursors.clear();
    }
    saved
        .cursors
        .retain(|table, _| tables.iter().any(|t| &t.name == table));
    done.has_cursor = !saved.cursors.is_empty();
    if !bytes.is_empty() || saved != catalog.saved {
        send(bytes, Some(saved))?;
    }
    Ok(done)
}

#[cfg(test)]
mod tests;
