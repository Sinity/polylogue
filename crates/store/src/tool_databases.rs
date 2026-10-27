//! Tool databases read in one SQLite transaction and emitted as row-dump
//! records. The writer stores these bytes as an ordinary input. Comparison
//! hashes and declared cursors live in the index, beside the parsed rows.
use crate::{Archive, FileFacts, InputKey, Observation};
use base64::Engine as _;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use rusqlite::types::ValueRef;
use serde_json::{Map, Value, json};
use std::{collections::HashMap, fs, path::{Path, PathBuf}, time::{Duration, Instant}};

mod catalog;
pub use catalog::{Comparison, SavedDatabase};

const BUSY_EXPORT: Duration = Duration::from_secs(3600);
const CURSOR_EXPORT: Duration = Duration::from_secs(180);
const VERIFY_EVERY_MS: i64 = 24 * 3600 * 1000;
/// A batching threshold, never a record-size limit.
pub const APPEND_BYTES: usize = 8 << 20;
pub type TableCursor = fn(&Path, &str) -> Option<&'static str>;

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error(transparent)] Io(#[from] std::io::Error),
    #[error(transparent)] Store(#[from] crate::StoreError),
    #[error("{0}: {1}")] Sqlite(PathBuf, rusqlite::Error),
    #[error("{0}")] Invalid(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DbFacts { db: Option<FileFacts>, wal: Option<FileFacts> }
impl DbFacts {
    fn of(db: &Path) -> Self {
        let facts = |p: &Path| fs::metadata(p).ok().map(|m| FileFacts::of(&m));
        let mut wal = db.as_os_str().to_owned(); wal.push("-wal");
        Self { db: facts(db), wal: facts(Path::new(&wal)) }
    }
}
#[derive(Default)]
struct DbState {
    last_seen: Option<DbFacts>, exported: Option<DbFacts>,
    exported_at: Option<Instant>, cursor_at: Option<Instant>,
}
#[derive(Default)]
pub struct Exporter { dbs: HashMap<PathBuf, DbState>, cursor: Option<TableCursor> }
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exported { pub rows: u64, pub gone: u64, pub visited: u64 }
impl Exporter {
    pub fn with_cursor(cursor: TableCursor) -> Self { Self { cursor: Some(cursor), ..Self::default() } }
    pub fn forget(&mut self, path: &Path) { self.dbs.remove(path); }
    /// `send` queues a complete group without waiting for its commit. The
    /// caller waits for all acknowledgments before starting another read of
    /// this input. Its queue bounds bytes awaiting the writer.
    pub fn export(
        &mut self, db: &Path, archive: &Archive, input: &InputKey, observation: &Observation,
        mut send: impl FnMut(Vec<u8>, Comparison) -> Result<(), ExportError>,
    ) -> Result<Option<Exported>, ExportError> {
        let now = DbFacts::of(db);
        let st = self.dbs.entry(db.to_owned()).or_default();
        if now.db.is_none() { *st = DbState::default(); return Ok(None); }
        let quiet = st.last_seen == Some(now);
        st.last_seen = Some(now);
        let compare = st.exported.is_none() || (st.exported != Some(now) &&
            (quiet || st.exported_at.is_some_and(|at| at.elapsed() >= BUSY_EXPORT)));
        let cursor_due = st.cursor_at.is_none_or(|at| at.elapsed() >= CURSOR_EXPORT);
        if !compare && !cursor_due { return Ok(None); }
        let done = write_export(db, archive, input, observation, self.cursor, compare, cursor_due, &mut send)?;
        if compare { st.exported = Some(now); st.exported_at = Some(Instant::now()); }
        st.cursor_at = Some(Instant::now());
        Ok(Some(done))
    }
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


#[allow(clippy::too_many_arguments)]
fn write_export(
    db: &Path, archive: &Archive, input: &InputKey, observation: &Observation,
    cursor: Option<TableCursor>, compare: bool, cursor_due: bool,
    send: &mut dyn FnMut(Vec<u8>, Comparison) -> Result<(), ExportError>,
) -> Result<Exported, ExportError> {
    let _busy = crate::timing::enter(crate::timing::Stage::Database);
    let sql_err = |e| ExportError::Sqlite(db.to_owned(), e);
    let source = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(sql_err)?;
    source.busy_timeout(Duration::from_secs(10)).map_err(sql_err)?;
    source.execute_batch("BEGIN").map_err(sql_err)?;
    let tables = tables(&source).map_err(sql_err)?;
    let catalog = catalog::Catalog::open(archive, &input.id)?;
    let mut saved = catalog.saved.clone();
    let facts = DbFacts::of(db).db.expect("source exists");
    let replaced = saved.source_file != Some((facts.device, facts.inode));
    let schema: Map<String,Value> = tables.iter().map(|t| (t.name.clone(), t.sql.clone().into())).collect();
    let schema = Value::Object(schema);
    let changed_schema = saved.schema.as_ref() != Some(&schema);
    let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
    let mut bytes = Vec::new();
    if changed_schema {
        serde_json::to_writer(&mut bytes, &json!({"db":db.file_name().map(|n| n.to_string_lossy()), "at_ms":at_ms, "schema":schema, "source":{"machine":observation.machine,"path":observation.path}})).map_err(std::io::Error::from)?;
        bytes.push(b'\n');
    }
    let mut done = Exported::default();
    for table in &tables {
        let declared = cursor.and_then(|c| c(db, &table.name));
        let previous = saved.cursors.get(&table.name).cloned();
        let full = replaced || changed_schema || declared.is_none() || previous.as_ref().is_none_or(|p| Some(p.column.as_str()) != declared || p.value.is_none() || at_ms.saturating_sub(p.verified_at) >= VERIFY_EVERY_MS);
        if !replaced && !changed_schema && ((declared.is_none() && !compare) || (declared.is_some() && !cursor_due)) { continue; }
        let quoted = table.name.replace('"', "\"\"");
        let select = if table.key.is_empty() {
            let rowid = ["rowid","_rowid_","oid"].into_iter().find(|name| !table.columns.iter().any(|c| c.eq_ignore_ascii_case(name))).ok_or_else(|| ExportError::Invalid(format!("table {} shadows all SQLite rowid names", table.name)))?;
            format!(r#"SELECT {rowid}, * FROM "{quoted}""#)
        } else { format!(r#"SELECT NULL, * FROM "{quoted}""#) };
        if let Some(column) = declared && !table.columns.iter().any(|c| c == column) { return Err(ExportError::Invalid(format!("declared cursor {column} is missing in {}", table.name))); }
        let select = if let Some(column) = declared.filter(|_| !full) {
            format!(r#"{select} WHERE "{}">{}"#, column.replace('"', "\"\""), previous.as_ref().and_then(|p| p.value).expect("stored cursor"))
        } else { select };
        let mut high = if full { None } else { previous.as_ref().and_then(|p| p.value) };
        let mut statement = source.prepare(&select).map_err(sql_err)?;
        let mut rows = statement.query([]).map_err(sql_err)?;
        while let Some(row) = rows.next().map_err(sql_err)? {
            done.visited += 1;
            let mut values = Map::new();
            for (i, column) in table.columns.iter().enumerate() { values.insert(column.clone(), value(row.get_ref(i+1).map_err(sql_err)?)?); }
            let key = if table.key.is_empty() { json!([value(row.get_ref(0).map_err(sql_err)?)?]) } else { Value::Array(table.key.iter().map(|&i| values[&table.columns[i]].clone()).collect()) };
            if let Some(column) = declared {
                let value = values.get(column).and_then(Value::as_i64).ok_or_else(|| ExportError::Invalid(format!("declared cursor {column} in {} is not an integer",table.name)))?;
                high = Some(high.map_or(value, |old:i64| old.max(value)));
            }
            let encoded = Value::Object(values);
            if catalog.changed(&table.name, &key, &encoded, full)? {
                serde_json::to_writer(&mut bytes, &json!({"t":table.name,"k":key,"h":format!("{:016x}",digest(&encoded.to_string())),"r":encoded})).map_err(std::io::Error::from)?;
                bytes.push(b'\n'); done.rows += 1;
                crate::fault::check("tool_export.after_row", &table.name)?;
            }
            if bytes.len() >= APPEND_BYTES {
                let update = Comparison::records(&bytes, None)?;
                send(std::mem::take(&mut bytes), update)?;
            }
        }
        if full {
            catalog.gone(&table.name, &mut |key| {
                serde_json::to_writer(&mut bytes, &json!({"t":table.name,"k":key,"kh":format!("{:016x}", row_key(&table.name, &key)),"gone":true})).map_err(std::io::Error::from)?;
                bytes.push(b'\n'); done.gone += 1;
                crate::fault::check("tool_export.after_gone", &table.name)?;
                if bytes.len() >= APPEND_BYTES { let update = Comparison::records(&bytes, None)?; send(std::mem::take(&mut bytes), update)?; }
                Ok(())
            })?;
        }
        if let Some(column) = declared { saved.cursors.insert(table.name.clone(), catalog::Cursor { column: column.into(), value: high, verified_at:if full { at_ms } else { previous.expect("stored cursor").verified_at } }); }
        else { saved.cursors.remove(&table.name); }
    }
    catalog.dropped(&tables.iter().map(|t|t.name.as_str()).collect::<Vec<_>>(), &mut |table,key| {
        serde_json::to_writer(&mut bytes,&json!({"t":table,"k":key,"kh":format!("{:016x}",row_key(table,&key)),"gone":true})).map_err(std::io::Error::from)?;
        bytes.push(b'\n'); done.gone += 1;
        if bytes.len() >= APPEND_BYTES { let update = Comparison::records(&bytes,None)?; send(std::mem::take(&mut bytes),update)?; } Ok(())
    })?;
    saved.schema = Some(schema);
    saved.source_file = Some((facts.device,facts.inode));
    saved.cursors.retain(|table,_|tables.iter().any(|t| &t.name==table));
    let update = Comparison::records(&bytes, Some(saved))?;
    send(bytes,update)?;
    Ok(done)
}

#[cfg(test)]
mod tests;
