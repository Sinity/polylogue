//! Read-only access to the predecessor's source catalog. Its bytes remain
//! external inputs; none of its repair or projection state is imported.
use crate::{Result, StoreError};
use rusqlite::{Connection, OpenFlags};
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Input {
    pub raw_id: String,
    pub origin: String,
    pub source_path: String,
    pub capture_mode: Option<String>,
    pub blob_hash: [u8; 32],
    pub blob_size: u64,
    pub revision_kind: String,
    pub predecessor: Option<String>,
    pub append_start: Option<u64>,
    pub append_end: Option<u64>,
}

pub fn inputs(
    root: &Path,
    mut capture: impl FnMut(&polylogue_model::Json) -> Result<()>,
) -> Result<Option<Vec<Input>>> {
    let catalog = root.join("source.db");
    if !catalog.is_file() || !root.join("blob").is_dir() {
        return Ok(None);
    }
    let conn = Connection::open_with_flags(
        catalog,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.execute_batch("BEGIN")?;
    let mut schema = conn.prepare("PRAGMA table_info(raw_sessions)")?;
    let columns = schema
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()?;
    if !["raw_id", "origin", "source_path", "blob_hash", "blob_size"]
        .iter()
        .all(|name| columns.contains(*name))
    {
        return Err(StoreError::Invalid(
            "predecessor source.db has no declared raw_sessions catalog".into(),
        ));
    }
    let schema: String = conn.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='table' AND name='raw_sessions'",
        [],
        |row| row.get(0),
    )?;
    capture(&serde_json::json!({"table":"raw_sessions","schema":schema}))?;
    let mut original = conn.prepare("SELECT * FROM raw_sessions ORDER BY raw_id")?;
    let names: Vec<_> = original
        .column_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut original_rows = original.query([])?;
    while let Some(row) = original_rows.next()? {
        let mut values = serde_json::Map::new();
        for (index, name) in names.iter().enumerate() {
            use rusqlite::types::ValueRef;
            let value = match row.get_ref(index)? {
                ValueRef::Null => serde_json::Value::Null,
                ValueRef::Integer(n) => n.into(),
                ValueRef::Real(n) => serde_json::Number::from_f64(n)
                    .map(serde_json::Value::Number)
                    .unwrap_or_else(
                        || serde_json::json!({"sqlite_real_bits":format!("{:016x}",n.to_bits())}),
                    ),
                ValueRef::Text(bytes) => match std::str::from_utf8(bytes) {
                    Ok(text) => text.into(),
                    Err(_) => serde_json::json!({"sqlite_text_hex":hex(bytes)}),
                },
                ValueRef::Blob(bytes) => serde_json::json!({"sqlite_blob_hex":hex(bytes)}),
            };
            values.insert(name.clone(), value);
        }
        capture(&serde_json::Value::Object(values))?;
    }
    let optional = |name: &str, default: &str| {
        if columns.contains(name) {
            name.to_owned()
        } else {
            default.to_owned()
        }
    };
    let sql = format!(
        "SELECT raw_id,origin,source_path,blob_hash,blob_size,{},{},{},{},{} FROM raw_sessions ORDER BY raw_id",
        optional("capture_mode", "NULL"),
        optional("revision_kind", "'unknown'"),
        optional("predecessor_raw_id", "NULL"),
        optional("append_start_offset", "NULL"),
        optional("append_end_offset", "NULL")
    );
    let mut rows = conn.prepare(&sql)?;
    let inputs = rows
        .query_map([], |row| {
            let bytes: Vec<u8> = row.get(3)?;
            let blob_hash = bytes.try_into().map_err(|bytes: Vec<u8>| {
                rusqlite::Error::FromSqlConversionFailure(
                    bytes.len(),
                    rusqlite::types::Type::Blob,
                    "SHA256 blob hash must have 32 bytes".into(),
                )
            })?;
            let size: i64 = row.get(4)?;
            let start: Option<i64> = row.get(8)?;
            let end: Option<i64> = row.get(9)?;
            if size < 0 || start.is_some_and(|n| n < 0) || end.is_some_and(|n| n < 0) {
                return Err(rusqlite::Error::InvalidQuery);
            }
            Ok(Input {
                raw_id: row.get(0)?,
                origin: row.get(1)?,
                source_path: row.get(2)?,
                blob_hash,
                blob_size: size as u64,
                capture_mode: row.get(5)?,
                revision_kind: row.get(6)?,
                predecessor: row.get(7)?,
                append_start: start.map(|n| n as u64),
                append_end: end.map(|n| n as u64),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Some(inputs))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
