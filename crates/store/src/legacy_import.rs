//! Read-only access to the predecessor's source catalog. Its raw sessions
//! become inputs with their source facts and append lineage; every row of
//! every table is kept verbatim, and none of its repair or projection state
//! is acted on.
use crate::{Result, StoreError};
use rusqlite::Connection;
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
    /// The predecessor's name for this revision of its source.
    pub source_revision: Option<String>,
    /// The revision this append continues, by that name; read when the
    /// predecessor recorded no `predecessor` row ID.
    pub predecessor_revision: Option<String>,
}

/// The predecessor's source catalog: `source.db` beside its `blob/` folder.
pub struct Catalog {
    conn: Connection,
    columns: std::collections::BTreeSet<String>,
    _copies: Option<tempfile::TempDir>,
}

impl Catalog {
    /// Opens the catalog in `root`, or `None` when `root` has no `source.db`
    /// and `blob/`. A catalog whose write-ahead log holds frames is copied
    /// with it into `scratch` and the copy is read.
    pub fn open(root: &Path, scratch: &Path) -> Result<Option<Self>> {
        let catalog = root.join("source.db");
        if !catalog.is_file() || !root.join("blob").is_dir() {
            return Ok(None);
        }
        let mut copies = None;
        let conn = crate::older_archive::open_source(&catalog, &mut copies, scratch)?;
        conn.execute_batch("BEGIN")?;
        let columns = crate::older_archive::tables_of(&conn)?
            .remove("raw_sessions")
            .unwrap_or_default()
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        if !["raw_id", "origin", "source_path", "blob_hash", "blob_size"]
            .iter()
            .all(|name| columns.contains(*name))
        {
            return Err(StoreError::Invalid(
                "predecessor source.db has no declared raw_sessions catalog".into(),
            ));
        }
        Ok(Some(Self {
            conn,
            columns,
            _copies: copies,
        }))
    }

    /// Writes every row of every table to `out`, one JSON object per line
    /// (`{"table": …, "row": {column: value}}`), values kept exactly.
    pub fn write_rows(&self, out: &mut dyn std::io::Write) -> Result<u64> {
        let mut written = 0;
        for (table, columns) in crate::older_archive::tables_of(&self.conn)? {
            let order = (1..=columns.len())
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let mut statement = self
                .conn
                .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY {order}"))?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let mut values = serde_json::Map::new();
                for (index, name) in columns.iter().enumerate() {
                    values.insert(
                        name.clone(),
                        crate::older_archive::json_value(row.get_ref(index)?),
                    );
                }
                serde_json::to_writer(
                    &mut *out,
                    &serde_json::json!({"table": table, "row": values}),
                )?;
                out.write_all(b"\n")?;
                written += 1;
            }
        }
        Ok(written)
    }

    pub fn inputs(&self) -> Result<Vec<Input>> {
        let optional = |name: &str, default: &str| {
            if self.columns.contains(name) {
                name.to_owned()
            } else {
                default.to_owned()
            }
        };
        let sql = format!(
            "SELECT raw_id,origin,source_path,blob_hash,blob_size,{},{},{},{},{},{},{} FROM raw_sessions ORDER BY raw_id",
            optional("capture_mode", "NULL"),
            optional("revision_kind", "'unknown'"),
            optional("predecessor_raw_id", "NULL"),
            optional("append_start_offset", "NULL"),
            optional("append_end_offset", "NULL"),
            optional("source_revision", "NULL"),
            optional("predecessor_source_revision", "NULL")
        );
        let mut rows = self.conn.prepare(&sql)?;
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
                    source_revision: row.get(10)?,
                    predecessor_revision: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(inputs)
    }
}
