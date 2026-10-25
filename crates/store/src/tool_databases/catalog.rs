use super::*;
use rusqlite::{OptionalExtension, params};

/// Disposable row comparison state. SQLite spills the index and transaction
/// journal to disk; Rust holds one decoded row at a time.
pub(super) struct Catalog {
    pub conn: Connection,
    pub schema: Option<u64>,
    pub binding: Option<super::binding::Binding>,
    unbound: bool,
}

fn invalid(reason: &str) -> ExportError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, reason).into()
}

impl Catalog {
    pub fn load(
        out: &Path,
        checkpoint: &mut dyn FnMut() -> Result<(), ExportError>,
    ) -> Result<Self, ExportError> {
        let conn = Connection::open("").map_err(|e| ExportError::Sqlite(out.into(), e))?;
        let mut catalog = Self {
            conn,
            schema: None,
            binding: None,
            unbound: false,
        };
        let sql_err = |e| ExportError::Sqlite(out.into(), e);
        catalog
            .conn
            .execute_batch(
                "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
             CREATE TABLE row_state(t TEXT NOT NULL, k TEXT NOT NULL, kh TEXT NOT NULL,
               h BLOB NOT NULL, seen INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(t,k)) WITHOUT ROWID;
             CREATE INDEX row_hash ON row_state(t,kh);
             BEGIN;",
            )
            .map_err(sql_err)?;
        match fs::File::open(out) {
            Ok(file) => {
                let mut reader = BufReader::new(file);
                let mut bytes = Vec::new();
                loop {
                    bytes.clear();
                    if reader.read_until(b'\n', &mut bytes)? == 0 {
                        break;
                    }
                    if bytes.last() != Some(&b'\n') {
                        return Err(invalid("incomplete SQLite export line"));
                    }
                    let value: Value =
                        serde_json::from_slice(&bytes).map_err(std::io::Error::from)?;
                    catalog.replay(out, value)?;
                    checkpoint()?;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        catalog.conn.execute_batch("COMMIT").map_err(sql_err)?;
        Ok(catalog)
    }

    pub(super) fn bind(
        &mut self,
        requested: Option<super::binding::Binding>,
    ) -> Result<(), ExportError> {
        match (&self.binding, requested) {
            (Some(previous), Some(current)) if previous == &current => Ok(()),
            (None, None) => Ok(()),
            (None, Some(current)) if !self.unbound => {
                self.binding = Some(current);
                Ok(())
            }
            _ => Err(invalid(
                "row dump source binding differs or is missing",
            )),
        }
    }

    fn replay(&mut self, out: &Path, v: Value) -> Result<(), ExportError> {
        let sql_err = |e| ExportError::Sqlite(out.into(), e);
        if let Some(schema) = v.get("schema").filter(|v| v.is_object()) {
            let binding = v
                .get("source")
                .filter(|value| !value.is_null())
                .map(|value| serde_json::from_value(value.clone()).map_err(std::io::Error::from))
                .transpose()?;
            self.bind(binding)?;
            self.unbound |= self.binding.is_none();
            self.schema = Some(digest(&schema.to_string()));
            return Ok(());
        }
        self.unbound |= self.binding.is_none();
        let t = v
            .get("t")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("SQLite export row has no table"))?;
        if v.get("gone") == Some(&Value::Bool(true)) {
            let k = v
                .get("k")
                .ok_or_else(|| invalid("SQLite removal has no key"))?;
            self.conn
                .execute(
                    "DELETE FROM row_state WHERE t=?1 AND k=?2",
                    params![t, k.to_string()],
                )
                .map_err(sql_err)?;
        } else {
            let k = v
                .get("k")
                .ok_or_else(|| invalid("SQLite export row has no key"))?;
            let row = v
                .get("r")
                .filter(|v| v.is_object())
                .ok_or_else(|| invalid("SQLite export row has no values"))?;
            let h = blake3::hash(row.to_string().as_bytes());
            self.conn
                .prepare_cached(
                    "INSERT INTO row_state(t,k,kh,h) VALUES (?1,?2,?3,?4)
                ON CONFLICT(t,k) DO UPDATE SET kh=excluded.kh,h=excluded.h",
                )
                .map_err(sql_err)?
                .execute(params![
                    t,
                    k.to_string(),
                    format!("{:016x}", row_key(t, k)),
                    h.as_bytes().as_slice()
                ])
                .map_err(sql_err)?;
        }
        Ok(())
    }
}

/// Mark a unique source key and compare its complete content hash.
pub(super) fn changed(
    conn: &Connection,
    t: &str,
    k: &Value,
    h: &blake3::Hash,
) -> rusqlite::Result<bool> {
    let key = k.to_string();
    let old: Option<(Vec<u8>, bool)> = conn
        .prepare_cached("SELECT h,seen FROM row_state WHERE t=?1 AND k=?2")?
        .query_row(params![t, key], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    if old.as_ref().is_some_and(|(_, seen)| *seen) {
        return Err(rusqlite::Error::InvalidParameterName(format!(
            "non-unique SQLite export key in {t}: {key}"
        )));
    }
    let changed = old.as_ref().is_none_or(|(old, _)| old != h.as_bytes());
    conn.prepare_cached(
        "INSERT INTO row_state(t,k,kh,h,seen) VALUES (?1,?2,?3,?4,1)
        ON CONFLICT(t,k) DO UPDATE SET h=excluded.h,seen=1",
    )?
    .execute(params![
        t,
        key,
        format!("{:016x}", row_key(t, k)),
        h.as_bytes().as_slice()
    ])?;
    Ok(changed)
}
