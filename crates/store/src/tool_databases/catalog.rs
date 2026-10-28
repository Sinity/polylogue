use super::*;
use rusqlite::{OptionalExtension, params};

/// Row comparison state, kept beside the capture as `<capture>.rows` and
/// stamped with the prefix it describes (its length and segment chain), so
/// a restart reuses it instead of replaying the whole capture. A catalog
/// whose stamp names another prefix is made again from the capture. Rust
/// holds one decoded row at a time.
pub(super) struct Catalog {
    pub conn: Connection,
    pub schema: Option<u64>,
    pub binding: Option<super::binding::Binding>,
    pub unbound: bool,
    /// The source database's facts at the export the stamp names.
    pub exported: Option<DbFacts>,
}

fn invalid(reason: &str) -> ExportError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, reason).into()
}

pub(super) fn rows_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".rows");
    name.into()
}

const SCHEMA: &str = "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
    PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
    CREATE TABLE IF NOT EXISTS row_state(t TEXT NOT NULL, k TEXT NOT NULL, kh TEXT NOT NULL,
      h BLOB NOT NULL, PRIMARY KEY(t,k)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS stamp(length INTEGER NOT NULL, chain BLOB NOT NULL,
      schema INTEGER, binding TEXT, unbound INTEGER NOT NULL, exported TEXT);";

/// The stamp row's values.
type Stamp = (
    i64,
    Vec<u8>,
    Option<i64>,
    Option<String>,
    bool,
    Option<String>,
);

impl Catalog {
    /// The catalog for the capture at `out` whose verified prefix is
    /// `prefix`: the kept one when its stamp names that prefix, or else one
    /// made again from the capture's lines.
    pub fn open(
        out: &Path,
        prefix: (u64, [u8; 32]),
        checkpoint: &mut dyn FnMut() -> Result<(), ExportError>,
    ) -> Result<Self, ExportError> {
        let path = rows_path(out);
        let sql_err = |e| ExportError::Sqlite(path.clone(), e);
        if path.exists() {
            let conn = Connection::open(&path).map_err(sql_err)?;
            let kept: Option<Stamp> = conn
                .execute_batch(SCHEMA)
                .and_then(|()| {
                    conn.query_row(
                        "SELECT length, chain, schema, binding, unbound, exported FROM stamp",
                        [],
                        |r| {
                            Ok((
                                r.get(0)?,
                                r.get(1)?,
                                r.get(2)?,
                                r.get(3)?,
                                r.get(4)?,
                                r.get(5)?,
                            ))
                        },
                    )
                    .optional()
                })
                .ok()
                .flatten();
            if let Some((length, chain, schema, binding, unbound, exported)) = kept
                && length == prefix.0 as i64
                && chain == prefix.1
            {
                return Ok(Self {
                    conn,
                    schema: schema.map(|s| s as u64),
                    binding: binding
                        .map(|b| serde_json::from_str(&b).map_err(std::io::Error::from))
                        .transpose()?,
                    unbound,
                    exported: exported.and_then(|e| DbFacts::decode(&e)),
                });
            }
            drop(conn);
            for suffix in ["", "-journal", "-wal", "-shm"] {
                let mut name = path.as_os_str().to_owned();
                name.push(suffix);
                match fs::remove_file(&name) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
        let conn = Connection::open(&path).map_err(sql_err)?;
        let mut catalog = Self {
            conn,
            schema: None,
            binding: None,
            unbound: false,
            exported: None,
        };
        catalog
            .conn
            .execute_batch(&format!("{SCHEMA} BEGIN;"))
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
        write_stamp(&catalog.conn, prefix, &catalog.stamp_values()).map_err(sql_err)?;
        catalog.conn.execute_batch("COMMIT").map_err(sql_err)?;
        Ok(catalog)
    }

    /// What the stamp records besides the prefix.
    pub fn stamp_values(&self) -> StampValues {
        StampValues {
            schema: self.schema,
            binding: self.binding.clone(),
            unbound: self.unbound,
            exported: self.exported,
        }
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
            _ => Err(invalid("row dump source binding differs or is missing")),
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

/// What a catalog's stamp records besides the prefix.
pub(super) struct StampValues {
    pub schema: Option<u64>,
    pub binding: Option<super::binding::Binding>,
    pub unbound: bool,
    pub exported: Option<DbFacts>,
}

/// Stamps a catalog, inside the caller's transaction, as describing `prefix`.
pub(super) fn write_stamp(
    conn: &Connection,
    prefix: (u64, [u8; 32]),
    values: &StampValues,
) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM stamp", [])?;
    conn.execute(
        "INSERT INTO stamp(length, chain, schema, binding, unbound, exported)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            prefix.0 as i64,
            prefix.1.as_slice(),
            values.schema.map(|s| s as i64),
            values
                .binding
                .as_ref()
                .map(|b| serde_json::to_string(b).unwrap_or_default()),
            values.unbound,
            values.exported.map(|e| e.encode()),
        ],
    )?;
    Ok(())
}

/// Starts an export's comparison: the keys it reads are noted in a
/// temporary table, so the kept rows are written only where they changed.
pub(super) fn start(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS seen(t TEXT NOT NULL, k TEXT NOT NULL,
           PRIMARY KEY(t,k)) WITHOUT ROWID;
         DELETE FROM temp.seen;",
    )
}

/// Notes a unique source key and compares its complete content hash.
pub(super) fn changed(
    conn: &Connection,
    t: &str,
    k: &Value,
    h: &blake3::Hash,
) -> rusqlite::Result<bool> {
    let key = k.to_string();
    let new = conn
        .prepare_cached("INSERT OR IGNORE INTO temp.seen(t,k) VALUES (?1,?2)")?
        .execute(params![t, key])?;
    if new == 0 {
        return Err(rusqlite::Error::InvalidParameterName(format!(
            "non-unique SQLite export key in {t}: {key}"
        )));
    }
    let old: Option<Vec<u8>> = conn
        .prepare_cached("SELECT h FROM row_state WHERE t=?1 AND k=?2")?
        .query_row(params![t, key], |r| r.get(0))
        .optional()?;
    let changed = old.as_ref().is_none_or(|old| old != h.as_bytes());
    if changed {
        conn.prepare_cached(
            "INSERT INTO row_state(t,k,kh,h) VALUES (?1,?2,?3,?4)
            ON CONFLICT(t,k) DO UPDATE SET h=excluded.h",
        )?
        .execute(params![
            t,
            key,
            format!("{:016x}", row_key(t, k)),
            h.as_bytes().as_slice()
        ])?;
    }
    Ok(changed)
}

/// The kept rows the export did not read: gone from the source.
pub(super) const GONE: &str = "SELECT t,k,kh FROM row_state r
    WHERE NOT EXISTS (SELECT 1 FROM temp.seen s WHERE s.t=r.t AND s.k=r.k) ORDER BY t,k";
