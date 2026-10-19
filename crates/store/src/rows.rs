//! Units of `archive.db` rows, as published to Sinex and restored from it.
//!
//! A unit is everything one archive write that Sinex must hold whole:
//! a revision with its input, the extents it added, the observations that
//! produced it and the blob rows (without inline bytes) it cites; a note with
//! its targets and relations; an excision; a setting; a delivery with its
//! events. Units are ordered by the archive's write counter (`commit_seq`).
//! Rows travel as JSON lines, one `{"table": …, "row": {…}}` per line, with
//! every column under its own name: integers and reals as numbers, text as
//! strings, blobs as `{"hex": …}`, null as null.
use rusqlite::types::{Value as Sql, ValueRef};
use rusqlite::{Connection, params_from_iter};
use serde_json::{Map, Value, json};

/// One unit of rows, before its rows are read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub commit_seq: i64,
    pub kind: &'static str,
    /// `<input>/<number>` for a revision, the row ID otherwise.
    pub key: String,
}

/// The units at or after `after` (exclusive), in write order.
pub fn units(archive: &Connection, after: i64) -> rusqlite::Result<Vec<Unit>> {
    let mut stmt = archive.prepare(
        "SELECT commit_seq, 'revision', input || '/' || number FROM revision WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'note', id FROM note WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'excision', id FROM excision WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'setting', id FROM setting WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'delivery', id FROM delivery WHERE commit_seq > ?1
         ORDER BY 1, 2, 3",
    )?;
    let rows = stmt.query_map([after], |r| {
        let kind: String = r.get(1)?;
        Ok(Unit {
            commit_seq: r.get(0)?,
            kind: match kind.as_str() {
                "revision" => "revision",
                "note" => "note",
                "excision" => "excision",
                "setting" => "setting",
                _ => "delivery",
            },
            key: r.get(2)?,
        })
    })?;
    rows.collect()
}

/// A unit's rows as JSON lines, and the input blobs they cite.
pub struct Batch {
    pub lines: Vec<u8>,
    pub rows: u64,
    pub blobs: Vec<[u8; 32]>,
}

/// Reads a unit's rows. Each query names its table, so a row is restored
/// into the table it came from.
pub fn read(archive: &Connection, unit: &Unit) -> rusqlite::Result<Batch> {
    let mut out = Batch {
        lines: Vec::new(),
        rows: 0,
        blobs: Vec::new(),
    };
    let key = [Sql::Text(unit.key.clone())];
    match unit.kind {
        "revision" => {
            let (input, number) = unit.key.rsplit_once('/').unwrap_or((&unit.key, "0"));
            let number: i64 = number.parse().unwrap_or(0);
            let args = [Sql::Text(input.to_owned()), Sql::Integer(number)];
            add(
                archive,
                &mut out,
                "input",
                "SELECT * FROM input WHERE id = ?1",
                &args[..1],
            )?;
            add(
                archive,
                &mut out,
                "revision",
                "SELECT * FROM revision WHERE input = ?1 AND number = ?2",
                &args,
            )?;
            // The extents this revision added to its chain: the ones from the
            // end of the chain's previous revision to its own end.
            let extents = "SELECT e.* FROM extent e JOIN revision r
                             ON r.input = e.input AND r.number = ?2 AND e.chain = r.chain
                           WHERE e.input = ?1 AND e.offset < r.length
                             AND e.offset >= COALESCE((SELECT MAX(p.length) FROM revision p
                                 WHERE p.input = r.input AND p.chain = r.chain AND p.number < r.number), 0)";
            // In Sinex every blob is a plain file in its content store, so
            // the restored row says so: no inline bytes, no delta base.
            let blobs_of_extents = format!(
                "SELECT b.hash, b.size, b.size AS stored_size, 'file' AS place,
                        'plain' AS encoding, NULL AS base, 'present' AS state, NULL AS data
                 FROM blob b WHERE b.hash IN (SELECT blob FROM ({extents}))"
            );
            add(archive, &mut out, "blob", &blobs_of_extents, &args)?;
            add(archive, &mut out, "extent", extents, &args)?;
            add(
                archive,
                &mut out,
                "machine",
                "SELECT m.* FROM machine m WHERE m.id IN
                   (SELECT machine FROM observation WHERE input = ?1 AND revision = ?2)",
                &args,
            )?;
            add(
                archive,
                &mut out,
                "observation",
                "SELECT * FROM observation WHERE input = ?1 AND revision = ?2",
                &args,
            )?;
        }
        "note" => {
            add(
                archive,
                &mut out,
                "note",
                "SELECT * FROM note WHERE id = ?1",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "note_target",
                "SELECT * FROM note_target WHERE note_id = ?1",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "note_relation",
                "SELECT * FROM note_relation WHERE note_id = ?1",
                &key,
            )?;
        }
        "excision" => add(
            archive,
            &mut out,
            "excision",
            "SELECT * FROM excision WHERE id = ?1",
            &key,
        )?,
        "setting" => add(
            archive,
            &mut out,
            "setting",
            "SELECT * FROM setting WHERE id = ?1",
            &key,
        )?,
        _ => {
            add(
                archive,
                &mut out,
                "delivery",
                "SELECT * FROM delivery WHERE id = ?1",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "delivery_event",
                "SELECT * FROM delivery_event WHERE delivery = ?1",
                &key,
            )?;
        }
    }
    Ok(out)
}

fn add(
    archive: &Connection,
    out: &mut Batch,
    table: &str,
    sql: &str,
    args: &[Sql],
) -> rusqlite::Result<()> {
    let mut stmt = archive.prepare(sql)?;
    let names: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    let mut rows = stmt.query(params_from_iter(args.iter()))?;
    while let Some(r) = rows.next()? {
        let mut row = Map::new();
        for (i, name) in names.iter().enumerate() {
            let value = match r.get_ref(i)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(n) => json!(n),
                ValueRef::Real(x) => json!(x),
                ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
                ValueRef::Blob(b) => {
                    if table == "blob"
                        && name == "hash"
                        && let Ok(h) = <[u8; 32]>::try_from(b)
                    {
                        out.blobs.push(h);
                    }
                    json!({ "hex": hex(b) })
                }
            };
            row.insert(name.clone(), value);
        }
        let line = json!({ "table": table, "row": row });
        serde_json::to_writer(&mut out.lines, &line).expect("JSON serializes");
        out.lines.push(b'\n');
        out.rows += 1;
    }
    Ok(())
}

/// Inserts a batch's rows into `archive.db`. A row already present is left
/// as it is, so restoring a unit twice changes nothing. Returns the rows
/// inserted.
pub fn apply(archive: &Connection, lines: &[u8]) -> crate::Result<u64> {
    let mut inserted = 0;
    for line in lines.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let v: Value = serde_json::from_slice(line)?;
        let table = v["table"]
            .as_str()
            .filter(|t| TABLES.contains(t))
            .ok_or_else(|| {
                crate::StoreError::Invalid(format!("a row names no known table: {v}"))
            })?;
        let row = v["row"]
            .as_object()
            .ok_or_else(|| crate::StoreError::Invalid("a row is not an object".into()))?;
        let cols: Vec<&String> = row.keys().collect();
        if cols
            .iter()
            .any(|c| !c.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_'))
        {
            return Err(crate::StoreError::Invalid(format!(
                "bad column name in {table}"
            )));
        }
        let sql = format!(
            "INSERT OR IGNORE INTO {table}({}) VALUES ({})",
            cols.iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            (1..=cols.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let values: Vec<Sql> = row.values().map(to_sql).collect::<Result<_, _>>()?;
        inserted += archive.execute(&sql, params_from_iter(values.iter()))? as u64;
    }
    Ok(inserted)
}

/// The tables a batch may restore into.
const TABLES: &[&str] = &[
    "input",
    "revision",
    "blob",
    "extent",
    "machine",
    "observation",
    "note",
    "note_target",
    "note_relation",
    "excision",
    "setting",
    "delivery",
    "delivery_event",
];

fn to_sql(v: &Value) -> crate::Result<Sql> {
    Ok(match v {
        Value::Null => Sql::Null,
        Value::Bool(b) => Sql::Integer(i64::from(*b)),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Sql::Integer(i),
            None => Sql::Real(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Sql::Text(s.clone()),
        Value::Object(o) => match o.get("hex").and_then(Value::as_str) {
            Some(h) => Sql::Blob(unhex(h)?),
            None => {
                return Err(crate::StoreError::Invalid(format!(
                    "unknown column value {v}"
                )));
            }
        },
        Value::Array(_) => {
            return Err(crate::StoreError::Invalid(format!(
                "unknown column value {v}"
            )));
        }
    })
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> crate::Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return Err(crate::StoreError::Invalid(format!("odd-length hex {s}")));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|_| crate::StoreError::Invalid(format!("bad hex {s}")))
        })
        .collect()
}

/// `archive.db` opened for publishing its rows.
pub struct Published {
    conn: Connection,
    blobs: hashed_blobs::Store,
}

impl Published {
    pub fn open(root: &std::path::Path) -> crate::Result<Self> {
        Ok(Self {
            conn: crate::files::open_read(&root.join("archive.db"))?,
            blobs: hashed_blobs::Store::open(root.to_path_buf(), vec![], 0o600)?,
        })
    }

    /// The archive's identity.
    pub fn archive_id(&self) -> crate::Result<String> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = 'archive_id'", [], |r| {
                r.get(0)
            })?)
    }

    pub fn units(&self, after: i64) -> crate::Result<Vec<Unit>> {
        Ok(units(&self.conn, after)?)
    }

    pub fn read(&self, unit: &Unit) -> crate::Result<Batch> {
        Ok(read(&self.conn, unit)?)
    }

    /// A stored blob's bytes, verified.
    pub fn blob(&self, hash: &[u8; 32]) -> crate::Result<Vec<u8>> {
        crate::blobs::read(&self.conn, &self.blobs, &polylogue_model::BlobHash(*hash))
    }
}

/// `archive.db` opened for restoring rows into it.
pub struct Restoring {
    conn: Connection,
}

impl Restoring {
    pub fn open(archive: &crate::Archive) -> crate::Result<Self> {
        Ok(Self {
            conn: crate::files::open_write(
                &archive.archive_db(),
                crate::files::cache::ARCHIVE_WRITER,
            )?,
        })
    }

    /// Inserts one unit's rows in one transaction; returns the rows inserted.
    pub fn apply(&mut self, lines: &[u8]) -> crate::Result<u64> {
        let tx = self.conn.transaction()?;
        let n = apply(&tx, lines)?;
        tx.commit()?;
        Ok(n)
    }

    /// Takes over the restored archive's identity and write counter.
    pub fn finish(&self, archive_id: &str, commit_seq: i64) -> crate::Result<()> {
        for (key, value) in [
            ("archive_id", archive_id.to_owned()),
            ("write_counter", commit_seq.to_string()),
        ] {
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [key, value.as_str()],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(
            "CREATE TABLE note (id TEXT PRIMARY KEY, kind TEXT, body BLOB, commit_seq INTEGER);
             CREATE TABLE note_target (note_id TEXT, ref TEXT, PRIMARY KEY (note_id, ref));
             CREATE TABLE note_relation (note_id TEXT, kind TEXT, other TEXT);",
        )
        .unwrap();
        c
    }

    #[test]
    fn a_unit_round_trips_and_restoring_it_twice_changes_nothing() {
        let a = archive();
        a.execute_batch(
            "INSERT INTO note VALUES ('n1', 'lesson', x'00ff', 7);
             INSERT INTO note_target VALUES ('n1', 'session:claude-code:s1');",
        )
        .unwrap();
        let unit = Unit {
            commit_seq: 7,
            kind: "note",
            key: "n1".into(),
        };
        let batch = read(&a, &unit).unwrap();
        assert_eq!(batch.rows, 2);
        let b = archive();
        assert_eq!(apply(&b, &batch.lines).unwrap(), 2);
        assert_eq!(apply(&b, &batch.lines).unwrap(), 0);
        let body: Vec<u8> = b
            .query_row("SELECT body FROM note WHERE id = 'n1'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(body, [0x00, 0xff]);
    }

    #[test]
    fn a_row_for_an_unknown_table_is_refused() {
        let b = archive();
        let line = br#"{"table":"sqlite_master","row":{"name":"x"}}"#;
        assert!(apply(&b, line).is_err());
    }
}
