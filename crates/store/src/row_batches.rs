//! `archive.db` rows grouped into row batches, for primary mode on Sinex.
//!
//! A row batch is what one archive write added and Sinex holds whole: a
//! revision with its input, extents, blob rows and observations; a note with
//! its targets and relations; an excision; a setting; a delivery; a later
//! observation; or a later delivery event. Batches are
//! ordered by the archive's write counter (`commit_seq`), then kind and key.
//! The archive only appends batches once Sinex holds it (excision refuses),
//! so a batch's position in that order never changes.
//!
//! The store lists batches and visits their rows as tables, column names and
//! values; `polylogue-sinex` owns how rows are encoded for Sinex. [`Restore`]
//! inserts decoded rows into an empty archive.
use crate::{Archive, Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::types::{Value as Sql, ValueRef};
use rusqlite::{Connection, MAIN_DB, params, params_from_iter};
use std::io::Read;
use std::path::Path;

/// The `meta` key set once Sinex holds the archive's rows.
const ON_SINEX: &str = "on_sinex";

/// One row batch's identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowBatch {
    pub commit_seq: i64,
    pub kind: String,
    /// `<input>/<number>` for a revision, a JSON array of the primary key for
    /// a delivery event, and the row ID otherwise.
    pub key: String,
}

/// One column value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl From<ValueRef<'_>> for Value {
    fn from(value: ValueRef<'_>) -> Self {
        match value {
            ValueRef::Null => Self::Null,
            ValueRef::Integer(n) => Self::Integer(n),
            ValueRef::Real(x) => Self::Real(x),
            ValueRef::Text(t) => Self::Text(String::from_utf8_lossy(t).into_owned()),
            ValueRef::Blob(b) => Self::Blob(b.to_vec()),
        }
    }
}

impl From<&Value> for Sql {
    fn from(value: &Value) -> Self {
        match value {
            Value::Null => Sql::Null,
            Value::Integer(n) => Sql::Integer(*n),
            Value::Real(x) => Sql::Real(*x),
            Value::Text(t) => Sql::Text(t.clone()),
            Value::Blob(b) => Sql::Blob(b.clone()),
        }
    }
}

/// One row of a batch.
pub struct Row<'a> {
    pub table: &'a str,
    pub columns: &'a [String],
    pub values: &'a [Value],
}

/// The tables row batches carry, and so the tables a restore may write.
const TABLES: &[&str] = &[
    "input",
    "revision",
    "blob",
    "extent",
    "machine",
    "observation",
    "observation_container",
    "note",
    "note_target",
    "note_relation",
    "excision",
    "excision_document",
    "setting",
    "delivery",
    "delivery_event",
];

/// The row batches one write added (`commit_seq = ?1`), keyed as [`RowBatch`].
const COMMIT_BATCHES: &str = "
    SELECT 'revision' AS kind, input || '/' || number AS key FROM revision WHERE commit_seq = ?1
    UNION ALL SELECT 'note', id FROM note WHERE commit_seq = ?1
    UNION ALL SELECT 'excision', id FROM excision WHERE commit_seq = ?1
    UNION ALL SELECT 'setting', id FROM setting WHERE commit_seq = ?1
    UNION ALL SELECT 'delivery', id FROM delivery WHERE commit_seq = ?1
    UNION ALL SELECT 'observation', id FROM observation WHERE commit_seq = ?1
    UNION ALL SELECT 'delivery_event', json_array(delivery, what, at)
        FROM delivery_event WHERE commit_seq = ?1";

/// The first write after `?1` that added a row batch. Rows with
/// `commit_seq = 0` belong to their parent's batch.
const NEXT_COMMIT: &str = "SELECT MIN(seq) FROM (
    SELECT MIN(commit_seq) AS seq FROM revision WHERE commit_seq > ?1
    UNION ALL SELECT MIN(commit_seq) FROM note WHERE commit_seq > ?1
    UNION ALL SELECT MIN(commit_seq) FROM excision WHERE commit_seq > ?1
    UNION ALL SELECT MIN(commit_seq) FROM setting WHERE commit_seq > ?1
    UNION ALL SELECT MIN(commit_seq) FROM delivery WHERE commit_seq > ?1
    UNION ALL SELECT MIN(commit_seq) FROM observation WHERE commit_seq > ?1
    UNION ALL SELECT MIN(commit_seq) FROM delivery_event WHERE commit_seq > ?1)";

/// In Sinex every blob is a plain file in its content store, so a batch's
/// blob rows say so: no inline bytes and no delta base.
const PLAIN_BLOB: &str = "SELECT hash, size, size AS stored_size, 'file' AS place,
    'plain' AS encoding, NULL AS base, 'present' AS state, NULL AS data FROM blob";

/// The extents a revision added to its chain: from the end of the chain's
/// previous revision to its own end.
const REVISION_EXTENTS: &str = "SELECT e.* FROM extent e JOIN revision r
      ON r.input = e.input AND r.number = ?2 AND e.chain = r.chain
    WHERE e.input = ?1 AND e.offset < r.length
      AND e.offset >= COALESCE((SELECT MAX(p.length) FROM revision p
          WHERE p.input = r.input AND p.chain = r.chain AND p.number < r.number), 0)";

/// One read transaction over `archive.db`, held for a whole Sinex pass.
pub struct Snapshot {
    conn: Connection,
    blobs: hashed_blobs::Store,
}

impl Snapshot {
    pub fn open(root: &Path) -> Result<Self> {
        let conn = crate::archive::open_read(&root.join("archive.db"))?;
        // Ordering every batch of a large archive may spill past the cache.
        conn.pragma_update(None, "temp_store", "FILE")?;
        conn.execute_batch("BEGIN")?;
        Ok(Self {
            conn,
            blobs: hashed_blobs::Store::open(root.to_path_buf(), vec![], 0o600)?,
        })
    }

    pub fn archive_id(&self) -> Result<String> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = 'archive_id'", [], |r| {
                r.get(0)
            })?)
    }

    /// The batches after `after`, in write order, a whole write at a time,
    /// until at least `limit` are listed or none are left. Each step reads
    /// one write through the `commit_seq` indexes, so paging through a large
    /// archive never sorts it.
    pub fn batches(&self, after: Option<&RowBatch>, limit: usize) -> Result<Vec<RowBatch>> {
        let mut out = Vec::new();
        let mut seq = 0;
        if let Some(after) = after {
            seq = after.commit_seq;
            self.commit(seq, Some((&after.kind, &after.key)), &mut out)?;
        }
        let mut next = self.conn.prepare_cached(NEXT_COMMIT)?;
        while out.len() < limit {
            let Some(found) = next.query_row([seq], |r| r.get::<_, Option<i64>>(0))? else {
                break;
            };
            seq = found;
            self.commit(seq, None, &mut out)?;
        }
        Ok(out)
    }

    /// The batches of write `seq`, after `(kind, key)` when given.
    fn commit(&self, seq: i64, after: Option<(&str, &str)>, out: &mut Vec<RowBatch>) -> Result<()> {
        let (kind, key) = after.unwrap_or(("", ""));
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT kind, key FROM ({COMMIT_BATCHES})
             WHERE (kind, key) > (?2, ?3) ORDER BY kind, key"
        ))?;
        let rows = statement.query_map(params![seq, kind, key], |r| {
            Ok(RowBatch {
                commit_seq: seq,
                kind: r.get(0)?,
                key: r.get(1)?,
            })
        })?;
        for batch in rows {
            out.push(batch?);
        }
        Ok(())
    }

    /// Visits a batch's rows, parents before the rows that refer to them. A
    /// batch whose bytes are still being written returns
    /// [`StoreError::InputUnfinished`]; a later pass sends it.
    pub fn rows(
        &self,
        batch: &RowBatch,
        mut visit: impl FnMut(Row<'_>) -> Result<()>,
    ) -> Result<()> {
        let mut add = |table: &str, sql: &str, args: &[Sql]| -> Result<()> {
            let mut statement = self.conn.prepare(sql)?;
            let columns: Vec<String> = statement
                .column_names()
                .iter()
                .map(|name| (*name).to_owned())
                .collect();
            let mut rows = statement.query(params_from_iter(args))?;
            let mut values = Vec::with_capacity(columns.len());
            while let Some(row) = rows.next()? {
                values.clear();
                for index in 0..columns.len() {
                    values.push(Value::from(row.get_ref(index)?));
                }
                visit(Row {
                    table,
                    columns: &columns,
                    values: &values,
                })?;
            }
            Ok(())
        };
        let key = [Sql::Text(batch.key.clone())];
        match batch.kind.as_str() {
            "revision" => {
                let (input, number) = revision_key(&batch.key)?;
                let args = [Sql::Text(input.to_owned()), Sql::Integer(number)];
                self.require_present(
                    input,
                    &format!("SELECT blob FROM ({REVISION_EXTENTS})"),
                    &args,
                )?;
                add("input", "SELECT * FROM input WHERE id = ?1", &args[..1])?;
                add(
                    "revision",
                    "SELECT * FROM revision WHERE input = ?1 AND number = ?2",
                    &args,
                )?;
                add(
                    "blob",
                    &format!("{PLAIN_BLOB} WHERE hash IN (SELECT blob FROM ({REVISION_EXTENTS}))"),
                    &args,
                )?;
                add("extent", REVISION_EXTENTS, &args)?;
                add(
                    "machine",
                    "SELECT id, host, first_seen FROM machine WHERE id IN (SELECT machine
                     FROM observation WHERE input = ?1 AND revision = ?2 AND commit_seq = 0)",
                    &args,
                )?;
                add(
                    "observation",
                    "SELECT * FROM observation
                     WHERE input = ?1 AND revision = ?2 AND commit_seq = 0 ORDER BY rowid",
                    &args,
                )?;
            }
            "observation" => {
                // Wait for the bytes of the observed input and of the container
                // it was read from.
                self.require_present(
                    &batch.key,
                    "SELECT blob FROM extent WHERE input IN (
                        SELECT input FROM observation WHERE id = ?1
                        UNION SELECT container_input FROM observation_container
                            WHERE observation_id = ?1)",
                    &key,
                )?;
                add(
                    "input",
                    "SELECT * FROM input WHERE id IN (SELECT input FROM observation WHERE id = ?1)",
                    &key,
                )?;
                add(
                    "machine",
                    "SELECT id, host, first_seen FROM machine
                     WHERE id IN (SELECT machine FROM observation WHERE id = ?1)",
                    &key,
                )?;
                add(
                    "observation",
                    "SELECT * FROM observation WHERE id = ?1",
                    &key,
                )?;
                add(
                    "observation_container",
                    "SELECT * FROM observation_container WHERE observation_id = ?1",
                    &key,
                )?;
            }
            "delivery_event" => {
                let (delivery, what, at): (String, String, i64) = serde_json::from_str(&batch.key)
                    .map_err(|_| invalid("invalid delivery event batch key"))?;
                add(
                    "delivery_event",
                    "SELECT * FROM delivery_event WHERE delivery = ?1 AND what = ?2 AND at = ?3",
                    &[Sql::Text(delivery), Sql::Text(what), Sql::Integer(at)],
                )?;
            }
            "note" => {
                add("note", "SELECT * FROM note WHERE id = ?1", &key)?;
                add(
                    "note_target",
                    "SELECT * FROM note_target WHERE note_id = ?1",
                    &key,
                )?;
                add(
                    "note_relation",
                    "SELECT * FROM note_relation WHERE note_id = ?1",
                    &key,
                )?;
            }
            "excision" => {
                let excised = "SELECT * FROM observation
                    WHERE class = 'excised' AND excision = ?1 AND commit_seq = 0";
                add(
                    "input",
                    &format!("SELECT * FROM input WHERE id IN (SELECT input FROM ({excised}))"),
                    &key,
                )?;
                add(
                    "machine",
                    &format!(
                        "SELECT id, host, first_seen FROM machine
                         WHERE id IN (SELECT machine FROM ({excised}))"
                    ),
                    &key,
                )?;
                add("excision", "SELECT * FROM excision WHERE id = ?1", &key)?;
                add(
                    "excision_document",
                    "SELECT * FROM excision_document WHERE excision_id = ?1 ORDER BY document_id",
                    &key,
                )?;
                add("observation", &format!("{excised} ORDER BY rowid"), &key)?;
            }
            "setting" => add("setting", "SELECT * FROM setting WHERE id = ?1", &key)?,
            "delivery" => {
                add("delivery", "SELECT * FROM delivery WHERE id = ?1", &key)?;
                add(
                    "delivery_event",
                    "SELECT * FROM delivery_event WHERE delivery = ?1 AND commit_seq = 0",
                    &key,
                )?;
            }
            kind => return Err(invalid(&format!("unknown row batch kind {kind}"))),
        }
        Ok(())
    }

    /// Refuses until every blob the query names is completely written.
    fn require_present(&self, input: &str, blobs: &str, args: &[Sql]) -> Result<()> {
        let unfinished: bool = self.conn.query_row(
            &format!(
                "SELECT EXISTS(SELECT 1 FROM blob WHERE hash IN ({blobs}) AND state <> 'present')"
            ),
            params_from_iter(args),
            |r| r.get(0),
        )?;
        if unfinished {
            return Err(unfinished_input(input));
        }
        Ok(())
    }

    /// A stored blob's length and its plain bytes. The reader does not check
    /// the hash; the caller does while copying.
    pub fn blob(&self, hash: &BlobHash) -> Result<(u64, Box<dyn Read + '_>)> {
        let (rowid, place, encoding, size, state): (i64, String, String, i64, String) =
            self.conn.query_row(
                "SELECT rowid, place, encoding, size, state FROM blob WHERE hash = ?1",
                [&hash.0[..]],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )?;
        if state != "present" {
            return Err(unfinished_input(&hash.to_string()));
        }
        if encoding != "plain" {
            return Err(invalid(&format!(
                "blob {hash} has {encoding} encoding, not plain"
            )));
        }
        let size = u64::try_from(size).map_err(|_| StoreError::BlobCorrupt(*hash))?;
        let reader: Box<dyn Read + '_> = match place.as_str() {
            "inline" => Box::new(zstd::stream::read::Decoder::new(
                self.conn.blob_open(MAIN_DB, "blob", "data", rowid, true)?,
            )?),
            "file" => Box::new(self.blobs.read_plain(
                &hashed_blobs::Hash(hash.0),
                Some(size),
                || Ok(()),
            )?),
            _ => return Err(invalid(&format!("unknown blob place {place}"))),
        };
        Ok((size, reader))
    }
}

/// Inserts the rows of every batch Sinex holds into an empty archive, in one
/// transaction. The batches may arrive in any order: foreign keys are checked
/// at commit. A row identical to one already inserted is skipped, so
/// receiving a batch twice changes nothing; a different row with the same
/// key refuses.
pub struct Restore {
    conn: Connection,
    archive: Archive,
    write_counter: i64,
    rows: u64,
}

impl Restore {
    pub fn begin(archive: &Archive) -> Result<Self> {
        let conn = crate::archive::open_write(
            &archive.archive_db(),
            crate::archive::cache::ARCHIVE_WRITER,
        )?;
        if !is_empty(&conn)? {
            return Err(invalid("a restore from Sinex needs an empty archive"));
        }
        conn.execute_batch("BEGIN IMMEDIATE; PRAGMA defer_foreign_keys = ON;")?;
        Ok(Self {
            conn,
            archive: archive.clone(),
            write_counter: 0,
            rows: 0,
        })
    }

    pub fn insert(&mut self, row: Row<'_>) -> Result<()> {
        if !TABLES.contains(&row.table) {
            return Err(invalid(&format!(
                "a row names no known table: {}",
                row.table
            )));
        }
        if row.columns.len() != row.values.len()
            || row.columns.iter().any(|column| {
                column.is_empty()
                    || !column
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
        {
            return Err(invalid(&format!("bad columns in a {} row", row.table)));
        }
        let values: Vec<Sql> = row.values.iter().map(Sql::from).collect();
        let columns = row.columns.join(", ");
        let placeholders = (1..=values.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let added = self.conn.execute(
            &format!(
                "INSERT OR IGNORE INTO {}({columns}) VALUES ({placeholders})",
                row.table
            ),
            params_from_iter(&values),
        )?;
        if added == 0 {
            let same = row
                .columns
                .iter()
                .enumerate()
                .map(|(i, column)| format!("{column} IS ?{}", i + 1))
                .collect::<Vec<_>>()
                .join(" AND ");
            let identical: bool = self.conn.query_row(
                &format!("SELECT EXISTS(SELECT 1 FROM {} WHERE {same})", row.table),
                params_from_iter(&values),
                |r| r.get(0),
            )?;
            if !identical {
                return Err(invalid(&format!(
                    "a restored {} row conflicts with another row with the same key",
                    row.table
                )));
            }
        }
        if let Some(index) = row.columns.iter().position(|c| c == "commit_seq")
            && let Value::Integer(seq) = row.values[index]
        {
            self.write_counter = self.write_counter.max(seq);
        }
        self.rows += added as u64;
        Ok(())
    }

    /// Commits the restored rows under the archive's identity and returns how
    /// many rows were inserted. The loop then builds `index.db` from them.
    pub fn finish(self, archive_id: &str) -> Result<u64> {
        if archive_id.is_empty() {
            return Err(invalid("the restored archive has no identity"));
        }
        let broken: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_foreign_key_check)",
            [],
            |r| r.get(0),
        )?;
        if broken {
            return Err(invalid(
                "the restored rows refer to rows Sinex does not hold",
            ));
        }
        for (key, value) in [
            ("archive_id", archive_id.to_owned()),
            ("write_counter", self.write_counter.to_string()),
            (ON_SINEX, "1".to_owned()),
        ] {
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [key, value.as_str()],
            )?;
        }
        self.conn.execute_batch("COMMIT")?;
        self.archive.prepare_index()?;
        Ok(self.rows)
    }
}

/// Records that Sinex holds this archive's rows. From then on the archive
/// only appends batches: excision refuses.
pub fn mark_on_sinex(archive: &Archive) -> Result<()> {
    let conn = crate::archive::open_write(&archive.archive_db(), crate::archive::cache::SETUP)?;
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?1, '1') ON CONFLICT(key) DO NOTHING",
        [ON_SINEX],
    )?;
    Ok(())
}

pub(crate) fn on_sinex(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM meta WHERE key = ?1)",
        [ON_SINEX],
        |r| r.get(0),
    )?)
}

/// No batch rows yet: only the identity and counters `meta` starts with.
fn is_empty(conn: &Connection) -> Result<bool> {
    for table in TABLES {
        let any: bool =
            conn.query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table})"), [], |r| {
                r.get(0)
            })?;
        if any {
            return Ok(false);
        }
    }
    Ok(true)
}

fn revision_key(key: &str) -> Result<(&str, i64)> {
    key.rsplit_once('/')
        .and_then(|(input, number)| Some((input, number.parse().ok()?)))
        .filter(|(_, number)| *number > 0)
        .ok_or_else(|| invalid("invalid revision batch key"))
}

fn unfinished_input(input: &str) -> StoreError {
    StoreError::InputUnfinished {
        input: input.to_owned(),
        reason: "its bytes are still being written".into(),
    }
}

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
