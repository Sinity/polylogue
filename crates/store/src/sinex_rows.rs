//! Units of `archive.db` rows, as published to Sinex and restored from it.
//!
//! A unit is everything one archive write that Sinex must hold whole:
//! a revision with its input, extents and blob rows (without inline bytes);
//! a note with its targets and relations; an excision; a setting; a delivery;
//! an observation; or a delivery event. Appendable facts own separate units.
//! Units are ordered by the archive's write counter (`commit_seq`).
//! Rows travel as JSON lines, one `{"table": …, "row": {…}}` per line, with
//! every column under its own name: integers and reals as numbers, text as
//! strings, blobs as `{"hex": …}`, null as null.
use serde::{Deserialize, Serialize};

mod excision;
pub(crate) use excision::{Capture, CurrentBatches, visit_retired_local_batches};
pub use excision::{ExcisionPublication, Replacement};
mod restore;
#[cfg(test)]
use restore::select;
pub use restore::{Catalog, RestoreUnit, Selection};

use rusqlite::types::Value as Sql;
use rusqlite::{Connection, params_from_iter};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use std::io::Seek;

/// One unit of rows, before its rows are read.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Unit {
    pub commit_seq: i64,
    pub kind: String,
    /// `<input>/<number>` for a revision, a JSON `[delivery,what,at]` tuple
    /// for a delivery event, and the row ID otherwise.
    pub key: String,
}

fn delivery_event_key(unit: &Unit) -> crate::Result<(String, String, i64)> {
    serde_json::from_str(&unit.key)
        .map_err(|_| crate::StoreError::Invalid("invalid delivery event unit key".into()))
}

/// Units strictly after `after`, in write order.
#[cfg(test)]
fn units(archive: &Connection, after: i64) -> crate::Result<Vec<Unit>> {
    let legacy: bool = archive.query_row(
        "SELECT EXISTS(SELECT 1 FROM excision WHERE json_type(refused_json, '$.publication') IS NULL)",
        [], |r| r.get(0),
    )?;
    if legacy {
        return Err(crate::StoreError::Invalid("primary publication cannot reconstruct unit identities for legacy excisions without replacement metadata".into()));
    }
    let mut stmt = archive.prepare(
        "SELECT commit_seq, 'revision', input || '/' || number FROM revision WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'note', id FROM note WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'excision', id FROM excision WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'setting', id FROM setting WHERE commit_seq > ?1
         UNION ALL SELECT commit_seq, 'delivery', id FROM delivery WHERE commit_seq > ?1
                UNION ALL SELECT commit_seq,'observation',id FROM observation WHERE commit_seq>?1
                UNION ALL SELECT commit_seq,'delivery_event',json_array(delivery,what,at) FROM delivery_event WHERE commit_seq>?1
         ORDER BY 1, 2, 3",
    )?;
    let rows = stmt.query_map([after], |r| {
        let kind: String = r.get(1)?;
        Ok(Unit {
            commit_seq: r.get(0)?,
            kind,
            key: r.get(2)?,
        })
    })?;
    let mut units: Vec<Unit> = rows.collect::<rusqlite::Result<_>>()?;
    excision::visit_publications(archive, |_, publication| {
        publication.each_replacement(|replacement| {
            if replacement.unit.commit_seq > after {
                units.push(replacement.unit.clone());
            }
            Ok(())
        })
    })?;
    units.sort();
    units.dedup();
    Ok(units)
}

mod batch;
pub use batch::{Batch, read};
mod inventory;
pub use inventory::Inventory;

/// Inserts rows, reusing identical rows already present. A conflicting row
/// refuses rather than silently retaining different content.
#[cfg(test)]
fn apply(archive: &Connection, lines: &[u8]) -> crate::Result<u64> {
    let mut inserted = 0;
    for line in lines.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        inserted += apply_row(archive, serde_json::from_slice(line)?)?;
    }
    Ok(inserted)
}

fn apply_row(archive: &Connection, v: Value) -> crate::Result<u64> {
    let table = v["table"]
        .as_str()
        .filter(|t| TABLES.contains(t))
        .ok_or_else(|| crate::StoreError::Invalid(format!("a row names no known table: {v}")))?;
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
    let added = archive.execute(&sql, params_from_iter(values.iter()))?;
    if added == 0 {
        let matching = format!(
            "SELECT EXISTS(SELECT 1 FROM {table} WHERE {})",
            cols.iter()
                .enumerate()
                .map(|(i, c)| format!("{c} IS ?{}", i + 1))
                .collect::<Vec<_>>()
                .join(" AND ")
        );
        let same: bool =
            archive.query_row(&matching, params_from_iter(values.iter()), |r| r.get(0))?;
        if !same {
            return Err(crate::StoreError::Invalid(format!(
                "restored {table} row conflicts with existing unrelated data"
            )));
        }
    }
    Ok(added as u64)
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
    "vector_deletion_policy",
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
    if !s.is_ascii() {
        return Err(crate::StoreError::Invalid("non-ASCII hex".into()));
    }
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

/// One read snapshot of `archive.db` for a publication pass.
pub struct Published {
    conn: Connection,
    blobs: hashed_blobs::Store,
}

impl Published {
    pub fn open(root: &std::path::Path) -> crate::Result<Self> {
        let conn = crate::archive::open_read(&root.join("archive.db"))?;
        // Large row-query sorts may spill beyond the connection cache.
        conn.pragma_update(None, "temp_store", "FILE")?;
        // One publication pass keeps the same view across its table queries.
        conn.execute_batch("BEGIN")?;
        Ok(Self {
            conn,
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

    pub fn inventory(&self) -> crate::Result<Inventory<'_>> {
        Inventory::new(self)
    }

    pub fn read(&self, unit: &Unit) -> crate::Result<Batch> {
        read(&self.conn, unit)
    }

    pub fn publication(&self, unit: &Unit) -> crate::Result<Option<ExcisionPublication>> {
        excision::publication(&self.conn, unit)
    }

    /// A frozen replacement batch, checked by its content address.
    pub fn replacement(&self, name: &str) -> crate::Result<std::fs::File> {
        let hash = excision::batch_hash(name)?;
        let reader = self.blobs.read_plain(&hash, None, || Ok(()))?;
        let mut file = tempfile::tempfile()?;
        crate::blobs::copy_verified(reader, &mut file, polylogue_model::BlobHash(hash.0), None)?;
        file.rewind()?;
        Ok(file)
    }

    /// Verify a source blob in private scratch, then place it in the target store.
    /// The returned length describes verified bytes, never an incomplete copy.
    pub fn copy_blob_to(
        &self,
        hash: &[u8; 32],
        target: &hashed_blobs::Store,
    ) -> crate::Result<u64> {
        use std::io::Seek;
        let expected = polylogue_model::BlobHash(*hash);
        let mut file = tempfile::tempfile()?;
        let copied = crate::blobs::copy_plain(&self.conn, &self.blobs, expected, &mut file)?;
        file.rewind()?;
        let installed = target.put_plain_checked::<crate::StoreError>(
            &mut file,
            crate::blobs::LEVEL,
            || Ok(()),
        )?;
        if installed.0 != *hash {
            return Err(crate::StoreError::BlobCorrupt(expected));
        }
        Ok(copied.bytes)
    }
}

/// `archive.db` opened for restoring rows into it.
pub struct Restoring {
    conn: Connection,
    archive: crate::Archive,
}

impl Restoring {
    pub fn open(archive: &crate::Archive) -> crate::Result<Self> {
        Ok(Self {
            archive: archive.clone(),
            conn: crate::archive::open_write(
                &archive.archive_db(),
                crate::archive::cache::ARCHIVE_WRITER,
            )?,
        })
    }

    /// Installs a complete selected view atomically. Payload lookup happens only
    /// after `Catalog::finish`, so intentionally retired before-images are never required.
    /// A same-archive update must include every local excision's exact manifest;
    /// an older confirmed view cannot overwrite unpublished local removals.
    /// Changed excisions prepare a fresh derived index; readers tied to a retired
    /// index refuse until they reopen a connection to the current file.
    pub fn install<R: std::io::Read>(
        &mut self,
        archive_id: &str,
        units: &Selection,
        fetch: impl FnMut(&RestoreUnit) -> crate::Result<Option<R>>,
    ) -> crate::Result<u64> {
        self.archive.validate_excision_indexes(&self.conn)?;
        // Canonical policies must survive power loss before vector deletion.
        let previous: i64 = self
            .conn
            .pragma_query_value(None, "synchronous", |r| r.get(0))?;
        self.conn
            .pragma_update(None, "synchronous", previous.max(2))?;
        let result = (|| {
            let inserted = restore::install(&mut self.conn, units, Some(archive_id), fetch)?;
            self.archive.reconcile_vectors(&mut self.conn)?;
            self.archive.prepare_index()?;
            Ok(inserted)
        })();
        self.conn
            .pragma_update(None, "synchronous", previous)
            .map_err(|e| crate::StoreError::ExcisionInterrupted(Box::new(e.into())))?;
        result
    }
}

#[cfg(test)]
mod tests;
