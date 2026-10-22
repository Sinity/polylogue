//! Disposable on-disk ordering and replacement selection for one restore.
use super::{RestoreUnit, Unit, invalid};
use crate::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde::ser::{Error, SerializeSeq};
use serde::{Serialize, Serializer};
use std::cell::Cell;

pub struct Catalog {
    pub(super) db: Connection,
    failed: bool,
}

impl Catalog {
    pub fn new() -> Result<Self> {
        let db = Connection::open("")?;
        db.execute_batch(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
            PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE confirmed(position INTEGER PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE selected(seq INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
                data TEXT NOT NULL, tombstone INTEGER NOT NULL,
                PRIMARY KEY(seq,kind,key)) WITHOUT ROWID;
            CREATE INDEX selected_fact ON selected(kind,key);
            CREATE TABLE excision_applied(id_hash BLOB PRIMARY KEY,policy_hash BLOB NOT NULL) WITHOUT ROWID;
            CREATE TABLE verified_source(hash TEXT PRIMARY KEY, bytes INTEGER NOT NULL CHECK(bytes>=0)) WITHOUT ROWID;
            CREATE TABLE blob_range(hash TEXT PRIMARY KEY, needed INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE allowed(kind TEXT NOT NULL, id TEXT NOT NULL,
                PRIMARY KEY(kind,id)) WITHOUT ROWID;
            BEGIN;",
        )?;
        Ok(Self { db, failed: false })
    }

    /// Confirmations may arrive out of order. Exact repeats are harmless;
    /// differing descriptors at one position refuse the complete restore.
    pub fn push(&mut self, position: u64, unit: &RestoreUnit) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "a refused confirmation invalidated this restore catalog",
            ));
        }
        let result = (|| {
            let position = i64::try_from(position)
                .ok()
                .filter(|p| *p > 0)
                .ok_or_else(|| invalid("confirmed position is outside the supported range"))?;
            let data = serde_json::to_string(unit)?;
            let inserted = self.db.execute(
                "INSERT OR IGNORE INTO confirmed VALUES (?1,?2)",
                params![position, data],
            )?;
            if inserted == 0 {
                let previous: String = self.db.query_row(
                    "SELECT data FROM confirmed WHERE position=?1",
                    [position],
                    |r| r.get(0),
                )?;
                if previous != data {
                    return Err(invalid("confirmed units disagree at the same position"));
                }
            }
            Ok(())
        })();
        self.failed = result.is_err();
        result
    }

    /// Check the complete order and fold later removals before any payload IO.
    pub fn finish(self) -> Result<Selection> {
        if self.failed {
            return Err(invalid(
                "a refused confirmation invalidated this restore catalog",
            ));
        }
        let mut count = 0;
        let mut previous = None;
        {
            let mut statement = self
                .db
                .prepare("SELECT position,data FROM confirmed ORDER BY position")?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let position: i64 = row.get(0)?;
                count += 1;
                if position != count {
                    return Err(invalid(&format!(
                        "Sinex holds unit {position} but not unit {count}"
                    )));
                }
                let original: RestoreUnit = serde_json::from_str(&row.get::<_, String>(1)?)?;
                let unit = &original.unit;
                if unit.commit_seq < 1
                    || unit.key.is_empty()
                    || ![
                        "revision",
                        "note",
                        "excision",
                        "setting",
                        "delivery",
                        "observation",
                        "delivery_event",
                    ]
                    .contains(&unit.kind.as_str())
                    || previous.as_ref().is_some_and(|p: &Unit| p >= unit)
                {
                    return Err(invalid(
                        "confirmed unit identities are not in strict archive order",
                    ));
                }
                previous = Some(unit.clone());
                if unit.kind == "excision" && original.excision.is_none() {
                    return Err(invalid("legacy excision has no replacement metadata"));
                }
                if let Some(hash) = &original.batch {
                    super::super::excision::batch_hash(hash)?;
                } else if original.rows != 0 {
                    return Err(invalid("a tombstone names rows"));
                }
                self.put(&original, false)?;
                if let Some(publication) = &original.excision {
                    if unit.kind != "excision" {
                        return Err(invalid("replacement metadata belongs to no excision"));
                    }
                    publication.validate()?;
                    for replacement in &publication.replacements {
                        let current = self.find(&replacement.unit)?;
                        if replacement.unit >= *unit || current.is_none() {
                            return Err(invalid(
                                "replacement names no earlier exact unit identity",
                            ));
                        }
                        let (current, authorized) = current.expect("checked current unit");
                        self.put(
                            &RestoreUnit {
                                unit: replacement.unit.clone(),
                                batch: replacement.batch.clone(),
                                bytes: Some(replacement.bytes),
                                rows: replacement.rows,
                                blobs: replacement.blobs.clone(),
                                excision: current.excision,
                            },
                            authorized || replacement.batch.is_none(),
                        )?;
                    }
                }
            }
        }
        let selection = Selection {
            catalog: self,
            len: count as u64,
            commit_seq: previous.map_or(0, |u| u.commit_seq),
            sources_verified: Cell::new(false),
        };
        selection.each(|current| {
            let authorized: bool = selection.catalog.db.query_row(
                "SELECT tombstone FROM selected WHERE seq=?1 AND kind=?2 AND key=?3",
                params![current.unit.commit_seq, current.unit.kind, current.unit.key],
                |r| r.get(0),
            )?;
            if (current.batch.is_none() && !authorized)
                || (current.batch.is_some() && current.rows == 0)
            {
                return Err(invalid("an empty unit has no confirmed excision tombstone"));
            }
            Ok(())
        })?;
        selection
            .catalog
            .db
            .execute_batch("DROP TABLE confirmed; COMMIT;")?;
        Ok(selection)
    }

    fn put(&self, unit: &RestoreUnit, authorized: bool) -> Result<()> {
        self.db.execute("INSERT INTO selected VALUES (?1,?2,?3,?4,?5)
            ON CONFLICT(seq,kind,key) DO UPDATE SET data=excluded.data,tombstone=excluded.tombstone",
            params![unit.unit.commit_seq,unit.unit.kind,unit.unit.key,serde_json::to_string(unit)?,authorized])?;
        Ok(())
    }

    fn find(&self, unit: &Unit) -> Result<Option<(RestoreUnit, bool)>> {
        let row = self
            .db
            .query_row(
                "SELECT data,tombstone FROM selected WHERE seq=?1 AND kind=?2 AND key=?3",
                params![unit.commit_seq, unit.kind, unit.key],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, bool>(1)?)),
            )
            .optional()?;
        row.map(|(data, authorized)| Ok((serde_json::from_str(&data)?, authorized)))
            .transpose()
    }
}

/// Only a complete validated catalog can be installed. Rows remain on disk;
/// iteration and lookup decode one descriptor at a time.
pub struct Selection {
    pub(super) catalog: Catalog,
    len: u64,
    commit_seq: i64,
    sources_verified: Cell<bool>,
}
impl Selection {
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn commit_seq(&self) -> i64 {
        self.commit_seq
    }

    /// Ask the caller to read each selected source through EOF, check its hash
    /// and return its byte count. Completion permits one install attempt;
    /// a new pass always repeats the reads and invalidates any earlier result.
    pub fn verify_source_blobs(&self, mut visit: impl FnMut(&str) -> Result<u64>) -> Result<()> {
        // Completion, not rollback of disposable scratch rows, authorizes use.
        self.sources_verified.set(false);
        let tx = self.catalog.db.unchecked_transaction()?;
        tx.execute("DELETE FROM verified_source", [])?;
        {
            let mut statement = tx.prepare(
                "SELECT DISTINCT b.value FROM selected s, json_each(s.data,'$.blobs') b ORDER BY b.value",
            )?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let name: String = row.get(0)?;
                let hash = super::super::excision::batch_hash(&name)?;
                let key = crate::sinex_rows::hex(&hash.0);
                let seen: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM verified_source WHERE hash=?1)",
                    [&key],
                    |r| r.get(0),
                )?;
                if seen {
                    continue;
                }
                let bytes = i64::try_from(visit(&name)?)
                    .map_err(|_| invalid("verified source length exceeds the supported range"))?;
                tx.execute(
                    "INSERT INTO verified_source VALUES (?1,?2)",
                    rusqlite::params![key, bytes],
                )?;
            }
        }
        tx.commit()?;
        self.sources_verified.set(true);
        Ok(())
    }

    pub(super) fn begin_install(&self) -> Result<()> {
        let verified = self.sources_verified.replace(false);
        let required: bool = self.catalog.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM selected s,json_each(s.data,'$.blobs'))",
            [],
            |r| r.get(0),
        )?;
        if required && !verified {
            return Err(invalid(
                "selected source bytes require completed verification before each install",
            ));
        }
        Ok(())
    }

    pub(super) fn find(&self, unit: &Unit) -> Result<Option<RestoreUnit>> {
        Ok(self.catalog.find(unit)?.map(|(unit, _)| unit))
    }

    pub(super) fn each_unit(&self, mut visit: impl FnMut(&Unit) -> Result<()>) -> Result<()> {
        let mut statement = self
            .catalog
            .db
            .prepare("SELECT seq,kind,key FROM selected ORDER BY seq,kind,key")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            visit(&Unit {
                commit_seq: row.get(0)?,
                kind: row.get(1)?,
                key: row.get(2)?,
            })?;
        }
        Ok(())
    }

    pub(in crate::sinex_rows) fn each(
        &self,
        mut visit: impl FnMut(&RestoreUnit) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self
            .catalog
            .db
            .prepare("SELECT data FROM selected ORDER BY seq,kind,key")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            visit(&serde_json::from_str(&row.get::<_, String>(0)?)?)?;
        }
        Ok(())
    }
}

// Keep the existing checkpoint representation: an ordered JSON array of units.
impl Serialize for Selection {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        let mut statement = self
            .catalog
            .db
            .prepare("SELECT data FROM selected ORDER BY seq,kind,key")
            .map_err(S::Error::custom)?;
        let mut rows = statement.query([]).map_err(S::Error::custom)?;
        while let Some(row) = rows.next().map_err(S::Error::custom)? {
            // Catalog writes use RestoreUnit's canonical codec. Borrow those
            // bytes so hashing the checkpoint does not decode large lists again.
            let data = row.get_ref(0).map_err(S::Error::custom)?;
            let data = data.as_str().map_err(S::Error::custom)?;
            let unit: &serde_json::value::RawValue =
                serde_json::from_str(data).map_err(S::Error::custom)?;
            sequence.serialize_element(unit)?;
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests;
