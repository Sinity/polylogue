//! Message selection survives deletion of the index rows that supplied it.
use crate::Result;
use rusqlite::Connection;
use serde::ser::{Error as _, SerializeSeq};
use serde::{Serialize, Serializer};

#[derive(Default)]
pub(super) struct SelectedMessages(Option<Connection>);

impl SelectedMessages {
    pub(super) fn insert(&mut self, id: &str) -> Result<()> {
        crate::fault::check("excision_messages.insert", id)?;
        if self.0.is_none() {
            let db = Connection::open("")?;
            db.execute_batch(
                "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
                PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                CREATE TABLE selected(id TEXT NOT NULL);
                CREATE INDEX selected_id ON selected(id);
                BEGIN;",
            )?;
            self.0 = Some(db);
        }
        self.0
            .as_ref()
            .expect("created message selection")
            .prepare_cached("INSERT INTO selected VALUES (?1)")?
            .execute([id])?;
        Ok(())
    }

    pub(super) fn count(&self) -> Result<u64> {
        match &self.0 {
            Some(db) => Ok(db
                .query_row("SELECT count(*) FROM selected", [], |r| r.get::<_, i64>(0))?
                as u64),
            None => Ok(0),
        }
    }

    pub(super) fn contains(&self, id: &str) -> Result<bool> {
        crate::fault::check("excision_messages.contains", id)?;
        match &self.0 {
            Some(db) => Ok(db
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM selected WHERE id=?1)")?
                .query_row([id], |r| r.get(0))?),
            None => Ok(false),
        }
    }
}

impl Serialize for SelectedMessages {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        if let Some(db) = &self.0 {
            let mut statement = db
                .prepare("SELECT id FROM selected ORDER BY id")
                .map_err(S::Error::custom)?;
            let mut rows = statement.query([]).map_err(S::Error::custom)?;
            while let Some(row) = rows.next().map_err(S::Error::custom)? {
                let id = row.get_ref(0).map_err(S::Error::custom)?;
                sequence.serialize_element(id.as_str().map_err(S::Error::custom)?)?;
            }
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests;
