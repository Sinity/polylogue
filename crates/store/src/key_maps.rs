//! Row lookups by natural key while their unique indexes are absent.
//! Candidate rows are read back and compared, including after rollback or removal.
use crate::{Result, archive::MemoryBudget};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension};
use std::ops::ControlFlow;
use std::sync::RwLock;
mod rows;
pub(crate) use rows::KeyRows;

pub(crate) struct KeyMaps {
    messages: RwLock<Rows>,
    events: RwLock<Rows>,
    versions: RwLock<Rows>,
}
struct Rows(KeyRows);
fn hash_of(key: &[u8]) -> [u8; 32] {
    *blake3::hash(key).as_bytes()
}
impl Rows {
    fn new(bytes: u64) -> Result<Self> {
        KeyRows::new(bytes).map(Self)
    }
    fn add(&mut self, hash: [u8; 32], pk: i64) -> Result<()> {
        self.0.set(hash, pk)
    }
    fn find(
        &self,
        conn: &Connection,
        check: &str,
        hash: [u8; 32],
        key: &[u8],
    ) -> Result<Option<i64>> {
        let mut found = None;
        self.0.visit(&hash, |pk| {
            let mut statement = conn.prepare_cached(check)?;
            let same = statement
                .query_row([pk], |r| {
                    Ok(match r.get_ref(0)? {
                        ValueRef::Text(value) | ValueRef::Blob(value) => value == key,
                        _ => false,
                    })
                })
                .optional()?;
            if same == Some(true) {
                found = Some(pk);
                Ok(ControlFlow::Break(()))
            } else {
                Ok(ControlFlow::Continue(()))
            }
        })?;
        Ok(found)
    }
}
impl Default for Rows {
    fn default() -> Self {
        Self::new(MemoryBudget::default().keys() / 9).expect("key rows")
    }
}
impl Default for KeyMaps {
    fn default() -> Self {
        Self {
            messages: RwLock::new(Rows::default()),
            events: RwLock::new(Rows::default()),
            versions: RwLock::new(Rows::default()),
        }
    }
}
impl KeyMaps {
    #[cfg(test)]
    pub(crate) fn load(index: &Connection) -> Result<Self> {
        Self::load_with_budget(index, MemoryBudget::default().keys() / 3)
    }
    pub(crate) fn load_with_budget(index: &Connection, bytes: u64) -> Result<Self> {
        // Nothing else sees the maps until they are returned, so they are
        // filled unlocked and wrapped last.
        let mut messages = Rows::new(bytes / 3)?;
        let mut events = Rows::new(bytes / 3)?;
        let mut versions = Rows::new(bytes / 3)?;
        for (table, values) in [("message", &mut messages), ("event", &mut events)] {
            let mut statement = index.prepare(&format!("SELECT id,pk FROM {table}"))?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let key: String = row.get(0)?;
                values.add(hash_of(key.as_bytes()), row.get(1)?)?;
            }
        }
        let mut statement = index.prepare("SELECT version_hash,pk FROM message_version")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            versions.add(hash_of(&hash), row.get(1)?)?;
        }
        Ok(Self {
            messages: RwLock::new(messages),
            events: RwLock::new(events),
            versions: RwLock::new(versions),
        })
    }
    pub(crate) fn message_version(&self, conn: &Connection, hash: &[u8]) -> Result<Option<i64>> {
        self.versions.read().expect("key map lock").find(
            conn,
            "SELECT version_hash FROM message_version WHERE pk=?1",
            hash_of(hash),
            hash,
        )
    }
    pub(crate) fn add_message_version(&self, hash: &[u8], pk: i64) -> Result<()> {
        self.versions
            .write()
            .expect("key map lock")
            .add(hash_of(hash), pk)
    }
    pub(crate) fn message(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.messages.read().expect("key map lock").find(
            conn,
            "SELECT id FROM message WHERE pk=?1",
            hash_of(id.as_bytes()),
            id.as_bytes(),
        )
    }
    pub(crate) fn add_message(&self, id: &str, pk: i64) -> Result<()> {
        self.messages
            .write()
            .expect("key map lock")
            .add(hash_of(id.as_bytes()), pk)
    }
    pub(crate) fn event(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.events.read().expect("key map lock").find(
            conn,
            "SELECT id FROM event WHERE pk=?1",
            hash_of(id.as_bytes()),
            id.as_bytes(),
        )
    }
    pub(crate) fn add_event(&self, id: &str, pk: i64) -> Result<()> {
        self.events
            .write()
            .expect("key map lock")
            .add(hash_of(id.as_bytes()), pk)
    }
}
#[cfg(test)]
mod tests;
