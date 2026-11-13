//! What a first build keeps in memory in place of the unique indexes on
//! natural keys: message IDs, event IDs and text hashes. A first build drops
//! those indexes and builds each once from a sorted scan when it ends
//! ([`crate::writer`]); inserting random keys into them rewrote pages all
//! over each index in every transaction, which grew with the archive.
//!
//! Message and event keys map to rows through the first eight bytes of
//! a BLAKE3 hash of the key. Every row found this way is read back and
//! compared with the key, so an entry left by a transaction that rolled back,
//! a row deleted since, or two keys sharing those bytes never yield a wrong
//! row.

use crate::Result;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Deref;
use std::sync::{Arc, RwLock};

#[derive(Default)]
pub(crate) struct Keys {
    messages: RwLock<Rows>,
    events: RwLock<Rows>,
}

/// The rows a table's keys were written as, by key hash.
#[derive(Default)]
struct Rows {
    /// The first row written under each hash.
    first: HashMap<u64, i64, BuildHasherDefault<Spread>>,
    /// Further rows under a hash already taken: a rewritten row, or another
    /// key with the same hash. Rare.
    more: HashMap<u64, Vec<i64>, BuildHasherDefault<Spread>>,
}

/// Hashes the map's keys, which are already uniformly spread hash bytes, as
/// themselves.
#[derive(Default)]
struct Spread(u64);

impl Hasher for Spread {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

fn hash_of(key: &[u8]) -> u64 {
    let digest = blake3::hash(key);
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("eight bytes"))
}

impl Rows {
    fn add(&mut self, hash: u64, pk: i64) {
        match self.first.entry(hash) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(pk);
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == pk => {}
            std::collections::hash_map::Entry::Occupied(_) => {
                let more = self.more.entry(hash).or_default();
                if !more.contains(&pk) {
                    more.push(pk);
                }
            }
        }
    }

    /// The row written under `hash` whose key, read by `check` (`?1` is the
    /// row's primary key), equals `key`.
    fn find(&self, conn: &Connection, check: &str, hash: u64, key: &[u8]) -> Result<Option<i64>> {
        let Some(first) = self.first.get(&hash) else {
            return Ok(None);
        };
        let mut statement = conn.prepare_cached(check)?;
        for pk in std::iter::once(first).chain(self.more.get(&hash).into_iter().flatten()) {
            let same = statement
                .query_row([pk], |r| {
                    Ok(match r.get_ref(0)? {
                        ValueRef::Text(found) | ValueRef::Blob(found) => found == key,
                        _ => false,
                    })
                })
                .optional()?;
            if same == Some(true) {
                return Ok(Some(*pk));
            }
        }
        Ok(None)
    }
}

impl Keys {
    pub(crate) fn load(index: &Connection) -> Result<Self> {
        let keys = Self::default();
        for (table, key, pk, rows) in [
            ("message", "id", "pk", &keys.messages),
            ("event", "id", "pk", &keys.events),
        ] {
            let mut statement = index.prepare(&format!("SELECT {key},{pk} FROM {table}"))?;
            let mut values = statement.query([])?;
            while let Some(row) = values.next()? {
                let value: String = row.get(0)?;
                let hash = hash_of(value.as_bytes());
                rows.write().expect("key map lock").add(hash, row.get(1)?);
            }
        }
        Ok(keys)
    }

    pub(crate) fn message(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.messages.read().expect("key map lock").find(
            conn,
            "SELECT id FROM message WHERE pk = ?1",
            hash_of(id.as_bytes()),
            id.as_bytes(),
        )
    }

    pub(crate) fn add_message(&self, id: &str, pk: i64) {
        self.messages
            .write()
            .expect("key map lock")
            .add(hash_of(id.as_bytes()), pk);
    }

    pub(crate) fn event(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.events.read().expect("key map lock").find(
            conn,
            "SELECT id FROM event WHERE pk = ?1",
            hash_of(id.as_bytes()),
            id.as_bytes(),
        )
    }

    pub(crate) fn add_event(&self, id: &str, pk: i64) {
        self.events
            .write()
            .expect("key map lock")
            .add(hash_of(id.as_bytes()), pk);
    }
}

/// Message/event keys are shared with resolvers; text identities belong only
/// to the serial writer and follow its index transaction.
#[derive(Default)]
pub(crate) struct FirstBuild {
    pub keys: Arc<Keys>,
    texts: RefCell<Option<HashMap<[u8; 32], i64>>>,
}

impl Deref for FirstBuild {
    type Target = Keys;
    fn deref(&self) -> &Keys {
        &self.keys
    }
}

impl FirstBuild {
    pub(crate) fn load(index: &Connection) -> Result<Self> {
        let first = Self {
            keys: Arc::new(Keys::load(index)?),
            texts: RefCell::new(None),
        };
        first.load_texts(index)?;
        Ok(first)
    }

    fn load_texts(&self, conn: &Connection) -> Result<()> {
        if self.texts.borrow().is_none() {
            let mut statement = conn.prepare("SELECT hash,id FROM text")?;
            let texts = statement
                .query_map([], |r| {
                    Ok((crate::encoding::Columns::hash(r, 0)?, r.get(1)?))
                })?
                .collect::<rusqlite::Result<HashMap<_, _>>>()?;
            *self.texts.borrow_mut() = Some(texts);
        }
        Ok(())
    }

    pub(crate) fn text(&self, conn: &Connection, hash: &[u8; 32]) -> Result<Option<i64>> {
        self.load_texts(conn)?;
        Ok(self
            .texts
            .borrow()
            .as_ref()
            .expect("loaded texts")
            .get(hash)
            .copied())
    }

    pub(crate) fn add_text(&self, hash: &[u8; 32], id: i64) {
        if let Some(texts) = self.texts.borrow_mut().as_mut() {
            texts.insert(*hash, id);
        }
    }

    pub(crate) fn remove_text(&self, hash: &[u8; 32]) {
        if let Some(texts) = self.texts.borrow_mut().as_mut() {
            texts.remove(hash);
        }
    }

    /// Discard mutations unless the owning index transaction commits. The
    /// next lookup reloads after SQLite has rolled the failed transaction back.
    pub(crate) fn text_transaction(&self) -> TextTransaction<'_> {
        TextTransaction {
            first: self,
            committed: false,
        }
    }
}

pub(crate) struct TextTransaction<'a> {
    first: &'a FirstBuild,
    committed: bool,
}
impl TextTransaction<'_> {
    pub(crate) fn commit(mut self) {
        self.committed = true;
    }
}
impl Drop for TextTransaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            *self.first.texts.borrow_mut() = None;
        }
    }
}

#[cfg(test)]
mod tests;
