//! What a first build keeps in memory in place of the unique indexes on
//! natural keys: message IDs, event IDs and text hashes. A first build drops
//! those indexes and builds each once from a sorted scan when it ends
//! ([`crate::writer`]); inserting random keys into them rewrote pages all
//! over each index in every transaction, which grew with the archive.
//!
//! A key maps to the row it was written as, through the first eight bytes of
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

#[derive(Default)]
pub(crate) struct FirstBuild {
    messages: RefCell<Rows>,
    events: RefCell<Rows>,
    texts: RefCell<Rows>,
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
                self.more.entry(hash).or_default().push(pk);
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

impl FirstBuild {
    pub(crate) fn message(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.messages.borrow().find(
            conn,
            "SELECT id FROM message WHERE pk = ?1",
            hash_of(id.as_bytes()),
            id.as_bytes(),
        )
    }

    pub(crate) fn add_message(&self, id: &str, pk: i64) {
        self.messages.borrow_mut().add(hash_of(id.as_bytes()), pk);
    }

    pub(crate) fn event(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.events.borrow().find(
            conn,
            "SELECT id FROM event WHERE pk = ?1",
            hash_of(id.as_bytes()),
            id.as_bytes(),
        )
    }

    pub(crate) fn add_event(&self, id: &str, pk: i64) {
        self.events.borrow_mut().add(hash_of(id.as_bytes()), pk);
    }

    /// A text by its BLAKE3 hash, whose first bytes are already spread.
    pub(crate) fn text(&self, conn: &Connection, hash: &[u8; 32]) -> Result<Option<i64>> {
        self.texts.borrow().find(
            conn,
            "SELECT hash FROM text WHERE id = ?1",
            text_key(hash),
            hash,
        )
    }

    pub(crate) fn add_text(&self, hash: &[u8; 32], id: i64) {
        self.texts.borrow_mut().add(text_key(hash), id);
    }
}

fn text_key(hash: &[u8; 32]) -> u64 {
    u64::from_le_bytes(hash[..8].try_into().expect("eight bytes"))
}

#[cfg(test)]
mod tests;
