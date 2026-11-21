//! Text identities belong to the serial writer, including IDs whose rows
//! temporarily lose their last reference. Sorted spill files keep these
//! identities without retaining a map that grows with the archive.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::key_maps::KeyRows;
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

/// The row key and whether its prepared payload was supplied in this transaction.
pub(crate) struct ResolvedText {
    pub id: i64,
    pub written: bool,
}

#[derive(Default)]
struct TextMark {
    indexed: i64,
    written: bool,
}

pub(crate) struct TextIds {
    /// A positive ID has a row; its negative form keeps the allocation after
    /// removal or rollback, so a later text cannot reuse that identity.
    known: RefCell<KeyRows>,
    marked: RefCell<HashMap<i64, TextMark>>,
    marked_entries: usize,
    in_transaction: Cell<bool>,
    groups: super::StoredGroups,
    next: Cell<i64>,
    reload: Cell<bool>,
    complete: Cell<bool>,
}

impl TextIds {
    pub(crate) fn new(conn: &Connection, preload: bool, bytes: u64) -> Result<Self> {
        let max: i64 = conn.scalar("SELECT coalesce(max(id),0) FROM text", [])?;
        let ids = Self {
            known: RefCell::new(KeyRows::new(bytes / 2)?),
            marked: RefCell::new(HashMap::new()),
            marked_entries: (bytes / 4 / 64).max(1) as usize,
            in_transaction: Cell::new(false),
            groups: super::StoredGroups::new(bytes / 4),
            next: Cell::new(next(max)?),
            reload: Cell::new(preload),
            complete: Cell::new(false),
        };
        ids.refresh(conn)?;
        Ok(ids)
    }

    fn refresh(&self, conn: &Connection) -> Result<()> {
        if !self.reload.get() {
            return Ok(());
        }
        let mut known = self.known.borrow_mut();
        known.transform_values(|id| -id.abs())?;
        let mut statement = conn.prepare("SELECT hash,id FROM text ORDER BY id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash = crate::encoding::Columns::hash(row, 0)?;
            let id: i64 = row.get(1)?;
            if known.get(&hash)?.is_some_and(|prior| prior.abs() != id) {
                return Err(StoreError::Invalid(
                    "a text hash has conflicting row identities".into(),
                ));
            }
            known.set(hash, id)?;
            self.next.set(self.next.get().max(next(id)?));
        }
        self.complete.set(true);
        self.reload.set(false);
        Ok(())
    }

    /// Resolves a complete text hash. An empty build has no earlier text to
    /// probe: its dictionary already includes every row written in the build.
    /// In an installed index, a prepared body first tries the unique insert;
    /// only a conflicting or bodyless lookup reads the stored hash index.
    pub(crate) fn resolve(
        &self,
        conn: &Connection,
        hash: &[u8; 32],
        body: Option<super::groups::Prepared<'_>>,
        membership: Option<&super::NewMembership<'_>>,
    ) -> Result<ResolvedText> {
        self.refresh(conn)?;
        let known = self.known.borrow().get(hash)?;
        if let Some(id) = known.filter(|id| *id > 0) {
            let written = self.in_transaction.get()
                && self
                    .marked
                    .borrow()
                    .get(&id)
                    .is_some_and(|mark| mark.written);
            return Ok(ResolvedText { id, written });
        }
        let allocated = known.map(i64::abs);
        if let Some(body) = body {
            let id = allocated.unwrap_or(self.next.get());
            let after = next(id)?;
            let bits = membership
                .and_then(|membership| membership.bits.get(hash))
                .copied()
                .unwrap_or(0);
            if super::groups::insert(conn, &self.groups, Some(id), hash, body, bits)?.is_some() {
                if bits != 0 {
                    if !membership.is_some_and(|membership| membership.at_build_end) {
                        for bit in [1, 2].into_iter().filter(|bit| bits & bit != 0) {
                            super::queue_entry(conn, id, bit)?;
                        }
                    }
                    self.mark(id, bits);
                }
                if self.in_transaction.get() {
                    self.mark(id, 0);
                    self.marked.borrow_mut().get_mut(&id).unwrap().written = true;
                }
                self.next.set(self.next.get().max(after));
                self.known.borrow_mut().set(*hash, id)?;
                return Ok(ResolvedText { id, written: true });
            }
        }
        if !self.complete.get() {
            let stored = conn
                .prepare_cached("SELECT id FROM text WHERE hash=?1")?
                .query_row([hash.as_slice()], |row| row.get::<_, i64>(0))
                .optional()?;
            if let Some(id) = stored {
                if allocated.is_some_and(|prior| prior != id) {
                    return Err(StoreError::Invalid(
                        "a text hash has conflicting row identities".into(),
                    ));
                }
                self.known.borrow_mut().set(*hash, id)?;
                self.next.set(self.next.get().max(next(id)?));
                return Ok(ResolvedText { id, written: false });
            }
        }
        Err(StoreError::Invalid(
            if body.is_some() {
                "allocated text identity conflicts with stored rows"
            } else {
                "a new text has no prepared bytes"
            }
            .into(),
        ))
    }

    #[cfg(test)]
    pub(crate) fn text(&self, conn: &Connection, hash: &[u8; 32]) -> Result<Option<i64>> {
        self.refresh(conn)?;
        Ok(self.known.borrow().get(hash)?.filter(|id| *id > 0))
    }

    /// A confirmed scope change discards cached identities; the allocator's
    /// high-water mark stays above every ID this process has used.
    pub(crate) fn discard_cached(&self) {
        self.known.borrow_mut().clear();
        self.marked.borrow_mut().clear();
        self.groups.clear();
        self.reload.set(true);
        self.complete.set(false);
    }

    pub(crate) fn removed(&self, hash: &[u8; 32], id: i64) -> Result<()> {
        self.groups.clear();
        self.known.borrow_mut().set(*hash, -id)?;
        self.marked.borrow_mut().remove(&id);
        Ok(())
    }

    pub(super) fn is_marked(&self, id: i64, bit: i64) -> bool {
        self.marked
            .borrow()
            .get(&id)
            .is_some_and(|mark| mark.indexed & bit != 0)
    }

    pub(super) fn mark(&self, id: i64, bit: i64) {
        let mut marked = self.marked.borrow_mut();
        // Forgetting a mark repeats membership SQL or stored-payload validation
        // instead of retaining a whole transaction's IDs.
        if marked.len() >= self.marked_entries && !marked.contains_key(&id) {
            marked.clear();
        }
        marked.entry(id).or_default().indexed |= bit;
    }

    pub(super) fn unmark(&self, id: i64, bit: i64) {
        if let Some(mark) = self.marked.borrow_mut().get_mut(&id) {
            mark.indexed &= !bit;
        }
    }

    pub(crate) fn transaction(&self) -> TextTransaction<'_> {
        self.marked.borrow_mut().clear();
        self.in_transaction.set(true);
        TextTransaction {
            texts: self,
            committed: false,
        }
    }
}

fn next(id: i64) -> Result<i64> {
    id.checked_add(1).or_invalid("text row identity exhausted")
}

pub(crate) struct TextTransaction<'a> {
    texts: &'a TextIds,
    committed: bool,
}
impl TextTransaction<'_> {
    pub(crate) fn commit(mut self) {
        self.committed = true;
        self.texts.groups.clear();
    }
}
impl Drop for TextTransaction<'_> {
    fn drop(&mut self) {
        self.texts.in_transaction.set(false);
        self.texts.marked.borrow_mut().clear();
        if !self.committed {
            self.texts.reload.set(true);
            self.texts.groups.clear();
        }
    }
}
