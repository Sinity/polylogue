//! Text identities belong to the serial writer, including IDs whose rows
//! temporarily lose their last reference. Reusing an ID follows its hash,
//! never whichever row number SQLite might next make available.
use crate::ConnExt as _;
use crate::{OptionExt as _, ResultExt as _};
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Identity {
    id: i64,
    present: bool,
}

/// The row key and whether this call wrote its prepared payload.
pub(crate) struct ResolvedText {
    pub id: i64,
    pub written: bool,
}

pub(crate) struct TextIds {
    known: RefCell<HashMap<[u8; 32], Identity>>,
    /// Search membership bits this writer knows are set on text rows, by ID,
    /// within the current transaction; forgotten when it ends.
    marked: RefCell<HashMap<i64, i64>>,
    groups: super::StoredGroups,
    next: Cell<i64>,
    reload: Cell<bool>,
}

impl TextIds {
    pub(crate) fn new(conn: &Connection, preload: bool) -> Result<Self> {
        let max: i64 = conn.scalar("SELECT coalesce(max(id),0) FROM text", [])?;
        let ids = Self {
            known: RefCell::new(HashMap::new()),
            marked: RefCell::new(HashMap::new()),
            groups: super::StoredGroups::default(),
            next: Cell::new(next(max)?),
            reload: Cell::new(preload),
        };
        ids.refresh(conn)?;
        Ok(ids)
    }

    fn refresh(&self, conn: &Connection) -> Result<()> {
        if !self.reload.get() {
            return Ok(());
        }
        let mut known = self.known.borrow_mut();
        for identity in known.values_mut() {
            identity.present = false;
        }
        let mut statement = conn.prepare("SELECT hash,id FROM text")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash = crate::encoding::Columns::hash(row, 0)?;
            let id: i64 = row.get(1)?;
            if known.get(&hash).is_some_and(|prior| prior.id != id) {
                return Err(StoreError::Invalid(
                    "a text hash has conflicting row identities".into(),
                ));
            }
            known.insert(hash, Identity { id, present: true });
            self.next.set(self.next.get().max(next(id)?));
        }
        self.reload.set(false);
        Ok(())
    }

    /// The row of text `hash`, written from `body` when it is not stored; a
    /// row written here joins the search indexes `membership` names for its
    /// hash.
    pub(crate) fn resolve(
        &self,
        conn: &Connection,
        hash: &[u8; 32],
        body: Option<super::groups::Prepared<'_>>,
        membership: Option<&super::NewMembership<'_>>,
    ) -> Result<ResolvedText> {
        self.refresh(conn)?;
        let known = self.known.borrow().get(hash).copied();
        if let Some(Identity { id, present: true }) = known {
            return Ok(ResolvedText { id, written: false });
        }
        let allocated = known.map(|identity| identity.id);
        if let Some(body) = body {
            let id = allocated.unwrap_or(self.next.get());
            let after = next(id)?;
            let bits = membership
                .and_then(|membership| membership.bits.get(hash))
                .copied()
                .unwrap_or(0);
            let inserted = super::groups::insert(conn, &self.groups, Some(id), hash, body, bits)?;
            if inserted.is_some() {
                if bits != 0 {
                    if !membership.is_some_and(|membership| membership.bulk) {
                        for bit in [1, 2].into_iter().filter(|bit| bits & bit != 0) {
                            super::queue_entry(conn, id, bit)?;
                        }
                    }
                    self.marked.borrow_mut().insert(id, bits);
                }
                self.next.set(self.next.get().max(after));
                self.known
                    .borrow_mut()
                    .insert(*hash, Identity { id, present: true });
                return Ok(ResolvedText { id, written: true });
            }
        }
        let stored = conn
            .prepare_cached("SELECT id FROM text WHERE hash=?1")?
            .query_row([hash.as_slice()], |r| r.get::<_, i64>(0))
            .optional()?;
        if let Some(id) = stored {
            if allocated.is_some_and(|prior| prior != id) {
                return Err(StoreError::Invalid(
                    "a text hash has conflicting row identities".into(),
                ));
            }
            self.known
                .borrow_mut()
                .insert(*hash, Identity { id, present: true });
            self.next.set(self.next.get().max(next(id)?));
            Ok(ResolvedText { id, written: false })
        } else {
            Err(StoreError::Invalid(
                if body.is_some() {
                    "allocated text identity conflicts with stored rows"
                } else {
                    "a new text has no prepared bytes"
                }
                .into(),
            ))
        }
    }

    #[cfg(test)]
    pub(crate) fn text(&self, conn: &Connection, hash: &[u8; 32]) -> Result<Option<i64>> {
        self.refresh(conn)?;
        if let Some(identity) = self.known.borrow().get(hash) {
            return Ok(identity.present.then_some(identity.id));
        }
        Ok(None)
    }

    /// A confirmed scope change discards identities from the old cache, while
    /// the allocator's high-water mark remains above all IDs already used.
    pub(crate) fn discard_cached(&self) {
        self.known.borrow_mut().clear();
        self.marked.borrow_mut().clear();
        self.groups.clear();
        self.reload.set(true);
    }

    /// Its group may have left with it.
    pub(crate) fn removed(&self, hash: &[u8; 32], id: i64) {
        self.groups.clear();
        self.known
            .borrow_mut()
            .insert(*hash, Identity { id, present: false });
        self.marked.borrow_mut().remove(&id);
    }

    /// Whether text `id` is known to be in the search index `bit` names.
    pub(super) fn is_marked(&self, id: i64, bit: i64) -> bool {
        self.marked
            .borrow()
            .get(&id)
            .is_some_and(|bits| bits & bit != 0)
    }

    pub(super) fn mark(&self, id: i64, bit: i64) {
        *self.marked.borrow_mut().entry(id).or_default() |= bit;
    }

    pub(super) fn unmark(&self, id: i64, bit: i64) {
        if let Some(bits) = self.marked.borrow_mut().get_mut(&id) {
            *bits &= !bit;
        }
    }

    pub(crate) fn transaction(&self) -> TextTransaction<'_> {
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
        self.texts.marked.borrow_mut().clear();
        if !self.committed {
            self.texts.reload.set(true);
            self.texts.groups.clear();
        }
    }
}
