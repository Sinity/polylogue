//! Text identities belong to the serial writer, including IDs whose rows
//! temporarily lose their last reference. Reusing an ID follows its hash,
//! never whichever row number SQLite might next make available.
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Identity {
    id: i64,
    present: bool,
}

pub(crate) struct TextIds {
    known: RefCell<HashMap<[u8; 32], Identity>>,
    next: Cell<i64>,
    reload: Cell<bool>,
}

impl TextIds {
    pub(crate) fn new(conn: &Connection, preload: bool) -> Result<Self> {
        let max: i64 = conn.query_row("SELECT coalesce(max(id),0) FROM text", [], |r| r.get(0))?;
        let ids = Self {
            known: RefCell::new(HashMap::new()),
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

    pub(crate) fn resolve(
        &self,
        conn: &Connection,
        hash: &[u8; 32],
        body: Option<&[u8]>,
    ) -> Result<i64> {
        self.refresh(conn)?;
        let known = self.known.borrow().get(hash).copied();
        if let Some(Identity { id, present: true }) = known {
            return Ok(id);
        }
        let allocated = known.map(|identity| identity.id);
        if let Some(body) = body {
            let id = allocated.unwrap_or(self.next.get());
            let after = next(id)?;
            let inserted = conn.prepare_cached("INSERT OR IGNORE INTO text(id,hash,body,indexed) VALUES (?1,?2,?3,0) RETURNING id")?
                .query_row(params![id,hash.as_slice(),body], |r| r.get::<_,i64>(0)).optional()?;
            if inserted.is_some() {
                self.next.set(self.next.get().max(after));
                self.known
                    .borrow_mut()
                    .insert(*hash, Identity { id, present: true });
                return Ok(id);
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
            Ok(id)
        } else {
            Err(StoreError::Invalid(
                "a new text has no prepared bytes".into(),
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
        self.reload.set(true);
    }

    pub(crate) fn removed(&self, hash: &[u8; 32], id: i64) {
        self.known
            .borrow_mut()
            .insert(*hash, Identity { id, present: false });
    }

    pub(crate) fn transaction(&self) -> TextTransaction<'_> {
        TextTransaction {
            texts: self,
            committed: false,
        }
    }
}

fn next(id: i64) -> Result<i64> {
    id.checked_add(1)
        .ok_or_else(|| StoreError::Invalid("text row identity exhausted".into()))
}

pub(crate) struct TextTransaction<'a> {
    texts: &'a TextIds,
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
            self.texts.reload.set(true);
        }
    }
}
