//! Text identities belong to the serial writer. A text row is found by its
//! hash, or inserted with the next identity. Identities only increase while
//! the writer runs, so one a removed or rolled-back text had is not given to
//! another text while a mark or a search entry may still name it.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

#[derive(Default)]
struct TextMark {
    indexed: i64,
}

pub(crate) struct TextIds {
    marked: RefCell<HashMap<i64, TextMark>>,
    marked_entries: usize,
    groups: super::StoredGroups,
    next: Cell<i64>,
    /// The table may hold rows this writer did not write (a confirmed scope
    /// change): the next identity is taken above them before it is used.
    reload: Cell<bool>,
}

impl TextIds {
    pub(crate) fn new(conn: &Connection, bytes: u64) -> Result<Self> {
        let ids = Self {
            marked: RefCell::new(HashMap::new()),
            marked_entries: (bytes / 4 / 64).max(1) as usize,
            groups: super::StoredGroups::new(bytes / 4),
            next: Cell::new(1),
            reload: Cell::new(true),
        };
        ids.refresh(conn)?;
        Ok(ids)
    }

    fn refresh(&self, conn: &Connection) -> Result<()> {
        if self.reload.replace(false) {
            let max: i64 = conn.scalar("SELECT coalesce(max(id),0) FROM text", [])?;
            self.next.set(self.next.get().max(next(max)?));
        }
        Ok(())
    }

    /// The row of a complete text hash: the stored one, or a new row holding
    /// the prepared body.
    pub(crate) fn resolve(
        &self,
        conn: &Connection,
        hash: &[u8; 32],
        body: Option<super::groups::Prepared<'_>>,
        membership: Option<&super::NewMembership<'_>>,
    ) -> Result<i64> {
        self.refresh(conn)?;
        let stored = conn
            .prepare_cached("SELECT id FROM text WHERE hash=?1")?
            .query_row([hash.as_slice()], |row| row.get::<_, i64>(0))
            .optional()?;
        if let Some(id) = stored {
            return Ok(id);
        }
        let body =
            body.ok_or_else(|| StoreError::Invalid("a new text has no prepared bytes".into()))?;
        let id = self.next.get();
        let bits = membership
            .and_then(|membership| membership.bits.get(hash))
            .copied()
            .unwrap_or(0);
        if super::groups::insert(conn, &self.groups, Some(id), hash, body, bits)?.is_none() {
            return Err(StoreError::Invalid(
                "allocated text identity conflicts with stored rows".into(),
            ));
        }
        if bits != 0 {
            if !membership.is_some_and(|membership| membership.at_build_end) {
                for bit in [1, 2].into_iter().filter(|bit| bits & bit != 0) {
                    super::queue_entry(conn, id, bit)?;
                }
            }
            self.mark(id, bits);
        }
        self.next.set(next(id)?);
        Ok(id)
    }

    /// A confirmed scope change replaced rows this writer did not write.
    pub(crate) fn discard_cached(&self) {
        self.marked.borrow_mut().clear();
        self.groups.clear();
        self.reload.set(true);
    }

    pub(crate) fn removed(&self, id: i64) {
        self.groups.clear();
        self.marked.borrow_mut().remove(&id);
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
            self.texts.groups.clear();
        }
    }
}
