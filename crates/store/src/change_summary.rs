//! What one index transaction changed in the rows sessions show. The code
//! that writes a message, event or link row records it here before it
//! changes the row; before the transaction commits, the session item index
//! is brought up to date from the summary alone, so a transaction's cost
//! follows what it changed, not the length of the sessions it touched. A
//! summary lives for one transaction and is never stored.
use crate::Result;
use rusqlite::{Connection, OptionalExtension};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct ChangeSummary {
    changed: RefCell<Changed>,
}

#[derive(Default)]
struct Changed {
    /// Messages whose row, blocks, place or visibility changed, with the
    /// session and parent each had before the change (none for a new one).
    messages: BTreeMap<i64, Option<(i64, Option<String>)>>,
    /// Events whose row changed, with the session each had before.
    events: BTreeMap<i64, Option<i64>>,
    /// Sessions whose links as a child changed, by session ID.
    link_children: BTreeSet<String>,
    /// Sessions to bring up to date from their first unit: a change the
    /// writer does not describe row by row.
    whole: BTreeSet<i64>,
}

/// One session's part of a summary.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SessionChange {
    /// Messages of the session, before or after the change, that changed.
    pub messages: BTreeSet<i64>,
    /// IDs of the parents those messages had before or have now: a new or
    /// removed reply changes whether its parent is a branch point.
    pub parents: BTreeSet<String>,
    pub events: BTreeSet<i64>,
    pub links: bool,
    pub whole: bool,
    /// A changed message of the session existed before the transaction:
    /// something already stored was rewritten, not only added to.
    pub rewritten: bool,
}

impl ChangeSummary {
    /// Starts the summary of a transaction on `conn`, which from now on
    /// also records the version bodies, values and texts whose references
    /// it removes ([`crate::versions::remove_unreferenced`]).
    pub(crate) fn start(conn: &Connection) -> Result<Self> {
        crate::versions::track_unreferenced(conn)?;
        Ok(Self {
            changed: RefCell::default(),
        })
    }

    /// Records a message before its row changes or is inserted.
    pub(crate) fn message(&self, conn: &Connection, pk: i64) -> Result<()> {
        if self.changed.borrow().messages.contains_key(&pk) {
            return Ok(());
        }
        let before = conn
            .prepare_cached("SELECT session_pk, parent_id FROM message WHERE pk = ?1")?
            .query_row([pk], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;
        self.changed.borrow_mut().messages.insert(pk, before);
        Ok(())
    }

    /// Records an event before its row changes or is inserted.
    pub(crate) fn event(&self, conn: &Connection, pk: i64) -> Result<()> {
        if self.changed.borrow().events.contains_key(&pk) {
            return Ok(());
        }
        let before = conn
            .prepare_cached("SELECT session_pk FROM event WHERE pk = ?1")?
            .query_row([pk], |row| row.get::<_, Option<i64>>(0))
            .optional()?
            .flatten();
        self.changed.borrow_mut().events.insert(pk, before);
        Ok(())
    }

    /// Records the child of a link before the link changes or is inserted.
    pub(crate) fn link_child(&self, child: &str) {
        self.changed
            .borrow_mut()
            .link_children
            .insert(child.to_owned());
    }

    /// Records a link by its row before it changes or is deleted.
    pub(crate) fn link(&self, conn: &Connection, pk: i64) -> Result<()> {
        let child: Option<String> = conn
            .prepare_cached("SELECT child_id FROM link WHERE pk = ?1")?
            .query_row([pk], |row| row.get(0))
            .optional()?;
        if let Some(child) = child {
            self.link_child(&child);
        }
        Ok(())
    }

    /// Records a session whose rows changed in a way not described here.
    pub(crate) fn whole_session(&self, pk: i64) {
        self.changed.borrow_mut().whole.insert(pk);
    }

    /// The changes by session, read after the rows changed. A message or
    /// event that moved between sessions belongs to both.
    pub(crate) fn sessions(&self, conn: &Connection) -> Result<BTreeMap<i64, SessionChange>> {
        let changed = self.changed.borrow();
        let mut out: BTreeMap<i64, SessionChange> = BTreeMap::new();
        let mut now =
            conn.prepare_cached("SELECT session_pk, parent_id FROM message WHERE pk = ?1")?;
        for (&pk, before) in &changed.messages {
            let after: Option<(i64, Option<String>)> = now
                .query_row([pk], |row| Ok((row.get(0)?, row.get(1)?)))
                .optional()?;
            for (session, parent) in before.iter().chain(after.iter()) {
                let change = out.entry(*session).or_default();
                change.messages.insert(pk);
                change.rewritten |= before.is_some();
                if let Some(parent) = parent {
                    change.parents.insert(parent.clone());
                }
            }
        }
        let mut now = conn.prepare_cached("SELECT session_pk FROM event WHERE pk = ?1")?;
        for (&pk, before) in &changed.events {
            let after: Option<i64> = now.query_row([pk], |row| row.get(0)).optional()?.flatten();
            for session in before.iter().chain(after.iter()) {
                out.entry(*session).or_default().events.insert(pk);
            }
        }
        let mut child = conn.prepare_cached("SELECT pk FROM session WHERE id = ?1")?;
        for id in &changed.link_children {
            if let Some(pk) = child.query_row([id], |row| row.get(0)).optional()? {
                out.entry(pk).or_default().links = true;
            }
        }
        for &pk in &changed.whole {
            out.entry(pk).or_default().whole = true;
        }
        Ok(out)
    }
}
