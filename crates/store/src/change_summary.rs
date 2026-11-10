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

/// A message's place in session order (`message_session`): its
/// `session_order` and ID.
pub(crate) type OrderKey = (String, String);

/// A message as it was before the transaction changed it.
struct Before {
    session: i64,
    parent: Option<String>,
    key: OrderKey,
}

#[derive(Default)]
struct Changed {
    /// Messages whose row, blocks, place or visibility changed, with the
    /// session, parent and place each had before the change (none for a
    /// new one).
    messages: BTreeMap<i64, Option<Before>>,
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
            .prepare_cached(
                "SELECT session_pk, parent_id, session_order, id
                 FROM message WHERE pk = ?1",
            )?
            .query_row([pk], |row| {
                Ok(Before {
                    session: row.get(0)?,
                    parent: row.get(1)?,
                    key: (row.get(2)?, row.get(3)?),
                })
            })
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

    /// The changed messages that were in `session_pk` before the
    /// transaction: every message a derivation changed, since derivations
    /// change messages without moving them.
    pub(crate) fn messages_before(&self, session_pk: i64) -> BTreeSet<i64> {
        self.changed
            .borrow()
            .messages
            .iter()
            .filter(|(_, before)| before.as_ref().is_some_and(|b| b.session == session_pk))
            .map(|(pk, _)| *pk)
            .collect()
    }

    /// The places in `session_pk`'s order that changed messages of it held
    /// before the transaction or hold now; `messages` are the session's
    /// changed messages ([`SessionChange::messages`]).
    pub(crate) fn places(
        &self,
        conn: &Connection,
        session_pk: i64,
        messages: &BTreeSet<i64>,
    ) -> Result<Vec<OrderKey>> {
        let changed = self.changed.borrow();
        let mut now = conn.prepare_cached(
            "SELECT session_order, id FROM message
             WHERE pk = ?1 AND session_pk = ?2",
        )?;
        let mut places = Vec::new();
        for pk in messages {
            if let Some(Some(before)) = changed.messages.get(pk)
                && before.session == session_pk
            {
                places.push(before.key.clone());
            }
            if let Some(key) = now
                .query_row([*pk, session_pk], |row| Ok((row.get(0)?, row.get(1)?)))
                .optional()?
            {
                places.push(key);
            }
        }
        Ok(places)
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
            let before = before.as_ref().map(|b| (b.session, b.parent.clone()));
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
