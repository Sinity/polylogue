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
mod counts;

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
    /// What it contributed to completion ([`crate::session_values`]).
    values: Vec<crate::session_values::Value>,
    counts: crate::derive::counts::Counts,
}

struct MessageChange {
    before: Option<Before>,
    /// The fresh insert already holds these values. Tool outcomes are read
    /// after pairing; other bodies need no reading back.
    installed: Option<Installed>,
}

/// What a message inserted in this transaction holds while its row is the
/// one the insert wrote: any later replacement or removal clears it.
struct Installed {
    session: i64,
    parent: Option<String>,
    counts: crate::derive::counts::Counts,
    has_calls: bool,
}

struct SessionBefore {
    script: bool,
    counts_known: bool,
}

#[derive(Default)]
struct Changed {
    /// Entries prepared from the selected typed content, kept only until commit.
    items: BTreeMap<i64, MessageItems>,
    /// Messages whose row, blocks, place or visibility changed, with the
    /// session, parent and place each had before the change (none for a
    /// new one).
    messages: BTreeMap<i64, MessageChange>,
    /// Events whose row changed, with the session each had before.
    events: BTreeMap<i64, Option<(i64, bool)>>,
    /// The session-wide rule for counting script calls, before any mutation.
    sessions: BTreeMap<i64, SessionBefore>,
    /// Sessions whose links as a child changed, by session ID.
    link_children: BTreeMap<String, String>,
    /// Sessions to bring up to date from their first unit: a change the
    /// writer does not describe row by row.
    whole: BTreeSet<i64>,
}

enum MessageItems {
    Metadata,
    Prepared(Vec<polylogue_model::read::IndexEntry>),
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
    pub(crate) fn prepare_items(&self, pk: i64, entries: Vec<polylogue_model::read::IndexEntry>) {
        self.changed
            .borrow_mut()
            .items
            .insert(pk, MessageItems::Prepared(entries));
    }

    pub(crate) fn has_prepared_items(&self, pk: i64) -> bool {
        matches!(
            self.changed.borrow().items.get(&pk),
            Some(MessageItems::Prepared(_))
        )
    }

    pub(crate) fn prepared_items(&self, pk: i64) -> Option<Vec<polylogue_model::read::IndexEntry>> {
        match self.changed.borrow().items.get(&pk) {
            Some(MessageItems::Prepared(entries)) => Some(entries.clone()),
            _ => None,
        }
    }

    pub(crate) fn reuses_items(&self, pk: i64) -> bool {
        let changed = self.changed.borrow();
        !changed.messages.contains_key(&pk)
            || matches!(changed.items.get(&pk), Some(MessageItems::Metadata))
    }

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
        self.record_message(conn, pk)?;
        self.changed.borrow_mut().items.remove(&pk);
        Ok(())
    }

    /// Placement and visibility change without changing the selected content.
    pub(crate) fn message_metadata(&self, conn: &Connection, pk: i64) -> Result<()> {
        self.record_message(conn, pk)?;
        self.replaced_message(pk)?;
        self.changed
            .borrow_mut()
            .items
            .entry(pk)
            .or_insert(MessageItems::Metadata);
        Ok(())
    }

    fn record_message(&self, conn: &Connection, pk: i64) -> Result<()> {
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
                    values: Vec::new(),
                    counts: crate::derive::counts::Counts::default(),
                })
            })
            .optional()?
            .map(|mut before| {
                before.values = crate::session_values::of_message(conn, pk, before.session)?;
                before.counts = crate::derive::counts::message(conn, pk)?
                    .ok_or_else(|| {
                        crate::StoreError::Invalid("changed message disappeared".into())
                    })?
                    .1;
                Ok::<_, crate::StoreError>(before)
            })
            .transpose()?;
        if let Some(before) = &before {
            self.session(conn, before.session)?;
        }
        self.changed.borrow_mut().messages.insert(
            pk,
            MessageChange {
                before,
                installed: None,
            },
        );
        Ok(())
    }

    pub(crate) fn new_message(&self, pk: i64) {
        self.changed
            .borrow_mut()
            .messages
            .entry(pk)
            .or_insert(MessageChange {
                before: None,
                installed: None,
            });
    }

    /// Records an event before its row changes or is inserted.
    pub(crate) fn event(&self, conn: &Connection, pk: i64) -> Result<()> {
        if self.changed.borrow().events.contains_key(&pk) {
            return Ok(());
        }
        let before = conn
            .prepare_cached("SELECT session_pk,kind='compaction' FROM event WHERE pk = ?1")?
            .query_row([pk], |row| <(Option<i64>, bool)>::try_from(row))
            .optional()?
            .and_then(|(session, compaction)| session.map(|session| (session, compaction)));
        if let Some((session, _)) = before {
            self.session(conn, session)?;
        }
        self.changed.borrow_mut().events.insert(pk, before);
        Ok(())
    }

    /// Records the child of a link before the link changes or is inserted.
    pub(crate) fn link_child(&self, conn: &Connection, child: &str) -> Result<()> {
        if !self.changed.borrow().link_children.contains_key(child) {
            let before = crate::lineage::relation_identity(conn, child)?;
            self.changed
                .borrow_mut()
                .link_children
                .insert(child.to_owned(), before);
        }
        Ok(())
    }

    /// Advance once per transaction only when its visible relations or their
    /// stored order changed. Reading the same values again is a no-op.
    pub(crate) fn finish_links(&self, conn: &Connection) -> Result<()> {
        for (child, before) in &self.changed.borrow().link_children {
            if *before != crate::lineage::relation_identity(conn, child)? {
                let previous: i64 = conn.prepare_cached(
                    "SELECT coalesce((SELECT CAST(value AS INTEGER) FROM meta WHERE key='relationship_revision'),0)",
                )?.query_row([], |row| row.get(0))?;
                let next = previous.checked_add(1).ok_or_else(|| {
                    crate::StoreError::Invalid("relationship revision exhausted".into())
                })?;
                crate::meta_set(conn, "relationship_revision", next.to_string())?;
                break;
            }
        }
        Ok(())
    }

    /// Records a link by its row before it changes or is deleted.
    pub(crate) fn link(&self, conn: &Connection, pk: i64) -> Result<()> {
        let child: Option<String> = conn
            .prepare_cached("SELECT child_id FROM link WHERE pk = ?1")?
            .query_row([pk], |row| row.get(0))
            .optional()?;
        if let Some(child) = child {
            self.link_child(conn, &child)?;
        }
        Ok(())
    }

    /// What a changed message contributed to completion before the
    /// transaction, when it was in `session_pk` then.
    pub(crate) fn values_before(
        &self,
        pk: i64,
        session_pk: i64,
    ) -> Vec<crate::session_values::Value> {
        match self
            .changed
            .borrow()
            .messages
            .get(&pk)
            .map(|change| &change.before)
        {
            Some(Some(before)) if before.session == session_pk => before.values.clone(),
            _ => Vec::new(),
        }
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
            if let Some(Some(before)) = changed.messages.get(pk).map(|change| &change.before)
                && before.session == session_pk
            {
                places.push(before.key.clone());
            }
            if let Some(key) = now
                .query_row([*pk, session_pk], |row| <(_, _)>::try_from(row))
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
        for (&pk, change) in &changed.messages {
            let before = &change.before;
            let after: Option<(i64, Option<String>)> = match &change.installed {
                Some(installed) => Some((installed.session, installed.parent.clone())),
                None => now
                    .query_row([pk], |row| <(_, _)>::try_from(row))
                    .optional()?,
            };
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
            for session in before
                .iter()
                .map(|(session, _)| session)
                .chain(after.iter())
            {
                out.entry(*session).or_default().events.insert(pk);
            }
        }
        let mut child = conn.prepare_cached("SELECT pk FROM session WHERE id = ?1")?;
        for id in changed.link_children.keys() {
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
