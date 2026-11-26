//! What one index transaction changed in the rows sessions show. The code
//! that writes a message, event or link row records it here before it
//! changes the row. When the transaction ends, each session it touched is
//! marked due for its summaries from the earliest row key its changes
//! reached ([`ChangeSummary::mark_due`]), and the session item index is
//! brought up to date from the summary alone, so a transaction's cost
//! follows what it changed, not the length of the sessions it touched. A
//! summary lives for one transaction and is never stored; nothing here is
//! a value of a session.
use crate::Result;
use crate::summary::Key;
use rusqlite::{Connection, OptionalExtension, params};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct ChangeSummary {
    changed: RefCell<Changed>,
    /// The sessions the transaction composes hold every message their
    /// events follow, or the message is one of these, shown in another
    /// session, or none shows it (a build's stream).
    sessions_alone: Option<std::sync::Arc<std::collections::HashSet<String>>>,
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

struct MessageChange {
    before: Option<Before>,
    /// The session a message new in this transaction was written in, when
    /// that session is composed whole: nothing about it is read back.
    whole_in: Option<i64>,
}

#[derive(Default)]
struct Changed {
    /// Entries prepared from the selected typed content, kept only until commit.
    items: BTreeMap<i64, MessageItems>,
    /// Messages whose row, blocks, place or visibility changed, with the
    /// session, parent and place each had before the change (none for a
    /// new one).
    messages: BTreeMap<i64, MessageChange>,
    /// Events whose row changed, with the session and row key each had
    /// before.
    events: BTreeMap<i64, Option<(i64, Key)>>,
    /// Sessions whose rows or readings the transaction touched.
    sessions: BTreeSet<i64>,
    /// Sessions whose links as a child changed, by session ID.
    link_children: BTreeMap<String, String>,
    /// Sessions to bring up to date from their first unit: a change the
    /// writer does not describe row by row.
    whole: BTreeSet<i64>,
}

enum MessageItems {
    Content,
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
        matches!(changed.items.get(&pk), None | Some(MessageItems::Metadata))
    }

    /// Starts the summary of a transaction.
    pub(crate) fn start(_conn: &Connection) -> Result<Self> {
        Ok(Self {
            changed: RefCell::default(),
            sessions_alone: None,
        })
    }

    /// Starts the summary of a transaction composing sessions each of which
    /// holds every message its events follow, unless the message is one of
    /// `elsewhere`, shown in another session, or none shows it: the item
    /// index finds those messages among the session's own.
    pub(crate) fn start_sessions_alone(
        conn: &Connection,
        elsewhere: std::sync::Arc<std::collections::HashSet<String>>,
    ) -> Result<Self> {
        Ok(Self {
            sessions_alone: Some(elsewhere),
            ..Self::start(conn)?
        })
    }

    /// When the transaction composes sessions alone, the messages their
    /// events follow that other sessions show.
    pub(crate) fn sessions_alone(&self) -> Option<&std::collections::HashSet<String>> {
        self.sessions_alone.as_deref()
    }

    /// Records a message before its row changes or is inserted.
    pub(crate) fn message(&self, conn: &Connection, pk: i64) -> Result<()> {
        self.record_message(conn, pk)?;
        self.changed
            .borrow_mut()
            .items
            .insert(pk, MessageItems::Content);
        Ok(())
    }

    /// Records a message the transaction composed without changing its row:
    /// its readings may have changed, which its session's summaries read.
    pub(crate) fn reached(&self, conn: &Connection, pk: i64) -> Result<()> {
        self.record_message(conn, pk)
    }

    /// Placement and visibility change without changing the selected content.
    pub(crate) fn message_metadata(&self, conn: &Connection, pk: i64) -> Result<()> {
        self.record_message(conn, pk)?;
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
                })
            })
            .optional()?;
        if let Some(before) = &before {
            self.session(before.session);
        }
        self.changed.borrow_mut().messages.insert(
            pk,
            MessageChange {
                before,
                whole_in: None,
            },
        );
        Ok(())
    }

    pub(crate) fn new_message(&self, pk: i64) {
        let mut changed = self.changed.borrow_mut();
        changed.messages.entry(pk).or_insert(MessageChange {
            before: None,
            whole_in: None,
        });
        changed.items.entry(pk).or_insert(MessageItems::Content);
    }

    /// Records a message new in this transaction, written in `session`,
    /// which the transaction composes whole ([`Self::whole_session`]).
    pub(crate) fn new_message_in(&self, pk: i64, session: i64) {
        let mut changed = self.changed.borrow_mut();
        changed.messages.entry(pk).or_insert(MessageChange {
            before: None,
            whole_in: Some(session),
        });
        changed.items.entry(pk).or_insert(MessageItems::Content);
    }

    /// Records an event the transaction inserted.
    pub(crate) fn new_event(&self, pk: i64) {
        self.changed.borrow_mut().events.entry(pk).or_insert(None);
    }

    /// Records an event before its row changes or is deleted.
    pub(crate) fn event(&self, conn: &Connection, pk: i64) -> Result<()> {
        if self.changed.borrow().events.contains_key(&pk) {
            return Ok(());
        }
        let before = conn
            .prepare_cached("SELECT session_pk, session_order, id FROM event WHERE pk = ?1")?
            .query_row([pk], |row| {
                <(Option<i64>, Option<String>, String)>::try_from(row)
            })
            .optional()?
            .and_then(|(session, order, id)| Some((session?, Key::event(order.as_deref()?, &id))));
        if let Some((session, _)) = &before {
            self.session(*session);
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
                    crate::StoreError::internal("relationship revision exhausted")
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

    /// Records a session whose rows changed in a way not described here.
    pub(crate) fn whole_session(&self, pk: i64) {
        self.changed.borrow_mut().whole.insert(pk);
    }

    /// Tool outcomes contribute to counts and measures. They leave the
    /// selected content and placement that the item index stores alone.
    pub(crate) fn item_sessions(&self, conn: &Connection) -> Result<BTreeMap<i64, SessionChange>> {
        self.changed_sessions(conn, |items| items.is_some())
    }

    fn changed_sessions(
        &self,
        conn: &Connection,
        includes: impl Fn(Option<&MessageItems>) -> bool,
    ) -> Result<BTreeMap<i64, SessionChange>> {
        let changed = self.changed.borrow();
        let mut out: BTreeMap<i64, SessionChange> = BTreeMap::new();
        let mut now =
            conn.prepare_cached("SELECT session_pk, parent_id FROM message WHERE pk = ?1")?;
        for (&pk, change) in &changed.messages {
            if !includes(changed.items.get(&pk)) {
                continue;
            }
            // A whole session is brought up to date from all its rows.
            if let Some(session) = change.whole_in.filter(|s| changed.whole.contains(s)) {
                out.entry(session).or_default().messages.insert(pk);
                continue;
            }
            let before = &change.before;
            let after: Option<(i64, Option<String>)> = now
                .query_row([pk], |row| <(_, _)>::try_from(row))
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

    /// Records that the transaction touched a session's rows or readings.
    pub(crate) fn session(&self, pk: i64) {
        self.changed.borrow_mut().sessions.insert(pk);
    }

    /// A result can change the count of a call in a different message: the
    /// messages holding the calls named `id` in `session` are recorded.
    pub(crate) fn call_result(&self, conn: &Connection, session: i64, id: &str) -> Result<()> {
        let messages = conn
            .prepare_cached(
                "SELECT b.message_pk FROM tool_call t JOIN block b ON b.pk=t.call_block_pk
                 WHERE t.session_pk=?1 AND t.call_id=?2",
            )?
            .query_map(params![session, id], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for message in messages {
            self.record_message(conn, message)?;
        }
        Ok(())
    }

    /// Ends the transaction's part in the sessions' summaries: each session
    /// it touched takes `counter` as the change of its own rows and is
    /// marked due from the earliest row key its changes reached, before or
    /// after them. A session composed whole is due from the start.
    pub(crate) fn mark_due(&self, conn: &Connection, counter: i64) -> Result<()> {
        let changed = self.changed.borrow();
        let mut reach: BTreeMap<i64, Key> = BTreeMap::new();
        let mut lower = |session: i64, key: Key| {
            let held = reach.entry(session).or_insert_with(Key::past_every_row);
            if key < *held {
                *held = key;
            }
        };
        for &session in changed.sessions.iter().chain(&changed.whole) {
            lower(session, Key::past_every_row());
        }
        for &session in &changed.whole {
            lower(session, Key::start());
        }
        let mut message_now =
            conn.prepare_cached("SELECT session_pk, session_order, id FROM message WHERE pk = ?1")?;
        for (&pk, change) in &changed.messages {
            if let Some(before) = &change.before {
                lower(before.session, Key::message(&before.key.0, &before.key.1));
            }
            if let Some((session, order, id)) = message_now
                .query_row([pk], |row| <(i64, String, String)>::try_from(row))
                .optional()?
            {
                lower(session, Key::message(&order, &id));
            }
        }
        let mut event_now =
            conn.prepare_cached("SELECT session_pk, session_order, id FROM event WHERE pk = ?1")?;
        for (&pk, before) in &changed.events {
            if let Some((session, key)) = before {
                lower(*session, key.clone());
            }
            if let Some((Some(session), Some(order), id)) = event_now
                .query_row([pk], |row| {
                    <(Option<i64>, Option<String>, String)>::try_from(row)
                })
                .optional()?
            {
                lower(session, Key::event(&order, &id));
            }
        }
        let mut exists =
            conn.prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?;
        let mut stamp = conn.prepare_cached("UPDATE session SET changed = ?2 WHERE pk = ?1")?;
        for (session, from) in reach {
            if !exists.query_row([session], |row| row.get::<_, bool>(0))? {
                continue;
            }
            stamp.execute(params![session, counter])?;
            crate::summary::mark(conn, session, &from)?;
        }
        Ok(())
    }
}
