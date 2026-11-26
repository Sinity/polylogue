//! What one index transaction changed in the rows sessions show. The code
//! that writes a message, event or link row records it here before it
//! changes the row. When the transaction ends, each session it touched is
//! marked due for its summaries from the earliest row key its changes
//! reached ([`ChangeSummary::mark_due`]). A summary lives for one
//! transaction and is never stored; nothing here is a value of a session.
use crate::Result;
use crate::summary::Key;
use rusqlite::{Connection, OptionalExtension, params};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct ChangeSummary {
    changed: RefCell<Changed>,
}

/// A message as it was before the transaction changed it.
struct Before {
    session: i64,
    parent: Option<String>,
    key: Key,
}

#[derive(Default)]
struct Changed {
    /// Messages whose row, blocks, place, readings or visibility changed,
    /// with the session, parent and place each had before the change (none
    /// for a new one).
    messages: BTreeMap<i64, Option<Before>>,
    /// Events whose row changed, with the session and row key each had
    /// before.
    events: BTreeMap<i64, Option<(i64, Key)>>,
    /// Sessions whose rows or readings the transaction touched.
    sessions: BTreeSet<i64>,
    /// Sessions whose links as a child changed, by session ID.
    link_children: BTreeMap<String, String>,
    /// Sessions due from their first row: a change the writer does not
    /// describe row by row.
    whole: BTreeSet<i64>,
}

impl ChangeSummary {
    /// Starts the summary of a transaction.
    pub(crate) fn start(_conn: &Connection) -> Result<Self> {
        Ok(Self {
            changed: RefCell::default(),
        })
    }

    /// Records a message before its row changes or is inserted, or one the
    /// transaction composed without changing its row: its readings may have
    /// changed, which its session's summaries read.
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
                    key: Key::message(&row.get::<_, String>(2)?, &row.get::<_, String>(3)?),
                })
            })
            .optional()?;
        if let Some(before) = &before {
            self.session(before.session);
        }
        self.changed.borrow_mut().messages.insert(pk, before);
        Ok(())
    }

    /// Records a message new in this transaction.
    pub(crate) fn new_message(&self, pk: i64) {
        self.changed.borrow_mut().messages.entry(pk).or_insert(None);
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
            self.message(conn, message)?;
        }
        Ok(())
    }

    /// Ends the transaction's part in the sessions' summaries: each session
    /// it touched takes `counter` as the change of its own rows and is
    /// marked due from the earliest row key its changes reached, before or
    /// after them. A session composed whole is due from the start. A reply
    /// that arrives, goes or moves can change whether its parent is a
    /// branch point, so the parent's key is reached too when the parent
    /// has more than one reply or the reply existed before.
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
        let mut message_now = conn.prepare_cached(
            "SELECT session_pk, session_order, id, parent_id FROM message WHERE pk = ?1",
        )?;
        let mut parent = conn.prepare_cached(
            "SELECT p.session_order, p.id,
                    (SELECT count(*) FROM (SELECT 1 FROM message c
                                           WHERE c.session_pk = ?1 AND c.parent_id = ?2
                                             AND c.duplicate_of IS NULL LIMIT 2))
             FROM message p WHERE p.id = ?2 AND p.session_pk = ?1",
        )?;
        let mut parent_of = |session: i64, id: &str, existed: bool| -> Result<Option<Key>> {
            Ok(parent
                .query_row(params![session, id], |row| {
                    <(String, String, i64)>::try_from(row)
                })
                .optional()?
                .filter(|(_, _, replies)| existed || *replies > 1)
                .map(|(order, id, _)| Key::message(&order, &id)))
        };
        for (&pk, before) in &changed.messages {
            if let Some(before) = before {
                lower(before.session, before.key.clone());
                if let Some(id) = &before.parent
                    && let Some(key) = parent_of(before.session, id, true)?
                {
                    lower(before.session, key);
                }
            }
            if let Some((session, order, id, parent)) = message_now
                .query_row([pk], |row| {
                    <(i64, String, String, Option<String>)>::try_from(row)
                })
                .optional()?
            {
                lower(session, Key::message(&order, &id));
                if let Some(parent) = parent
                    && let Some(key) = parent_of(session, &parent, before.is_some())?
                {
                    lower(session, key);
                }
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
