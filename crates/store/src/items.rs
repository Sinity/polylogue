//! The session item index (`session_item`, `session_item_state`). Every write
//! transaction that changes a session's rows brings that session's entries
//! up to date before it commits, so the index is never behind the rows.
//!
//! A transaction's [change summary](crate::change_summary) names the
//! messages, events and links it changed. From it the refresh finds the
//! earliest entry those changes can reach and rebuilds the session's units
//! (message versions and events, in reading order) from the last timed
//! message before it; entries before that message are not read. Of the
//! rebuilt units, those equal to the ones already indexed keep their entries.
//! When the difference is only new units at the end, `append_seq` moves;
//! when an indexed unit changed, moved or disappeared, or the links that
//! compose the session changed, `rewrite_version` moves and outstanding
//! cursors go stale.
//!
//! An event that names no message but has a time goes after the last
//! message at or before its time. To place such events from a suffix alone,
//! each message's first entry keeps the latest message time up to it
//! (`time_bound`): the events placed at or after a timed message are exactly
//! those whose time is at least its bound.
use crate::change_summary::{ChangeSummary, SessionChange};
use crate::encoding::{Columns, restore_text, unpack_block, unpack_text};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    Message, MessageId, PromptKind, Role, SessionId, Usage,
    read::{IndexEntry, MessagePlace, event_entry, message_entries, microcompacted_calls},
};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeSet, HashMap, HashSet};

/// A session's entries and the two counters cursors bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredItems {
    pub session_pk: i64,
    pub append_seq: u64,
    pub rewrite_version: u64,
    pub entries: Vec<IndexEntry>,
}

/// The counters alone, read in O(1) to validate a cached index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemVersions {
    pub append_seq: u64,
    pub rewrite_version: u64,
}

enum Unit {
    Message {
        pk: i64,
        version: Vec<u8>,
        place: MessagePlace,
        time_bound: Option<i64>,
    },
    Event {
        pk: i64,
        id: String,
        kind: String,
        at: Option<i64>,
        data_bytes: u64,
    },
}
impl Unit {
    /// What makes this unit's entries what they are, without its body.
    fn identity(&self) -> Vec<u8> {
        let mut digest = blake3::Hasher::new();
        match self {
            Self::Message {
                pk, version, place, ..
            } => {
                digest.update(b"message\0");
                digest.update(&pk.to_le_bytes());
                digest.update(version);
                digest.update(&[
                    match place.active_path {
                        None => 2,
                        Some(true) => 1,
                        Some(false) => 0,
                    },
                    u8::from(place.branch_point),
                ]);
            }
            Self::Event { pk, kind, .. } => {
                digest.update(b"event\0");
                digest.update(&pk.to_le_bytes());
                digest.update(kind.as_bytes());
            }
        }
        digest.finalize().as_bytes().to_vec()
    }

    /// The columns of the unit's first entry that locate it.
    fn columns(&self) -> (Option<i64>, Option<i64>, Option<i64>) {
        match self {
            Self::Message { pk, time_bound, .. } => (Some(*pk), None, *time_bound),
            Self::Event { pk, .. } => (None, Some(*pk), None),
        }
    }
}

/// An indexed message entry the rebuild starts from: entries before it stay.
#[derive(Debug, Clone, Copy)]
struct Anchor {
    seq: i64,
    pk: i64,
    time_bound: Option<i64>,
}

/// A message row as the units read it.
struct MessageRow {
    pk: i64,
    id: String,
    version: Vec<u8>,
    active: Option<bool>,
    branch_point: bool,
    at: Option<i64>,
}

/// An event row as the units read it.
struct EventRow {
    pk: i64,
    id: String,
    kind: String,
    at: Option<i64>,
    data_bytes: u64,
    after: Option<i64>,
    seq: i64,
}

impl EventRow {
    fn unit(self) -> Unit {
        Unit::Event {
            pk: self.pk,
            id: self.id,
            kind: self.kind,
            at: self.at,
            data_bytes: self.data_bytes,
        }
    }
}

const MESSAGE_COLUMNS: &str = "m.pk, m.id,
    coalesce(s.version_hash, (SELECT CAST(count(*) || ':' || coalesce(max(b.pk), 0) AS BLOB)
                              FROM block b WHERE b.message_pk = m.pk)),
    m.on_active_path, m.at,
    EXISTS(SELECT 1 FROM message c WHERE c.session_pk = m.session_pk AND c.parent_id = m.id
           AND c.duplicate_of IS NULL LIMIT 1 OFFSET 1)";
const EVENT_COLUMNS: &str = "e.pk, e.id, e.kind, e.at, length(e.data), e.after_message_pk, e.seq";

fn message_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MessageRow> {
    Ok(MessageRow {
        pk: row.get(0)?,
        id: row.get(1)?,
        version: row.get(2)?,
        active: row.get::<_, Option<i64>>(3)?.map(|value| value != 0),
        at: row.get(4)?,
        branch_point: row.get(5)?,
    })
}

fn event_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<EventRow> {
    Ok(EventRow {
        pk: row.get(0)?,
        id: row.get(1)?,
        kind: row.get(2)?,
        at: row.get(3)?,
        data_bytes: row.get::<_, i64>(4)?.max(0) as u64,
        after: row.get(5)?,
        seq: row.get(6)?,
    })
}

/// The session's units in reading order from `anchor` (from the first
/// message without one): messages by source position; each event after the
/// message it follows, a timed event that names no message after the last
/// message at or before its time, and the rest at the end. `extra` names
/// changed events to place, in case nothing else finds them.
fn units_from(
    conn: &Connection,
    session_pk: i64,
    anchor: Option<Anchor>,
    extra: &BTreeSet<i64>,
) -> Result<Vec<Unit>> {
    // A message with no stored version (rows written directly by the
    // fixture builder) is identified by its block rows.
    let rows: Vec<MessageRow> = match anchor {
        None => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                 ORDER BY m.order_input_id, m.seq, m.record_position, m.id"
            ))?
            .query_map([session_pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
        Some(anchor) => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message a, message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
                 WHERE a.pk = ?2 AND m.session_pk = ?1 AND m.duplicate_of IS NULL
                   AND (m.order_input_id, m.seq, m.record_position, m.id)
                       >= (a.order_input_id, a.seq, a.record_position, a.id)
                 ORDER BY m.order_input_id, m.seq, m.record_position, m.id"
            ))?
            .query_map(params![session_pk, anchor.pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
    };
    let events = events_from(conn, session_pk, anchor, &rows, extra)?;
    let loaded: HashSet<i64> = rows.iter().map(|row| row.pk).collect();
    let mut after: HashMap<i64, Vec<Unit>> = HashMap::new();
    let mut timed: Vec<(i64, Unit)> = Vec::new();
    let mut unplaced = Vec::new();
    let mut orphaned = Vec::new();
    let mut shown_before = conn.prepare_cached(
        "SELECT session_pk = ?2 AND duplicate_of IS NULL FROM message WHERE pk = ?1",
    )?;
    for event in events {
        match (event.after, event.at) {
            (Some(message), _) if loaded.contains(&message) => {
                after.entry(message).or_default().push(event.unit());
            }
            (Some(message), _) => {
                // Shown before the anchor: it is placed there, not here.
                let shown = anchor.is_some()
                    && shown_before
                        .query_row(params![message, session_pk], |row| row.get::<_, bool>(0))
                        .optional()?
                        .unwrap_or(false);
                if !shown {
                    orphaned.push(event.unit());
                }
            }
            (None, Some(at)) => {
                if anchor.is_none_or(|anchor| anchor.time_bound.is_none_or(|bound| at >= bound)) {
                    timed.push((at, event.unit()));
                }
            }
            (None, None) => unplaced.push(event.unit()),
        }
    }
    // An event that names no message (a hook line) goes after the last
    // message at or before its time, so a later message still appends.
    timed.sort_by_key(|(at, _)| *at);
    let mut timed = timed.into_iter().peekable();
    // The time of the next message that has one, for each message.
    let mut next_times = vec![None; rows.len()];
    for index in (0..rows.len().saturating_sub(1)).rev() {
        next_times[index] = rows[index + 1].at.or(next_times[index + 1]);
    }
    let mut out = Vec::with_capacity(rows.len());
    let mut time_bound = anchor.and_then(|anchor| anchor.time_bound);
    for (index, row) in rows.iter().enumerate() {
        // The anchor's bound is stored; the first message's is none.
        if index > 0 {
            time_bound = match (time_bound, row.at) {
                (Some(bound), Some(at)) => Some(bound.max(at)),
                (bound, at) => bound.or(at),
            };
        }
        out.push(Unit::Message {
            pk: row.pk,
            version: row.version.clone(),
            place: MessagePlace {
                // Unknown and on the path read the same; only off the path
                // changes what a reading shows.
                active_path: (row.active == Some(false)).then_some(false),
                branch_point: row.branch_point,
            },
            time_bound,
        });
        if let Some(events) = after.remove(&row.pk) {
            out.extend(events);
        }
        // Timed events before the next timed message belong here; those
        // after the last timed message go at the end.
        if let Some(next) = next_times[index] {
            while timed.peek().is_some_and(|(at, _)| *at < next) {
                out.push(timed.next().expect("peeked").1);
            }
        }
    }
    out.extend(timed.map(|(_, unit)| unit));
    // Events whose message is not shown (a duplicate, or not stored) keep
    // their order at the end with the events that name no message.
    orphaned.sort_by_key(|unit| match unit {
        Unit::Event { pk, .. } | Unit::Message { pk, .. } => *pk,
    });
    out.extend(unplaced);
    out.extend(orphaned);
    Ok(out)
}

/// The events that can be placed from `anchor` on, in stored order: those
/// following a loaded message, those naming no message, those indexed from
/// the anchor on (a message they follow may have stopped being shown), and
/// the changed ones.
fn events_from(
    conn: &Connection,
    session_pk: i64,
    anchor: Option<Anchor>,
    rows: &[MessageRow],
    extra: &BTreeSet<i64>,
) -> Result<Vec<EventRow>> {
    let Some(anchor) = anchor else {
        return Ok(conn
            .prepare_cached(&format!(
                "SELECT {EVENT_COLUMNS} FROM event e WHERE e.session_pk = ?1 ORDER BY e.seq, e.id"
            ))?
            .query_map([session_pk], event_row)?
            .collect::<rusqlite::Result<_>>()?);
    };
    let mut found: HashMap<i64, EventRow> = HashMap::new();
    let mut following = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e
         WHERE e.after_message_id = ?1 AND e.after_message_pk = ?2 AND e.session_pk = ?3"
    ))?;
    for row in rows {
        for event in following.query_map(params![row.id, row.pk, session_pk], event_row)? {
            let event = event?;
            found.insert(event.pk, event);
        }
    }
    // Timed ones from the anchor's bound on, and the untimed ones.
    let mut timed = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e INDEXED BY event_unanchored
         WHERE e.session_pk = ?1 AND e.after_message_pk IS NULL AND e.at >= coalesce(?2, -9223372036854775808)"
    ))?;
    let mut untimed = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e INDEXED BY event_unanchored
         WHERE e.session_pk = ?1 AND e.after_message_pk IS NULL AND e.at IS NULL"
    ))?;
    for event in timed
        .query_map(params![session_pk, anchor.time_bound], event_row)?
        .chain(untimed.query_map([session_pk], event_row)?)
    {
        let event = event?;
        found.insert(event.pk, event);
    }
    let indexed: Vec<i64> = conn
        .prepare_cached(
            "SELECT event_pk FROM session_item
             WHERE session_pk = ?1 AND seq >= ?2 AND event_pk IS NOT NULL",
        )?
        .query_map(params![session_pk, anchor.seq], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut by_pk = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e WHERE e.pk = ?1 AND e.session_pk = ?2"
    ))?;
    for pk in indexed.into_iter().chain(extra.iter().copied()) {
        if found.contains_key(&pk) {
            continue;
        }
        if let Some(event) = by_pk
            .query_row(params![pk, session_pk], event_row)
            .optional()?
        {
            found.insert(pk, event);
        }
    }
    let mut events: Vec<EventRow> = found.into_values().collect();
    events.sort_by(|a, b| (a.seq, &a.id).cmp(&(b.seq, &b.id)));
    Ok(events)
}

/// A session's entries computed from its rows alone, without the stored
/// index: what the stored index must always equal.
#[cfg(test)]
pub(crate) fn computed_entries(conn: &Connection, session_pk: i64) -> Result<Vec<IndexEntry>> {
    let mut out = Vec::new();
    for unit in units_from(conn, session_pk, None, &BTreeSet::new())? {
        out.extend(unit_entries(conn, &unit)?);
    }
    Ok(out)
}

/// Brings every session a transaction changed up to date from its change
/// summary; returns the number of units read to do so.
pub(crate) fn refresh(conn: &Connection, changes: &ChangeSummary) -> Result<u64> {
    let mut read = 0;
    let mut exists = conn.prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?;
    for (session_pk, change) in changes.sessions(conn)? {
        if exists.query_row([session_pk], |row| row.get::<_, bool>(0))? {
            read += refresh_session(conn, session_pk, &change)?;
        }
    }
    Ok(read)
}

/// Bring up to date every session whose rows changed since it was indexed:
/// rows the fixture builder writes directly. The fixture builder writes rows
/// the writer would refuse (message IDs without an origin, partial block
/// bodies) for tests that never read them as sessions; such a session is
/// left unindexed, so only a session read of it is refused.
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn refresh_stale(conn: &Connection) -> Result<()> {
    let sessions: Vec<i64> = conn
        .prepare(
            "SELECT s.pk FROM session s LEFT JOIN session_item_state t ON t.session_pk = s.pk
             WHERE t.session_pk IS NULL OR t.indexed_changed IS NOT s.changed ORDER BY s.pk",
        )?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for pk in sessions {
        conn.execute_batch("SAVEPOINT fixture_items")?;
        match rebuild(conn, pk, None, &BTreeSet::new()) {
            Ok(_) => conn.execute_batch("RELEASE fixture_items")?,
            Err(error) => {
                conn.execute_batch("ROLLBACK TO fixture_items; RELEASE fixture_items")?;
                if !matches!(
                    error,
                    StoreError::Invalid(_) | StoreError::StoredValue { .. } | StoreError::Json(_)
                ) {
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

/// Where a session's rebuild starts.
enum Start {
    /// No unit changed: only the composing links are compared.
    Links,
    Whole,
    From(Anchor),
}

fn refresh_session(conn: &Connection, session_pk: i64, change: &SessionChange) -> Result<u64> {
    let indexed: bool = conn
        .prepare_cached("SELECT EXISTS(SELECT 1 FROM session_item_state WHERE session_pk = ?1)")?
        .query_row([session_pk], |row| row.get(0))?;
    let start = if !indexed || change.whole {
        Start::Whole
    } else {
        start(conn, session_pk, change)?
    };
    match start {
        Start::Links => {
            if change.links {
                relink(conn, session_pk)?;
            }
            Ok(0)
        }
        // A rebuild compares the links itself.
        Start::Whole => rebuild(conn, session_pk, None, &change.events),
        Start::From(anchor) => rebuild(conn, session_pk, Some(anchor), &change.events),
    }
}

/// The indexed entry of a message or an event, in this session.
fn header(conn: &Connection, session_pk: i64, column: &str, pk: i64) -> Result<Option<i64>> {
    Ok(conn
        .prepare_cached(&format!(
            "SELECT seq FROM session_item WHERE {column} = ?1 AND session_pk = ?2"
        ))?
        .query_row(params![pk, session_pk], |row| row.get(0))
        .optional()?)
}

/// The last timed message entry the session's changes cannot reach, which
/// the rebuild starts from.
fn start(conn: &Connection, session_pk: i64, change: &SessionChange) -> Result<Start> {
    let end: i64 = conn
        .prepare_cached("SELECT coalesce(max(seq) + 1, 0) FROM session_item WHERE session_pk = ?1")?
        .query_row([session_pk], |row| row.get(0))?;
    // The latest entry the rebuild may start from.
    let mut limit: Option<i64> = None;
    let keep = |limit: &mut Option<i64>, seq: i64| {
        *limit = Some(limit.map_or(seq, |old: i64| old.min(seq)));
    };
    // A change at `seq` (a unit there changes, or a unit goes before it).
    let lower = |limit: &mut Option<i64>, seq: i64| keep(limit, seq - 1);
    let mut shown = conn.prepare_cached(
        "SELECT order_input_id, seq, record_position, id FROM message
         WHERE pk = ?1 AND session_pk = ?2 AND duplicate_of IS NULL",
    )?;
    let mut later = conn.prepare_cached(
        "SELECT pk FROM message WHERE session_pk = ?1 AND duplicate_of IS NULL
           AND (order_input_id, seq, record_position, id) > (?2, ?3, ?4, ?5)
         ORDER BY order_input_id, seq, record_position, id",
    )?;
    for &pk in &change.messages {
        if let Some(seq) = header(conn, session_pk, "message_pk", pk)? {
            lower(&mut limit, seq);
        }
        let order: Option<(String, i64, i64, String)> = shown
            .query_row(params![pk, session_pk], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .optional()?;
        let Some(order) = order else { continue };
        // Its entries go before the first unchanged message after it.
        let mut place = end;
        let mut rows = later.query(params![session_pk, order.0, order.1, order.2, order.3])?;
        while let Some(row) = rows.next()? {
            let next: i64 = row.get(0)?;
            if change.messages.contains(&next) {
                continue;
            }
            if let Some(seq) = header(conn, session_pk, "message_pk", next)? {
                place = seq;
                break;
            }
        }
        lower(&mut limit, place);
    }
    let mut parent = conn.prepare_cached(
        "SELECT pk FROM message WHERE id = ?1 AND session_pk = ?2 AND duplicate_of IS NULL",
    )?;
    for id in &change.parents {
        if let Some(pk) = parent
            .query_row(params![id, session_pk], |row| row.get::<_, i64>(0))
            .optional()?
            && let Some(seq) = header(conn, session_pk, "message_pk", pk)?
        {
            lower(&mut limit, seq);
        }
    }
    let mut event = conn.prepare_cached(
        "SELECT e.after_message_pk, e.at,
                EXISTS(SELECT 1 FROM message m WHERE m.pk = e.after_message_pk
                       AND m.session_pk = e.session_pk AND m.duplicate_of IS NULL)
         FROM event e WHERE e.pk = ?1 AND e.session_pk = ?2",
    )?;
    for &pk in &change.events {
        if let Some(seq) = header(conn, session_pk, "event_pk", pk)? {
            lower(&mut limit, seq);
        }
        let now: Option<(Option<i64>, Option<i64>, bool)> = event
            .query_row(params![pk, session_pk], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .optional()?;
        match now {
            None => {}
            // After the message it follows, which stays as it was unless it
            // changed itself.
            Some((Some(message), _, true)) => {
                if let Some(seq) = header(conn, session_pk, "message_pk", message)? {
                    keep(&mut limit, seq);
                }
            }
            Some((None, Some(at), _)) => keep(&mut limit, timed_place(conn, session_pk, at)?),
            // At the end.
            Some(_) => lower(&mut limit, end),
        }
    }
    let Some(limit) = limit else {
        return Ok(Start::Links);
    };
    let anchor = conn
        .prepare_cached(
            "SELECT i.seq, i.message_pk, i.time_bound FROM session_item i
             JOIN message m ON m.pk = i.message_pk
             WHERE i.session_pk = ?1 AND i.seq <= ?2 AND i.message_pk IS NOT NULL
               AND m.at IS NOT NULL
             ORDER BY i.seq DESC LIMIT 1",
        )?
        .query_row(params![session_pk, limit], |row| {
            Ok(Anchor {
                seq: row.get(0)?,
                pk: row.get(1)?,
                time_bound: row.get(2)?,
            })
        })
        .optional()?;
    Ok(anchor.map_or(Start::Whole, Start::From))
}

/// The entry of the last timed message an event at `at` that names no
/// message cannot precede: the last one whose bound is at most `at`.
fn timed_place(conn: &Connection, session_pk: i64, at: i64) -> Result<i64> {
    Ok(conn
        .prepare_cached(
            "SELECT i.seq FROM session_item i JOIN message m ON m.pk = i.message_pk
             WHERE i.session_pk = ?1 AND i.message_pk IS NOT NULL AND m.at IS NOT NULL
               AND (i.time_bound IS NULL OR i.time_bound <= ?2)
             ORDER BY i.seq DESC LIMIT 1",
        )?
        .query_row(params![session_pk, at], |row| row.get(0))
        .optional()?
        .unwrap_or(0))
}

/// Compares the session's composing links with those indexed; a change
/// stales outstanding cursors without changing any entry.
fn relink(conn: &Connection, session_pk: i64) -> Result<()> {
    let links = links_identity(conn, session_pk)?;
    conn.prepare_cached(
        "UPDATE session_item_state
         SET rewrite_version = max(rewrite_version + 1, (SELECT changed FROM session WHERE pk = ?1)),
             links = ?2
         WHERE session_pk = ?1 AND links IS NOT ?2",
    )?
    .execute(params![session_pk, links])?;
    Ok(())
}

/// Rebuilds the session's entries from `anchor` (from its first unit without
/// one), keeping those that did not change; returns the units read.
fn rebuild(
    conn: &Connection,
    session_pk: i64,
    anchor: Option<Anchor>,
    changed_events: &BTreeSet<i64>,
) -> Result<u64> {
    let changed: i64 = conn
        .prepare_cached("SELECT changed FROM session WHERE pk = ?1")?
        .query_row([session_pk], |row| row.get(0))?;
    let first_seq = anchor.map_or(0, |anchor| anchor.seq);
    let units = units_from(conn, session_pk, anchor, changed_events)?;
    let identities: Vec<Vec<u8>> = units.iter().map(Unit::identity).collect();
    let indexed: Vec<(i64, Vec<u8>)> = conn
        .prepare_cached(
            "SELECT seq, unit FROM session_item
             WHERE session_pk = ?1 AND seq >= ?2 AND unit IS NOT NULL ORDER BY seq",
        )?
        .query_map(params![session_pk, first_seq], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let state: Option<(i64, i64, Vec<u8>)> = conn
        .prepare_cached(
            "SELECT append_seq, rewrite_version, links FROM session_item_state WHERE session_pk = ?1",
        )?
        .query_row([session_pk], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .optional()?;
    let links = links_identity(conn, session_pk)?;
    let mut same = indexed
        .iter()
        .zip(&identities)
        .take_while(|((_, stored), current)| stored == *current)
        .count();
    let links_moved = state
        .as_ref()
        .is_some_and(|(_, _, stored)| stored.as_slice() != links.as_slice());
    let mut insert = conn.prepare_cached(
        "INSERT INTO session_item(session_pk, seq, unit, message_pk, event_pk, time_bound, entry)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    let mut next: i64 = conn
        .prepare_cached("SELECT coalesce(max(seq) + 1, 0) FROM session_item WHERE session_pk = ?1")?
        .query_row([session_pk], |row| row.get(0))?;
    let mut grown = false;
    let mut header_moved = false;
    // A live session's last message grows block by block. When the only
    // difference is new blocks after the last indexed message's entries,
    // that is an append; outstanding cursors stay valid.
    if same + 1 == indexed.len() && same < units.len() {
        let (first_seq, _) = indexed[same];
        let stored = stored_entries(conn, session_pk, first_seq)?;
        let current = unit_entries(conn, &units[same])?;
        if extends(&stored, &current) {
            let (_, _, time_bound) = units[same].columns();
            conn.prepare_cached(
                "UPDATE session_item SET unit = ?3, time_bound = ?4, entry = ?5
                 WHERE session_pk = ?1 AND seq = ?2",
            )?
            .execute(params![
                session_pk,
                first_seq,
                identities[same],
                time_bound,
                serde_json::to_vec(&current[0])?
            ])?;
            for entry in &current[stored.len()..] {
                insert.execute(params![
                    session_pk,
                    next,
                    None::<Vec<u8>>,
                    None::<i64>,
                    None::<i64>,
                    None::<i64>,
                    serde_json::to_vec(entry)?
                ])?;
                next += 1;
            }
            grown = current.len() > stored.len();
            header_moved = current[0] != stored[0];
            same += 1;
        }
    }
    let rewritten = same < indexed.len() || links_moved;
    let appended = grown || header_moved || same < units.len();
    // Keep entries before the first changed unit.
    if let Some((seq, _)) = indexed.get(same) {
        conn.prepare_cached("DELETE FROM session_item WHERE session_pk = ?1 AND seq >= ?2")?
            .execute(params![session_pk, seq])?;
        next = *seq;
    }
    for (unit, identity) in units.iter().zip(&identities).skip(same) {
        let (message_pk, event_pk, time_bound) = unit.columns();
        for (offset, entry) in unit_entries(conn, unit)?.into_iter().enumerate() {
            let first = offset == 0;
            insert.execute(params![
                session_pk,
                next,
                first.then_some(identity.as_slice()),
                message_pk.filter(|_| first),
                event_pk.filter(|_| first),
                time_bound.filter(|_| first),
                serde_json::to_vec(&entry)?
            ])?;
            next += 1;
        }
    }
    // A rewrite version is never below the write counter of the change that
    // set it, so a session removed and stored again never repeats a version
    // an earlier cursor saw.
    let (append_seq, rewrite_version) = match state {
        None => (i64::from(appended), changed.max(1)),
        Some((append_seq, rewrite_version, _)) => (
            append_seq + i64::from(appended && !rewritten),
            if rewritten {
                (rewrite_version + 1).max(changed)
            } else {
                rewrite_version
            },
        ),
    };
    conn.prepare_cached(
        "INSERT INTO session_item_state(session_pk, append_seq, rewrite_version, links, indexed_changed)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(session_pk) DO UPDATE SET append_seq = excluded.append_seq,
           rewrite_version = excluded.rewrite_version, links = excluded.links,
           indexed_changed = excluded.indexed_changed",
    )?
    .execute(params![session_pk, append_seq, rewrite_version, links, changed])?;
    Ok(units.len() as u64)
}

fn unit_entries(conn: &Connection, unit: &Unit) -> Result<Vec<IndexEntry>> {
    match unit {
        Unit::Message { pk, place, .. } => {
            let (message, ordinals) = message_body(conn, *pk)?;
            Ok(message_entries(&message, &ordinals, *place))
        }
        Unit::Event {
            pk,
            id,
            kind,
            at,
            data_bytes,
        } => {
            // Only a microcompaction's data is read: it names the results it
            // removed from the context.
            let cleared = if kind == "microcompaction" {
                let data: Option<Vec<u8>> = conn
                    .prepare_cached("SELECT data FROM event WHERE pk = ?1")?
                    .query_row([pk], |row| row.get(0))
                    .optional()?;
                match data {
                    Some(packed) => microcompacted_calls(&crate::encoding::unpack_json::<
                        serde_json::Value,
                    >(&packed)?),
                    None => Vec::new(),
                }
            } else {
                Vec::new()
            };
            Ok(vec![event_entry(id, kind, *at, *data_bytes, cleared)])
        }
    }
}

/// A message's row and its blocks with their original ordinals: the fields
/// the index estimates from. Structured tool data is not read.
fn message_body(conn: &Connection, pk: i64) -> Result<(Message, Vec<u64>)> {
    let (id, session, role, prompt_kind, model, at, tokens) = conn
        .prepare_cached(
            "SELECT s.id, m.role, m.prompt_kind, m.model, m.at,
                    m.input_tokens, m.cache_read_tokens, m.cache_write_tokens, m.id
             FROM message m JOIN session s ON s.pk = m.session_pk WHERE m.pk = ?1",
        )?
        .query_row([pk], |row| {
            Ok((
                row.id::<MessageId>(8)?,
                row.id::<SessionId>(0)?,
                row.name::<Role>(1)?,
                row.optional_name::<PromptKind>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.optional_time_ms(4)?,
                [
                    row.get::<_, Option<u64>>(5)?,
                    row.get::<_, Option<u64>>(6)?,
                    row.get::<_, Option<u64>>(7)?,
                ],
            ))
        })?;
    let mut blocks = Vec::new();
    let mut ordinals = Vec::new();
    let mut statement = conn.prepare_cached(
        "SELECT b.body, t.body, coalesce(CAST(b.ordinal AS INTEGER), b.idx)
         FROM block b LEFT JOIN text t ON t.id = b.text_id
         WHERE b.message_pk = ?1 ORDER BY b.idx",
    )?;
    let mut rows = statement.query([pk])?;
    while let Some(row) = rows.next()? {
        let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
        if let Some(text) = row.get::<_, Option<Vec<u8>>>(1)? {
            restore_text(&mut block, unpack_text(&text)?);
        }
        blocks.push(block);
        ordinals.push(row.get::<_, u64>(2)?);
    }
    let message = Message {
        id,
        session,
        parent: None,
        role,
        at,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model,
        prompt_kind,
        blocks,
        usage: Usage {
            input: tokens[0],
            cached_input: tokens[1],
            cache_write: tokens[2],
            ..Usage::default()
        },
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    };
    Ok((message, ordinals))
}

/// The links that compose a session as a child: a change to them changes
/// what a composed read of the session shows.
fn links_identity(conn: &Connection, session_pk: i64) -> Result<Vec<u8>> {
    let mut digest = blake3::Hasher::new();
    let mut statement = conn.prepare_cached(
        "SELECT l.parent_id, l.kind, l.how, coalesce(l.branch_point, ''), coalesce(l.spawning_call, '')
         FROM link l JOIN session s ON s.id = l.child_id
         WHERE s.pk = ?1 ORDER BY l.pk",
    )?;
    let mut rows = statement.query([session_pk])?;
    while let Some(row) = rows.next()? {
        for column in 0..5 {
            digest.update(row.get::<_, String>(column)?.as_bytes());
            digest.update(b"\0");
        }
    }
    Ok(digest.finalize().as_bytes().to_vec())
}

/// The stored entries of a session from `from` on.
fn stored_entries(conn: &Connection, session_pk: i64, from: i64) -> Result<Vec<IndexEntry>> {
    let mut statement = conn.prepare_cached(
        "SELECT entry FROM session_item WHERE session_pk = ?1 AND seq >= ?2 ORDER BY seq",
    )?;
    let mut rows = statement.query(params![session_pk, from])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(decode(&row.get::<_, Vec<u8>>(0)?)?);
    }
    Ok(out)
}

/// Whether `current` is `stored` with entries added at the end. The header's
/// reported context size is a figure shown on it, not a position, so a
/// usage record arriving later does not count as a rewrite.
fn extends(stored: &[IndexEntry], current: &[IndexEntry]) -> bool {
    if stored.is_empty() || current.len() < stored.len() {
        return false;
    }
    let mut header = current[0].clone();
    header.context_tokens = stored[0].context_tokens;
    header == stored[0] && current[1..stored.len()] == stored[1..]
}

fn decode(bytes: &[u8]) -> Result<IndexEntry> {
    Ok(serde_json::from_slice(bytes)?)
}

fn counter(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| StoreError::Invalid("negative item index counter".into()))
}

/// A link that puts a parent's messages before a session's own: a fork or
/// a resume, with the parent message the child continues from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposingLink {
    pub parent: String,
    pub how: String,
    pub branch_point: Option<String>,
}

/// The rows whose source record a raw reading shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordTable {
    Message,
    Event,
}
impl RecordTable {
    fn name(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::Event => "event",
        }
    }
}

/// The messages of one context reading, by reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextMessages {
    pub in_context: Vec<MessageId>,
    pub evicted: Vec<MessageId>,
    pub complete: bool,
}

/// One stored link between two sessions, as the reading needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionLink {
    pub parent: String,
    pub child: String,
    /// `subagent`, `continuation` or `reference`.
    pub kind: String,
    pub how: String,
    pub branch_point: Option<String>,
    pub spawning_call: Option<String>,
}

/// Which way a lineage walk goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lineage {
    Ancestors,
    Descendants,
}

/// A record's stored bytes, as the source wrote them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredRecord {
    pub input: String,
    pub bytes: Vec<u8>,
}

impl Reader<'_> {
    fn links_where(&self, column: &str, session: &str) -> Result<Vec<SessionLink>> {
        Ok(self
            .conn()?
            .prepare_cached(&format!(
                "SELECT parent_id, child_id, kind, how, branch_point, spawning_call FROM link
                 WHERE {column} = ?1 ORDER BY pk"
            ))?
            .query_map([session], |row| {
                Ok(SessionLink {
                    parent: row.get(0)?,
                    child: row.get(1)?,
                    kind: row.get(2)?,
                    how: row.get(3)?,
                    branch_point: row.get(4)?,
                    spawning_call: row.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// A session's own message count, as derived, when it is stored and
    /// counted.
    pub fn own_message_count(&self, id: &SessionId) -> Result<Option<u64>> {
        Ok(self
            .conn()?
            .prepare_cached("SELECT message_count FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |row| row.get::<_, Option<i64>>(0))
            .optional()?
            .flatten()
            .and_then(|count| u64::try_from(count).ok()))
    }

    /// Links from `id` to its children, in the order they were stored.
    pub fn child_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        self.links_where("parent_id", &id.to_string())
    }

    /// Links from `id`'s parents to it, in the order they were stored.
    pub fn parent_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        self.links_where("child_id", &id.to_string())
    }

    /// Every session reached from `id` through links other than
    /// `reference`, transitively, each once; a cycle ends where it closes.
    pub fn lineage(&self, id: &SessionId, direction: Lineage) -> Result<Vec<String>> {
        let mut seen = std::collections::HashSet::from([id.to_string()]);
        let mut queue = std::collections::VecDeque::from([id.to_string()]);
        let mut out = Vec::new();
        while let Some(current) = queue.pop_front() {
            self.check()?;
            let links = match direction {
                Lineage::Ancestors => self.links_where("child_id", &current)?,
                Lineage::Descendants => self.links_where("parent_id", &current)?,
            };
            for link in links.into_iter().filter(|link| link.kind != "reference") {
                let next = match direction {
                    Lineage::Ancestors => link.parent,
                    Lineage::Descendants => link.child,
                };
                if seen.insert(next.clone()) {
                    out.push(next.clone());
                    queue.push_back(next);
                }
            }
        }
        Ok(out)
    }

    /// The stored bytes of one message's or event's source record. When a
    /// parser recorded no end, the record runs to the next record of the
    /// same input chain, or to how far that chain was parsed.
    pub fn stored_record(&self, table: RecordTable, id: &str) -> Result<Option<StoredRecord>> {
        let table = table.name();
        let conn = self.conn()?;
        // input pk, input ID, then the record's chain, start and end.
        type Located = (i64, String, Option<i64>, Option<i64>, Option<i64>);
        let found: Option<Located> = conn
            .query_row(
                &format!(
                    "SELECT x.input_pk, i.id, x.source_chain, x.source_start, x.source_end
                     FROM {table} x JOIN input_ref i ON i.pk = x.input_pk WHERE x.id = ?1"
                ),
                [id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((input_pk, input, Some(chain), Some(start), end)) = found else {
            return Ok(None);
        };
        let end = match end.filter(|end| *end > start) {
            Some(end) => end,
            None => conn.query_row(
                "SELECT coalesce(
                    (SELECT min(s) FROM (
                        SELECT source_start AS s FROM message_source WHERE input_pk = ?1 AND chain = ?2 AND source_start > ?3
                        UNION ALL
                        SELECT source_start FROM event WHERE input_pk = ?1 AND source_chain = ?2 AND source_start > ?3)),
                    (SELECT parsed_upto FROM parse_state WHERE input_pk = ?1 AND chain = ?2),
                    ?3)",
                params![input_pk, chain, start],
                |row| row.get(0),
            )?,
        };
        let (Ok(start), Ok(end)) = (u64::try_from(start), u64::try_from(end)) else {
            return Err(StoreError::Invalid(format!(
                "{id} has a negative source range"
            )));
        };
        let bytes = self.stored_range(&input, chain, start, end, &self.blob_reader()?)?;
        Ok(Some(StoredRecord { input, bytes }))
    }

    /// The messages in the model's context at `message` and the ones the
    /// newest compaction before it removed, read from the owning session's
    /// item index. This is what the `in_context_at:` and `evicted_at:` query
    /// terms select.
    pub fn context_messages(&self, message: &MessageId) -> Result<Option<ContextMessages>> {
        let Some(session) = self.message_session(message)? else {
            return Ok(None);
        };
        let Some(items) = self.item_index(&session)? else {
            return Ok(None);
        };
        let header = polylogue_model::read::EntryKey::header(message);
        let Some(at) = items.entries.iter().position(|entry| entry.key == header) else {
            return Ok(None);
        };
        let parts = [polylogue_model::read::Part {
            session,
            start: 0,
            end: items.entries.len(),
            inherited: false,
            joined_by: None,
        }];
        let context = polylogue_model::read::context_at(
            &parts,
            |_| &items.entries,
            polylogue_model::read::Place { part: 0, entry: at },
        );
        let messages = |places: &[polylogue_model::read::Place]| -> Vec<MessageId> {
            let mut out: Vec<MessageId> = Vec::new();
            for place in places {
                if let Some(id) = items.entries[place.entry].key.message()
                    && out.last() != Some(id)
                {
                    out.push(id.clone());
                }
            }
            out
        };
        Ok(Some(ContextMessages {
            in_context: messages(&context.members),
            evicted: messages(&context.evicted),
            complete: context.complete,
        }))
    }

    /// The links that compose `id` from a parent, in the order they were stored.
    pub fn composing_links(&self, id: &SessionId) -> Result<Vec<ComposingLink>> {
        Ok(self
            .conn()?
            .prepare_cached(
                "SELECT parent_id, how, branch_point FROM link
                 WHERE child_id = ?1 AND kind = 'continuation' AND how IN ('fork', 'resume')
                 ORDER BY pk",
            )?
            .query_map([id.to_string()], |row| {
                Ok(ComposingLink {
                    parent: row.get(0)?,
                    how: row.get(1)?,
                    branch_point: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// A session's whole item index, or `None` when the session is not
    /// stored. One range scan.
    pub fn item_index(&self, id: &SessionId) -> Result<Option<StoredItems>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(None);
        };
        let Some(versions) = self.item_versions_of(pk)? else {
            return Err(StoreError::Invalid(format!(
                "session {id} has no item index"
            )));
        };
        Ok(Some(StoredItems {
            session_pk: pk,
            append_seq: versions.append_seq,
            rewrite_version: versions.rewrite_version,
            entries: self.item_entries_from(pk, 0)?,
        }))
    }

    /// The entries from `seq` on: what an append added to a cached index.
    pub fn item_entries_from(&self, session_pk: i64, seq: u64) -> Result<Vec<IndexEntry>> {
        let seq = i64::try_from(seq)
            .map_err(|_| StoreError::Invalid("item index position out of range".into()))?;
        stored_entries(self.conn()?, session_pk, seq)
    }

    pub fn item_versions_of(&self, session_pk: i64) -> Result<Option<ItemVersions>> {
        self.conn()?
            .prepare_cached(
                "SELECT append_seq, rewrite_version FROM session_item_state WHERE session_pk = ?1",
            )?
            .query_row([session_pk], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?
            .map(|(append_seq, rewrite_version)| {
                Ok(ItemVersions {
                    append_seq: counter(append_seq)?,
                    rewrite_version: counter(rewrite_version)?,
                })
            })
            .transpose()
    }
}
