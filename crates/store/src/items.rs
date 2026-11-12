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
    read::{
        EntryClass, IndexEntry, ItemCounts, MessagePlace, entry_classes, event_entry,
        message_entries,
    },
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

/// What a session reading needs of a session before reading its entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemState {
    pub session_pk: i64,
    pub versions: ItemVersions,
    /// How many entries the session has.
    pub entries: u64,
    /// The counts over every entry.
    pub counts: ItemCounts,
}

/// One stored entry, with its place and class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEntry {
    pub seq: u64,
    pub class: EntryClass,
    pub entry: IndexEntry,
}

/// The kinds of entry a reading finds through their own index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemLookup {
    Prompt,
    Compaction,
    Microcompaction,
}

/// Entries decoded by session item reads in this process: what a test
/// counts to show a reading reads only near its page.
#[cfg(any(test, feature = "fixture"))]
static ENTRIES_DECODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(any(test, feature = "fixture"))]
pub fn entries_decoded() -> u64 {
    ENTRIES_DECODED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The counters alone, read in O(1).
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
            Self::Event {
                pk,
                id,
                kind,
                at,
                data_bytes,
            } => {
                digest.update(b"event\0");
                digest.update(&pk.to_le_bytes());
                for value in [id, kind] {
                    digest.update(&(value.len() as u64).to_le_bytes());
                    digest.update(value.as_bytes());
                }
                digest.update(&[u8::from(at.is_some())]);
                digest.update(&at.unwrap_or(0).to_le_bytes());
                digest.update(&data_bytes.to_le_bytes());
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

/// Whether an event's stored item already describes its current metadata.
/// Older item fingerprints are repaired by the ordinary session rewrite.
pub(crate) fn event_is_current(
    conn: &Connection,
    pk: i64,
    session: i64,
    id: &str,
    kind: &str,
    at: Option<i64>,
    data_bytes: u64,
) -> Result<bool> {
    let expected = Unit::Event {
        pk,
        id: id.into(),
        kind: kind.into(),
        at,
        data_bytes,
    }
    .identity();
    let stored: Option<Vec<u8>> = conn
        .prepare_cached("SELECT unit FROM session_item WHERE event_pk=?1 AND session_pk=?2")?
        .query_row(params![pk, session], |row| row.get(0))
        .optional()?;
    Ok(stored.as_ref() == Some(&expected))
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
                 ORDER BY m.session_order, m.id"
            ))?
            .query_map([session_pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
        Some(anchor) => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message a, message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
                 WHERE a.pk = ?2 AND m.session_pk = ?1 AND m.duplicate_of IS NULL
                   AND (m.session_order, m.id)
                       >= (a.session_order, a.id)
                 ORDER BY m.session_order, m.id"
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

/// What the writer stores beside each entry: its class, call ID and, on a
/// unit's first entry, the counts before it; then the entry count and the
/// counts over every entry.
#[cfg(test)]
pub(crate) type StoredColumns = (
    Vec<(EntryClass, Option<String>, Option<ItemCounts>)>,
    u64,
    ItemCounts,
);

#[cfg(test)]
pub(crate) fn stored_columns(conn: &Connection, session_pk: i64) -> Result<StoredColumns> {
    let mut statement = conn.prepare(
        "SELECT class, call_id, counts FROM session_item WHERE session_pk = ?1 ORDER BY seq",
    )?;
    let mut rows = statement.query([session_pk])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push((
            class_of(row.get(0)?)?,
            row.get(1)?,
            row.get::<_, Option<Vec<u8>>>(2)?
                .map(|counts| decode_counts(&counts))
                .transpose()?,
        ));
    }
    let (entries, counts): (i64, Vec<u8>) = conn.query_row(
        "SELECT entries, counts FROM session_item_state WHERE session_pk = ?1",
        [session_pk],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((out, counter(entries)?, decode_counts(&counts)?))
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
    let counter = crate::session_values::write_counter(conn)?;
    for pk in sessions {
        crate::session_values::rebuild(conn, pk, counter)?;
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
        "SELECT session_order, id FROM message
         WHERE pk = ?1 AND session_pk = ?2 AND duplicate_of IS NULL",
    )?;
    let mut later = conn.prepare_cached(
        "SELECT pk FROM message WHERE session_pk = ?1 AND duplicate_of IS NULL
           AND (session_order, id) > (?2, ?3)
         ORDER BY session_order, id",
    )?;
    let mut first_changed: Option<(String, String)> = None;
    for &pk in &change.messages {
        if let Some(seq) = header(conn, session_pk, "message_pk", pk)? {
            lower(&mut limit, seq);
        }
        let order: Option<(String, String)> = shown
            .query_row(params![pk, session_pk], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .optional()?;
        if let Some(order) = order
            && first_changed.as_ref().is_none_or(|first| &order < first)
        {
            first_changed = Some(order);
        }
    }
    // Message headers follow this same order. The first unchanged successor
    // after the earliest changed message is also the earliest insertion
    // boundary; scanning from every changed message repeats a batch's tail.
    if let Some(order) = first_changed {
        let mut place = end;
        let mut rows = later.query(params![session_pk, order.0, order.1])?;
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
    let state: Option<(i64, i64, Vec<u8>, Vec<u8>)> = conn
        .prepare_cached(
            "SELECT append_seq, rewrite_version, links, counts FROM session_item_state
             WHERE session_pk = ?1",
        )?
        .query_row([session_pk], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .optional()?;
    let links = links_identity(conn, session_pk)?;
    let mut same = indexed
        .iter()
        .zip(&identities)
        .take_while(|((_, stored), current)| stored == *current)
        .count();
    let links_moved = state
        .as_ref()
        .is_some_and(|(_, _, stored, _)| stored.as_slice() != links.as_slice());
    let mut insert = conn.prepare_cached(
        "INSERT INTO session_item(session_pk, seq, unit, message_pk, event_pk, time_bound, entry,
                                  class, call_id, counts)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;
    // The counts before the first unit written below: those stored on the
    // first unit replaced, or after every entry when units are only added.
    let mut running = match indexed.get(same) {
        Some((seq, _)) => unit_counts(conn, session_pk, *seq)?,
        None => match &state {
            Some((_, _, _, counts)) => decode_counts(counts)?,
            None => ItemCounts::default(),
        },
    };
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
            let classes = entry_classes(&current);
            // A new block can change what its header's class says.
            conn.prepare_cached(
                "UPDATE session_item SET unit = ?3, time_bound = ?4, entry = ?5, class = ?6
                 WHERE session_pk = ?1 AND seq = ?2",
            )?
            .execute(params![
                session_pk,
                first_seq,
                identities[same],
                time_bound,
                serde_json::to_vec(&current[0])?,
                classes[0].bits()
            ])?;
            for (entry, class) in current.iter().zip(&classes).skip(stored.len()) {
                insert.execute(params![
                    session_pk,
                    next,
                    None::<Vec<u8>>,
                    None::<i64>,
                    None::<i64>,
                    None::<i64>,
                    serde_json::to_vec(entry)?,
                    class.bits(),
                    entry.call_id,
                    None::<Vec<u8>>
                ])?;
                next += 1;
            }
            running = unit_counts(conn, session_pk, first_seq)?;
            for (entry, class) in current.iter().zip(&classes) {
                running.add(entry, *class);
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
        let entries = unit_entries(conn, unit)?;
        let before = encode_counts(&running);
        for (offset, (entry, class)) in entries.iter().zip(entry_classes(&entries)).enumerate() {
            let first = offset == 0;
            insert.execute(params![
                session_pk,
                next,
                first.then_some(identity.as_slice()),
                message_pk.filter(|_| first),
                event_pk.filter(|_| first),
                time_bound.filter(|_| first),
                serde_json::to_vec(entry)?,
                class.bits(),
                entry.call_id,
                first.then_some(before.as_slice())
            ])?;
            running.add(entry, class);
            next += 1;
        }
    }
    // A rewrite version is never below the write counter of the change that
    // set it, so a session removed and stored again never repeats a version
    // an earlier cursor saw.
    let (append_seq, rewrite_version) = match state {
        None => (i64::from(appended), changed.max(1)),
        Some((append_seq, rewrite_version, _, _)) => (
            append_seq + i64::from(appended && !rewritten),
            if rewritten {
                (rewrite_version + 1).max(changed)
            } else {
                rewrite_version
            },
        ),
    };
    conn.prepare_cached(
        "INSERT INTO session_item_state(session_pk, append_seq, rewrite_version, links, entries,
                                        counts, indexed_changed)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(session_pk) DO UPDATE SET append_seq = excluded.append_seq,
           rewrite_version = excluded.rewrite_version, links = excluded.links,
           entries = excluded.entries, counts = excluded.counts,
           indexed_changed = excluded.indexed_changed",
    )?
    .execute(params![
        session_pk,
        append_seq,
        rewrite_version,
        links,
        next,
        encode_counts(&running),
        changed
    ])?;
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
            // Context boundaries carry normalized retained/removed references.
            let data: Option<Vec<u8>> = if matches!(kind.as_str(), "compaction" | "microcompaction")
            {
                conn.prepare_cached("SELECT data FROM event WHERE pk=?1")?
                    .query_row([pk], |row| row.get(0))
                    .optional()?
            } else {
                None
            };
            let cleared = if kind == "microcompaction" {
                data.as_deref()
                    .map(|packed| {
                        crate::encoding::unpack_json::<polylogue_model::MicrocompactionBoundary>(
                            packed,
                        )
                        .map(|boundary| boundary.removed_calls)
                    })
                    .transpose()?
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let mut entry = event_entry(id, kind, *at, *data_bytes, cleared);
            if kind == "compaction" {
                entry.compaction = data
                    .as_deref()
                    .map(crate::encoding::unpack_json::<polylogue_model::CompactionBoundary>)
                    .transpose()?;
            }
            Ok(vec![entry])
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
        "SELECT l.parent_id, l.kind, l.how, coalesce(l.branch_point, ''), coalesce(l.spawning_call, ''), CAST(l.copies_prefix AS TEXT)
         FROM session_link l JOIN session s ON s.id = l.child_id
         WHERE s.pk = ?1 ORDER BY l.parent_id,l.kind,l.how,l.branch_point,l.spawning_call",
    )?;
    let mut rows = statement.query([session_pk])?;
    while let Some(row) = rows.next()? {
        for column in 0..6 {
            let value = row.get::<_, Option<String>>(column)?;
            digest.update(&[u8::from(value.is_some())]);
            if let Some(value) = value {
                digest.update(value.as_bytes());
            }
            digest.update(b"\0");
        }
    }
    let mut copies = conn.prepare_cached("SELECT DISTINCT p.parent_id FROM prefix_copy p JOIN session s ON s.id=p.child_id WHERE s.pk=?1 ORDER BY p.parent_id")?;
    let parents = copies.query_map([session_pk], |row| row.get::<_, String>(0))?;
    for parent in parents {
        digest.update(b"copied-prefix\0");
        digest.update(parent?.as_bytes());
        digest.update(b"\0");
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

/// The counts stored on the unit whose first entry is at `seq`.
fn unit_counts(conn: &Connection, session_pk: i64, seq: i64) -> Result<ItemCounts> {
    let counts: Option<Vec<u8>> = conn
        .prepare_cached("SELECT counts FROM session_item WHERE session_pk = ?1 AND seq = ?2")?
        .query_row(params![session_pk, seq], |row| row.get(0))
        .optional()?
        .flatten();
    match counts {
        Some(counts) => decode_counts(&counts),
        None => Err(StoreError::Invalid(format!(
            "session item {seq} is not the first entry of a unit"
        ))),
    }
}

/// Counts as stored: unsigned LEB128 numbers, the running full-form tokens
/// first, then each class with a count above zero and its count, by class.
fn encode_counts(counts: &ItemCounts) -> Vec<u8> {
    fn number(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }
    let mut out = Vec::with_capacity(4 + counts.classes.len() * 4);
    number(&mut out, counts.tokens_full);
    for (class, count) in &counts.classes {
        number(&mut out, u64::from(class.bits()));
        number(&mut out, *count);
    }
    out
}

fn decode_counts(bytes: &[u8]) -> Result<ItemCounts> {
    let invalid = || StoreError::Invalid("malformed session item counts".into());
    let number = |at: &mut usize| -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *bytes.get(*at).ok_or_else(invalid)?;
            *at += 1;
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(invalid())
    };
    let mut at = 0;
    let mut counts = ItemCounts {
        tokens_full: number(&mut at)?,
        ..ItemCounts::default()
    };
    while at < bytes.len() {
        let class = u16::try_from(number(&mut at)?)
            .ok()
            .and_then(EntryClass::from_bits)
            .ok_or_else(invalid)?;
        counts.classes.insert(class, number(&mut at)?);
    }
    Ok(counts)
}

/// The partial index and condition that find one kind of entry.
fn lookup_filter(kind: ItemLookup) -> &'static str {
    match kind {
        ItemLookup::Prompt => "INDEXED BY session_item_prompt WHERE class & 63 = 32",
        ItemLookup::Compaction => "INDEXED BY session_item_compaction WHERE class & 15 = 9",
        ItemLookup::Microcompaction => {
            "INDEXED BY session_item_microcompaction WHERE class & 15 = 10"
        }
    }
}

fn class_of(bits: i64) -> Result<EntryClass> {
    u16::try_from(bits)
        .ok()
        .and_then(EntryClass::from_bits)
        .ok_or_else(|| StoreError::Invalid(format!("unknown session item class {bits}")))
}

fn position(seq: u64) -> Result<i64> {
    i64::try_from(seq).map_err(|_| StoreError::Invalid("item index position out of range".into()))
}

fn counter(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| StoreError::Invalid("negative item index counter".into()))
}

/// Recorded parent history: an anchored branch or messages copied into the
/// child's input. Physical copying does not create a provider relationship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InheritedPrefix {
    pub parent: String,
    pub copies_prefix: bool,
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

/// A record's stored bytes, as the source wrote them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredRecord {
    pub input: String,
    pub bytes: Vec<u8>,
}

impl Reader<'_> {
    /// Source-declared copied prefixes of this physical session, independent
    /// of whether a provider relationship is recorded.
    pub fn copied_prefixes(&self, id: &SessionId) -> Result<Vec<SessionId>> {
        Ok(self
            .conn()?
            .prepare_cached(
                "SELECT parent_id FROM prefix_copy WHERE child_id=?1 UNION SELECT parent_id FROM session_prefix WHERE child_id=?1 AND copies_prefix=1 AND branch_point IS NULL ORDER BY parent_id",
            )?
            .query_map([id.to_string()], |row| row.id::<SessionId>(0))?
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

    /// The recorded prefixes that a physical reading of `id` inherits.
    pub fn inherited_prefixes(&self, id: &SessionId) -> Result<Vec<InheritedPrefix>> {
        Ok(self
            .conn()?
            .prepare_cached(
                "SELECT parent_id,branch_point,copies_prefix FROM session_prefix
                 WHERE child_id=?1 ORDER BY parent_id,branch_point",
            )?
            .query_map([id.to_string()], |row| {
                Ok(InheritedPrefix {
                    parent: row.get(0)?,
                    copies_prefix: row.get(2)?,
                    branch_point: row.get(1)?,
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
            entries: stored_entries(self.conn()?, pk, 0)?,
        }))
    }

    /// What a session reading needs of a session before reading any of its
    /// entries, or `None` when the session is not stored.
    pub fn item_state(&self, id: &SessionId) -> Result<Option<ItemState>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(None);
        };
        let state: Option<(i64, i64, i64, Vec<u8>)> = self
            .conn()?
            .prepare_cached(
                "SELECT append_seq, rewrite_version, entries, counts FROM session_item_state
                 WHERE session_pk = ?1",
            )?
            .query_row([pk], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .optional()?;
        let Some((append_seq, rewrite_version, entries, counts)) = state else {
            return Err(StoreError::Invalid(format!(
                "session {id} has no item index"
            )));
        };
        Ok(Some(ItemState {
            session_pk: pk,
            versions: ItemVersions {
                append_seq: counter(append_seq)?,
                rewrite_version: counter(rewrite_version)?,
            },
            entries: counter(entries)?,
            counts: decode_counts(&counts)?,
        }))
    }

    /// Entries in `from..until`, forward from `from` or backward from
    /// `until`, at most `limit` of those whose class `wanted` accepts. Rows
    /// it does not accept are passed over without being decoded.
    pub fn item_window(
        &self,
        session_pk: i64,
        range: std::ops::Range<u64>,
        forward: bool,
        limit: usize,
        wanted: impl Fn(EntryClass) -> bool,
    ) -> Result<Vec<StoredEntry>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(if forward {
            "SELECT seq, class, entry FROM session_item
             WHERE session_pk = ?1 AND seq >= ?2 AND seq < ?3 ORDER BY seq"
        } else {
            "SELECT seq, class, entry FROM session_item
             WHERE session_pk = ?1 AND seq >= ?2 AND seq < ?3 ORDER BY seq DESC"
        })?;
        let mut rows = statement.query(params![
            session_pk,
            position(range.start)?,
            position(range.end)?
        ])?;
        let mut out = Vec::new();
        while out.len() < limit
            && let Some(row) = rows.next()?
        {
            let class = class_of(row.get(1)?)?;
            if !wanted(class) {
                continue;
            }
            #[cfg(any(test, feature = "fixture"))]
            ENTRIES_DECODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.push(StoredEntry {
                seq: counter(row.get(0)?)?,
                class,
                entry: decode(&row.get::<_, Vec<u8>>(2)?)?,
            });
        }
        Ok(out)
    }

    /// The counts of the entries before `seq`: those stored on its unit's
    /// first entry and the unit's entries before it.
    pub fn item_counts_before(&self, session_pk: i64, seq: u64) -> Result<ItemCounts> {
        let conn = self.conn()?;
        let unit: Option<(i64, Vec<u8>)> = conn
            .prepare_cached(
                "SELECT seq, counts FROM session_item
                 WHERE session_pk = ?1 AND seq <= ?2 AND counts IS NOT NULL
                 ORDER BY seq DESC LIMIT 1",
            )?
            .query_row(params![session_pk, position(seq)?], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .optional()?;
        let Some((first, counts)) = unit else {
            return Ok(ItemCounts::default());
        };
        let mut counts = decode_counts(&counts)?;
        let first = counter(first)?;
        for stored in self.item_window(session_pk, first..seq, true, usize::MAX, |_| true)? {
            counts.add(&stored.entry, stored.class);
        }
        Ok(counts)
    }

    /// Where an entry sits in a session's index.
    pub fn item_place(
        &self,
        session_pk: i64,
        key: &polylogue_model::read::EntryKey,
    ) -> Result<Option<u64>> {
        use polylogue_model::read::EntryKey;
        let conn = self.conn()?;
        match key {
            EntryKey::Event { event } => Ok(conn
                .prepare_cached(
                    "SELECT i.seq FROM event e JOIN session_item i
                       ON i.event_pk = e.pk AND i.session_pk = e.session_pk
                     WHERE e.id = ?1 AND e.session_pk = ?2",
                )?
                .query_row(params![event, session_pk], |row| row.get::<_, i64>(0))
                .optional()?
                .map(counter)
                .transpose()?),
            EntryKey::Message { message, block } => {
                let Some(header) = self.item_header(session_pk, message)? else {
                    return Ok(None);
                };
                if block.is_none() {
                    return Ok(Some(header));
                }
                let end = self.item_unit_end(session_pk, header)?;
                Ok(self
                    .item_window(session_pk, header..end, true, usize::MAX, |_| true)?
                    .into_iter()
                    .find(|stored| &stored.entry.key == key)
                    .map(|stored| stored.seq))
            }
        }
    }

    /// The header entry of a message in a session's index.
    fn item_header(&self, session_pk: i64, message: &MessageId) -> Result<Option<u64>> {
        self.conn()?
            .prepare_cached(
                "SELECT i.seq FROM message m JOIN session_item i
                   ON i.message_pk = m.pk AND i.session_pk = m.session_pk
                 WHERE m.id = ?1 AND m.session_pk = ?2",
            )?
            .query_row(params![message.to_string(), session_pk], |row| {
                row.get::<_, i64>(0)
            })
            .optional()?
            .map(counter)
            .transpose()
    }

    /// The end of the message whose header is at `header`: the next header,
    /// or the end of the session. Events after the message come before it.
    pub fn item_unit_end(&self, session_pk: i64, header: u64) -> Result<u64> {
        let conn = self.conn()?;
        let next: Option<i64> = conn
            .prepare_cached(
                "SELECT seq FROM session_item
                 WHERE session_pk = ?1 AND seq > ?2 AND message_pk IS NOT NULL
                 ORDER BY seq LIMIT 1",
            )?
            .query_row(params![session_pk, position(header)?], |row| row.get(0))
            .optional()?;
        match next {
            Some(next) => counter(next),
            None => Ok(self.item_entry_count(session_pk)?),
        }
    }

    fn item_entry_count(&self, session_pk: i64) -> Result<u64> {
        let entries: Option<i64> = self
            .conn()?
            .prepare_cached("SELECT entries FROM session_item_state WHERE session_pk = ?1")?
            .query_row([session_pk], |row| row.get(0))
            .optional()?;
        counter(entries.unwrap_or(0))
    }

    /// A message's header entry and the end of the message: after its last
    /// entry and the events that follow it, before the next message.
    pub fn item_message_span(
        &self,
        session_pk: i64,
        message: &MessageId,
    ) -> Result<Option<(u64, u64)>> {
        match self.item_header(session_pk, message)? {
            Some(header) => Ok(Some((header, self.item_unit_end(session_pk, header)?))),
            None => Ok(None),
        }
    }

    /// The first (or with `forward` false, the last) entry in `range` of
    /// one kind a reading looks up: a person's prompt, a compaction or a
    /// microcompaction.
    pub fn item_find(
        &self,
        session_pk: i64,
        kind: ItemLookup,
        range: std::ops::Range<u64>,
        forward: bool,
    ) -> Result<Option<u64>> {
        let filter = lookup_filter(kind);
        let order = if forward { "" } else { " DESC" };
        let found: Option<i64> = self
            .conn()?
            .prepare_cached(&format!(
                "SELECT seq FROM session_item {filter}
                   AND session_pk = ?1 AND seq >= ?2 AND seq < ?3 ORDER BY seq{order} LIMIT 1"
            ))?
            .query_row(
                params![session_pk, position(range.start)?, position(range.end)?],
                |row| row.get(0),
            )
            .optional()?;
        found.map(counter).transpose()
    }

    /// Every entry in `range` of one kind a reading looks up, in order.
    pub fn item_find_all(
        &self,
        session_pk: i64,
        kind: ItemLookup,
        range: std::ops::Range<u64>,
    ) -> Result<Vec<u64>> {
        let filter = lookup_filter(kind);
        self.conn()?
            .prepare_cached(&format!(
                "SELECT seq FROM session_item {filter}
                   AND session_pk = ?1 AND seq >= ?2 AND seq < ?3 ORDER BY seq"
            ))?
            .query_map(
                params![session_pk, position(range.start)?, position(range.end)?],
                |row| row.get::<_, i64>(0),
            )?
            .map(|seq| counter(seq?))
            .collect()
    }

    /// The entries of a session with a provider call ID: its call and the
    /// call's results.
    pub fn item_calls(&self, session_pk: i64, call_id: &str) -> Result<Vec<StoredEntry>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT seq, class, entry FROM session_item
             WHERE session_pk = ?1 AND call_id = ?2 ORDER BY seq",
        )?;
        let mut rows = statement.query(params![session_pk, call_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            #[cfg(any(test, feature = "fixture"))]
            ENTRIES_DECODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.push(StoredEntry {
                seq: counter(row.get(0)?)?,
                class: class_of(row.get(1)?)?,
                entry: decode(&row.get::<_, Vec<u8>>(2)?)?,
            });
        }
        Ok(out)
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

#[cfg(test)]
mod tests;
