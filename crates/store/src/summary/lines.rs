//! Transcript lines (`transcript_line`, `transcript_call`): per session,
//! the ordered lines a transcript can show, with token estimates and
//! running counts, written by the session summary from the session's
//! assembled rows. The writer only marks the session due with the earliest
//! row key its changes reached; nothing here runs in its transaction.
//!
//! The lines are kept from a stored state: the lines themselves, each
//! unit's first line holding its row's key (`row_key`) and the counts
//! before it, and the session's [`LineState`]. A run finds the earliest
//! line a change at or after the due key can reach, and rebuilds the
//! session's units (message versions and events, in reading order) from the
//! last timed message at or before it; lines before that message are not
//! read. Of the rebuilt units, those equal to the ones already written keep
//! their lines; a message whose content is unchanged takes its lines from
//! the stored ones, so only new or changed messages are read from their
//! blocks. When the difference is only new units at the end, `append_seq`
//! moves; when a written unit changed, moved or disappeared, or the links
//! that compose the session changed, `rewrite_version` moves and
//! outstanding cursors go stale.
//!
//! An event that names no message but has a time goes after the last
//! message at or before its time. To place such events from a suffix alone,
//! each message's first line keeps the latest message time up to it
//! (`time_bound`): the events placed at or after a timed message are exactly
//! those whose time is at least its bound.

use super::fold::Key;
use crate::encoding::Columns;
use crate::items::entry_bytes::{EntryOf, decode, encode};
use crate::items::{decode_counts, encode_counts, read_entries};
use crate::row::{Row as _, row};
use crate::{Result, StoreError};
use polylogue_model::read::{
    IndexEntry, ItemCounts, MessagePlace, entry_classes, event_entry, message_entries,
};
use polylogue_model::{Message, MessageId, PromptKind, Role, SessionId, Usage};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Its key in `summary_state`.
pub(crate) const NAME: &str = "transcript_lines";
/// Bumped when the rules change; every session's lines are written again.
pub(crate) const VERSION: i64 = 1;

row! {
    /// A session's line counters and totals, stored as the `state` of its
    /// `transcript_lines` row in `summary_state`: what cursors bind and
    /// readings page with.
    pub(crate) struct LineState {
        /// Moves when lines are added at the end.
        pub append_seq: i64,
        /// Moves when anything already written changes; stales cursors.
        pub rewrite_version: i64,
        /// Identity of the links that compose this session as a child.
        pub links: Vec<u8>,
        /// How many lines the session has.
        pub entries: i64,
        /// `transcript_line.counts` over every line.
        pub counts: Vec<u8>,
    }
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<LineState>();
}

/// The session's line state, when its lines were written.
pub(crate) fn state(conn: &Connection, session_pk: i64) -> Result<Option<LineState>> {
    let bytes: Option<Option<Vec<u8>>> = conn
        .prepare_cached("SELECT state FROM summary_state WHERE name = ?1 AND session_pk = ?2")?
        .query_row(params![NAME, session_pk], |r| r.get(0))
        .optional()?;
    bytes
        .flatten()
        .map(|bytes| LineState::from_bytes(&mut bytes.as_slice()))
        .transpose()
}

/// What a run computed, for the writer to apply ([`apply`]): the lines
/// from `from` on, which replace those stored there, and the new state.
pub(crate) struct Lines {
    /// The first line replaced; `None` when no line changes.
    from: Option<i64>,
    lines: Vec<Line>,
    /// Kept units' first lines whose rows' keys moved, with the new key.
    rekeys: Vec<(i64, Vec<u8>)>,
    state: LineState,
    /// The units the run read from the session's rows.
    pub read: u64,
    /// The line the run started from, and the session's lines before it.
    pub start: i64,
    pub before: i64,
}

impl Lines {
    /// How many lines the run writes.
    #[cfg(test)]
    pub(crate) fn written(&self) -> usize {
        self.lines.len()
    }
}

struct Line {
    seq: i64,
    /// On a unit's first line: its identity, message or event, time bound,
    /// row key and the counts before it.
    head: Option<Head>,
    entry: Vec<u8>,
    class: i64,
    call_id: Option<String>,
}

struct Head {
    unit: Vec<u8>,
    message_pk: Option<i64>,
    event_pk: Option<i64>,
    time_bound: Option<i64>,
    row_key: Vec<u8>,
    counts: Vec<u8>,
}

/// Writes a run's lines and state; the summary's token check has passed.
pub(crate) fn apply(tx: &Connection, session_pk: i64, lines: &Lines) -> Result<()> {
    let mut rekey = tx.prepare_cached(
        "UPDATE transcript_line SET row_key = ?3 WHERE session_pk = ?1 AND seq = ?2",
    )?;
    for (seq, key) in &lines.rekeys {
        rekey.execute(params![session_pk, seq, key])?;
    }
    drop(rekey);
    if let Some(from) = lines.from {
        let calls: Vec<(String, i64)> = tx
            .prepare_cached(
                "SELECT call_id, seq FROM transcript_line
                 WHERE session_pk = ?1 AND seq >= ?2 AND call_id IS NOT NULL",
            )?
            .query_map(params![session_pk, from], |r| <(String, i64)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        let mut forget = tx.prepare_cached(
            "DELETE FROM transcript_call WHERE session_pk = ?1 AND call_id = ?2 AND seq = ?3",
        )?;
        for (call, seq) in calls {
            forget.execute(params![session_pk, call, seq])?;
        }
        tx.prepare_cached("DELETE FROM transcript_line WHERE session_pk = ?1 AND seq >= ?2")?
            .execute(params![session_pk, from])?;
        let mut head = tx.prepare_cached(
            "INSERT INTO transcript_line(session_pk, seq, unit, message_pk, event_pk, time_bound,
                                         row_key, entry, class, call_id, counts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )?;
        let mut later = tx.prepare_cached(
            "INSERT INTO transcript_line(session_pk, seq, entry, class, call_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        let mut call = tx.prepare_cached(
            "INSERT INTO transcript_call(session_pk, call_id, seq, class) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for line in &lines.lines {
            match &line.head {
                Some(h) => head.execute(params![
                    session_pk,
                    line.seq,
                    h.unit,
                    h.message_pk,
                    h.event_pk,
                    h.time_bound,
                    h.row_key,
                    line.entry,
                    line.class,
                    line.call_id,
                    h.counts
                ])?,
                None => later.execute(params![
                    session_pk,
                    line.seq,
                    line.entry,
                    line.class,
                    line.call_id
                ])?,
            };
            if let Some(id) = &line.call_id {
                call.execute(params![session_pk, id, line.seq, line.class])?;
            }
        }
    }
    let mut state = Vec::new();
    lines.state.to_bytes(&mut state);
    tx.prepare_cached(
        "INSERT INTO summary_state(name, session_pk, version, state) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(name, session_pk) DO UPDATE SET version = excluded.version, state = excluded.state",
    )?
    .execute(params![NAME, session_pk, VERSION.to_string(), state])?;
    Ok(())
}

/// Computes the session's lines from the last line a change at or after
/// `from` can reach, reading on `conn`; nothing is written.
pub(crate) fn compute(conn: &Connection, session_pk: i64, from: &Key) -> Result<Lines> {
    let state = state(conn, session_pk)?;
    let start = match &state {
        None => Start::Whole,
        Some(_) if from.is_start() => Start::Whole,
        Some(state) => start(conn, session_pk, from, state.entries)?,
    };
    let links = links_identity(conn, session_pk)?;
    let changed: i64 = conn
        .prepare_cached("SELECT changed FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| r.get(0))?;
    match (start, state) {
        (Start::Links, Some(state)) => Ok(Lines {
            start: state.entries,
            before: state.entries,
            from: None,
            lines: Vec::new(),
            rekeys: Vec::new(),
            state: if state.links == links {
                state
            } else {
                LineState {
                    rewrite_version: (state.rewrite_version + 1).max(changed),
                    links,
                    ..state
                }
            },
            read: 0,
        }),
        (Start::Links, None) => Err(StoreError::internal("a session without lines has a start")),
        (Start::Whole, state) => rebuild(conn, session_pk, None, from, state, links, changed),
        (Start::From(point), state) => {
            rebuild(conn, session_pk, Some(point), from, state, links, changed)
        }
    }
}

/// Where a session's run starts.
enum Start {
    /// No unit was reached: only the composing links are compared.
    Links,
    Whole,
    From(StartPoint),
}

/// A written message line the run starts from: lines before it stay.
#[derive(Debug, Clone, Copy)]
struct StartPoint {
    seq: i64,
    pk: i64,
    time_bound: Option<i64>,
}

/// The last timed message line a change at or after `from` cannot reach,
/// which the run starts from; `end` is the session's line count.
fn start(conn: &Connection, session_pk: i64, from: &Key, end: i64) -> Result<Start> {
    // A cause outside the session's rows reached none of them.
    if *from >= Key::past_every_row() {
        return Ok(Start::Links);
    }
    // The latest line the run may start from.
    let mut limit: Option<i64> = None;
    let keep = |limit: &mut Option<i64>, seq: i64| {
        *limit = Some(limit.map_or(seq, |old: i64| old.min(seq)));
    };
    // A change at `seq` (a unit there changes, or a unit goes before it).
    let lower = |limit: &mut Option<i64>, seq: i64| keep(limit, seq - 1);
    // Units written from rows at or after `from`: those rows changed,
    // moved or went.
    let reached: Option<i64> = conn
        .prepare_cached(
            "SELECT min(seq) FROM transcript_line INDEXED BY transcript_line_row
             WHERE session_pk = ?1 AND row_key >= ?2",
        )?
        .query_row(params![session_pk, from.0], |r| r.get(0))?;
    if let Some(seq) = reached {
        lower(&mut limit, seq);
    }
    // A message at or after `from` not yet written goes before the first
    // unit reached, or at the end.
    let order = from.order();
    let messages: bool = conn
        .prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM message INDEXED BY message_session
                           WHERE session_pk = ?1 AND session_order >= ?2 AND duplicate_of IS NULL)",
        )?
        .query_row(params![session_pk, order], |r| r.get(0))?;
    if messages {
        lower(&mut limit, end);
    }
    // Events at or after `from`, placed where they go now.
    let mut events = conn.prepare_cached(
        "SELECT e.id, e.session_order, (SELECT m.pk FROM message m WHERE m.id = e.after_message_id),
                e.at,
                EXISTS(SELECT 1 FROM message m WHERE m.id = e.after_message_id
                       AND m.session_pk = e.session_pk AND m.duplicate_of IS NULL)
         FROM event e INDEXED BY event_session_order
         WHERE e.session_pk = ?1 AND e.session_order >= ?2",
    )?;
    let mut rows = events.query(params![session_pk, order])?;
    // A later time places an event at the same line or a later one, so the
    // earliest time among events placed by time gives the earliest place.
    let mut earliest_timed: Option<i64> = None;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let event_order: String = row.get(1)?;
        if Key::event(&event_order, &id) < *from {
            continue;
        }
        let message: Option<i64> = row.get(2)?;
        let at: Option<i64> = row.get(3)?;
        let shown: bool = row.get(4)?;
        match (message, at, shown) {
            // After the message it follows, which stays as it was unless it
            // changed itself.
            (Some(message), _, true) => {
                if let Some(seq) = header(conn, session_pk, message)? {
                    keep(&mut limit, seq);
                }
            }
            (None, Some(at), _) => {
                earliest_timed = Some(earliest_timed.map_or(at, |old| old.min(at)));
            }
            // At the end.
            _ => lower(&mut limit, end),
        }
    }
    drop(rows);
    if let Some(at) = earliest_timed {
        keep(&mut limit, timed_place(conn, session_pk, at)?);
    }
    let Some(limit) = limit else {
        return Ok(Start::Links);
    };
    let point = conn
        .prepare_cached(
            "SELECT i.seq, i.message_pk, i.time_bound FROM transcript_line i
             JOIN message m ON m.pk = i.message_pk
             WHERE i.session_pk = ?1 AND i.seq <= ?2 AND i.message_pk IS NOT NULL
               AND m.at IS NOT NULL
             ORDER BY i.seq DESC LIMIT 1",
        )?
        .query_row(params![session_pk, limit], |r| {
            Ok(StartPoint {
                seq: r.get(0)?,
                pk: r.get(1)?,
                time_bound: r.get(2)?,
            })
        })
        .optional()?;
    Ok(point.map_or(Start::Whole, Start::From))
}

/// The first line of a message in this session.
fn header(conn: &Connection, session_pk: i64, message_pk: i64) -> Result<Option<i64>> {
    Ok(conn
        .prepare_cached(
            "SELECT seq FROM transcript_line WHERE message_pk = ?1 AND session_pk = ?2",
        )?
        .query_row(params![message_pk, session_pk], |r| r.get(0))
        .optional()?)
}

/// The line of the last timed message an event at `at` that names no
/// message cannot precede: the last one whose bound is at most `at`.
fn timed_place(conn: &Connection, session_pk: i64, at: i64) -> Result<i64> {
    Ok(conn
        .prepare_cached(
            "SELECT i.seq FROM transcript_line i JOIN message m ON m.pk = i.message_pk
             WHERE i.session_pk = ?1 AND i.message_pk IS NOT NULL AND m.at IS NOT NULL
               AND (i.time_bound IS NULL OR i.time_bound <= ?2)
             ORDER BY i.seq DESC LIMIT 1",
        )?
        .query_row(params![session_pk, at], |r| r.get(0))
        .optional()?
        .unwrap_or(0))
}

/// A unit: a message version or an event, in reading order.
enum Unit {
    Message {
        pk: i64,
        id: String,
        key: Key,
        at: Option<i64>,
        /// Names the message's content and the row fields its lines read.
        content: [u8; 32],
        place: MessagePlace,
        time_bound: Option<i64>,
    },
    Event {
        pk: i64,
        id: String,
        key: Key,
        kind: String,
        at: Option<i64>,
        data_bytes: u64,
        data: Option<Vec<u8>>,
    },
    /// An unchanged event the run takes from its line as stored.
    Kept {
        pk: i64,
        id: String,
        key: Key,
        identity: Vec<u8>,
        entry: Vec<u8>,
        call_id: Option<String>,
    },
}

impl Unit {
    /// The identity of the lines this unit stores. It names the message or
    /// event by its natural key, not its row key, so an index holds the same
    /// identities whatever order its rows were written in. A message's is
    /// its content's identity followed by its place and time, which its
    /// lines take from its row.
    fn identity(&self) -> Result<Vec<u8>> {
        match self {
            Self::Message {
                content, place, at, ..
            } => {
                let mut out = content.to_vec();
                out.extend([
                    match place.active_path {
                        None => 2,
                        Some(true) => 1,
                        Some(false) => 0,
                    },
                    u8::from(place.branch_point),
                ]);
                if let Some(at) = at {
                    out.extend(at.to_be_bytes());
                }
                Ok(out)
            }
            Self::Event {
                id,
                kind,
                at,
                data_bytes,
                data,
                ..
            } => event_identity(
                id,
                &event_item(id, kind, *at, *data_bytes, data.as_deref())?,
            ),
            Self::Kept { identity, .. } => Ok(identity.clone()),
        }
    }

    /// The columns of the unit's first line that locate it.
    fn columns(&self) -> (Option<i64>, Option<i64>, Option<i64>) {
        match self {
            Self::Message { pk, time_bound, .. } => (Some(*pk), None, *time_bound),
            Self::Event { pk, .. } | Self::Kept { pk, .. } => (None, Some(*pk), None),
        }
    }

    fn key(&self) -> &Key {
        match self {
            Self::Message { key, .. } | Self::Event { key, .. } | Self::Kept { key, .. } => key,
        }
    }
}

/// A message row as the units read it.
struct MessageRow {
    pk: i64,
    id: String,
    order: String,
    content: [u8; 32],
    active: Option<bool>,
    branch_point: bool,
    at: Option<i64>,
}

/// An event as the units place it: where it goes, and what it stores.
struct EventRow {
    pk: i64,
    id: String,
    order: String,
    at: Option<i64>,
    after: Option<i64>,
    seq: i64,
    body: EventBody,
}

enum EventBody {
    /// Read from the event row.
    Read {
        kind: String,
        data_bytes: u64,
        data: Option<Vec<u8>>,
        /// Where `data` is stored, until it is read.
        data_at: Option<crate::content::Stretch>,
    },
    /// An unchanged event: its stored identity and entry stand for its row.
    Kept {
        identity: Vec<u8>,
        entry: Vec<u8>,
        call_id: Option<String>,
    },
}

impl EventRow {
    fn unit(self) -> Unit {
        let key = Key::event(&self.order, &self.id);
        match self.body {
            EventBody::Read {
                kind,
                data_bytes,
                data,
                ..
            } => Unit::Event {
                pk: self.pk,
                id: self.id,
                key,
                kind,
                at: self.at,
                data_bytes,
                data,
            },
            EventBody::Kept {
                identity,
                entry,
                call_id,
            } => Unit::Kept {
                pk: self.pk,
                id: self.id,
                key,
                identity,
                entry,
                call_id,
            },
        }
    }
}

/// The message columns a unit reads: its content is named by its stored
/// version (or, for rows written without one, its block rows), the blocks
/// shown, and every row field its lines are made from.
const MESSAGE_COLUMNS: &str = "m.pk, m.id, m.session_order,
    coalesce(m.version_hash, (SELECT CAST(count(*) || ':' || coalesce(max(b.pk), 0) AS BLOB)
                              FROM block b WHERE b.message_pk = m.pk)),
    m.role, m.prompt_kind, m.input_tokens, m.cache_read_tokens, m.cache_write_tokens,
    m.on_active_path, m.at,
    EXISTS(SELECT 1 FROM message c WHERE c.session_pk = m.session_pk AND c.parent_id = m.id
           AND c.duplicate_of IS NULL LIMIT 1 OFFSET 1),
    (SELECT group_concat(o, ',') FROM (SELECT coalesce(CAST(b.ordinal AS INTEGER), b.idx) AS o
                                      FROM block b WHERE b.message_pk = m.pk ORDER BY b.idx)),
    m.model";

/// The columns an event is read with. The data a line needs is read only
/// for the kinds whose line holds it (see [`read_data`]).
const EVENT_COLUMNS: &str = "e.pk, e.id, e.kind, e.at, e.data_length,
    (SELECT m.pk FROM message m WHERE m.id = e.after_message_id), e.seq,
    CASE WHEN e.kind IN ('compaction','microcompaction') THEN e.content_group END,
    e.content_start + e.content_length, e.data_length, e.session_order";

fn message_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MessageRow> {
    let id: String = row.get(1)?;
    let mut digest = blake3::Hasher::new();
    digest.update(b"message\0");
    digest.update(id.as_bytes());
    digest.update(b"\0");
    digest.update(&row.get::<_, Vec<u8>>(3)?);
    // Role, prompt kind and model: what the header's estimate and author
    // are made from.
    for column in [4, 5, 13] {
        let value: Option<String> = row.get(column)?;
        digest.update(b"\0");
        digest.update(value.as_deref().unwrap_or("").as_bytes());
    }
    // The blocks shown: readings of one version can each hold some.
    let shown: Option<String> = row.get(12)?;
    digest.update(b"\0");
    digest.update(shown.as_deref().unwrap_or("").as_bytes());
    for column in [6, 7, 8] {
        let value: Option<i64> = row.get(column)?;
        digest.update(&value.map_or([0xFF; 9], |n| {
            let mut bytes = [0; 9];
            bytes[1..].copy_from_slice(&n.to_be_bytes());
            bytes
        }));
    }
    Ok(MessageRow {
        pk: row.get(0)?,
        id,
        order: row.get(2)?,
        content: *digest.finalize().as_bytes(),
        active: row.get::<_, Option<i64>>(9)?.map(|value| value != 0),
        at: row.get(10)?,
        branch_point: row.get(11)?,
    })
}

/// What the line runs of this process read and wrote: what a test counts
/// to show a run's cost follows what its change reached.
#[cfg(any(test, feature = "fixture"))]
pub(crate) mod counted {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Event rows read from the event table.
    pub(super) static EVENT_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Units (messages and events) a run placed.
    pub(super) static UNITS: AtomicU64 = AtomicU64::new(0);
    /// Messages whose lines were made from their blocks, not taken from
    /// their stored lines.
    pub(super) static BODIES: AtomicU64 = AtomicU64::new(0);
    /// Lines written.
    pub(super) static LINES: AtomicU64 = AtomicU64::new(0);

    /// The counts so far: event rows, units, message bodies, lines written.
    #[cfg(test)]
    pub(crate) fn now() -> [u64; 4] {
        [&EVENT_ROWS, &UNITS, &BODIES, &LINES].map(|n| n.load(Ordering::Relaxed))
    }

    pub(super) fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }
}

fn event_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<EventRow> {
    #[cfg(any(test, feature = "fixture"))]
    counted::add(&counted::EVENT_ROWS, 1);
    Ok(EventRow {
        pk: row.get(0)?,
        id: row.get(1)?,
        at: row.get(3)?,
        after: row.get(5)?,
        seq: row.get(6)?,
        order: row.get(10)?,
        body: EventBody::Read {
            kind: row.get(2)?,
            data_bytes: row.get::<_, i64>(4)?.max(0) as u64,
            data: None,
            data_at: match row.get::<_, Option<i64>>(7)? {
                Some(group) => Some(crate::content::Stretch {
                    group,
                    start: row.get(8)?,
                    length: row.get(9)?,
                }),
                None => None,
            },
        },
    })
}

/// Reads the data of the events whose line holds it.
fn read_data<'e>(
    conn: &Connection,
    events: impl IntoIterator<Item = &'e mut EventRow>,
) -> Result<()> {
    let mut reader = crate::content::Reader::default();
    for event in events {
        if let EventBody::Read { data, data_at, .. } = &mut event.body
            && let Some(at) = data_at.take()
        {
            *data = Some(reader.bytes(conn, at)?.to_vec());
        }
    }
    Ok(())
}

/// The session's units in reading order from `point` (from the first
/// message without one): messages by their place; each event after the
/// message it follows, a timed event that names no message after the last
/// message at or before its time, and the rest at the end. `extra` names
/// changed events to place, in case nothing else finds them.
fn units_from(
    conn: &Connection,
    session_pk: i64,
    point: Option<StartPoint>,
    extra: &BTreeSet<i64>,
) -> Result<Vec<Unit>> {
    let rows: Vec<MessageRow> = match point {
        None => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message m INDEXED BY message_session
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                 ORDER BY m.session_order, m.id"
            ))?
            .query_map([session_pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
        Some(point) => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message a, message m INDEXED BY message_session
                 WHERE a.pk = ?2 AND m.session_pk = ?1 AND m.duplicate_of IS NULL
                   AND (m.session_order, m.id) >= (a.session_order, a.id)
                 ORDER BY m.session_order, m.id"
            ))?
            .query_map(params![session_pk, point.pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
    };
    let events = events_from(conn, session_pk, point, &rows, extra)?;
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
                // Shown before the start point: it is placed there, not here.
                let shown = point.is_some()
                    && shown_before
                        .query_row(params![message, session_pk], |row| row.get::<_, bool>(0))
                        .optional()?
                        .unwrap_or(false);
                if !shown {
                    orphaned.push(event.unit());
                }
            }
            (None, Some(at)) => {
                if point.is_none_or(|point| point.time_bound.is_none_or(|bound| at >= bound)) {
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
    let mut time_bound = point.and_then(|point| point.time_bound);
    for (index, row) in rows.into_iter().enumerate() {
        // The start point's bound is stored; the first message's is none.
        if index > 0 {
            time_bound = match (time_bound, row.at) {
                (Some(bound), Some(at)) => Some(bound.max(at)),
                (bound, at) => bound.or(at),
            };
        }
        let pk = row.pk;
        out.push(Unit::Message {
            pk,
            key: Key::message(&row.order, &row.id),
            id: row.id,
            at: row.at,
            content: row.content,
            place: MessagePlace {
                // Unknown and on the path read the same; only off the path
                // changes what a reading shows.
                active_path: (row.active == Some(false)).then_some(false),
                branch_point: row.branch_point,
            },
            time_bound,
        });
        if let Some(events) = after.remove(&pk) {
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
    // their order (by place in their input, then ID) at the end with the
    // events that name no message; their row keys depend on the order the
    // rows were written in, so they order nothing.
    out.extend(unplaced);
    out.extend(orphaned);
    Ok(out)
}

/// The events that can be placed from `point` on, in stored order: those
/// following a loaded message, the changed ones, and the rest of those
/// written from the start point on. The rest are unchanged, so they are
/// placed after the start point again (nothing before it changed); their
/// place comes from the event placement index and what they store from
/// their lines, so the event table is read only for what the change
/// reached.
fn events_from(
    conn: &Connection,
    session_pk: i64,
    point: Option<StartPoint>,
    rows: &[MessageRow],
    extra: &BTreeSet<i64>,
) -> Result<Vec<EventRow>> {
    let Some(point) = point else {
        let mut events: Vec<EventRow> = conn
            .prepare_cached(&format!(
                "SELECT {EVENT_COLUMNS} FROM event e INDEXED BY event_session
                 WHERE e.session_pk = ?1 ORDER BY e.seq, e.id"
            ))?
            .query_map([session_pk], event_row)?
            .collect::<rusqlite::Result<_>>()?;
        read_data(conn, &mut events)?;
        return Ok(events);
    };
    let mut found: HashMap<i64, EventRow> = HashMap::new();
    let mut following = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e INDEXED BY event_after_id
         WHERE e.after_message_id = ?1 AND e.session_pk = ?2"
    ))?;
    for row in rows {
        for event in following.query_map(params![row.id, session_pk], event_row)? {
            let event = event?;
            found.insert(event.pk, event);
        }
    }
    let mut by_pk = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e WHERE e.pk = ?1 AND e.session_pk = ?2"
    ))?;
    for &pk in extra {
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
    // Each written event's key, with its stored identity, entry and call ID.
    let written: Vec<(i64, EventBody)> = conn
        .prepare_cached(
            "SELECT event_pk, unit, entry, call_id FROM transcript_line
             WHERE session_pk = ?1 AND seq >= ?2 AND event_pk IS NOT NULL",
        )?
        .query_map(params![session_pk, point.seq], |row| {
            Ok((
                row.get(0)?,
                EventBody::Kept {
                    identity: row.get(1)?,
                    entry: row.get(2)?,
                    call_id: row.get(3)?,
                },
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut placed = conn.prepare_cached(
        "SELECT e.id, e.at, (SELECT m.pk FROM message m WHERE m.id = e.after_message_id), e.seq,
                e.session_order
         FROM event e INDEXED BY event_placement WHERE e.session_pk = ?1 AND e.pk = ?2",
    )?;
    for (pk, body) in written {
        if found.contains_key(&pk) || extra.contains(&pk) {
            continue;
        }
        // An unchanged written event is still in the session; one that is
        // not was changed, and its row key reached the due mark.
        let event = placed
            .query_row(params![session_pk, pk], |row| {
                Ok(EventRow {
                    pk,
                    id: row.get(0)?,
                    at: row.get(1)?,
                    after: row.get(2)?,
                    seq: row.get(3)?,
                    order: row.get(4)?,
                    body,
                })
            })
            .optional()?
            .ok_or_else(|| {
                StoreError::internal(format!(
                    "written event {pk} left session {session_pk} unmarked"
                ))
            })?;
        found.insert(pk, event);
    }
    let mut events: Vec<EventRow> = found.into_values().collect();
    events.sort_by(|a, b| (a.seq, &a.id).cmp(&(b.seq, &b.id)));
    read_data(conn, &mut events)?;
    Ok(events)
}

/// The events a change at or after `from` reached: those whose row key is
/// at or after it now, and those written from a row key at or after it.
fn reached_events(conn: &Connection, session_pk: i64, from: &Key) -> Result<BTreeSet<i64>> {
    let mut out = BTreeSet::new();
    let mut now = conn.prepare_cached(
        "SELECT pk, id, session_order FROM event INDEXED BY event_session_order
         WHERE session_pk = ?1 AND session_order >= ?2",
    )?;
    let mut rows = now.query(params![session_pk, from.order()])?;
    while let Some(row) = rows.next()? {
        let order: String = row.get(2)?;
        if Key::event(&order, &row.get::<_, String>(1)?) >= *from {
            out.insert(row.get(0)?);
        }
    }
    drop(rows);
    let written = conn
        .prepare_cached(
            "SELECT event_pk FROM transcript_line INDEXED BY transcript_line_row
             WHERE session_pk = ?1 AND row_key >= ?2 AND event_pk IS NOT NULL",
        )?
        .query_map(params![session_pk, from.0], |r| r.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    out.extend(written);
    Ok(out)
}

/// A unit already written from the run's start on: where it is and its
/// identity.
struct Written {
    seq: i64,
    unit: Vec<u8>,
    message_pk: Option<i64>,
    event_pk: Option<i64>,
    row_key: Option<Vec<u8>>,
}

/// Computes the session's lines from `point` (from its first unit without
/// one), keeping those that did not change.
fn rebuild(
    conn: &Connection,
    session_pk: i64,
    point: Option<StartPoint>,
    from: &Key,
    state: Option<LineState>,
    links: Vec<u8>,
    changed: i64,
) -> Result<Lines> {
    let first_seq = point.map_or(0, |point| point.seq);
    let extra = match point {
        Some(_) => reached_events(conn, session_pk, from)?,
        None => BTreeSet::new(),
    };
    let units = units_from(conn, session_pk, point, &extra)?;
    let identities: Vec<Vec<u8>> = units.iter().map(Unit::identity).collect::<Result<_>>()?;
    let written: Vec<Written> = conn
        .prepare_cached(
            "SELECT seq, unit, message_pk, event_pk, row_key FROM transcript_line
             WHERE session_pk = ?1 AND seq >= ?2 AND unit IS NOT NULL ORDER BY seq",
        )?
        .query_map(params![session_pk, first_seq], |row| {
            Ok(Written {
                seq: row.get(0)?,
                unit: row.get(1)?,
                message_pk: row.get(2)?,
                event_pk: row.get(3)?,
                row_key: row.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let end = state.as_ref().map_or(0, |state| state.entries);
    // Where each written message is, so an unchanged one keeps its lines.
    let mut stored: HashMap<i64, (i64, i64, &[u8])> = HashMap::new();
    for (n, unit) in written.iter().enumerate() {
        if let Some(pk) = unit.message_pk {
            let until = written.get(n + 1).map_or(end, |next| next.seq);
            stored.insert(pk, (unit.seq, until, &unit.unit));
        }
    }
    let same = written
        .iter()
        .zip(&identities)
        .take_while(|(written, identity)| written.unit == **identity)
        .count();
    // A kept unit whose row moved in the session's order keeps its lines
    // and takes its row's new key: a stale key would hide a later change
    // from the run's start.
    let rekeys: Vec<(i64, Vec<u8>)> = written
        .iter()
        .zip(&units)
        .take(same)
        .filter(|(written, unit)| written.row_key.as_deref() != Some(&unit.key().0[..]))
        .map(|(written, unit)| (written.seq, unit.key().0.clone()))
        .collect();
    let pending = units
        .iter()
        .skip(same)
        .map(|unit| unit_entries(conn, session_pk, unit, &stored))
        .collect::<Result<Vec<_>>>()?;
    let links_moved = state
        .as_ref()
        .is_some_and(|state| state.links.as_slice() != links.as_slice());
    // The first line written below and the counts before it: those stored
    // on the first unit replaced, or after every line when units are only
    // added.
    let (mut next, mut running) = match written.get(same) {
        Some(unit) => (unit.seq, unit_counts(conn, session_pk, unit.seq)?),
        None => (
            end,
            match &state {
                Some(state) => decode_counts(&state.counts)?,
                None => ItemCounts::default(),
            },
        ),
    };
    let replaced = written.get(same).map(|unit| unit.seq);
    let mut grown = false;
    let mut header_moved = false;
    let mut extended = false;
    // A live session's last message grows block by block. When the only
    // difference is new blocks after the last written message's lines,
    // that is an append; outstanding cursors stay valid.
    if same + 1 == written.len()
        && same < units.len()
        && (written[same].message_pk, written[same].event_pk) == {
            let (message, event, _) = units[same].columns();
            (message, event)
        }
    {
        let stored = crate::items::stored_entries(conn, session_pk, written[same].seq)?;
        let current = &pending[0];
        if extends(&stored, current) {
            grown = current.len() > stored.len();
            header_moved = current[0] != stored[0];
            extended = true;
        }
    }
    let after = same + usize::from(extended);
    let rewritten = after < written.len() || links_moved;
    let appended = grown || header_moved || after < units.len();
    let mut lines = Vec::new();
    for ((unit, identity), entries) in units.iter().zip(&identities).skip(same).zip(&pending) {
        let (message_pk, event_pk, time_bound) = unit.columns();
        let counts = encode_counts(&running);
        for (offset, (entry, class)) in entries.iter().zip(entry_classes(entries)).enumerate() {
            lines.push(Line {
                seq: next,
                head: (offset == 0).then(|| Head {
                    unit: identity.clone(),
                    message_pk,
                    event_pk,
                    time_bound,
                    row_key: unit.key().0.clone(),
                    counts: counts.clone(),
                }),
                entry: if offset == 0 {
                    encode(entry, None)?
                } else {
                    encode(entry, Some(&entries[0]))?
                },
                class: i64::from(class.bits()),
                call_id: entry.call_id.clone(),
            });
            running.add(entry, class);
            next += 1;
        }
    }
    #[cfg(any(test, feature = "fixture"))]
    {
        counted::add(&counted::UNITS, units.len() as u64);
        counted::add(&counted::LINES, lines.len() as u64);
    }
    // A rewrite version is never below the write counter of the change that
    // set it, so a session removed and stored again never repeats a version
    // an earlier cursor saw.
    let (append_seq, rewrite_version) = match &state {
        None => (i64::from(appended), changed.max(1)),
        Some(state) => (
            state.append_seq + i64::from(appended && !rewritten),
            if rewritten {
                (state.rewrite_version + 1).max(changed)
            } else {
                state.rewrite_version
            },
        ),
    };
    Ok(Lines {
        from: (!lines.is_empty() || replaced.is_some()).then(|| replaced.unwrap_or(end)),
        lines,
        rekeys,
        state: LineState {
            append_seq,
            rewrite_version,
            links,
            entries: next,
            counts: encode_counts(&running),
        },
        read: units.len() as u64,
        start: first_seq,
        before: end,
    })
}

/// The counts stored on the unit whose first line is at `seq`.
fn unit_counts(conn: &Connection, session_pk: i64, seq: i64) -> Result<ItemCounts> {
    let counts: Option<Vec<u8>> = conn
        .prepare_cached("SELECT counts FROM transcript_line WHERE session_pk = ?1 AND seq = ?2")?
        .query_row(params![session_pk, seq], |row| row.get(0))
        .optional()?
        .flatten();
    match counts {
        Some(counts) => decode_counts(&counts),
        None => Err(StoreError::stored_data(format!(
            "transcript line {seq} is not the first line of a unit"
        ))),
    }
}

/// A unit's entries: a message's from its stored lines when its content
/// is the one they were written from, otherwise from its block rows; an
/// event's from its row or its stored line.
fn unit_entries(
    conn: &Connection,
    session_pk: i64,
    unit: &Unit,
    stored: &HashMap<i64, (i64, i64, &[u8])>,
) -> Result<Vec<IndexEntry>> {
    match unit {
        Unit::Message {
            pk,
            id,
            at,
            content,
            place,
            ..
        } => {
            let mut entries = match stored.get(pk) {
                Some((seq, until, identity)) if identity.get(..32) == Some(&content[..]) => {
                    read_entries(conn, session_pk, *seq..*until, true, usize::MAX, |_| true)?
                        .into_iter()
                        .map(|stored| stored.entry)
                        .collect()
                }
                _ => {
                    #[cfg(any(test, feature = "fixture"))]
                    counted::add(&counted::BODIES, 1);
                    let (message, ordinals) = message_body(conn, *pk)?;
                    message_entries(&message, &ordinals, MessagePlace::default())
                }
            };
            if entries.is_empty() {
                return Err(StoreError::stored_data(format!(
                    "message {id} has no transcript lines"
                )));
            }
            let id = MessageId::parse(id).map_err(StoreError::stored_data)?;
            for entry in &mut entries {
                if let polylogue_model::read::EntryKey::Message { message, .. } = &mut entry.key {
                    *message = id.clone();
                }
                entry.at = *at;
                entry.active_path = place.active_path;
                entry.branch_point = place.branch_point;
            }
            Ok(entries)
        }
        Unit::Event {
            id,
            kind,
            at,
            data_bytes,
            data,
            ..
        } => Ok(vec![event_item(
            id,
            kind,
            *at,
            *data_bytes,
            data.as_deref(),
        )?]),
        Unit::Kept {
            id, entry, call_id, ..
        } => Ok(vec![decode(
            entry,
            EntryOf::Event(id.clone()),
            call_id.clone(),
        )?]),
    }
}

/// The event entry decides both the stored line and its identity.
fn event_item(
    id: &str,
    kind: &str,
    at: Option<i64>,
    data_bytes: u64,
    data: Option<&[u8]>,
) -> Result<IndexEntry> {
    let cleared = if kind == "microcompaction" {
        data.map(crate::encoding::read_json::<polylogue_model::MicrocompactionBoundary>)
            .transpose()?
            .map(|boundary| boundary.removed_calls)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut entry = event_entry(id, kind, at, data_bytes, cleared);
    if kind == "compaction" {
        entry.compaction = data
            .map(crate::encoding::read_json::<polylogue_model::CompactionBoundary>)
            .transpose()?;
    }
    Ok(entry)
}

fn event_identity(id: &str, entry: &IndexEntry) -> Result<Vec<u8>> {
    let mut digest = blake3::Hasher::new();
    digest.update(b"event\0");
    digest.update(id.as_bytes());
    digest.update(b"\0");
    digest.update(&encode(entry, None)?);
    Ok(digest.finalize().as_bytes().to_vec())
}

/// A message's row and its blocks with their original ordinals: the fields
/// its lines are estimated from. Structured tool data is not read.
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
    let mut statement = conn.prepare_cached(&format!(
        "SELECT b.pk, coalesce(CAST(b.ordinal AS INTEGER), b.idx), {}
         FROM block b {} WHERE b.message_pk = ?1 ORDER BY b.idx",
        crate::ordered_blocks::COLUMNS,
        crate::ordered_blocks::TEXT_JOIN
    ))?;
    let mut rows = statement.query([pk])?;
    let mut reader = crate::content::Reader::default();
    while let Some(row) = rows.next()? {
        blocks.push(crate::ordered_blocks::decode(
            conn,
            &mut reader,
            row,
            2,
            false,
        )?);
        ordinals.push(row.get::<_, u64>(1)?);
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

/// A session's lines computed from its rows alone, without the stored
/// lines: what the stored lines must always equal.
#[cfg(test)]
pub(crate) fn computed_entries(conn: &Connection, session_pk: i64) -> Result<Vec<IndexEntry>> {
    let mut out = Vec::new();
    for unit in units_from(conn, session_pk, None, &BTreeSet::new())? {
        match unit {
            Unit::Message { pk, place, .. } => {
                let (message, ordinals) = message_body(conn, pk)?;
                out.extend(message_entries(&message, &ordinals, place));
            }
            Unit::Event {
                id,
                kind,
                at,
                data_bytes,
                data,
                ..
            } => {
                out.push(event_item(&id, &kind, at, data_bytes, data.as_deref())?);
            }
            Unit::Kept { .. } => {
                return Err(StoreError::internal(
                    "a session read whole takes every event from its row",
                ));
            }
        }
    }
    Ok(out)
}

/// Writes the lines of every session changed since `written` records it
/// (its `changed` then), from its first row, on a connection
/// that holds rows written directly (the fixture builder) and no due
/// marks. The fixture builder writes rows the writer would refuse (message
/// IDs without an origin, partial block bodies) for tests that never read
/// them as sessions; such a session is left without lines, so only a
/// session read of it is refused.
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn write_fixture_lines(
    conn: &Connection,
    written: &mut std::collections::HashMap<i64, i64>,
) -> Result<()> {
    let sessions: Vec<(i64, i64)> = conn
        .prepare("SELECT pk, changed FROM session ORDER BY pk")?
        .query_map([], |row| <(i64, i64)>::try_from(row))?
        .filter(|row| !matches!(row, Ok((pk, changed)) if written.get(pk) == Some(changed)))
        .collect::<rusqlite::Result<_>>()?;
    // One transaction, as the writer applies a round.
    conn.execute_batch("SAVEPOINT fixture_lines")?;
    let result = (|| {
        for &(pk, _) in &sessions {
            match compute(conn, pk, &Key::start()) {
                Ok(lines) => apply(conn, pk, &lines)?,
                Err(
                    StoreError::Invalid(_)
                    | StoreError::Refused(_)
                    | StoreError::Internal(_)
                    | StoreError::StoredValue { .. }
                    | StoreError::Json(_),
                ) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => conn.execute_batch("RELEASE fixture_lines")?,
        Err(error) => {
            conn.execute_batch("ROLLBACK TO fixture_lines; RELEASE fixture_lines")?;
            return Err(error);
        }
    }
    written.extend(sessions);
    Ok(())
}

/// What is stored beside each line: its class, call ID and, on a unit's
/// first line, the counts before it; then the line count and the counts
/// over every line.
#[cfg(test)]
pub(crate) type StoredColumns = (
    Vec<(
        polylogue_model::read::EntryClass,
        Option<String>,
        Option<ItemCounts>,
    )>,
    u64,
    ItemCounts,
);

#[cfg(test)]
pub(crate) fn stored_columns(conn: &Connection, session_pk: i64) -> Result<StoredColumns> {
    let mut statement = conn.prepare(
        "SELECT class, call_id, counts FROM transcript_line WHERE session_pk = ?1 ORDER BY seq",
    )?;
    let mut rows = statement.query([session_pk])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let bits: i64 = row.get(0)?;
        out.push((
            u16::try_from(bits)
                .ok()
                .and_then(polylogue_model::read::EntryClass::from_bits)
                .ok_or_else(|| StoreError::stored_data("unknown transcript line class"))?,
            row.get(1)?,
            row.get::<_, Option<Vec<u8>>>(2)?
                .map(|counts| decode_counts(&counts))
                .transpose()?,
        ));
    }
    let state = state(conn, session_pk)?
        .ok_or_else(|| StoreError::stored_data("a session has no transcript lines"))?;
    Ok((
        out,
        u64::try_from(state.entries).unwrap_or(0),
        decode_counts(&state.counts)?,
    ))
}
