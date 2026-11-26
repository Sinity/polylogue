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
use crate::ConnExt as _;
use crate::change_summary::{ChangeSummary, SessionChange};
use crate::encoding::Columns;
use crate::{Reader, Result, StoreError};
use entry_bytes::{EntryOf, decode, encode};
#[cfg(any(test, feature = "fixture"))]
use polylogue_model::{Message, Usage};
use polylogue_model::{
    MessageId, PromptKind, Role, SessionId,
    read::{
        EntryClass, IndexEntry, ItemCounts, MessagePlace, entry_classes, event_entry,
        message_entries,
    },
};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub(crate) mod entry_bytes;

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

/// Event rows the item refresh read from the event table in this process:
/// what a test counts to show a refresh reads only what its write reached.
#[cfg(any(test, feature = "fixture"))]
static EVENT_ROWS_READ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(any(test, feature = "fixture"))]
pub fn event_rows_read() -> u64 {
    EVENT_ROWS_READ.load(std::sync::atomic::Ordering::Relaxed)
}

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

/// Production uses worker entries. Raw-row fixtures decode only units that
/// need rebuilding, rather than preparing every prior message again.
#[derive(Clone, Copy)]
enum MessageEntries<'a> {
    Prepared(&'a ChangeSummary),
    #[cfg(any(test, feature = "fixture"))]
    Fixture,
}

/// Where the message an event follows is looked for.
#[derive(Debug, Clone, Copy)]
enum Placement<'a> {
    /// In the whole index, by its ID.
    Index,
    /// Among the session's own messages, then among these, which other
    /// sessions show: a build composing sessions from its stream knows
    /// which messages its events follow outside their session, so nothing
    /// is looked up by ID in an index it is still writing.
    SessionAlone(&'a std::collections::HashSet<String>),
}

/// The row key of a message only known to be shown in another session: no
/// row has it.
const OTHER_SESSION: i64 = 0;

impl<'a> MessageEntries<'a> {
    fn placement(self) -> Placement<'a> {
        match self {
            Self::Prepared(changes) => match changes.sessions_alone() {
                Some(elsewhere) => Placement::SessionAlone(elsewhere),
                None => Placement::Index,
            },
            #[cfg(any(test, feature = "fixture"))]
            Self::Fixture => Placement::Index,
        }
    }

    fn has_prepared_items(self, pk: i64) -> bool {
        match self {
            Self::Prepared(changes) => changes.has_prepared_items(pk),
            #[cfg(any(test, feature = "fixture"))]
            Self::Fixture => false,
        }
    }
}

enum Unit {
    Message {
        pk: i64,
        id: String,
        at: Option<i64>,
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
        data: Option<Vec<u8>>,
    },
    /// An unchanged event the refresh takes from the index as stored.
    Kept {
        pk: i64,
        id: String,
        identity: Vec<u8>,
        entry: Vec<u8>,
        call_id: Option<String>,
    },
}
impl Unit {
    /// The identity of the entries this unit stores. It names the message or
    /// event by its natural key, not its row key, so an index holds the same
    /// identities whatever order its rows were written in.
    fn identity(&self) -> Result<Vec<u8>> {
        let mut digest = blake3::Hasher::new();
        match self {
            Self::Message {
                id, version, place, ..
            } => {
                digest.update(b"message\0");
                digest.update(id.as_bytes());
                digest.update(b"\0");
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
                id,
                kind,
                at,
                data_bytes,
                data,
                ..
            } => {
                return event_identity(
                    id,
                    &event_item(id, kind, *at, *data_bytes, data.as_deref())?,
                );
            }
            Self::Kept { identity, .. } => return Ok(identity.clone()),
        }
        Ok(digest.finalize().as_bytes().to_vec())
    }

    /// The columns of the unit's first entry that locate it.
    fn columns(&self) -> (Option<i64>, Option<i64>, Option<i64>) {
        match self {
            Self::Message { pk, time_bound, .. } => (Some(*pk), None, *time_bound),
            Self::Event { pk, .. } | Self::Kept { pk, .. } => (None, Some(*pk), None),
        }
    }
}

/// An indexed message entry the rebuild starts from: entries before it stay.
#[derive(Debug, Clone, Copy)]
struct StartPoint {
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

/// An event as the units place it: where it goes, and what it stores.
struct EventRow {
    pk: i64,
    id: String,
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
        match self.body {
            EventBody::Read {
                kind,
                data_bytes,
                data,
                ..
            } => Unit::Event {
                pk: self.pk,
                id: self.id,
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
                identity,
                entry,
                call_id,
            },
        }
    }
}

const MESSAGE_COLUMNS: &str = "m.pk, m.id,
    coalesce(m.version_hash, (SELECT CAST(count(*) || ':' || coalesce(max(b.pk), 0) AS BLOB)
                              FROM block b WHERE b.message_pk = m.pk)),
    m.on_active_path, m.at,
    EXISTS(SELECT 1 FROM message c WHERE c.session_pk = m.session_pk AND c.parent_id = m.id
           AND c.duplicate_of IS NULL LIMIT 1 OFFSET 1)";
/// The columns an event is read with, `$after` finding the message it
/// follows. The data an entry needs is read only for the kinds whose entry
/// holds it (see [`read_data`]).
macro_rules! event_columns {
    ($after:literal) => {
        concat!(
            "e.pk, e.id, e.kind, e.at, e.data_length, ",
            $after,
            ", e.seq,
    CASE WHEN e.kind IN ('compaction','microcompaction') THEN e.content_group END,
    e.content_start + e.content_length, e.data_length, e.after_message_id"
        )
    };
}
const EVENT_COLUMNS: &str =
    event_columns!("(SELECT m.pk FROM message m WHERE m.id = e.after_message_id)");
/// [`EVENT_COLUMNS`] for a session composed alone, whose events' messages
/// are found from their IDs by the caller ([`Placement::SessionAlone`]).
const EVENT_COLUMNS_ALONE: &str = event_columns!("NULL");

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
    #[cfg(any(test, feature = "fixture"))]
    EVENT_ROWS_READ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(EventRow {
        pk: row.get(0)?,
        id: row.get(1)?,
        at: row.get(3)?,
        after: row.get(5)?,
        seq: row.get(6)?,
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

/// Reads the data of the events whose entry holds it.
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

/// The session's units in reading order from `start_point` (from the first
/// message without one): messages by source position; each event after the
/// message it follows, a timed event that names no message after the last
/// message at or before its time, and the rest at the end. `extra` names
/// changed events to place, in case nothing else finds them.
fn units_from(
    conn: &Connection,
    session_pk: i64,
    start_point: Option<StartPoint>,
    extra: &BTreeSet<i64>,
    placement: Placement<'_>,
) -> Result<Vec<Unit>> {
    // A message with no stored version (rows written directly by the
    // fixture builder) is identified by its block rows.
    let rows: Vec<MessageRow> = match start_point {
        None => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message m
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                 ORDER BY m.session_order, m.id"
            ))?
            .query_map([session_pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
        Some(start_point) => conn
            .prepare_cached(&format!(
                "SELECT {MESSAGE_COLUMNS}
                 FROM message a, message m
                 WHERE a.pk = ?2 AND m.session_pk = ?1 AND m.duplicate_of IS NULL
                   AND (m.session_order, m.id)
                       >= (a.session_order, a.id)
                 ORDER BY m.session_order, m.id"
            ))?
            .query_map(params![session_pk, start_point.pk], message_row)?
            .collect::<rusqlite::Result<_>>()?,
    };
    let events = events_from(conn, session_pk, start_point, &rows, extra, placement)?;
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
                let shown = start_point.is_some()
                    && shown_before
                        .query_row(params![message, session_pk], |row| row.get::<_, bool>(0))
                        .optional()?
                        .unwrap_or(false);
                if !shown {
                    orphaned.push(event.unit());
                }
            }
            (None, Some(at)) => {
                if start_point.is_none_or(|start_point| {
                    start_point.time_bound.is_none_or(|bound| at >= bound)
                }) {
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
    let mut time_bound = start_point.and_then(|start_point| start_point.time_bound);
    for (index, row) in rows.iter().enumerate() {
        // The start point's bound is stored; the first message's is none.
        if index > 0 {
            time_bound = match (time_bound, row.at) {
                (Some(bound), Some(at)) => Some(bound.max(at)),
                (bound, at) => bound.or(at),
            };
        }
        out.push(Unit::Message {
            pk: row.pk,
            id: row.id.clone(),
            at: row.at,
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
    // their order (by place in their input, then ID) at the end with the
    // events that name no message; their row keys depend on the order the
    // rows were written in, so they order nothing.
    out.extend(unplaced);
    out.extend(orphaned);
    Ok(out)
}

/// The events that can be placed from `start_point` on, in stored order: those
/// following a loaded message, the changed ones, and the rest of those indexed
/// from the start point on. The rest are unchanged, so they are placed after
/// the start point again (nothing before it changed); their place comes from
/// the event placement index and what they store from the item index, so the
/// event table is read only for what the write reached.
fn events_from(
    conn: &Connection,
    session_pk: i64,
    start_point: Option<StartPoint>,
    rows: &[MessageRow],
    extra: &BTreeSet<i64>,
    placement: Placement<'_>,
) -> Result<Vec<EventRow>> {
    let Some(start_point) = start_point else {
        if let Placement::SessionAlone(elsewhere) = placement {
            // `rows` are every message of the session: an event whose
            // message is not among them follows one another session shows,
            // or none.
            let pks: HashMap<&str, i64> =
                rows.iter().map(|row| (row.id.as_str(), row.pk)).collect();
            let mut statement = conn.prepare_cached(&format!(
                "SELECT {EVENT_COLUMNS_ALONE} FROM event e WHERE e.session_pk = ?1 ORDER BY e.seq, e.id"
            ))?;
            let mut found = statement.query([session_pk])?;
            let mut events = Vec::new();
            while let Some(row) = found.next()? {
                let mut event = event_row(row)?;
                event.after = row
                    .get_ref(10)?
                    .as_str_or_null()
                    .map_err(rusqlite::Error::from)?
                    .and_then(|id| {
                        pks.get(id)
                            .copied()
                            .or(elsewhere.contains(id).then_some(OTHER_SESSION))
                    });
                events.push(event);
            }
            read_data(conn, &mut events)?;
            return Ok(events);
        }
        let mut events: Vec<EventRow> = conn
            .prepare_cached(&format!(
                "SELECT {EVENT_COLUMNS} FROM event e WHERE e.session_pk = ?1 ORDER BY e.seq, e.id"
            ))?
            .query_map([session_pk], event_row)?
            .collect::<rusqlite::Result<_>>()?;
        read_data(conn, &mut events)?;
        return Ok(events);
    };
    let mut found: HashMap<i64, EventRow> = HashMap::new();
    let mut following = conn.prepare_cached(&format!(
        "SELECT {EVENT_COLUMNS} FROM event e
         WHERE e.after_message_id = ?1 AND e.session_pk = ?3"
    ))?;
    for row in rows {
        for event in following.query_map(params![row.id, row.pk, session_pk], event_row)? {
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
    // Each indexed event's key, with its stored identity, entry and call ID.
    let indexed: Vec<(i64, EventBody)> = conn
        .prepare_cached(
            "SELECT event_pk, unit, entry, call_id FROM session_item
             WHERE session_pk = ?1 AND seq >= ?2 AND event_pk IS NOT NULL",
        )?
        .query_map(params![session_pk, start_point.seq], |row| {
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
        "SELECT e.id, e.at, (SELECT m.pk FROM message m WHERE m.id = e.after_message_id), e.seq
         FROM event e INDEXED BY event_placement WHERE e.session_pk = ?1 AND e.pk = ?2",
    )?;
    for (pk, body) in indexed {
        if found.contains_key(&pk) || extra.contains(&pk) {
            continue;
        }
        // An unchanged indexed event is still in the session; one that is
        // not was changed, and the change summary names it.
        let event = placed
            .query_row(params![session_pk, pk], |row| {
                Ok(EventRow {
                    pk,
                    id: row.get(0)?,
                    at: row.get(1)?,
                    after: row.get(2)?,
                    seq: row.get(3)?,
                    body,
                })
            })
            .optional()?
            .ok_or_else(|| {
                StoreError::internal(format!(
                    "indexed event {pk} left session {session_pk} unchanged"
                ))
            })?;
        found.insert(pk, event);
    }
    let mut events: Vec<EventRow> = found.into_values().collect();
    events.sort_by(|a, b| (a.seq, &a.id).cmp(&(b.seq, &b.id)));
    read_data(conn, &mut events)?;
    Ok(events)
}

/// A session's entries computed from its rows alone, without the stored
/// index: what the stored index must always equal.
#[cfg(test)]
pub(crate) fn computed_entries(conn: &Connection, session_pk: i64) -> Result<Vec<IndexEntry>> {
    let mut out = Vec::new();
    for unit in units_from(conn, session_pk, None, &BTreeSet::new(), Placement::Index)? {
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
        |row| <(_, _)>::try_from(row),
    )?;
    Ok((out, counter(entries)?, decode_counts(&counts)?))
}

/// Brings every session a transaction changed up to date from its change
/// summary; returns the number of units read to do so.
pub(crate) fn refresh(conn: &Connection, changes: &ChangeSummary) -> Result<u64> {
    let mut read = 0;
    let mut exists = conn.prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?;
    for (session_pk, change) in changes.item_sessions(conn)? {
        if exists.query_row([session_pk], |row| row.get::<_, bool>(0))? {
            read += refresh_session(conn, session_pk, &change, changes)?;
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
    // Match the writer's transaction boundary. Opening fixture reads must not
    // commit each session's item and value rows as independent transactions.
    conn.execute_batch("SAVEPOINT fixture_refresh")?;
    let result = (|| {
        for pk in sessions {
            conn.execute_batch("SAVEPOINT fixture_items")?;
            let result = rebuild(conn, pk, None, &BTreeSet::new(), MessageEntries::Fixture);
            match result {
                Ok(_) => conn.execute_batch("RELEASE fixture_items")?,
                Err(error) => {
                    conn.execute_batch("ROLLBACK TO fixture_items; RELEASE fixture_items")?;
                    if !matches!(
                        error,
                        StoreError::Invalid(_)
                            | StoreError::Refused(_)
                            | StoreError::Internal(_)
                            | StoreError::StoredValue { .. }
                            | StoreError::Json(_)
                    ) {
                        return Err(error);
                    }
                }
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => conn.execute_batch("RELEASE fixture_refresh")?,
        Err(error) => {
            conn.execute_batch("ROLLBACK TO fixture_refresh; RELEASE fixture_refresh")?;
            return Err(error);
        }
    }
    Ok(())
}

/// Where a session's rebuild starts.
enum Start {
    /// No unit changed: only the composing links are compared.
    Links,
    Whole,
    From(StartPoint),
}

fn refresh_session(
    conn: &Connection,
    session_pk: i64,
    change: &SessionChange,
    prepared: &ChangeSummary,
) -> Result<u64> {
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
        Start::Whole => rebuild(
            conn,
            session_pk,
            None,
            &change.events,
            MessageEntries::Prepared(prepared),
        ),
        Start::From(start_point) => rebuild(
            conn,
            session_pk,
            Some(start_point),
            &change.events,
            MessageEntries::Prepared(prepared),
        ),
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
    let mut parents: BTreeSet<&str> = change.parents.iter().map(String::as_str).collect();
    for &pk in &change.messages {
        if let Some(seq) = header(conn, session_pk, "message_pk", pk)? {
            lower(&mut limit, seq);
        }
        let order: Option<(String, String)> = shown
            .query_row(params![pk, session_pk], |row| <(_, _)>::try_from(row))
            .optional()?;
        if let Some(order) = order {
            // Its own header and insertion bounds already cover this parent.
            parents.remove(order.1.as_str());
            if first_changed.as_ref().is_none_or(|first| &order < first) {
                first_changed = Some(order);
            }
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
    for id in parents {
        if let Some(pk) = parent
            .query_row(params![id, session_pk], |row| row.get::<_, i64>(0))
            .optional()?
            && let Some(seq) = header(conn, session_pk, "message_pk", pk)?
        {
            lower(&mut limit, seq);
        }
    }
    let mut event = conn.prepare_cached(
        "SELECT (SELECT m.pk FROM message m WHERE m.id = e.after_message_id), e.at,
                EXISTS(SELECT 1 FROM message m WHERE m.id = e.after_message_id
                       AND m.session_pk = e.session_pk AND m.duplicate_of IS NULL)
         FROM event e WHERE e.pk = ?1 AND e.session_pk = ?2",
    )?;
    // A later time places an event at the same entry or a later one, so the
    // earliest time among events placed by time gives the earliest place.
    let mut earliest_timed: Option<i64> = None;
    for &pk in &change.events {
        if let Some(seq) = header(conn, session_pk, "event_pk", pk)? {
            lower(&mut limit, seq);
        }
        let now: Option<(Option<i64>, Option<i64>, bool)> = event
            .query_row(params![pk, session_pk], |row| <(_, _, _)>::try_from(row))
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
            Some((None, Some(at), _)) => {
                earliest_timed = Some(earliest_timed.map_or(at, |old| old.min(at)));
            }
            // At the end.
            Some(_) => lower(&mut limit, end),
        }
    }
    if let Some(at) = earliest_timed {
        keep(&mut limit, timed_place(conn, session_pk, at)?);
    }
    let Some(limit) = limit else {
        return Ok(Start::Links);
    };
    let start_point = conn
        .prepare_cached(
            "SELECT i.seq, i.message_pk, i.time_bound FROM session_item i
             JOIN message m ON m.pk = i.message_pk
             WHERE i.session_pk = ?1 AND i.seq <= ?2 AND i.message_pk IS NOT NULL
               AND m.at IS NOT NULL
             ORDER BY i.seq DESC LIMIT 1",
        )?
        .query_row(params![session_pk, limit], |row| {
            Ok(StartPoint {
                seq: row.get(0)?,
                pk: row.get(1)?,
                time_bound: row.get(2)?,
            })
        })
        .optional()?;
    Ok(start_point.map_or(Start::Whole, Start::From))
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

/// Rebuilds the session's entries from `start_point` (from its first unit without
/// one), keeping those that did not change; returns the units read.
fn rebuild(
    conn: &Connection,
    session_pk: i64,
    start_point: Option<StartPoint>,
    changed_events: &BTreeSet<i64>,
    prepared: MessageEntries<'_>,
) -> Result<u64> {
    let changed: i64 = conn
        .prepare_cached("SELECT changed FROM session WHERE pk = ?1")?
        .query_row([session_pk], |row| row.get(0))?;
    let first_seq = start_point.map_or(0, |start_point| start_point.seq);
    let units = units_from(
        conn,
        session_pk,
        start_point,
        changed_events,
        prepared.placement(),
    )?;
    let identities: Vec<Vec<u8>> = units.iter().map(Unit::identity).collect::<Result<_>>()?;
    let indexed: Vec<(i64, Vec<u8>)> = conn
        .prepare_cached(
            "SELECT seq, unit FROM session_item
             WHERE session_pk = ?1 AND seq >= ?2 AND unit IS NOT NULL ORDER BY seq",
        )?
        .query_map(params![session_pk, first_seq], |row| {
            <(_, _)>::try_from(row)
        })?
        .collect::<rusqlite::Result<_>>()?;
    let state: Option<(i64, i64, Vec<u8>, Vec<u8>)> = conn
        .prepare_cached(
            "SELECT append_seq, rewrite_version, links, counts FROM session_item_state
             WHERE session_pk = ?1",
        )?
        .query_row([session_pk], |row| <(_, _, _, _)>::try_from(row))
        .optional()?;
    let links = links_identity(conn, session_pk)?;
    let mut same = 0;
    for (((_, stored), identity), unit) in indexed.iter().zip(&identities).zip(&units) {
        if stored != identity {
            break;
        }
        if let Unit::Message { pk, .. } = unit
            && prepared.has_prepared_items(*pk)
            && stored_message_entries(conn, *pk)? != unit_entries(conn, unit, prepared)?
        {
            break;
        }
        same += 1;
    }
    // Capture stored metadata before deleting a rewritten suffix. New content
    // comes from worker preparation; placement-only changes reuse stored entries.
    let pending = units
        .iter()
        .skip(same)
        .map(|unit| unit_entries(conn, unit, prepared))
        .collect::<Result<Vec<_>>>()?;
    let links_moved = state
        .as_ref()
        .is_some_and(|(_, _, stored, _)| stored.as_slice() != links.as_slice());
    // A unit's first entry, and each later one, which holds only its entry.
    let mut insert = conn.prepare_cached(
        "INSERT INTO session_item(session_pk, seq, unit, message_pk, event_pk, time_bound, entry,
                                  class, call_id, counts)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;
    let mut insert_later = conn.prepare_cached(
        "INSERT INTO session_item(session_pk, seq, entry, class, call_id) VALUES (?1, ?2, ?3, ?4, ?5)",
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
    let mut extended = false;
    // A live session's last message grows block by block. When the only
    // difference is new blocks after the last indexed message's entries,
    // that is an append; outstanding cursors stay valid. The update below
    // keeps the row's message or event, so only that same row can grow.
    if same + 1 == indexed.len()
        && same < units.len()
        && located_by(conn, session_pk, indexed[same].0)? == located_columns(&units[same])
    {
        let (first_seq, _) = indexed[same];
        let stored = stored_entries(conn, session_pk, first_seq)?;
        let current = &pending[0];
        if extends(&stored, current) {
            let (_, _, time_bound) = units[same].columns();
            let classes = entry_classes(current);
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
                encode(&current[0], None)?,
                classes[0].bits()
            ])?;
            for (entry, class) in current.iter().zip(&classes).skip(stored.len()) {
                insert_later.execute(params![
                    session_pk,
                    next,
                    encode(entry, Some(&current[0]))?,
                    class.bits(),
                    entry.call_id,
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
            extended = true;
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
    for ((unit, identity), entries) in units
        .iter()
        .zip(&identities)
        .skip(same)
        .zip(pending.iter().skip(usize::from(extended)))
    {
        let (message_pk, event_pk, time_bound) = unit.columns();
        let before = encode_counts(&running);
        for (offset, (entry, class)) in entries.iter().zip(entry_classes(entries)).enumerate() {
            if offset == 0 {
                insert.execute(params![
                    session_pk,
                    next,
                    identity,
                    message_pk,
                    event_pk,
                    time_bound,
                    encode(entry, None)?,
                    class.bits(),
                    entry.call_id,
                    before
                ])?;
            } else {
                insert_later.execute(params![
                    session_pk,
                    next,
                    encode(entry, Some(&entries[0]))?,
                    class.bits(),
                    entry.call_id,
                ])?;
            }
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

/// The message and event a unit's first row names.
fn located_by(conn: &Connection, session_pk: i64, seq: i64) -> Result<(Option<i64>, Option<i64>)> {
    Ok(conn
        .prepare_cached(
            "SELECT message_pk, event_pk FROM session_item WHERE session_pk = ?1 AND seq = ?2",
        )?
        .query_row(params![session_pk, seq], |row| <(_, _)>::try_from(row))?)
}

fn located_columns(unit: &Unit) -> (Option<i64>, Option<i64>) {
    let (message_pk, event_pk, _) = unit.columns();
    (message_pk, event_pk)
}

fn unit_entries(
    conn: &Connection,
    unit: &Unit,
    prepared: MessageEntries<'_>,
) -> Result<Vec<IndexEntry>> {
    match unit {
        Unit::Message {
            pk, id, at, place, ..
        } => {
            let mut entries = match prepared {
                MessageEntries::Prepared(changes) => match changes.prepared_items(*pk) {
                    Some(entries) => entries,
                    None if changes.reuses_items(*pk) => stored_message_entries(conn, *pk)?,
                    None => {
                        return Err(StoreError::internal(
                            "changed message items were not prepared",
                        ));
                    }
                },
                #[cfg(any(test, feature = "fixture"))]
                MessageEntries::Fixture => {
                    let (message, ordinals) = message_body(conn, *pk)?;
                    message_entries(&message, &ordinals, MessagePlace::default())
                }
            };
            if entries.is_empty() {
                return Err(StoreError::stored_data(
                    "changed message items were not prepared",
                ));
            }
            let id = MessageId::parse(id).map_err(crate::StoreError::stored_data)?;
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

/// The event entry decides both the stored items and their fingerprint.
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

/// Item metadata kept for one current unit, without visiting payload rows.
fn stored_message_entries(conn: &Connection, pk: i64) -> Result<Vec<IndexEntry>> {
    let bounds: Option<(i64, i64, Option<i64>)> = conn
        .prepare_cached(
            "SELECT head.session_pk,head.seq,
           (SELECT next.seq FROM session_item next WHERE next.session_pk=head.session_pk
             AND next.seq>head.seq AND next.unit IS NOT NULL ORDER BY next.seq LIMIT 1)
         FROM session_item head WHERE head.message_pk=?1",
        )?
        .query_row([pk], |row| <(_, _, _)>::try_from(row))
        .optional()?;
    let Some((session, from, until)) = bounds else {
        return Ok(Vec::new());
    };
    let until = until.unwrap_or(i64::MAX);
    Ok(
        read_entries(conn, session, from..until, true, usize::MAX, |_| true)?
            .into_iter()
            .map(|stored| stored.entry)
            .collect(),
    )
}

/// A visibility derivation may reveal content whose item rows are no longer
/// present. Its worker prepares the shown value under its snapshot;
/// the existing derive token protects these entries when the writer applies it.
pub(crate) fn prepare_stored_message(conn: &Connection, pk: i64) -> Result<Vec<IndexEntry>> {
    let (mut message, block_selection) = crate::writer::shown_message(conn, pk)?
        .ok_or_else(|| StoreError::internal("revealed message has no kept selected version"))?;
    conn.prepare_cached(
        "SELECT m.id,s.id,m.role,m.prompt_kind,m.model,m.at,m.input_tokens,m.cache_read_tokens,m.cache_write_tokens
         FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.pk=?1"
    )?.query_row([pk], |row| {
        message.id = row.id::<MessageId>(0)?;
        message.session = row.id::<SessionId>(1)?;
        message.role = row.name::<Role>(2)?;
        message.prompt_kind = row.optional_name::<PromptKind>(3)?;
        message.model = row.get(4)?;
        message.at = row.optional_time_ms(5)?;
        message.usage.input = row.get(6)?;
        message.usage.cached_input = row.get(7)?;
        message.usage.cache_write = row.get(8)?;
        Ok(())
    })?;
    Ok(message_entries(
        &message,
        &block_selection.block_ordinals,
        MessagePlace::default(),
    ))
}

/// A message's row and its blocks with their original ordinals: the fields
/// the index estimates from. Structured tool data is not read.
#[cfg(any(test, feature = "fixture"))]
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

/// The stored entries of a session from `from` on.
pub(crate) fn stored_entries(
    conn: &Connection,
    session_pk: i64,
    from: i64,
) -> Result<Vec<IndexEntry>> {
    Ok(
        read_entries(conn, session_pk, from..i64::MAX, true, usize::MAX, |_| true)?
            .into_iter()
            .map(|stored| stored.entry)
            .collect(),
    )
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

/// What a read of stored entries selects: the row, and the message or
/// event its unit's first row names.
const ENTRY_SELECT: &str =
    "SELECT i.seq, i.class, i.entry, i.call_id, m.id, e.id, i.unit IS NOT NULL FROM session_item i";
const ENTRY_JOINS: &str =
    "LEFT JOIN message m ON m.pk = i.message_pk LEFT JOIN event e ON e.pk = i.event_pk";

/// The entries in `range` of one session, forward from its start or
/// backward from its end, at most `limit` of those whose class `wanted`
/// accepts. Rows it does not accept are passed over without being decoded.
fn read_entries(
    conn: &Connection,
    session_pk: i64,
    range: std::ops::Range<i64>,
    forward: bool,
    limit: usize,
    wanted: impl Fn(EntryClass) -> bool,
) -> Result<Vec<StoredEntry>> {
    let order = if forward { "" } else { " DESC" };
    let mut statement = conn.prepare_cached(&format!(
        "{ENTRY_SELECT} {ENTRY_JOINS}
         WHERE i.session_pk = ?1 AND i.seq >= ?2 AND i.seq < ?3 ORDER BY i.seq{order}"
    ))?;
    let mut rows = statement.query(params![session_pk, range.start, range.end])?;
    let mut run = Run::new(conn, session_pk);
    while run.rows.len() < limit
        && let Some(row) = rows.next()?
    {
        run.visit(row, &wanted)?;
    }
    run.finish()
}

/// The first row of a message unit: what the message's later entries take
/// from it.
struct MessageStart {
    /// Parsed only when a later entry or the row itself is decoded.
    id: String,
    entry: Vec<u8>,
    call_id: Option<String>,
}

/// Where a kept row's entry comes from.
enum RowPlace {
    /// A message's first row, kept among the unit starts.
    MessageStart,
    Event(String, Vec<u8>),
    /// A later entry of a message.
    Block(Vec<u8>),
}

struct KeptRow {
    seq: i64,
    class: EntryClass,
    call_id: Option<String>,
    place: RowPlace,
}

/// The rows of one contiguous run of a session's index, read in either
/// direction. A later entry of a message is decoded with the message's
/// first row: the nearest unit start at or before it in the run, or, for
/// the entries before the run's first unit start, the one the index holds
/// before the run.
struct Run<'c> {
    conn: &'c Connection,
    session_pk: i64,
    /// Every unit start the run passed, by place: a message's first row,
    /// or `None` for an event.
    starts: BTreeMap<i64, Option<MessageStart>>,
    rows: Vec<KeptRow>,
}

impl<'c> Run<'c> {
    fn new(conn: &'c Connection, session_pk: i64) -> Self {
        Self {
            conn,
            session_pk,
            starts: BTreeMap::new(),
            rows: Vec::new(),
        }
    }

    /// Reads one row of [`ENTRY_SELECT`], keeping it when `wanted` accepts
    /// its class.
    fn visit(
        &mut self,
        row: &rusqlite::Row<'_>,
        wanted: &impl Fn(EntryClass) -> bool,
    ) -> Result<()> {
        let seq: i64 = row.get(0)?;
        let class = class_of(row.get(1)?)?;
        let keep = wanted(class);
        let message: Option<String> = row.get(4)?;
        let event: Option<String> = row.get(5)?;
        let starts_unit: bool = row.get(6)?;
        let place = match (message, event) {
            (Some(id), _) => {
                self.starts.insert(
                    seq,
                    Some(MessageStart {
                        id,
                        entry: row.get(2)?,
                        call_id: row.get(3)?,
                    }),
                );
                RowPlace::MessageStart
            }
            (None, Some(id)) => {
                self.starts.insert(seq, None);
                if !keep {
                    return Ok(());
                }
                RowPlace::Event(id, row.get(2)?)
            }
            (None, None) if starts_unit => {
                return Err(StoreError::stored_data(format!(
                    "session item {seq} names no stored message or event"
                )));
            }
            (None, None) => {
                if !keep {
                    return Ok(());
                }
                RowPlace::Block(row.get(2)?)
            }
        };
        if keep {
            self.rows.push(KeptRow {
                seq,
                class,
                call_id: row.get(3)?,
                place,
            });
        }
        Ok(())
    }

    /// The kept rows' entries, in the order they were read.
    fn finish(mut self) -> Result<Vec<StoredEntry>> {
        let mut headers: HashMap<i64, IndexEntry> = HashMap::new();
        let mut out = Vec::with_capacity(self.rows.len());
        for row in std::mem::take(&mut self.rows) {
            let entry = match row.place {
                RowPlace::MessageStart => self.header(&mut headers, row.seq)?.clone(),
                RowPlace::Event(id, entry) => decode(&entry, EntryOf::Event(id), row.call_id)?,
                RowPlace::Block(entry) => {
                    let start = match self.starts.range(..=row.seq).next_back() {
                        Some((start, _)) => *start,
                        None => self.start_before(row.seq)?,
                    };
                    let header = self.header(&mut headers, start)?;
                    decode(&entry, EntryOf::Header(header), row.call_id)?
                }
            };
            out.push(StoredEntry {
                seq: counter(row.seq)?,
                class: row.class,
                entry,
            });
        }
        Ok(out)
    }

    /// The decoded header of the message whose first row is at `start`.
    fn header<'h>(
        &self,
        headers: &'h mut HashMap<i64, IndexEntry>,
        start: i64,
    ) -> Result<&'h IndexEntry> {
        if let std::collections::hash_map::Entry::Vacant(vacant) = headers.entry(start) {
            let Some(Some(message)) = self.starts.get(&start) else {
                return Err(StoreError::stored_data(format!(
                    "session item {start} is not the first entry of a message"
                )));
            };
            vacant.insert(decode(
                &message.entry,
                EntryOf::Message(message_id(&message.id)?),
                message.call_id.clone(),
            )?);
        }
        Ok(&headers[&start])
    }

    /// Finds the unit start at or before `seq` in the index, records it and
    /// returns its place.
    fn start_before(&mut self, seq: i64) -> Result<i64> {
        // Its place, entry, call ID and message ID.
        type StartRow = (i64, Vec<u8>, Option<String>, Option<String>);
        let found: Option<StartRow> = self
            .conn
            .prepare_cached(
                "SELECT i.seq, i.entry, i.call_id, m.id
                 FROM session_item i LEFT JOIN message m ON m.pk = i.message_pk
                 WHERE i.session_pk = ?1 AND i.seq <= ?2 AND i.unit IS NOT NULL
                 ORDER BY i.seq DESC LIMIT 1",
            )?
            .query_row(params![self.session_pk, seq], |row| {
                <(_, _, _, _)>::try_from(row)
            })
            .optional()?;
        let Some((start, entry, call_id, Some(id))) = found else {
            return Err(StoreError::stored_data(format!(
                "session item {seq} follows no message header"
            )));
        };
        self.starts
            .insert(start, Some(MessageStart { id, entry, call_id }));
        Ok(start)
    }
}

fn message_id(id: &str) -> Result<MessageId> {
    MessageId::parse(id).map_err(crate::StoreError::stored_data)
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
        None => Err(StoreError::stored_data(format!(
            "session item {seq} is not the first entry of a unit"
        ))),
    }
}

/// Counts as stored: unsigned LEB128 numbers, the running full-form tokens
/// first, then each class with a count above zero and its count, by class.
fn encode_counts(counts: &ItemCounts) -> Vec<u8> {
    use entry_bytes::number;
    let mut out = Vec::with_capacity(4 + counts.classes.len() * 4);
    number(&mut out, counts.tokens_full);
    for (class, count) in &counts.classes {
        number(&mut out, u64::from(class.bits()));
        number(&mut out, *count);
    }
    out
}

fn decode_counts(bytes: &[u8]) -> Result<ItemCounts> {
    let malformed = || StoreError::stored_data("malformed session item counts");
    let mut input = entry_bytes::Input { bytes, at: 0 };
    let mut counts = ItemCounts {
        tokens_full: input.leb128().ok_or_else(malformed)?,
        ..ItemCounts::default()
    };
    while !input.done() {
        let class = input
            .leb128()
            .and_then(|bits| u16::try_from(bits).ok())
            .and_then(EntryClass::from_bits)
            .ok_or_else(malformed)?;
        let count = input.leb128().ok_or_else(malformed)?;
        if counts.classes.insert(class, count).is_some() {
            return Err(malformed());
        }
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
        .ok_or_else(|| StoreError::stored_data(format!("unknown session item class {bits}")))
}

fn position(seq: u64) -> Result<i64> {
    crate::to_i64(seq, "item index position out of range")
}

fn counter(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| crate::StoreError::stored(0, "negative item index counter"))
}

/// Recorded parent history: a branch from a branch point, or messages
/// copied into the child's input. Copying bytes does not create a provider relationship.
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
        let conn = self.conn()?;
        let mut parents: BTreeSet<String> = conn
            .prepare_cached("SELECT DISTINCT parent_id FROM prefix_copy WHERE child_id=?1")?
            .query_map([id.to_string()], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        parents.extend(
            crate::lineage::inherited_prefixes(conn, &id.to_string())?
                .into_iter()
                .filter(|(_, branch_point, copies)| *copies && branch_point.is_none())
                .map(|(parent, _, _)| parent),
        );
        parents
            .iter()
            .map(|parent| SessionId::parse(parent).map_err(StoreError::stored_data))
            .collect()
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
                |row| <(_, _, _, _, _)>::try_from(row),
            )
            .optional()?;
        let Some((input_pk, input, Some(chain), Some(start), end)) = found else {
            return Ok(None);
        };
        let end = match end.filter(|end| *end > start) {
            Some(end) => end,
            None => conn.scalar("SELECT coalesce(
                    (SELECT min(s) FROM (
                        SELECT source_start AS s FROM message_parse WHERE input_pk = ?1 AND chain = ?2 AND source_start > ?3
                        UNION ALL
                        SELECT source_start FROM event WHERE input_pk = ?1 AND source_chain = ?2 AND source_start > ?3)),
                    (SELECT parsed_to FROM parse_state WHERE input_pk = ?1 AND chain = ?2),
                    ?3)", params![input_pk, chain, start])?,
        };
        let (Ok(start), Ok(end)) = (u64::try_from(start), u64::try_from(end)) else {
            return Err(StoreError::stored_data(format!(
                "{id} has a negative source range"
            )));
        };
        let bytes = self.stored_range(&input, chain, start, end, &self.blob_reader()?)?;
        Ok(Some(StoredRecord { input, bytes }))
    }

    /// The recorded prefixes that a physical reading of `id` inherits.
    pub fn inherited_prefixes(&self, id: &SessionId) -> Result<Vec<InheritedPrefix>> {
        Ok(
            crate::lineage::inherited_prefixes(self.conn()?, &id.to_string())?
                .into_iter()
                .map(|(parent, branch_point, copies_prefix)| InheritedPrefix {
                    parent,
                    copies_prefix,
                    branch_point,
                })
                .collect(),
        )
    }

    /// A session's whole item index, or `None` when the session is not
    /// stored. One range scan.
    pub fn item_index(&self, id: &SessionId) -> Result<Option<StoredItems>> {
        let Some(crate::row::SessionPk(pk)) = self.session_pk(id)? else {
            return Ok(None);
        };
        let Some(versions) = self.item_versions_of(pk)? else {
            return Err(StoreError::internal(format!(
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
        let Some(crate::row::SessionPk(pk)) = self.session_pk(id)? else {
            return Ok(None);
        };
        let state: Option<(i64, i64, i64, Vec<u8>)> = self
            .conn()?
            .prepare_cached(
                "SELECT append_seq, rewrite_version, entries, counts FROM session_item_state
                 WHERE session_pk = ?1",
            )?
            .query_row([pk], |row| <(_, _, _, _)>::try_from(row))
            .optional()?;
        let Some((append_seq, rewrite_version, entries, counts)) = state else {
            return Err(StoreError::stored_data(format!(
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
        let out = read_entries(
            self.conn()?,
            session_pk,
            position(range.start)?..position(range.end)?,
            forward,
            limit,
            wanted,
        )?;
        #[cfg(any(test, feature = "fixture"))]
        ENTRIES_DECODED.fetch_add(out.len() as u64, std::sync::atomic::Ordering::Relaxed);
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
                <(_, _)>::try_from(row)
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
        // Named, because without table statistics the planner reads every
        // entry of the session through the primary key instead.
        let mut statement = conn.prepare_cached(&format!(
            "{ENTRY_SELECT} INDEXED BY session_item_call {ENTRY_JOINS}
             WHERE i.session_pk = ?1 AND i.call_id = ?2 ORDER BY i.seq"
        ))?;
        let mut rows = statement.query(params![session_pk, call_id])?;
        let mut out = Vec::new();
        // The rows are apart, so each is its own run.
        while let Some(row) = rows.next()? {
            #[cfg(any(test, feature = "fixture"))]
            ENTRIES_DECODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut run = Run::new(conn, session_pk);
            run.visit(row, &|_| true)?;
            out.extend(run.finish()?);
        }
        Ok(out)
    }

    /// Where the call with each of `call_ids` sits in a session's index, for
    /// those the session holds: one statement, and no entry is decoded.
    pub fn item_call_places(
        &self,
        session_pk: i64,
        call_ids: &[&str],
    ) -> Result<HashMap<String, u64>> {
        let mut places = HashMap::new();
        if call_ids.is_empty() {
            return Ok(places);
        }
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT i.call_id, i.seq, i.class
             FROM json_each(?2) c CROSS JOIN session_item i INDEXED BY session_item_call
             WHERE i.session_pk = ?1 AND i.call_id = c.value",
        )?;
        let mut rows = statement.query(params![session_pk, serde_json::to_string(call_ids)?])?;
        while let Some(row) = rows.next()? {
            if !class_of(row.get(2)?)?.is_call() {
                continue;
            }
            let seq = counter(row.get(1)?)?;
            places
                .entry(row.get::<_, String>(0)?)
                .and_modify(|held: &mut u64| *held = (*held).min(seq))
                .or_insert(seq);
        }
        Ok(places)
    }

    pub fn item_versions_of(&self, session_pk: i64) -> Result<Option<ItemVersions>> {
        self.conn()?
            .prepare_cached(
                "SELECT append_seq, rewrite_version FROM session_item_state WHERE session_pk = ?1",
            )?
            .query_row([session_pk], |row| <(_, _)>::try_from(row))
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
