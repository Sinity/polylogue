//! The session item index (`session_item`, `session_item_state`). Every write
//! transaction that changes a session's rows brings that session's entries
//! up to date before it commits, so the index is never behind the rows.
//!
//! A refresh compares the session's units (message versions and events, in
//! reading order) with the units already indexed. Entries before the first
//! difference stay; entries from there on are rebuilt from the bodies. When
//! the difference is only new units at the end, `append_seq` moves; when an
//! indexed unit changed, moved or disappeared, or the links that compose the
//! session changed, `rewrite_version` moves and outstanding cursors go stale.
use crate::encoding::{NamedColumns, restore_text, unpack_block, unpack_text};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    Message, MessageId, PromptKind, Role, SessionId, Usage,
    read::{IndexEntry, MessagePlace, event_entry, message_entries, microcompacted_calls},
};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashMap;

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
        id: String,
        version: Vec<u8>,
        place: MessagePlace,
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
}

/// The session's units in reading order: messages by source position; each
/// event after the message it follows, and events without one at the end.
fn units(conn: &Connection, session_pk: i64) -> Result<Vec<Unit>> {
    struct Row {
        pk: i64,
        id: String,
        version: Vec<u8>,
        active: Option<bool>,
        parent: Option<String>,
        at: Option<i64>,
    }
    // A message with no stored version (rows written directly by the
    // fixture builder) is identified by its block rows.
    let rows: Vec<Row> = conn
        .prepare_cached(
            "SELECT m.pk, m.id,
                    coalesce(s.version_hash, (SELECT CAST(count(*) || ':' || coalesce(max(b.pk), 0) AS BLOB)
                                              FROM block b WHERE b.message_pk = m.pk)),
                    m.on_active_path, m.parent_id, m.at
             FROM message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
             WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
             ORDER BY m.order_input_id, m.seq, m.record_position, m.id",
        )?
        .query_map([session_pk], |row| {
            Ok(Row {
                pk: row.get(0)?,
                id: row.get(1)?,
                version: row.get(2)?,
                active: row.get::<_, Option<i64>>(3)?.map(|value| value != 0),
                parent: row.get(4)?,
                at: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut replies: HashMap<&str, u32> = HashMap::new();
    for row in &rows {
        if let Some(parent) = &row.parent {
            *replies.entry(parent.as_str()).or_default() += 1;
        }
    }
    let mut after: HashMap<i64, Vec<Unit>> = HashMap::new();
    let mut timed: Vec<(i64, Unit)> = Vec::new();
    let mut unplaced = Vec::new();
    let mut events = conn.prepare_cached(
        "SELECT e.pk, e.id, e.kind, e.at, e.after_message_pk, length(e.data)
         FROM event e WHERE e.session_pk = ?1 ORDER BY e.seq, e.id",
    )?;
    let mut event_rows = events.query([session_pk])?;
    while let Some(row) = event_rows.next()? {
        let unit = Unit::Event {
            pk: row.get(0)?,
            id: row.get(1)?,
            kind: row.get(2)?,
            at: row.get(3)?,
            data_bytes: row.get::<_, i64>(5)?.max(0) as u64,
        };
        let at: Option<i64> = row.get(3)?;
        match (row.get::<_, Option<i64>>(4)?, at) {
            (Some(message), _) => after.entry(message).or_default().push(unit),
            (None, Some(at)) => timed.push((at, unit)),
            (None, None) => unplaced.push(unit),
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
    for (index, row) in rows.iter().enumerate() {
        let next_at = next_times[index];
        out.push(Unit::Message {
            pk: row.pk,
            id: row.id.clone(),
            version: row.version.clone(),
            place: MessagePlace {
                // Unknown and on the path read the same; only off the path
                // changes what a reading shows.
                active_path: (row.active == Some(false)).then_some(false),
                branch_point: replies.get(row.id.as_str()).is_some_and(|n| *n > 1),
            },
        });
        if let Some(events) = after.remove(&row.pk) {
            out.extend(events);
        }
        // Timed events before the next timed message belong here; those
        // after the last timed message go at the end.
        if let Some(next) = next_at {
            while timed.peek().is_some_and(|(at, _)| *at < next) {
                out.push(timed.next().expect("peeked").1);
            }
        }
    }
    out.extend(timed.map(|(_, unit)| unit));
    // Events whose message is not shown (a duplicate, or not stored) keep
    // their order at the end with the events that name no message.
    let mut orphaned: Vec<Unit> = after.into_values().flatten().collect();
    orphaned.sort_by_key(|unit| match unit {
        Unit::Event { pk, .. } => *pk,
        Unit::Message { pk, .. } => *pk,
    });
    out.extend(unplaced);
    out.extend(orphaned);
    Ok(out)
}

fn unit_entries(conn: &Connection, unit: &Unit) -> Result<Vec<IndexEntry>> {
    match unit {
        Unit::Message { pk, id, place, .. } => {
            let (message, ordinals) = message_body(conn, *pk, id)?;
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
fn message_body(conn: &Connection, pk: i64, id: &str) -> Result<(Message, Vec<u64>)> {
    let invalid = |what: &str| StoreError::Invalid(format!("message {id}: {what}"));
    let id = MessageId::parse(id).map_err(|_| invalid("its ID does not parse"))?;
    let (session, role, prompt_kind, model, at, tokens) = conn
        .prepare_cached(
            "SELECT s.id, m.role, m.prompt_kind, m.model, m.at,
                    m.input_tokens, m.cache_read_tokens, m.cache_write_tokens
             FROM message m JOIN session s ON s.pk = m.session_pk WHERE m.pk = ?1",
        )?
        .query_row([pk], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.name::<Role>(1)?,
                row.optional_name::<PromptKind>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<i64>>(4)?,
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
        ordinals.push(
            u64::try_from(row.get::<_, i64>(2)?)
                .map_err(|_| invalid("a block has a negative position"))?,
        );
    }
    let message = Message {
        id,
        session: SessionId::parse(&session)
            .map_err(|_| invalid("its session ID does not parse"))?,
        parent: None,
        role,
        at: crate::read::ts(at),
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

/// A session's entries computed from its rows alone, without the stored
/// index: what the stored index must always equal.
#[cfg(test)]
pub(crate) fn computed_entries(conn: &Connection, session_pk: i64) -> Result<Vec<IndexEntry>> {
    let mut out = Vec::new();
    for unit in units(conn, session_pk)? {
        out.extend(unit_entries(conn, &unit)?);
    }
    Ok(out)
}

/// Bring every session this transaction changed up to date.
pub(crate) fn refresh_changed(conn: &Connection, counter: i64) -> Result<()> {
    let sessions: Vec<i64> = conn
        .prepare_cached("SELECT pk FROM session WHERE changed = ?1")?
        .query_map([counter], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for pk in sessions {
        refresh_session(conn, pk)?;
    }
    Ok(())
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
        match refresh_session(conn, pk) {
            Ok(()) => conn.execute_batch("RELEASE fixture_items")?,
            Err(error) => {
                conn.execute_batch("ROLLBACK TO fixture_items; RELEASE fixture_items")?;
                if !matches!(error, StoreError::Invalid(_) | StoreError::Json(_)) {
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

/// Bring one session's entries up to date with its rows.
pub(crate) fn refresh_session(conn: &Connection, session_pk: i64) -> Result<()> {
    let changed: i64 = conn.query_row(
        "SELECT changed FROM session WHERE pk = ?1",
        [session_pk],
        |row| row.get(0),
    )?;
    let units = units(conn, session_pk)?;
    let identities: Vec<Vec<u8>> = units.iter().map(Unit::identity).collect();
    let indexed: Vec<(i64, Vec<u8>)> = conn
        .prepare_cached(
            "SELECT seq, unit FROM session_item
             WHERE session_pk = ?1 AND unit IS NOT NULL ORDER BY seq",
        )?
        .query_map([session_pk], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let state: Option<(i64, i64, Vec<u8>)> = conn
        .query_row(
            "SELECT append_seq, rewrite_version, links FROM session_item_state WHERE session_pk = ?1",
            [session_pk],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
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
        "INSERT INTO session_item(session_pk, seq, unit, entry) VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut next: i64 = conn.query_row(
        "SELECT coalesce(max(seq) + 1, 0) FROM session_item WHERE session_pk = ?1",
        [session_pk],
        |row| row.get(0),
    )?;
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
            conn.prepare_cached(
                "UPDATE session_item SET unit = ?3, entry = ?4 WHERE session_pk = ?1 AND seq = ?2",
            )?
            .execute(params![
                session_pk,
                first_seq,
                identities[same],
                serde_json::to_vec(&current[0])?
            ])?;
            for entry in &current[stored.len()..] {
                insert.execute(params![
                    session_pk,
                    next,
                    None::<Vec<u8>>,
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
        for (offset, entry) in unit_entries(conn, unit)?.into_iter().enumerate() {
            insert.execute(params![
                session_pk,
                next,
                (offset == 0).then_some(identity.as_slice()),
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
    Ok(())
}

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
