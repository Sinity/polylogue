//! Reading transcript lines (`transcript_line`, `transcript_call`), which the
//! session summary writes ([`crate::summary::lines`]): a session's lines in
//! a range either way, their counts before a place, the lines of one kind,
//! and a call's lines. A read of a session whose summary is due asks the
//! summary worker for that session first and waits for it, so a read made
//! right after a commit shows what the commit wrote.
use crate::ConnExt as _;
use crate::summary::lines::LineState;
use crate::{Reader, Result, StoreError};
use entry_bytes::{EntryOf, decode};
use polylogue_model::{
    MessageId, SessionId,
    read::{EntryClass, IndexEntry, ItemCounts},
};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) mod entry_bytes;

/// A session's lines and the two counters cursors bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredItems {
    pub session_pk: i64,
    pub append_seq: u64,
    pub rewrite_version: u64,
    pub entries: Vec<IndexEntry>,
}

/// What a session reading needs of a session before reading its lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemState {
    pub session_pk: i64,
    pub versions: ItemVersions,
    /// How many lines the session has.
    pub entries: u64,
    /// The counts over every line.
    pub counts: ItemCounts,
}

/// One stored line, with its place and class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEntry {
    pub seq: u64,
    pub class: EntryClass,
    pub entry: IndexEntry,
}

/// The kinds of line a reading finds through their own index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemLookup {
    Prompt,
    Compaction,
    Microcompaction,
}

/// Lines decoded by transcript reads in this process: what a test counts
/// to show a reading reads only near its page.
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

impl ItemVersions {
    fn of(state: &LineState) -> Result<Self> {
        Ok(Self {
            append_seq: counter(state.append_seq)?,
            rewrite_version: counter(state.rewrite_version)?,
        })
    }
}

/// How many times one reader waits for summaries before it reads the lines
/// as stored: a session written to faster than its summary is computed
/// does not hold a read without a deadline forever.
const SUMMARY_WAITS: u64 = 8;

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

/// What a read of stored entries selects: the row, and the message or
/// event its unit's first row names.
const ENTRY_SELECT: &str = "SELECT i.seq, i.class, i.entry, i.call_id, m.id, e.id, i.unit IS NOT NULL FROM transcript_line i";
const ENTRY_JOINS: &str =
    "LEFT JOIN message m ON m.pk = i.message_pk LEFT JOIN event e ON e.pk = i.event_pk";

/// The entries in `range` of one session, forward from its start or
/// backward from its end, at most `limit` of those whose class `wanted`
/// accepts. Rows it does not accept are passed over without being decoded.
pub(crate) fn read_entries(
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
                    "transcript line {seq} names no stored message or event"
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
                    "transcript line {start} is not the first line of a message"
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
                 FROM transcript_line i LEFT JOIN message m ON m.pk = i.message_pk
                 WHERE i.session_pk = ?1 AND i.seq <= ?2 AND i.unit IS NOT NULL
                 ORDER BY i.seq DESC LIMIT 1",
            )?
            .query_row(params![self.session_pk, seq], |row| {
                <(_, _, _, _)>::try_from(row)
            })
            .optional()?;
        let Some((start, entry, call_id, Some(id))) = found else {
            return Err(StoreError::stored_data(format!(
                "transcript line {seq} follows no message header"
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

/// Counts as stored: unsigned LEB128 numbers, the running full-form tokens
/// first, then each class with a count above zero and its count, by class.
pub(crate) fn encode_counts(counts: &ItemCounts) -> Vec<u8> {
    use entry_bytes::number;
    let mut out = Vec::with_capacity(4 + counts.classes.len() * 4);
    number(&mut out, counts.tokens_full);
    for (class, count) in &counts.classes {
        number(&mut out, u64::from(class.bits()));
        number(&mut out, *count);
    }
    out
}

pub(crate) fn decode_counts(bytes: &[u8]) -> Result<ItemCounts> {
    let malformed = || StoreError::stored_data("malformed transcript line counts");
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
        ItemLookup::Prompt => "INDEXED BY transcript_line_prompt WHERE class & 63 = 32",
        ItemLookup::Compaction => "INDEXED BY transcript_line_compaction WHERE class & 15 = 9",
        ItemLookup::Microcompaction => {
            "INDEXED BY transcript_line_microcompaction WHERE class & 15 = 10"
        }
    }
}

fn class_of(bits: i64) -> Result<EntryClass> {
    u16::try_from(bits)
        .ok()
        .and_then(EntryClass::from_bits)
        .ok_or_else(|| StoreError::stored_data(format!("unknown transcript line class {bits}")))
}

fn position(seq: u64) -> Result<i64> {
    crate::to_i64(seq, "transcript line position out of range")
}

fn counter(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| crate::StoreError::stored(0, "negative transcript line counter"))
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
                    (SELECT parsed_upto FROM parse_state WHERE input_pk = ?1 AND chain = ?2),
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

    /// A session's lines, or `None` when the session is not stored. One
    /// range scan, after the session's summary when it is due.
    pub fn item_index(&self, id: &SessionId) -> Result<Option<StoredItems>> {
        let Some(crate::row::SessionPk(pk)) = self.session_pk(id)? else {
            return Ok(None);
        };
        self.summarized(pk)?;
        let Some(state) = crate::summary::lines::state(self.conn()?, pk)? else {
            return Err(StoreError::stored_data(format!(
                "session {id} has no transcript lines"
            )));
        };
        let versions = ItemVersions::of(&state)?;
        Ok(Some(StoredItems {
            session_pk: pk,
            append_seq: versions.append_seq,
            rewrite_version: versions.rewrite_version,
            entries: stored_entries(self.conn()?, pk, 0)?,
        }))
    }

    /// What a session reading needs of a session before reading any of its
    /// lines, or `None` when the session is not stored: read after the
    /// session's summary when it is due.
    pub fn item_state(&self, id: &SessionId) -> Result<Option<ItemState>> {
        let Some(crate::row::SessionPk(pk)) = self.session_pk(id)? else {
            return Ok(None);
        };
        self.summarized(pk)?;
        let Some(state) = crate::summary::lines::state(self.conn()?, pk)? else {
            return Err(StoreError::stored_data(format!(
                "session {id} has no transcript lines"
            )));
        };
        Ok(Some(ItemState {
            session_pk: pk,
            versions: ItemVersions::of(&state)?,
            entries: counter(state.entries)?,
            counts: decode_counts(&state.counts)?,
        }))
    }

    /// Waits for the summary of a session that is due: the summary worker
    /// of this process is asked for it, this reader's read ends, and the
    /// next access starts a read that sees what the summary wrote. Without
    /// a worker (no writer in this process, or a build in progress), the
    /// lines are read as stored. The wait ends with the reader's limit.
    fn summarized(&self, session_pk: i64) -> Result<()> {
        while self.restarts() < SUMMARY_WAITS {
            let due: bool = self
                .conn()?
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM summary_due WHERE session_pk = ?1)")?
                .query_row([session_pk], |row| row.get(0))?;
            if !due {
                return Ok(());
            }
            let Some(answer) = self.archive().summaries.ask(session_pk) else {
                return Ok(());
            };
            self.restart()?;
            if !answer.wait(self.limit(), self.archive().stop())? {
                return Ok(());
            }
        }
        Ok(())
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
                "SELECT seq, counts FROM transcript_line
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
                    "SELECT i.seq FROM event e JOIN transcript_line i
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
                "SELECT i.seq FROM message m JOIN transcript_line i
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
                "SELECT seq FROM transcript_line
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
        crate::summary::lines::state(self.conn()?, session_pk)?
            .map_or(Ok(0), |state| counter(state.entries))
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
                "SELECT seq FROM transcript_line {filter}
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
                "SELECT seq FROM transcript_line {filter}
                   AND session_pk = ?1 AND seq >= ?2 AND seq < ?3 ORDER BY seq"
            ))?
            .query_map(
                params![session_pk, position(range.start)?, position(range.end)?],
                |row| row.get::<_, i64>(0),
            )?
            .map(|seq| counter(seq?))
            .collect()
    }

    /// The lines of a session with a provider call ID: its call and the
    /// call's results, found through `transcript_call` and read by place.
    pub fn item_calls(&self, session_pk: i64, call_id: &str) -> Result<Vec<StoredEntry>> {
        let conn = self.conn()?;
        let seqs: Vec<i64> = conn
            .prepare_cached(
                "SELECT seq FROM transcript_call WHERE session_pk = ?1 AND call_id = ?2 ORDER BY seq",
            )?
            .query_map(params![session_pk, call_id], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut line = conn.prepare_cached(&format!(
            "{ENTRY_SELECT} {ENTRY_JOINS} WHERE i.session_pk = ?1 AND i.seq = ?2"
        ))?;
        let mut out = Vec::new();
        // The lines are apart, so each is its own run.
        for seq in seqs {
            #[cfg(any(test, feature = "fixture"))]
            ENTRIES_DECODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut rows = line.query(params![session_pk, seq])?;
            let row = rows.next()?.ok_or_else(|| {
                StoreError::stored_data(format!("transcript call line {seq} is not stored"))
            })?;
            let mut run = Run::new(conn, session_pk);
            run.visit(row, &|_| true)?;
            drop(rows);
            out.extend(run.finish()?);
        }
        Ok(out)
    }

    /// Where the call with each of `call_ids` sits in a session's lines, for
    /// those the session holds: one statement over `transcript_call`, and no
    /// line is read.
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
            "SELECT c.call_id, c.seq, c.class
             FROM json_each(?2) j CROSS JOIN transcript_call c
             WHERE c.session_pk = ?1 AND c.call_id = j.value",
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
}

#[cfg(test)]
mod tests;
