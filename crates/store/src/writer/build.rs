//! The batch build of a private index (docs/architecture.md, one path).
//! The engine reads each chain whole, on several workers; each worker
//! writes a chain's reading into a scratch index with the code upkeep uses,
//! then moves its rows out into one sorted stream ([`BuildWorker`], over
//! [`crate::sorted_runs`]) and clears the scratch. The writer merges every
//! worker's runs in one pass ([`Inner::build`]): inputs first, then text
//! groups, then texts made one per hash, with each reading's text keys
//! joined to the texts by the same merge, then every reading table in key
//! order, with no secondary index to keep. It then creates the indexes,
//! composes every session once with the composition upkeep uses
//! ([`super::shown`]), writes the search entries and installs the file.
//! Nothing the build writes is looked up to decide what to write, and a
//! build that does not finish is discarded and starts again from the stored
//! bytes ([`crate::archive::index_build`]).
use super::*;
use crate::archive::build_progress::{CountUnit, StepCount};
use crate::archive::index_build::Paths;
use crate::sorted_runs::{Merger, Sorter};
use rusqlite::types::{Value, ValueRef};
use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::time::Instant;

/// The scratch index's page cache.
const SCRATCH_CACHE_KIB: i64 = 64 << 10;

/// Sessions composed per transaction.
const SESSIONS_PER_TRANSACTION: usize = 512;

/// Rows the merge writes per transaction, at most.
const ROWS_PER_TRANSACTION: u64 = 200_000;

/// How long one step of the writer's build work runs before it commits
/// and lets the writer serve what waits.
const STEP_TIME: std::time::Duration = std::time::Duration::from_millis(250);

/// The streams of a build, by the first byte of a record's key.
const INPUTS: u8 = 0;
const GROUPS: u8 = 1;
/// Texts and the reading text rows that hold them, by hash: a text's own
/// record (`TEXT`) comes before the rows that name it (`TEXT_HOLDER`).
const TEXTS: u8 = 2;
const TEXT: u8 = 0;
const TEXT_HOLDER: u8 = 1;
/// The reading tables, in name order, from this byte on.
const TABLES: u8 = 3;

/// What decides which sessions are composed from the stream, read before
/// it: sessions whose composition reaches other sessions' readings
/// ([`FALLBACK`]), and each message ID with the session stating it
/// ([`MESSAGE_ID`]), an event's message with its session ([`FOLLOWED`]) or
/// an event ID with its session ([`EVENT_ID`]).
const MARKS: u8 = 253;
const FALLBACK: u8 = 0;
const MESSAGE_ID: u8 = 1;
const FOLLOWED: u8 = 2;
const EVENT_ID: u8 = 3;
/// The readings composition takes, grouped by the session stating them:
/// its facts' kinds, each message's readings with their values, its event
/// readings and its calls' background outcomes, in that order.
const COMPOSE: u8 = 254;
const FACTS: u8 = 0;
const MESSAGE: u8 = 1;
const EVENT: u8 = 2;
const OUTCOME: u8 = 3;

/// The tables composition writes, and those of their indexes kept while it
/// composes from the stream: those it reads (its rows in order, the replies
/// to each message and the message an event follows, for its item index,
/// counts and values), and those whose first column is the session, message
/// or event a row belongs to. Composition makes those keys in the order it
/// writes, so each such entry lands at its index's end and costs no read;
/// made after the stream, each would scan its whole table again. Their
/// other indexes are made after the stream.
const SHOWN_TABLES: &[&str] = &[
    "message",
    "message_route",
    "block",
    "tool_call",
    "event",
    "session_item",
];
const STREAM_READS: &[&str] = &["message_id"];
const COMPOSED_IN_ORDER: &[&str] = &["session_pk", "message_pk", "event_pk"];

/// The table an index statement is on, and its first column.
fn index_target(sql: &str) -> Option<(&str, &str)> {
    let (_, on) = sql.split_once(" ON ")?;
    let (table, columns) = on.split_once('(')?;
    let first = columns.split([',', ')']).next()?;
    Some((table.trim().trim_matches('"'), first.trim()))
}

/// Whether an index on a shown table is kept while composition writes it.
fn kept_while_composing(name: &str, sql: &str) -> bool {
    STREAM_READS.contains(&name)
        || index_target(sql).is_some_and(|(_, first)| COMPOSED_IN_ORDER.contains(&first))
}

/// The indexes a build's load concerns, as (name, statement, table): every
/// one on a stored table but the one that makes an equal text group one row.
const LOAD_INDEXES: &str =
    "SELECT name,sql,tbl_name FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL
     AND name <> 'text_group_hash'
     AND tbl_name NOT IN (SELECT name FROM sqlite_schema WHERE sql LIKE 'CREATE VIRTUAL TABLE%')";

fn load_indexes(conn: &Connection) -> Result<Vec<(String, String, String)>> {
    Ok(conn
        .prepare(LOAD_INDEXES)?
        .query_map([], |r| <(String, String, String)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?)
}

/// [`load_indexes`] as a new build file declares them, so that an index an
/// earlier round dropped is made again from its declaration.
fn declared_load_indexes() -> Result<Vec<(String, String, String)>> {
    let declared = Connection::open_in_memory()?;
    crate::schema::ensure_build_index(&declared)?;
    load_indexes(&declared)
}

/// Messages composed from the stream per transaction, at session ends.
const MESSAGES_PER_TRANSACTION: usize = 20_000;

/// One phase of a build, as its record shows it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BuildPhase {
    pub name: String,
    pub seconds: f64,
    /// Bytes this process read from and wrote to storage during the phase.
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// The process's peak resident memory when the phase ended.
    pub peak_memory: u64,
    /// What the phase handled: chains, rows, sessions or texts.
    pub items: u64,
}

/// The steps the writer runs in a build, in order, as `status`, the log and
/// the docs name them.
pub const SORTING_TEXTS: &str = "sorting texts";
pub const WRITING_TABLES: &str = "writing tables";
pub const INDEXING: &str = "indexing";
pub const ASSEMBLING_SESSIONS: &str = "assembling sessions";
pub const FINISHING_INDEXES: &str = "finishing indexes";
pub const SEARCH_INDEX: &str = "search index";
pub const INSTALLING: &str = "installing";
pub const STEPS: &[&str] = &[
    SORTING_TEXTS,
    WRITING_TABLES,
    INDEXING,
    ASSEMBLING_SESSIONS,
    FINISHING_INDEXES,
    SEARCH_INDEX,
    INSTALLING,
];

/// What a writer step counts as it works.
fn counted(step: &str) -> Option<CountUnit> {
    match step {
        SORTING_TEXTS | WRITING_TABLES => Some(CountUnit::Records),
        INDEXING => Some(CountUnit::Indexes),
        ASSEMBLING_SESSIONS => Some(CountUnit::Sessions),
        _ => None,
    }
}

/// Measures one phase from the process's storage counters.
pub struct PhaseClock {
    name: &'static str,
    unit: Option<CountUnit>,
    started: Instant,
    io: (u64, u64),
}

impl PhaseClock {
    /// Starts measuring step `name`, whose items are `unit`.
    pub fn start(name: &'static str, unit: Option<CountUnit>) -> Self {
        Self {
            name,
            unit,
            started: Instant::now(),
            io: process_io(),
        }
    }

    pub fn end(self, items: u64) -> BuildPhase {
        let (read, written) = process_io();
        let phase = BuildPhase {
            name: self.name.to_owned(),
            seconds: self.started.elapsed().as_secs_f64(),
            bytes_read: read.saturating_sub(self.io.0),
            bytes_written: written.saturating_sub(self.io.1),
            peak_memory: peak_memory(),
            items,
        };
        crate::log!(
            info,
            "index: {} took {:.1} s; {}{:.1} MB read, {:.1} MB written; peak memory {:.0} MB",
            phase.name,
            phase.seconds,
            self.unit
                .map(|unit| format!("{} {}; ", phase.items, unit.word()))
                .unwrap_or_default(),
            phase.bytes_read as f64 / 1e6,
            phase.bytes_written as f64 / 1e6,
            phase.peak_memory as f64 / 1e6,
        );
        phase
    }
}

/// Bytes this process read from and wrote to storage (Linux `/proc`).
fn process_io() -> (u64, u64) {
    let Ok(io) = std::fs::read_to_string("/proc/self/io") else {
        return (0, 0);
    };
    let field = |name: &str| {
        io.lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0)
    };
    (field("read_bytes:"), field("write_bytes:"))
}

/// The process's peak resident memory in bytes (Linux `/proc`).
fn peak_memory() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("VmHWM:"))
                .and_then(|value| {
                    value
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse::<u64>()
                        .ok()
                })
        })
        .map_or(0, |kib| kib * 1024)
}

/// A reading table moved out of the scratch index: its name, columns, the
/// column holding the input key, and the insert that says how rows from
/// several chains combine.
#[derive(Clone)]
struct Table {
    name: String,
    columns: Vec<String>,
    input: Option<usize>,
    /// The statement that inserts one row into the build file.
    insert: String,
}

impl Table {
    fn new(name: String, columns: Vec<String>, conflict: &'static str) -> Self {
        let input = columns.iter().position(|c| c == "input_pk");
        let names = columns
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(",");
        let marks = (1..=columns.len())
            .map(|n| format!("?{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let insert = format!("INSERT {conflict} INTO main.\"{name}\"({names}) VALUES ({marks})");
        Self {
            name,
            columns,
            input,
            insert,
        }
    }
}

/// The reading tables of an index file, in name order. A reading's rows
/// each carry their input and chain, and conflict with no other chain's; a
/// repository lookup is the last one read; rows keyed by content across
/// chains are written once. Inputs and texts are streams of their own.
fn tables(conn: &Connection) -> Result<Vec<Table>> {
    let names: Vec<String> = conn
        .prepare(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'
             AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut found = Vec::new();
    for name in names {
        let columns: Vec<String> = conn
            .prepare(&format!(
                "SELECT name FROM pragma_table_info('{name}') ORDER BY cid"
            ))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let input = columns.iter().position(|c| c == "input_pk");
        let conflict = match name.as_str() {
            "input_ref" | "text" | "text_group" | "parse_text" => continue,
            "text_fragment" | "wire_declaration" => "OR IGNORE",
            "repo_lookup" => "OR REPLACE",
            _ if input.is_some() => "",
            _ => continue,
        };
        found.push(Table::new(name, columns, conflict));
    }
    if found.len() + TABLES as usize > MARKS as usize {
        return Err(StoreError::Invalid(
            "too many reading tables for one stream".into(),
        ));
    }
    Ok(found)
}

/// A row as stream bytes: each value tagged with its type.
fn encode_row(values: &[ValueRef<'_>], out: &mut Vec<u8>) {
    for value in values {
        match value {
            ValueRef::Null => out.push(0),
            ValueRef::Integer(n) => {
                out.push(1);
                out.extend_from_slice(&n.to_be_bytes());
            }
            ValueRef::Real(f) => {
                out.push(2);
                out.extend_from_slice(&f.to_be_bytes());
            }
            ValueRef::Text(bytes) | ValueRef::Blob(bytes) => {
                out.push(if matches!(value, ValueRef::Text(_)) {
                    3
                } else {
                    4
                });
                crate::sorted_runs::put_varint(out, bytes.len() as u64);
                out.extend_from_slice(bytes);
            }
        }
    }
}

fn decode_row(mut bytes: &[u8]) -> Result<Vec<Value>> {
    let bad = || StoreError::Invalid("a build stream holds a malformed row".into());
    let mut values = Vec::new();
    while let Some((&tag, rest)) = bytes.split_first() {
        bytes = rest;
        let fixed = |bytes: &mut &[u8]| -> Result<[u8; 8]> {
            let (head, rest) = bytes.split_first_chunk::<8>().ok_or_else(bad)?;
            *bytes = rest;
            Ok(*head)
        };
        values.push(match tag {
            0 => Value::Null,
            1 => Value::Integer(i64::from_be_bytes(fixed(&mut bytes)?)),
            2 => Value::Real(f64::from_be_bytes(fixed(&mut bytes)?)),
            3 | 4 => {
                let (mut length, mut shift) = (0u64, 0);
                loop {
                    let (&byte, rest) = bytes.split_first().ok_or_else(bad)?;
                    bytes = rest;
                    length |= u64::from(byte & 0x7f) << shift;
                    if byte < 0x80 {
                        break;
                    }
                    shift += 7;
                }
                let length = usize::try_from(length).map_err(|_| bad())?;
                if bytes.len() < length {
                    return Err(bad());
                }
                let (body, rest) = bytes.split_at(length);
                bytes = rest;
                if tag == 3 {
                    Value::Text(String::from_utf8(body.to_vec()).map_err(|_| bad())?)
                } else {
                    Value::Blob(body.to_vec())
                }
            }
            _ => return Err(bad()),
        });
    }
    Ok(values)
}

/// The key of a reading row: its input, chain and place in the chain.
fn row_key(stream: u8, input: &str, chain: i64, seq: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(input.len() + 14);
    key.push(stream);
    key.extend_from_slice(input.as_bytes());
    key.push(0);
    key.extend_from_slice(&chain.to_be_bytes());
    key.extend_from_slice(&seq.to_be_bytes());
    key
}

/// The key of a session's composition records of `kind`.
fn compose_key(session: &str, kind: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(session.len() + 48);
    key.push(COMPOSE);
    key.extend_from_slice(session.as_bytes());
    key.push(0);
    key.push(kind);
    key
}

/// The key of one chain's reading of an item of a session.
fn reading_key(session: &str, kind: u8, item: &str, input: &str, chain: i64) -> Vec<u8> {
    let mut key = compose_key(session, kind);
    key.extend_from_slice(item.as_bytes());
    key.push(0);
    key.extend_from_slice(input.as_bytes());
    key.push(0);
    key.extend_from_slice(&chain.to_be_bytes());
    key
}

fn mark_key(kind: u8, parts: &[&str]) -> Vec<u8> {
    let mut key = vec![MARKS, kind];
    for part in parts {
        key.extend_from_slice(part.as_bytes());
        key.push(0);
    }
    key
}

/// The NUL-ended string at the start of `bytes`, and what follows it.
fn split_id(bytes: &[u8]) -> Result<(&str, &[u8])> {
    let bad = || StoreError::Invalid("a build stream key is malformed".into());
    let end = bytes.iter().position(|&b| b == 0).ok_or_else(bad)?;
    let id = std::str::from_utf8(&bytes[..end]).map_err(|_| bad())?;
    Ok((id, &bytes[end + 1..]))
}

/// The input ID at the start of a reading row's key (after its stream byte).
fn key_input(key: &[u8]) -> Result<&str> {
    let end = key
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| StoreError::Invalid("a build stream key names no input".into()))?;
    std::str::from_utf8(&key[..end])
        .map_err(|_| StoreError::Invalid("a build stream key names no input".into()))
}

/// One worker of a build: chains' readings written as upkeep writes them,
/// into a scratch index cleared after each chain, and moved out as sorted
/// runs of one stream.
pub struct BuildWorker {
    conn: Connection,
    register: Connection,
    blobs: Arc<crate::blobs::Blobs>,
    tables: Vec<Table>,
    sorter: Sorter,
    number: u32,
    groups: u64,
    scratch: PathBuf,
    packer: crate::encoding::RowPacker,
    /// How many bytes of prepared values a chain keeps (see
    /// [`OpenChain::values`]).
    value_memory: usize,
}

impl BuildWorker {
    /// Creates worker `number` of the build in progress, sorting in at most
    /// `memory` bytes.
    pub fn create(archive: &Archive, number: u64, memory: usize) -> Result<Self> {
        if archive.is_primary() || !archive.index_build.building() {
            return Err(StoreError::Invalid(
                "a build worker belongs to a standalone build in progress".into(),
            ));
        }
        let number = u32::try_from(number)
            .map_err(|_| StoreError::Invalid("too many build workers".into()))?;
        let paths = Paths::of(&archive.root);
        std::fs::create_dir_all(&paths.runs)?;
        let scratch = paths.runs.join(format!("scratch-{number:06}.db"));
        crate::archive::index_build::remove_run(&scratch)?;
        let conn = Connection::open(&scratch)?;
        conn.pragma_update(None, "journal_mode", "off")?;
        conn.pragma_update(None, "synchronous", "off")?;
        conn.pragma_update(None, "temp_store", "memory")?;
        conn.pragma_update(None, "cache_size", -SCRATCH_CACHE_KIB)?;
        crate::schema::ensure_index(&conn)?;
        // A chain's rows leave the scratch index table by table; the build
        // file's references are those of its own load.
        conn.pragma_update(None, "foreign_keys", false)?;
        let tables = tables(&conn)?;
        let uri = format!("file:{}?mode=ro", archive.archive_db().display());
        conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        crate::source::views(&conn)?;
        let register = crate::archive::open_read(&archive.archive_db())?;
        Ok(Self {
            conn,
            register,
            blobs: Arc::new(archive.blobs()?),
            tables,
            sorter: Sorter::new(&paths.runs, format!("w{number:06}"), memory.max(32 << 20)),
            number,
            groups: 0,
            scratch,
            packer: crate::encoding::RowPacker::new()?,
            value_memory: memory / 4,
        })
    }

    /// Opens chain `chain` of `input`, read from its first record to
    /// `length`: its steps go into the scratch index as they are parsed, in
    /// one transaction, and [`OpenChain::finish`] moves its rows into the
    /// worker's stream. Dropping it unfinished leaves nothing of the chain.
    pub fn begin<'w>(
        &'w mut self,
        writer: &'w WriterSender,
        input: &InputKey,
        chain: i64,
        length: u64,
    ) -> Result<OpenChain<'w>> {
        crate::fault::check("index.build.prepare", &input.id)?;
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        Ok(OpenChain {
            steps: super::chain::ChainSteps::new(input, chain, length),
            values: std::collections::HashMap::new(),
            value_room: self.value_memory,
            changes,
            register: &self.register,
            tx,
            input: input.id.clone(),
            chain,
            writer,
            blobs: &self.blobs,
            out: Out {
                sorter: &mut self.sorter,
                tables: &self.tables,
                number: self.number,
                groups: &mut self.groups,
                packer: &mut self.packer,
            },
        })
    }

    /// Ends the worker; its runs, for the writer to load.
    pub fn finish(self) -> Result<Vec<PathBuf>> {
        let Self {
            conn,
            sorter,
            scratch,
            ..
        } = self;
        conn.close().map_err(|(_, error)| error)?;
        crate::archive::index_build::remove_run(&scratch)?;
        Ok(sorter.finish()?)
    }
}

/// A chain being read into a build worker's scratch index.
pub struct OpenChain<'w> {
    steps: super::chain::ChainSteps,
    /// Each message reading's value as its last step prepared it, with the
    /// layout it was written with, by message ID.
    values: std::collections::HashMap<String, (Vec<u8>, Prepared)>,
    /// Bytes of values the chain may still keep; a reading whose value did
    /// not fit is read back.
    value_room: usize,
    changes: crate::change_summary::ChangeSummary,
    register: &'w Connection,
    tx: Transaction<'w>,
    input: String,
    chain: i64,
    writer: &'w WriterSender,
    blobs: &'w Arc<crate::blobs::Blobs>,
    out: Out<'w>,
}

impl super::chain::ChainSink for OpenChain<'_> {
    fn step(&mut self, step: &mut ParseStep) -> Result<()> {
        if step.prepared != Default::default() {
            return Err(StoreError::Invalid(
                "a chain reading needs complete parsed values".into(),
            ));
        }
        if !step.output.blobs.is_empty() {
            return Err(StoreError::Invalid(
                "decoded bytes must be stored before their chain reading".into(),
            ));
        }
        step.counts.check()?;
        step.reset = self.steps.is_first();
        step.prepare();
        let w = super::chain::ReadingWrite {
            register: self.register,
            counter: 1,
            primary: false,
            changes: &self.changes,
        };
        self.steps
            .write(&self.tx, &w, step, &mut super::readings::Touched::default())?;
        // The values composition takes, kept from what the step prepared
        // rather than read back from the scratch index.
        for (row, prepared) in step.output.messages.iter().zip(&mut step.prepared.messages) {
            let layout = crate::encoding::message_layout(&prepared.block_selection)?;
            let block_count = prepared.block_selection.block_count;
            let size = prepared.blocks.iter().map(Vec::len).sum::<usize>()
                + prepared.data.iter().flatten().map(Vec::len).sum::<usize>()
                + prepared.items.len() * std::mem::size_of::<polylogue_model::read::IndexEntry>();
            let id = row.message.id.to_string();
            if size > self.value_room {
                continue;
            }
            self.value_room -= size;
            let (items, content) = prepared.take_value();
            self.values.insert(
                id,
                (
                    layout,
                    Prepared {
                        block_count,
                        items,
                        words: super::shown::text_words(&row.message),
                        content,
                    },
                ),
            );
        }
        Ok(())
    }

    fn decoded(&mut self, batch: crate::staging::PreparedBatch) -> Result<()> {
        if !batch.batch.blobs.is_empty() {
            for result in self
                .writer
                .store_group(vec![batch], Arc::clone(self.blobs))?
            {
                result?;
            }
        }
        Ok(())
    }
}

impl OpenChain<'_> {
    /// Ends the chain: moves its rows into the worker's stream and clears
    /// the scratch index.
    pub fn finish(mut self) -> Result<()> {
        self.steps.finish()?;
        move_out(
            &self.tx,
            &mut self.out,
            &self.input,
            self.chain,
            &mut self.values,
        )?;
        self.tx.commit()?;
        Ok(())
    }
}

/// Where a worker moves a chain's rows.
struct Out<'a> {
    sorter: &'a mut Sorter,
    tables: &'a [Table],
    number: u32,
    groups: &'a mut u64,
    packer: &'a mut crate::encoding::RowPacker,
}

/// Moves the chain's rows from the scratch index into the stream and
/// clears it.
fn move_out(
    tx: &Transaction<'_>,
    out: &mut Out<'_>,
    input: &str,
    chain: i64,
    values: &mut std::collections::HashMap<String, (Vec<u8>, Prepared)>,
) -> Result<()> {
    let mut value = Vec::new();
    let (format, route): (String, String) = tx.query_row(
        "SELECT format,route FROM input_ref WHERE id=?1",
        [input],
        |r| <(String, String)>::try_from(r),
    )?;
    let mut key = vec![INPUTS];
    key.extend_from_slice(input.as_bytes());
    key.push(0);
    encode_row(
        &[
            ValueRef::from(format.as_str()),
            ValueRef::from(route.as_str()),
        ],
        &mut value,
    );
    out.sorter.push(&key, &value)?;
    // Text groups take the worker's next numbers; a group's record is its
    // hash followed by its body.
    let mut groups = std::collections::HashMap::new();
    {
        let mut statement = tx.prepare_cached("SELECT id,hash,body FROM text_group ORDER BY id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let local: i64 = row.get(0)?;
            let seq = *out.groups;
            *out.groups += 1;
            groups.insert(local, seq);
            let mut key = vec![GROUPS];
            key.extend_from_slice(&out.number.to_be_bytes());
            key.extend_from_slice(&seq.to_be_bytes());
            value.clear();
            value.extend_from_slice(row.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?);
            value.extend_from_slice(row.get_ref(2)?.as_blob().map_err(rusqlite::Error::from)?);
            out.sorter.push(&key, &value)?;
        }
    }
    {
        let mut statement = tx.prepare_cached(
            "SELECT t.hash,t.normalized_hash,t.group_id,t.start,t.length,g.hash FROM text t
             JOIN text_group g ON g.id=t.group_id
             WHERE t.id IN (SELECT text_id FROM parse_text)",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let group = groups[&row.get::<_, i64>(2)?];
            let mut key = vec![TEXTS];
            key.extend_from_slice(&row.get::<_, Vec<u8>>(0)?);
            key.push(TEXT);
            value.clear();
            encode_row(
                &[
                    ValueRef::Integer(i64::from(out.number)),
                    ValueRef::Integer(group as i64),
                    row.get_ref(3)?,
                    row.get_ref(4)?,
                    row.get_ref(1)?,
                    row.get_ref(5)?,
                ],
                &mut value,
            );
            out.sorter.push(&key, &value)?;
        }
    }
    let mut seq = 0u32;
    {
        // A message's text rows also carry the key of its reading in the
        // composition stream, which the load gives each its text key under.
        let mut statement = tx.prepare_cached(
            "SELECT p.chain,p.document,p.item,p.idx,t.hash,m.session_id FROM parse_text p
             JOIN text t ON t.id=p.text_id
             LEFT JOIN message_parse m ON p.document=0 AND m.input_pk=p.input_pk AND m.chain=p.chain
                 AND m.message_id=p.item
             ORDER BY p.document,p.item,p.idx",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let mut key = vec![TEXTS];
            key.extend_from_slice(&row.get::<_, Vec<u8>>(4)?);
            key.push(TEXT_HOLDER);
            key.extend_from_slice(&row_key(0, input, chain, seq)[1..]);
            seq += 1;
            let reading = row
                .get::<_, Option<String>>(5)?
                .map(|session| -> Result<Vec<u8>> {
                    Ok(reading_key(
                        &session,
                        MESSAGE,
                        &row.get::<_, String>(2)?,
                        input,
                        chain,
                    ))
                })
                .transpose()?;
            value.clear();
            encode_row(
                &[
                    ValueRef::Null,
                    row.get_ref(0)?,
                    row.get_ref(1)?,
                    row.get_ref(2)?,
                    row.get_ref(3)?,
                    ValueRef::Null,
                    reading.as_deref().map_or(ValueRef::Null, ValueRef::Blob),
                ],
                &mut value,
            );
            out.sorter.push(&key, &value)?;
        }
    }
    compose_records(tx, out, input, &route, chain, values)?;
    for (n, table) in out.tables.iter().enumerate() {
        let stream = TABLES + n as u8;
        let mut statement = tx.prepare_cached(&format!("SELECT * FROM \"{}\"", table.name))?;
        let mut rows = statement.query([])?;
        let mut seq = 0u32;
        while let Some(row) = rows.next()? {
            let values = (0..table.columns.len())
                .map(|i| {
                    if Some(i) == table.input {
                        Ok(ValueRef::Null)
                    } else {
                        row.get_ref(i)
                    }
                })
                .collect::<rusqlite::Result<Vec<_>>>()?;
            value.clear();
            encode_row(&values, &mut value);
            out.sorter
                .push(&row_key(stream, input, chain, seq), &value)?;
            seq += 1;
        }
    }
    for table in out.tables.iter().map(|table| table.name.as_str()).chain([
        "parse_text",
        "text",
        "text_group",
    ]) {
        tx.execute(&format!("DELETE FROM \"{table}\""), [])?;
    }
    Ok(())
}

/// Moves the chain's readings out into the composition stream, grouped
/// by the session stating them, each message reading with its value
/// (docs/architecture.md, one path).
fn compose_records(
    tx: &Transaction<'_>,
    out: &mut Out<'_>,
    input: &str,
    route: &str,
    chain: i64,
    values: &mut std::collections::HashMap<String, (Vec<u8>, Prepared)>,
) -> Result<()> {
    let mut value = Vec::new();
    let input_pk: i64 = tx.query_row("SELECT pk FROM input_ref WHERE id=?1", [input], |r| {
        r.get(0)
    })?;
    {
        let mut statement = tx
            .prepare_cached("SELECT session_id,json_extract(facts,'$.kind') FROM session_record")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let mut key = compose_key(&row.get::<_, String>(0)?, FACTS);
            key.extend_from_slice(input.as_bytes());
            key.push(0);
            key.extend_from_slice(&chain.to_be_bytes());
            value.clear();
            encode_row(&[row.get_ref(1)?], &mut value);
            out.sorter.push(&key, &value)?;
        }
    }
    {
        // In each chain's order, so that its texts are read group by group.
        let mut statement = tx.prepare_cached(
            "SELECT message_id,session_id,source_start,source_end,seq,record_position,order_time,route,
                 completion,id_kind='content',hash,body,layout FROM message_parse ORDER BY seq,record_position",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let message: String = row.get(0)?;
            let session: String = row.get(1)?;
            out.sorter
                .push(&mark_key(MESSAGE_ID, &[&message, &session]), &[])?;
            // The value a step prepared is the reading's when the reading is
            // that step's statement: the same version and blocks. A reading
            // joined from several statements is read back.
            let prepared = values.remove(&message).filter(|(layout, prepared)| {
                row.get_ref(10).is_ok_and(|hash| {
                    hash.as_blob()
                        .is_ok_and(|hash| hash == prepared.content.hash)
                }) && row
                    .get_ref(12)
                    .is_ok_and(|stored| stored.as_blob().is_ok_and(|stored| stored == layout))
            });
            let reading = match prepared {
                Some((_, prepared)) => prepared,
                None => {
                    let (read, selection) = super::readings::read_message(
                        tx, input_pk, chain, &message,
                    )?
                    .ok_or_else(|| StoreError::Invalid("a message reading has no body".into()))?;
                    Prepared::of(super::shown::ReadingValue::of(read, selection, out.packer)?)
                }
            };
            if reading.partial() {
                // Its value joins other readings of its version: the
                // session is composed from the index.
                out.sorter.push(&mark_key(FALLBACK, &[&session]), &[])?;
                continue;
            }
            let items = crate::items::entry_bytes::encode_message(&reading.items)?;
            // The body's blocks are the packed blocks and data the record
            // carries after it: the record holds the rest of the message.
            let head = message_head(row.get_ref(11)?.as_blob().map_err(rusqlite::Error::from)?)?;
            let mut values: Vec<ValueRef<'_>> = (2..13)
                .map(|n| row.get_ref(n))
                .collect::<rusqlite::Result<_>>()?;
            values[9] = ValueRef::Blob(&head);
            values.push(ValueRef::Integer(reading.words));
            values.push(ValueRef::Blob(&items));
            for ((block, data), text) in reading
                .content
                .blocks
                .iter()
                .zip(&reading.content.data)
                .zip(&reading.content.texts)
            {
                values.push(ValueRef::Blob(block));
                values.push(data.as_deref().map_or(ValueRef::Null, ValueRef::Blob));
                values.push(
                    text.as_ref()
                        .map_or(ValueRef::Null, |hash| ValueRef::Blob(hash)),
                );
            }
            value.clear();
            encode_row(&values, &mut value);
            out.sorter.push(
                &reading_key(&session, MESSAGE, &message, input, chain),
                &value,
            )?;
        }
    }
    {
        let mut statement = tx.prepare_cached(
            "SELECT event_id,session_id,route,hash,fields,data,seq,source_start,source_end,after_message_id
             FROM event_parse WHERE session_id IS NOT NULL",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let (event, session): (String, String) = (row.get(0)?, row.get(1)?);
            if let Some(after) = row.get::<_, Option<String>>(9)? {
                out.sorter
                    .push(&mark_key(FOLLOWED, &[&after, &session]), &[])?;
            }
            out.sorter
                .push(&mark_key(EVENT_ID, &[&event, &session]), &[])?;
            let values: Vec<ValueRef<'_>> = (2..9)
                .map(|n| row.get_ref(n))
                .collect::<rusqlite::Result<_>>()?;
            value.clear();
            encode_row(&values, &mut value);
            out.sorter
                .push(&reading_key(&session, EVENT, &event, input, chain), &value)?;
        }
    }
    {
        let mut statement =
            tx.prepare_cached("SELECT session_id,call_id,position,body FROM call_outcome")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let (session, call): (String, String) = (row.get(0)?, row.get(1)?);
            value.clear();
            encode_row(
                &[row.get_ref(2)?, row.get_ref(3)?, ValueRef::from(route)],
                &mut value,
            );
            out.sorter
                .push(&reading_key(&session, OUTCOME, &call, input, chain), &value)?;
        }
    }
    Ok(())
}

/// A packed message body as JSON without its blocks.
fn message_head(body: &[u8]) -> Result<Vec<u8>> {
    let mut message: serde_json::Value = crate::encoding::unpack_json(body)?;
    let blocks = message
        .get_mut("blocks")
        .ok_or_else(|| StoreError::Invalid("a message body has no blocks".into()))?;
    *blocks = serde_json::Value::Array(Vec::new());
    Ok(serde_json::to_vec(&message)?)
}

/// What composing a message reading takes from its value: whether it holds
/// every block of its version, its index entries, its words and its packed
/// content (see [`super::shown::ReadingValue`]).
struct Prepared {
    block_count: u64,
    items: Vec<polylogue_model::read::IndexEntry>,
    words: i64,
    content: crate::encoding::PreparedContent,
}

impl Prepared {
    fn of(value: super::shown::ReadingValue) -> Self {
        Self {
            block_count: value.selection.block_count,
            items: value.items,
            words: value.words,
            content: value.content,
        }
    }

    /// Whether the reading holds only some of its version's blocks.
    fn partial(&self) -> bool {
        self.block_count != self.content.blocks.len() as u64
    }
}

/// What decides which sessions a build composes from the index rather than
/// from the stream: those whose readings reach other sessions' readings.
#[derive(Default)]
struct Fallback {
    sessions: BTreeSet<String>,
    /// Message and event IDs of readings upkeep wrote into the build file
    /// while the build ran.
    messages: BTreeSet<String>,
    events: HashSet<String>,
}

impl Fallback {
    /// The sessions and messages of the readings upkeep wrote into the build
    /// file before its load: those sessions are composed from the index.
    fn upkept(conn: &Connection) -> Result<Self> {
        let strings = |sql: &str| -> Result<BTreeSet<String>> {
            Ok(conn
                .prepare(sql)?
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?)
        };
        Ok(Self {
            sessions: strings(
                "SELECT session_id FROM session_record UNION SELECT session_id FROM message_parse
                 UNION SELECT session_id FROM event_parse WHERE session_id IS NOT NULL
                 UNION SELECT session_id FROM session_alias UNION SELECT alias FROM session_alias
                 UNION SELECT session_id FROM call_outcome",
            )?,
            messages: strings("SELECT DISTINCT message_id FROM message_parse")?,
            events: conn
                .prepare("SELECT DISTINCT event_id FROM event_parse WHERE session_id IS NOT NULL")?
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?,
        })
    }
}

/// What the load leaves for composition: the rest of the merge, which holds
/// the composition stream, each message reading's text keys, and the
/// sessions composed from the index.
pub(super) struct Loaded {
    merged: Merger,
    texts: Merger,
    fallback: BTreeSet<String>,
    shared_events: HashSet<String>,
}

/// Where a build's composition stands between its steps
/// ([`Inner::compose_step`]); it holds no transaction or statement.
pub(super) struct ComposeState {
    step: Step,
    /// Indexes to make once the stream is composed.
    indexes: Vec<String>,
    /// The triggers that keep `row_count`, dropped while composition writes
    /// and made again once every table is counted.
    counting: Vec<String>,
    /// The composition stream, decoded ahead on a thread of its own.
    pieces: std::sync::mpsc::Receiver<Result<Piece>>,
    stream_done: bool,
    /// Sessions composed from the index instead of the stream, which the
    /// stream reader passes over.
    fallback: Vec<String>,
    aliases: Arc<super::identity::Aliases>,
    /// Event IDs several sessions state: the stream leaves them for the
    /// index to compose from every session's readings.
    shared_events: Arc<HashSet<String>>,
    /// Events left by the stream for the index to compose.
    deferred: Vec<(Option<String>, String)>,
    /// Sessions composed so far, and of how many: every session a reading
    /// states and every name composed from the index.
    pub(super) sessions: u64,
    pub(super) total: u64,
}

enum Step {
    Links,
    Stream,
    /// The shown tables' indexes the stream does not read.
    Indexes,
    Index {
        next: usize,
    },
    /// Counts each counted table and makes its triggers again.
    Count,
    Done,
}

impl ComposeState {
    /// The composition of a loaded build file.
    pub(super) fn new(loaded: Loaded, indexes: Vec<String>, index: &Connection) -> Result<Self> {
        let Loaded {
            merged,
            texts,
            mut fallback,
            shared_events,
        } = loaded;
        let aliases = Arc::new(super::identity::Aliases::load(index)?);
        // Composing runs no trigger per row: each counted table is counted
        // once when composition ends.
        let counting: Vec<(String, String)> = index
            .prepare(
                "SELECT name,sql FROM sqlite_schema
                 WHERE type='trigger' AND sql LIKE '%UPDATE row_count %'",
            )?
            .query_map([], |r| <(String, String)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        for (name, _) in &counting {
            index.execute(&format!("DROP TRIGGER \"{name}\""), [])?;
        }
        // The sessions a copy name involves are decided across sessions.
        fallback.extend(
            index
                .prepare(
                    "SELECT alias FROM session_alias WHERE alias<>session_id
                     UNION SELECT session_id FROM session_alias WHERE alias<>session_id",
                )?
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        let mut total: u64 = index.scalar(
            "SELECT count(*) FROM (SELECT DISTINCT session_id FROM session_record)",
            [],
        )?;
        for name in &fallback {
            if !index.exists(
                "SELECT EXISTS(SELECT 1 FROM session_record WHERE session_id=?1)",
                [name],
            )? {
                total += 1;
            }
        }
        Ok(Self {
            step: Step::Links,
            indexes,
            counting: counting.into_iter().map(|(_, sql)| sql).collect(),
            pieces: read_stream(merged, texts, fallback.clone(), Arc::clone(&aliases)),
            stream_done: false,
            fallback: fallback.into_iter().collect(),
            aliases,
            shared_events: Arc::new(shared_events),
            deferred: Vec::new(),
            sessions: 0,
            total,
        })
    }
}

/// A build the writer advances one bounded step at a time, between the
/// messages it serves: each step commits its own transaction and the job
/// keeps its position, so nothing a step did is done again.
pub(super) struct BuildJob {
    stage: Stage,
    announce: Option<std::sync::mpsc::Sender<&'static str>>,
    reply: SyncSender<Result<Vec<BuildPhase>>>,
    phases: Vec<BuildPhase>,
    clock: Option<(PhaseClock, u64)>,
    /// The sessions composition will assemble, known once it starts.
    sessions: Option<u64>,
    /// The counted step that ended in this build step, with its final
    /// count, for `status` to record before the step that follows.
    ended: Option<(&'static str, u64, CountUnit)>,
    /// The indexes dropped for the load, recreated after it; the shown
    /// tables' indexes composition does not keep are made after the
    /// streamed sessions (see [`SHOWN_TABLES`]).
    indexes: Vec<String>,
    after_stream: Vec<String>,
    /// The full-text files, filled while the build composes.
    pub(super) search: Option<crate::text_rows::SearchFill>,
    /// Set once the file is complete and being installed: a failure then
    /// leaves it for a later finish, rather than discarding it.
    pub(super) installing: bool,
}

enum Stage {
    Start(Vec<PathBuf>),
    Load(Box<Load>),
    TextHolders(Box<TextHolders>),
    Indexes(usize, Box<Loaded>),
    Compose(Box<ComposeState>),
    Deferred(Vec<String>),
    FullText,
    Install,
}

/// The merge of the workers' runs, where it stands.
struct Load {
    merged: Merger,
    tables: Vec<Table>,
    held: std::iter::Peekable<std::vec::IntoIter<(Vec<u8>, i64)>>,
    next_group: i64,
    /// The key the load's first group takes.
    first_group: i64,
    /// One bit for each group the load read, from `first_group` on, set for
    /// a group not stored because a group with the same body already was (by
    /// another worker, another chain or upkeep): its texts are placed at the
    /// same bytes of the stored one, found by the group hash they carry.
    shared_groups: Vec<u64>,
    bases: std::collections::HashMap<u32, i64>,
    holders: Option<Sorter>,
    input: Option<(String, i64)>,
    current_text: Option<(Vec<u8>, i64)>,
    stream: Option<u8>,
    /// Each message reading's text keys, by its key in the composition
    /// stream.
    reading_texts: Option<Sorter>,
    /// Sessions composed from the index, the message and event IDs of
    /// readings upkeep wrote meanwhile, the copy names, the last message and
    /// event ID seen with its session, and the event IDs several sessions
    /// state (see [`MARKS`]).
    fallback: BTreeSet<String>,
    upkept_messages: BTreeSet<String>,
    upkept_events: HashSet<String>,
    aliases: Option<super::identity::Aliases>,
    last_message: Option<(String, String)>,
    last_event: Option<(String, String)>,
    shared_events: HashSet<String>,
}

/// The reading text rows, joined to their texts, being loaded.
struct TextHolders {
    merged: Merger,
    input: Option<(String, i64)>,
    table: Table,
    composing: Box<Loaded>,
}

impl BuildJob {
    pub(super) fn new(
        runs: Vec<PathBuf>,
        announce: Option<std::sync::mpsc::Sender<&'static str>>,
        reply: SyncSender<Result<Vec<BuildPhase>>>,
    ) -> Self {
        Self {
            stage: Stage::Start(runs),
            announce,
            reply,
            phases: Vec::new(),
            clock: None,
            sessions: None,
            ended: None,
            indexes: Vec::new(),
            after_stream: Vec::new(),
            search: None,
            installing: false,
        }
    }

    /// Starts `name` unless it is the phase already running, ending the one
    /// before.
    fn phase(&mut self, name: &'static str) {
        if self
            .clock
            .as_ref()
            .is_some_and(|(clock, _)| clock.name == name)
        {
            return;
        }
        if let Some((clock, items)) = self.clock.take() {
            if let Some(unit) = clock.unit {
                self.ended = Some((clock.name, items, unit));
            }
            self.phases.push(clock.end(items));
        }
        if let Some(announce) = &self.announce {
            let _ = announce.send(name);
        }
        self.clock = Some((PhaseClock::start(name, counted(name)), 0));
    }

    fn count(&mut self, items: u64) {
        if let Some((_, counted)) = &mut self.clock {
            *counted += items;
        }
    }

    /// Answers the caller: the phases, or why the build stopped.
    pub(super) fn finish(mut self, result: Result<()>) {
        if let Some((clock, items)) = self.clock.take() {
            self.phases.push(clock.end(items));
        }
        let _ = self.reply.send(result.map(|()| self.phases));
    }
}

impl Inner {
    /// Runs the next step of `job`; true once the build file is installed.
    pub(super) fn build_step(&mut self, job: &mut BuildJob) -> Result<bool> {
        if self.sinex.is_some() || !self.location.index_build.building() {
            return Err(StoreError::Invalid(
                "no standalone build is in progress".into(),
            ));
        }
        let stage = std::mem::replace(&mut job.stage, Stage::Install);
        job.stage = match stage {
            Stage::Start(runs) => {
                job.phase(if runs.is_empty() {
                    INDEXING
                } else {
                    SORTING_TEXTS
                });
                self.index.flush_prepared_statement_cache();
                let upkept = Fallback::upkept(&self.index)?;
                let present = load_indexes(&self.index)?;
                // A load drops every index but the one that makes an equal
                // group one row (see `Load::shared_groups`); the build makes
                // the declared indexes the file lacks.
                if !runs.is_empty() {
                    for (name, ..) in &present {
                        self.index.execute(&format!("DROP INDEX \"{name}\""), [])?;
                    }
                }
                let kept = load_indexes(&self.index)?;
                for (name, sql, table) in declared_load_indexes()? {
                    if kept.iter().any(|(held, ..)| *held == name) {
                        continue;
                    }
                    if SHOWN_TABLES.contains(&table.as_str()) && !kept_while_composing(&name, &sql)
                    {
                        job.after_stream.push(sql);
                    } else {
                        job.indexes.push(sql);
                    }
                }
                // Of the indexes a build makes at its end, those composition
                // fills in order are made before it; the end finds them made.
                for sql in crate::schema::built_at_end_sql()? {
                    if index_target(&sql).is_some_and(|(table, first)| {
                        SHOWN_TABLES.contains(&table) && COMPOSED_IN_ORDER.contains(&first)
                    }) {
                        job.indexes.push(sql);
                    }
                }
                job.search = Some(crate::text_rows::SearchFill::start(&self.index)?);
                if runs.is_empty() {
                    // Nothing to load: no text is merged and no row moves.
                    let runs_dir = Paths::of(&self.location.root).runs;
                    Stage::Indexes(
                        0,
                        Box::new(Loaded {
                            merged: Merger::open(Vec::new(), &runs_dir, 0)?,
                            texts: Merger::open(Vec::new(), &runs_dir, 0)?,
                            fallback: upkept.sessions,
                            shared_events: HashSet::new(),
                        }),
                    )
                } else {
                    Stage::Load(Box::new(self.start_load(runs, upkept)?))
                }
            }
            Stage::Load(mut load) => {
                if self.load_step(job, &mut load)? {
                    Stage::Load(load)
                } else {
                    crate::fault::check("index.build.loading", "")?;
                    let paths = Paths::of(&self.location.root);
                    let holders = load.holders.take().expect("the text holders' sorter");
                    let reading_texts = load
                        .reading_texts
                        .take()
                        .expect("the reading texts' sorter");
                    let load = *load;
                    let composing = Box::new(Loaded {
                        texts: Merger::open(
                            reading_texts.finish()?,
                            &paths.runs,
                            self.location.memory_budget().side_merge_memory(),
                        )?,
                        merged: load.merged,
                        fallback: load.fallback,
                        shared_events: load.shared_events,
                    });
                    Stage::TextHolders(Box::new(TextHolders {
                        merged: Merger::open(
                            holders.finish()?,
                            &paths.runs,
                            self.location.memory_budget().side_merge_memory(),
                        )?,
                        input: None,
                        composing,
                        table: Table::new(
                            "parse_text".into(),
                            ["input_pk", "chain", "document", "item", "idx", "text_id"]
                                .map(String::from)
                                .to_vec(),
                            "",
                        ),
                    }))
                }
            }
            Stage::TextHolders(mut holders) => {
                if self.holders_step(job, &mut holders)? {
                    Stage::TextHolders(holders)
                } else {
                    self.index.pragma_update(None, "foreign_keys", true)?;
                    job.phase(INDEXING);
                    Stage::Indexes(0, holders.composing)
                }
            }
            Stage::Indexes(next, loaded) => match job.indexes.get(next) {
                Some(sql) => {
                    self.index.execute(sql, [])?;
                    job.count(1);
                    Stage::Indexes(next + 1, loaded)
                }
                None => {
                    crate::fault::check("index.build.composing", "")?;
                    job.phase(ASSEMBLING_SESSIONS);
                    let indexes = std::mem::take(&mut job.after_stream);
                    let state = ComposeState::new(*loaded, indexes, &self.index)?;
                    job.sessions = Some(state.total);
                    Stage::Compose(Box::new(state))
                }
            },
            Stage::Compose(mut state) => {
                let before = state.sessions;
                let done = self.compose_step(&mut state)?;
                job.count(state.sessions - before);
                if !done {
                    Stage::Compose(state)
                } else {
                    drop(state);
                    crate::archive::index_build::remove_runs(&Paths::of(&self.location.root))?;
                    job.phase(FINISHING_INDEXES);
                    // Composition wrote shown rows only: the search
                    // membership of every text comes from the blocks now,
                    // and from here on membership and the trigram index
                    // follow the shown rows as they do in an installed index.
                    let tx = self
                        .index
                        .transaction_with_behavior(TransactionBehavior::Immediate)?;
                    super::shown::mark_searched_texts(&tx)?;
                    tx.execute_batch(crate::schema::SEARCH_TRIGGERS)?;
                    tx.commit()?;
                    job.search
                        .as_mut()
                        .ok_or_else(|| StoreError::internal("a build has no full-text fill"))?
                        .settle(&self.index)?;
                    let started = std::time::Instant::now();
                    self.index.execute(
                        "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
                        [],
                    )?;
                    crate::timing::add(crate::timing::Stage::TrigramIndex, started.elapsed());
                    let mut deferred = crate::schema::built_at_end_sql()?;
                    deferred.reverse();
                    Stage::Deferred(deferred)
                }
            }
            Stage::Deferred(mut deferred) => match deferred.pop() {
                Some(sql) => {
                    let started = std::time::Instant::now();
                    self.index.execute(&sql, [])?;
                    crate::timing::add(crate::timing::Stage::SecondaryIndexes, started.elapsed());
                    crate::fault::check("index.deferred", &self.location.root.to_string_lossy())?;
                    Stage::Deferred(deferred)
                }
                None => {
                    job.phase(SEARCH_INDEX);
                    Stage::FullText
                }
            },
            // The fill's threads finish what composition left; the writer
            // serves other work between waits.
            Stage::FullText => {
                let search = job
                    .search
                    .as_mut()
                    .ok_or_else(|| StoreError::internal("a build has no full-text fill"))?;
                if search.wait(STEP_TIME)? {
                    job.search = None;
                    job.phase(INSTALLING);
                    Stage::Install
                } else {
                    Stage::FullText
                }
            }
            Stage::Install => {
                job.installing = true;
                self.install_build()?;
                self.search.ready();
                self.commits
                    .committed(crate::completion::Committed::CONTENT);
                return Ok(true);
            }
        };
        let ended = job.ended.take();
        let running = job
            .clock
            .as_ref()
            .and_then(|(clock, done)| clock.unit.map(|unit| (clock.name, *done, unit)));
        for (step, done, unit) in ended.into_iter().chain(running) {
            let total = match step {
                INDEXING => Some(job.indexes.len() as u64),
                ASSEMBLING_SESSIONS => job.sessions,
                _ => None,
            };
            self.location
                .record_build_count(step, StepCount { done, total, unit });
        }
        Ok(false)
    }

    /// Opens the merge of every worker's runs.
    fn start_load(&mut self, runs: Vec<PathBuf>, upkept: Fallback) -> Result<Load> {
        let paths = Paths::of(&self.location.root);
        std::fs::create_dir_all(&paths.runs)?;
        // Texts upkeep wrote into the build file while the build ran keep
        // their rows; a run's equal text takes that row.
        let held: Vec<(Vec<u8>, i64)> = self
            .index
            .prepare("SELECT hash,id FROM text ORDER BY hash")?
            .query_map([], |r| <(Vec<u8>, i64)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        let next_group: i64 = self
            .index
            .scalar("SELECT coalesce(max(id),0)+1 FROM text_group", [])?;
        // References are checked by the indexes and composition after the
        // load; tables arrive in name order, not in reference order.
        self.index.pragma_update(None, "foreign_keys", false)?;
        Ok(Load {
            merged: Merger::open(
                runs,
                &paths.runs,
                self.location.memory_budget().run_merge_memory(),
            )?,
            tables: tables(&self.index)?,
            held: held.into_iter().peekable(),
            shared_groups: Vec::new(),
            next_group,
            first_group: next_group,
            bases: std::collections::HashMap::new(),
            holders: Some(Sorter::new(
                &paths.runs,
                "text-holders".into(),
                (self.location.memory_budget().total() / 8) as usize,
            )),
            input: None,
            current_text: None,
            stream: None,
            reading_texts: Some(Sorter::new(
                &paths.runs,
                "reading-texts".into(),
                (self.location.memory_budget().total() / 16) as usize,
            )),
            fallback: upkept.sessions,
            upkept_messages: upkept.messages,
            upkept_events: upkept.events,
            aliases: None,
            last_message: None,
            last_event: None,
            shared_events: HashSet::new(),
        })
    }

    /// Loads up to a transaction's worth of records; false once the runs
    /// are read.
    fn load_step(&mut self, job: &mut BuildJob, load: &mut Load) -> Result<bool> {
        let started = std::time::Instant::now();
        let t = Transaction::new_unchecked(&self.index, TransactionBehavior::Immediate)?;
        let mut written = 0u64;
        let mut values = Vec::new();
        let more = loop {
            if written == ROWS_PER_TRANSACTION
                || (written % 1024 == 1023 && started.elapsed() > STEP_TIME)
            {
                break true;
            }
            // The composition stream is read by composition.
            if load
                .merged
                .peek()?
                .is_none_or(|(key, _)| key.first() == Some(&COMPOSE))
            {
                break false;
            }
            let Some((key, value)) = load.merged.next()? else {
                break false;
            };
            let (&kind, rest) = key
                .split_first()
                .ok_or_else(|| StoreError::Invalid("a build stream holds an empty key".into()))?;
            if load.stream != Some(kind) {
                job.phase(match kind {
                    INPUTS | GROUPS | TEXTS => SORTING_TEXTS,
                    _ => WRITING_TABLES,
                });
                load.stream = Some(kind);
            }
            match kind {
                INPUTS => {
                    let row = decode_row(value)?;
                    t.prepare_cached(
                        "INSERT INTO input_ref(id,format,route) VALUES (?1,?2,?3)
                         ON CONFLICT(id) DO UPDATE SET format=excluded.format,route=excluded.route
                         WHERE format IS NOT excluded.format OR route IS NOT excluded.route",
                    )?
                    .execute(params![key_input(rest)?, row[0], row[1]])?;
                }
                GROUPS => {
                    let worker = u32::from_be_bytes(rest[..4].try_into().expect("four bytes"));
                    let seq = i64::from_be_bytes(rest[4..12].try_into().expect("eight bytes"));
                    load.bases.entry(worker).or_insert(load.next_group - seq);
                    if value.len() < 32 {
                        return Err(StoreError::Invalid("a text group has no hash".into()));
                    }
                    let (hash, body) = value.split_at(32);
                    let stored = t
                        .prepare_cached(
                            "INSERT INTO text_group(id,hash,body) VALUES (?1,?2,?3)
                             ON CONFLICT(hash) DO NOTHING",
                        )?
                        .execute(params![load.next_group, hash, body])?;
                    if stored == 0 {
                        let at = (load.next_group - load.first_group) as usize;
                        if load.shared_groups.len() <= at / 64 {
                            load.shared_groups.resize(at / 64 + 1, 0);
                        }
                        load.shared_groups[at / 64] |= 1 << (at % 64);
                    }
                    load.next_group += 1;
                }
                TEXTS => {
                    let (hash, rest) = rest.split_at(32);
                    while load
                        .held
                        .next_if(|(stored, _)| stored.as_slice() < hash)
                        .is_some()
                    {}
                    if load
                        .current_text
                        .as_ref()
                        .is_none_or(|(same, _)| same.as_slice() != hash)
                    {
                        load.current_text = load
                            .held
                            .next_if(|(stored, _)| stored.as_slice() == hash)
                            .map(|(_, id)| (hash.to_vec(), id));
                    }
                    match rest.first() {
                        Some(&TEXT) => {
                            if load.current_text.is_none() {
                                let row = decode_row(value)?;
                                let (Value::Integer(worker), Value::Integer(seq)) =
                                    (&row[0], &row[1])
                                else {
                                    return Err(StoreError::Invalid(
                                        "a text names no group".into(),
                                    ));
                                };
                                let base = load.bases.get(&(*worker as u32)).ok_or_else(|| {
                                    StoreError::Invalid("a text's group is not in the build".into())
                                })?;
                                let mut group = base + seq;
                                let at = (group - load.first_group) as usize;
                                if load
                                    .shared_groups
                                    .get(at / 64)
                                    .is_some_and(|bits| bits & (1 << (at % 64)) != 0)
                                {
                                    group = t
                                        .prepare_cached("SELECT id FROM text_group WHERE hash=?1")?
                                        .query_row([&row[5]], |r| r.get(0))?;
                                }
                                t.prepare_cached(
                                    "INSERT INTO text(hash,normalized_hash,group_id,start,length)
                                     VALUES (?1,?2,?3,?4,?5)",
                                )?
                                .execute(params![hash, row[4], group, row[2], row[3]])?;
                                load.current_text = Some((hash.to_vec(), t.last_insert_rowid()));
                            }
                        }
                        Some(&TEXT_HOLDER) => {
                            let (_, id) = load.current_text.as_ref().ok_or_else(|| {
                                StoreError::Invalid("a reading names a text the build lacks".into())
                            })?;
                            let mut row = decode_row(value)?;
                            // A message's text: its reading's key in the
                            // composition stream, then its block.
                            if let Some(Value::Blob(mut reading)) = row.pop() {
                                let Value::Integer(idx) = row[4] else {
                                    return Err(StoreError::Invalid(
                                        "a reading's text names no block".into(),
                                    ));
                                };
                                reading.extend_from_slice(&(idx as u32).to_be_bytes());
                                load.reading_texts
                                    .as_mut()
                                    .expect("the reading texts' sorter")
                                    .push(&reading, &id.to_be_bytes())?;
                            }
                            *row.last_mut().expect("a text column") = Value::Integer(*id);
                            values.clear();
                            let refs: Vec<ValueRef<'_>> = row.iter().map(ValueRef::from).collect();
                            encode_row(&refs, &mut values);
                            let mut key = vec![0];
                            key.extend_from_slice(&rest[1..]);
                            load.holders
                                .as_mut()
                                .expect("the text holders' sorter")
                                .push(&key, &values)?;
                        }
                        _ => return Err(StoreError::Invalid("a text record of no kind".into())),
                    }
                }
                MARKS => {
                    // Every reading table is loaded by now.
                    let aliases = match &mut load.aliases {
                        Some(aliases) => aliases,
                        None => load.aliases.insert(super::identity::Aliases::load(&t)?),
                    };
                    let (&mark, rest) = rest
                        .split_first()
                        .ok_or_else(|| StoreError::Invalid("a build mark of no kind".into()))?;
                    match mark {
                        FALLBACK => {
                            load.fallback.insert(split_id(rest)?.0.to_owned());
                        }
                        MESSAGE_ID | FOLLOWED => {
                            let (message, rest) = split_id(rest)?;
                            let session = split_id(rest)?.0;
                            // A message one of whose names is a copy's is
                            // shown as its session decides.
                            let named = super::identity::message_session(message)
                                .is_some_and(|named| aliases.involve(&named));
                            if named {
                                load.fallback.insert(session.to_owned());
                            }
                            if mark == MESSAGE_ID {
                                // A message stated under several sessions, or
                                // also by a reading upkeep wrote.
                                if load.upkept_messages.contains(message) {
                                    load.fallback.insert(session.to_owned());
                                }
                                if let Some((last, other)) = &load.last_message
                                    && last == message
                                    && other != session
                                {
                                    load.fallback.insert(other.clone());
                                    load.fallback.insert(session.to_owned());
                                }
                                load.last_message = Some((message.to_owned(), session.to_owned()));
                            }
                        }
                        EVENT_ID => {
                            // An event stated under several sessions, or
                            // also by a reading upkeep wrote, is composed
                            // from all its readings once the stream is.
                            let (event, rest) = split_id(rest)?;
                            let session = split_id(rest)?.0;
                            if load.upkept_events.contains(event)
                                || load
                                    .last_event
                                    .as_ref()
                                    .is_some_and(|(last, other)| last == event && other != session)
                            {
                                load.shared_events.insert(event.to_owned());
                            }
                            load.last_event = Some((event.to_owned(), session.to_owned()));
                        }
                        _ => return Err(StoreError::Invalid("a build mark of no kind".into())),
                    }
                }
                _ => {
                    let table = load.tables.get(usize::from(kind - TABLES)).ok_or_else(|| {
                        StoreError::Invalid("a build stream names no table".into())
                    })?;
                    let mut row = decode_row(value)?;
                    if let Some(at) = table.input {
                        row[at] = Value::Integer(input_pk(&t, &mut load.input, key_input(rest)?)?);
                    }
                    insert(&t, table, &row)?;
                }
            }
            written += 1;
        };
        if !more {
            // Groups whose every text another worker also held.
            t.execute(
                "DELETE FROM text_group WHERE id NOT IN (SELECT group_id FROM text)",
                [],
            )?;
        }
        t.commit()?;
        job.count(written);
        Ok(more)
    }

    /// Loads up to a transaction's worth of reading text rows; false once
    /// they are all loaded.
    fn holders_step(&mut self, job: &mut BuildJob, holders: &mut TextHolders) -> Result<bool> {
        let started = std::time::Instant::now();
        let t = Transaction::new_unchecked(&self.index, TransactionBehavior::Immediate)?;
        let mut written = 0u64;
        let more = loop {
            if written == ROWS_PER_TRANSACTION
                || (written % 1024 == 1023 && started.elapsed() > STEP_TIME)
            {
                break true;
            }
            let Some((key, value)) = holders.merged.next()? else {
                break false;
            };
            let mut row = decode_row(value)?;
            row[0] = Value::Integer(input_pk(&t, &mut holders.input, key_input(&key[1..])?)?);
            insert(&t, &holders.table, &row)?;
            written += 1;
        };
        t.commit()?;
        job.count(written);
        Ok(more)
    }

    /// Composes the next part of a build with the composition upkeep uses
    /// (docs/architecture.md, one path), in a transaction of its own: links
    /// and documents first, from the index, so each session's item index
    /// sees its links; then sessions from the stream, grouped as they come;
    /// then, from the index, the sessions whose readings reach other
    /// sessions' readings, with the events about inputs and the events
    /// placed after another session's message. True once all is composed.
    fn compose_step(&mut self, state: &mut ComposeState) -> Result<bool> {
        self.stop.check()?;
        match state.step {
            Step::Links => {
                let mut touched = super::readings::Touched::default();
                touched.links.extend(
                    self.index
                        .prepare("SELECT DISTINCT statement FROM link_parse")?
                        .query_map([], |r| r.get::<_, Vec<u8>>(0))?
                        .collect::<rusqlite::Result<Vec<_>>>()?,
                );
                touched.documents.extend(
                    self.index
                        .prepare("SELECT DISTINCT document_id FROM document_parse")?
                        .query_map([], |r| r.get::<_, String>(0))?
                        .collect::<rusqlite::Result<Vec<_>>>()?,
                );
                self.compose_from_index(touched)?;
                state.step = Step::Stream;
            }
            Step::Stream => {
                self.compose_stream(state)?;
                if state.stream_done {
                    state.step = Step::Indexes;
                }
            }
            Step::Indexes => match state.indexes.pop() {
                Some(sql) => {
                    self.index.execute(&sql, [])?;
                }
                None => {
                    state.step = Step::Index { next: 0 };
                }
            },
            Step::Index { next } => {
                let chunk = &state.fallback[next.min(state.fallback.len())
                    ..(next + SESSIONS_PER_TRANSACTION).min(state.fallback.len())];
                let mut touched = super::readings::Touched::default();
                touched.whole.extend(chunk.iter().cloned());
                touched.sessions.extend(chunk.iter().cloned());
                for session in chunk {
                    for row in self
                        .index
                        .prepare_cached(
                            "SELECT session_id,call_id FROM call_outcome WHERE session_id=?1",
                        )?
                        .query_map([session], |r| <(String, String)>::try_from(r))?
                    {
                        touched.calls.insert(row?);
                    }
                    for row in self
                        .index
                        .prepare_cached(
                            "SELECT fields,data,kind FROM event_parse WHERE session_id=?1
                             ORDER BY at,input_pk,chain,seq",
                        )?
                        .query_map([session], |r| <(Vec<u8>, Vec<u8>, String)>::try_from(r))?
                    {
                        let (fields, data, kind) = row?;
                        if polylogue_model::live::is_live_event(&kind) {
                            let mut event: Event = crate::encoding::unpack_json(&fields)?;
                            event.data = crate::encoding::unpack_json(&data)?;
                            touched.live.push(event);
                        }
                    }
                }
                state.sessions += chunk.len() as u64;
                if next == 0 {
                    // What no session holds: input events; and events that
                    // follow another session's message.
                    for row in self
                        .index
                        .prepare("SELECT event_id FROM event_parse WHERE session_id IS NULL")?
                        .query_map([], |r| r.get::<_, String>(0))?
                    {
                        touched.events.insert((None, row?));
                    }
                    touched.events.extend(std::mem::take(&mut state.deferred));
                }
                self.compose_from_index(touched)?;
                let next = next + SESSIONS_PER_TRANSACTION;
                state.step = if next < state.fallback.len() {
                    Step::Index { next }
                } else {
                    Step::Count
                };
            }
            Step::Count => {
                let tx = self
                    .index
                    .transaction_with_behavior(TransactionBehavior::Immediate)?;
                let tables: Vec<String> = tx
                    .prepare("SELECT name FROM row_count")?
                    .query_map([], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?;
                for table in tables {
                    tx.execute(
                        &format!(
                            "UPDATE row_count SET n=(SELECT count(*) FROM \"{table}\") WHERE name=?1"
                        ),
                        [&table],
                    )?;
                }
                for sql in std::mem::take(&mut state.counting) {
                    tx.execute(&sql, [])?;
                }
                tx.commit()?;
                state.step = Step::Done;
            }
            Step::Done => {}
        }
        Ok(matches!(state.step, Step::Done))
    }

    /// Composes sessions from the stream, in one transaction, until about
    /// [`MESSAGES_PER_TRANSACTION`] messages or [`SESSIONS_PER_TRANSACTION`]
    /// sessions are written.
    fn compose_stream(&mut self, state: &mut ComposeState) -> Result<()> {
        // A session from the stream refers only to rows the build wrote.
        self.index.pragma_update(None, "foreign_keys", false)?;
        let result = self.compose_stream_transaction(state);
        self.index.pragma_update(None, "foreign_keys", true)?;
        result
    }

    fn compose_stream_transaction(&mut self, state: &mut ComposeState) -> Result<()> {
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        let context = super::shown::Context {
            tx: &tx,
            counter,
            changes: &changes,
        };
        let mut composer = super::shown::Composer::new(
            &context,
            Arc::clone(&state.aliases),
            Arc::clone(&state.shared_events),
        )?;
        let mut inputs = InputState::default();
        let mut kinds = std::collections::HashMap::new();
        let (mut messages, mut sessions) = (0usize, 0usize);
        // A transaction ends with a session, once it holds enough or took
        // its share of the writer's time.
        let started = std::time::Instant::now();
        let mut open = false;
        while open
            || (messages < MESSAGES_PER_TRANSACTION
                && sessions < SESSIONS_PER_TRANSACTION
                && started.elapsed() < STEP_TIME)
        {
            let piece = match state.pieces.recv() {
                Ok(piece) => piece?,
                // The reader ended: every session of the stream is composed.
                Err(_) => {
                    state.stream_done = true;
                    break;
                }
            };
            match piece {
                Piece::Begin => {
                    composer.begin();
                    kinds.clear();
                    open = true;
                }
                Piece::Facts { input, chain, kind } => {
                    composer.facts();
                    let (pk, _) = inputs.of(&tx, &input)?;
                    kinds.insert((pk, chain), kind);
                }
                Piece::Message { id, mut readings } => {
                    for reading in &mut readings {
                        let (pk, finished) = inputs.of(&tx, &reading.input)?;
                        reading.input_pk = pk;
                        if let Some(given) = &mut reading.given {
                            given.kind = kinds.get(&(pk, reading.chain)).copied().flatten();
                            given.superseded =
                                finished.is_some_and(|newest| newest > reading.chain);
                        }
                    }
                    composer.message(&id, readings)?;
                    messages += 1;
                }
                Piece::Event(mut event) => {
                    event.input_pk = inputs.of(&tx, &event.input)?.0;
                    composer.event(event);
                }
                Piece::Outcome(mut outcome) => {
                    outcome.input_pk = inputs.of(&tx, &outcome.input)?.0;
                    composer.outcome(outcome);
                }
                Piece::End(session) => {
                    composer.end(&session)?;
                    sessions += 1;
                    open = false;
                }
            }
        }
        let (_, mut late) = composer.finish();
        state.deferred.append(&mut late);
        state.sessions += sessions as u64;
        rows::finish_summaries(&tx, &changes, counter)?;
        tx.commit()?;
        Ok(())
    }

    /// Composes what `touched` names from the readings in the index, as
    /// upkeep does, in one transaction.
    fn compose_from_index(&mut self, touched: super::readings::Touched) -> Result<()> {
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        rows::finish(&tx, touched, &changes, counter)?;
        tx.commit()?;
        Ok(())
    }
}

/// The input a session's readings last named: its key in the build file
/// and its newest chain read to its end or waiting for more. A session's
/// readings come from few inputs, so each is looked up once per run of it.
#[derive(Default)]
struct InputState {
    id: String,
    pk: i64,
    finished: Option<i64>,
}

impl InputState {
    fn of(&mut self, tx: &Transaction<'_>, id: &str) -> Result<(i64, Option<i64>)> {
        if self.id != id || self.id.is_empty() {
            let (pk, finished): (i64, Option<i64>) = tx
                .prepare_cached(
                    "SELECT i.pk,(SELECT max(p.chain) FROM parse_state p WHERE p.input_pk=i.pk
                         AND p.status IN ('done','waiting'))
                     FROM input_ref i WHERE i.id=?1",
                )?
                .query_row([id], |r| <(i64, Option<i64>)>::try_from(r))
                .optional()?
                .ok_or_else(|| {
                    StoreError::Invalid("a reading names an input the build lacks".into())
                })?;
            *self = Self {
                id: id.to_owned(),
                pk,
                finished,
            };
        }
        Ok((self.pk, self.finished))
    }
}

/// What the stream reader hands composition, in stream order: each session
/// begun, its facts' kinds, each message's readings with their values, its
/// event readings and outcomes, then its end. Input keys, kinds and whether
/// a newer chain supersedes a reading are filled in by composition, which
/// reads the build file.
pub(super) enum Piece {
    Begin,
    Facts {
        input: String,
        chain: i64,
        kind: Option<polylogue_model::SessionKind>,
    },
    Message {
        id: String,
        readings: Vec<super::shown::MessageReading>,
    },
    Event(super::shown::EventRow),
    Outcome(super::shown::OutcomeRow),
    End(String),
}

/// Pieces decoded ahead of composition.
const PIECES_AHEAD: usize = 1024;

/// Reads the composition stream on a thread of its own, merging it with
/// each message reading's text keys and decoding its records, so that the
/// writer only composes. Sessions composed from the index are passed over.
fn read_stream(
    mut merged: Merger,
    mut texts: Merger,
    skipped: BTreeSet<String>,
    aliases: Arc<super::identity::Aliases>,
) -> std::sync::mpsc::Receiver<Result<Piece>> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(PIECES_AHEAD);
    let spawned = std::thread::Builder::new()
        .name("polylogue-compose-stream".into())
        .spawn(move || {
            let mut send = |piece: Piece| sender.send(Ok(piece)).is_ok();
            let result = (|| -> Result<()> {
                while let Some((key, _)) = merged.peek()? {
                    if key.first() != Some(&COMPOSE) {
                        return Err(StoreError::Invalid(
                            "a build stream holds records after its composition".into(),
                        ));
                    }
                    let session = split_id(&key[1..])?.0.to_owned();
                    if skipped.contains(&session) || aliases.involve(&session) {
                        while merged
                            .peek()?
                            .is_some_and(|(key, _)| of_session(key, &session))
                        {
                            merged.next()?;
                        }
                        continue;
                    }
                    if !send(Piece::Begin)
                        || !read_session(&mut merged, &mut texts, &session, &mut send)?
                    {
                        return Ok(());
                    }
                    if !send(Piece::End(session)) {
                        return Ok(());
                    }
                }
                Ok(())
            })();
            if let Err(error) = result {
                let _ = sender.send(Err(error));
            }
        });
    if let Err(error) = spawned {
        let (sender, failed) = std::sync::mpsc::sync_channel(1);
        let _ = sender.send(Err(error.into()));
        return failed;
    }
    receiver
}

/// Whether `key` is one of `session`'s composition records.
fn of_session(key: &[u8], session: &str) -> bool {
    key.len() > session.len() + 1
        && key[0] == COMPOSE
        && &key[1..=session.len()] == session.as_bytes()
        && key[session.len() + 1] == 0
}

/// Decodes one session's records into pieces; false once composition
/// stopped taking them.
fn read_session(
    merged: &mut Merger,
    texts: &mut Merger,
    session: &str,
    send: &mut impl FnMut(Piece) -> bool,
) -> Result<bool> {
    use super::shown::{EventRow, Given, MessageReading, OutcomeRow, ReadingValue};
    let bad = || StoreError::Invalid("a composition record is malformed".into());
    let mut pending: Option<(String, Vec<MessageReading>)> = None;
    let prefix = session.len() + 2;
    while let Some((key, value)) = merged.peek()? {
        if !of_session(key, session) {
            break;
        }
        let (key, value) = (key.to_vec(), value.to_vec());
        merged.next()?;
        let kind = key[prefix];
        let rest = &key[prefix + 1..];
        let row = decode_row(&value)?;
        // (item, input, chain) of a reading; facts name no item.
        let (item, input, chain) = if kind == FACTS {
            let (input, chain) = split_id(rest)?;
            (String::new(), input.to_owned(), chain)
        } else {
            let (item, rest) = split_id(rest)?;
            let (input, chain) = split_id(rest)?;
            (item.to_owned(), input.to_owned(), chain)
        };
        let chain = i64::from_be_bytes(chain.try_into().map_err(|_| bad())?);
        if kind != MESSAGE
            && let Some((id, readings)) = pending.take()
            && !send(Piece::Message { id, readings })
        {
            return Ok(false);
        }
        let int = |n: usize| match row.get(n) {
            Some(Value::Integer(v)) => Ok(*v),
            _ => Err(bad()),
        };
        let int_or_null = |n: usize| match row.get(n) {
            Some(Value::Integer(v)) => Ok(Some(*v)),
            Some(Value::Null) => Ok(None),
            _ => Err(bad()),
        };
        let text = |n: usize| match row.get(n) {
            Some(Value::Text(v)) => Ok(v.as_str()),
            _ => Err(bad()),
        };
        let blob = |n: usize| match row.get(n) {
            Some(Value::Blob(v)) => Ok(v.as_slice()),
            _ => Err(bad()),
        };
        let route = |n: usize| {
            polylogue_model::Route::from_name(text(n)?)
                .ok_or_else(|| StoreError::stored(0, "unknown stored route"))
        };
        match kind {
            FACTS => {
                let kind = match row.first() {
                    Some(Value::Text(kind)) => Some(
                        polylogue_model::SessionKind::from_name(kind)
                            .ok_or_else(|| StoreError::stored(0, "unknown stored session kind"))?,
                    ),
                    _ => None,
                };
                if !send(Piece::Facts { input, chain, kind }) {
                    return Ok(false);
                }
            }
            MESSAGE => {
                if pending.as_ref().is_some_and(|(id, _)| *id != item)
                    && let Some((id, readings)) = pending.take()
                    && !send(Piece::Message { id, readings })
                {
                    return Ok(false);
                }
                // Columns: start, end, seq, position, order time, route,
                // completion, content-named, hash, the message without its
                // blocks, layout, words, items; then each block's packed
                // body, data and text hash. The message's blocks are read
                // back from their packed bodies as a block row is, without
                // their data: composition writes each block's data from the
                // record and reads none from the message.
                let hash = blob(8)?.to_vec();
                if row.len() < 13 || (row.len() - 13) % 3 != 0 {
                    return Err(bad());
                }
                let blocks = (row.len() - 13) / 3;
                let mut message: Message = crate::encoding::read_json(blob(9)?)?;
                message.blocks = (0..blocks)
                    .map(|n| crate::encoding::unpack_block(blob(13 + 3 * n)?))
                    .collect::<Result<_>>()?;
                let selection =
                    crate::encoding::message_block_selection(&hash, Some(blob(10)?), blocks)?;
                let mut content = crate::encoding::PreparedContent {
                    hash: selection.source_version.0,
                    blocks: Vec::with_capacity(blocks),
                    texts: Vec::with_capacity(blocks),
                    data: Vec::with_capacity(blocks),
                };
                for n in 0..blocks {
                    let at = 13 + 3 * n;
                    content.blocks.push(blob(at)?.to_vec());
                    content.data.push(match &row[at + 1] {
                        Value::Blob(data) => Some(data.clone()),
                        _ => None,
                    });
                    content.texts.push(match &row[at + 2] {
                        Value::Blob(hash) => Some(hash.as_slice().try_into().map_err(|_| bad())?),
                        _ => None,
                    });
                }
                // The text keys the load gave this reading's text rows.
                // Both streams are in key order; the text rows of readings
                // composed from the index are passed over.
                let reading = key.as_slice();
                let mut ids = vec![None; blocks];
                while let Some((text_key, id)) = texts.peek()? {
                    if text_key.len() == reading.len() + 4 && text_key.starts_with(reading) {
                        let idx = u32::from_be_bytes(
                            text_key[reading.len()..].try_into().expect("four bytes"),
                        );
                        let id = i64::from_be_bytes(id.try_into().map_err(|_| bad())?);
                        *ids.get_mut(idx as usize).ok_or_else(bad)? = Some(id);
                    } else if text_key >= reading {
                        break;
                    }
                    texts.next()?;
                }
                if ids
                    .iter()
                    .zip(&content.texts)
                    .any(|(id, text)| id.is_some() != text.is_some())
                {
                    return Err(StoreError::Invalid(
                        "a message reading's text rows differ from its blocks".into(),
                    ));
                }
                let items = crate::items::entry_bytes::decode_message(blob(12)?, &message.id)?;
                let value = ReadingValue {
                    message,
                    words: int(11)?,
                    selection,
                    items,
                    content,
                    text_ids: Some(ids),
                };
                let reading = MessageReading {
                    input_pk: 0,
                    input,
                    chain,
                    message: item.clone(),
                    session: session.to_owned(),
                    start: int(0)?,
                    end: int(1)?,
                    seq: int(2)?,
                    position: int(3)?,
                    order_time: int_or_null(4)?,
                    route: route(5)?,
                    completion: polylogue_model::Completion::from_name(text(6)?)
                        .ok_or_else(|| StoreError::stored(0, "unknown stored completion"))?,
                    content: int(7)? != 0,
                    hash,
                    given: Some(Given {
                        kind: None,
                        superseded: false,
                        value,
                    }),
                };
                pending
                    .get_or_insert_with(|| (item, Vec::new()))
                    .1
                    .push(reading);
            }
            EVENT => {
                if !send(Piece::Event(EventRow {
                    input,
                    input_pk: 0,
                    chain,
                    event_id: item,
                    route: route(0)?,
                    hash: blob(1)?.to_vec(),
                    fields: blob(2)?.to_vec(),
                    data: blob(3)?.to_vec(),
                    seq: int(4)?,
                    start: int_or_null(5)?,
                    end: int_or_null(6)?,
                })) {
                    return Ok(false);
                }
            }
            OUTCOME => {
                if !send(Piece::Outcome(OutcomeRow {
                    input_pk: 0,
                    chain,
                    position: int(0)?,
                    input,
                    route: route(2)?,
                    call_id: item,
                    body: text(1)?.to_owned(),
                })) {
                    return Ok(false);
                }
            }
            _ => return Err(bad()),
        }
    }
    if let Some((id, readings)) = pending.take() {
        return Ok(send(Piece::Message { id, readings }));
    }
    Ok(true)
}

/// The build file's key of `id`, looked up once per input as the merge
/// passes it in key order.
fn input_pk(tx: &Transaction<'_>, current: &mut Option<(String, i64)>, id: &str) -> Result<i64> {
    if let Some((held, pk)) = current
        && held == id
    {
        return Ok(*pk);
    }
    let pk: i64 = tx
        .prepare_cached("SELECT pk FROM input_ref WHERE id=?1")?
        .query_row([id], |r| r.get(0))?;
    *current = Some((id.to_owned(), pk));
    Ok(pk)
}

fn insert(tx: &Transaction<'_>, table: &Table, row: &[Value]) -> Result<()> {
    tx.prepare_cached(&table.insert)?
        .execute(rusqlite::params_from_iter(row))?;
    Ok(())
}
