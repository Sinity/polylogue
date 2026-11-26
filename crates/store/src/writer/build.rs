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

mod composers;
use crate::archive::index_build::Paths;
use crate::sorted_runs::{Merger, Sorter};
use rusqlite::types::{Value, ValueRef};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The cache of a worker's content-group file, at most; the rest of a
/// worker's memory is its sort buffer.
const SCRATCH_CACHE: usize = 64 << 20;

/// A worker's scratch index past this size after a chain is opened again,
/// so that a large chain's rows leave no memory behind them.
const SCRATCH_KEPT: i64 = 64 << 20;

/// Sessions composed per transaction.
const SESSIONS_PER_TRANSACTION: usize = 512;

/// Rows the merge writes per transaction, at most.
const ROWS_PER_TRANSACTION: u64 = 200_000;

/// How long one step of the writer's build work runs before it commits
/// and lets the writer serve what waits.
const STEP_TIME: std::time::Duration = std::time::Duration::from_millis(250);

/// The streams of a build, by the first byte of a record's key.
const INPUTS: u8 = 0;
/// Content groups, by worker and the worker's number for the group; a
/// group's record is its kind ([`TEXT_GROUP`] or [`READING_GROUP`]), hash and
/// body. Until the load gives it its key, a row names a group by
/// [`provisional`].
const GROUPS: u8 = 1;
const TEXT_GROUP: u8 = 0;
const READING_GROUP: u8 = 1;
/// Texts and the reading text rows that hold them, by hash: a text's own
/// record (`TEXT`) comes before the rows that name it (`TEXT_HOLDER`).
const TEXTS: u8 = 2;
const TEXT: u8 = 0;
const TEXT_HOLDER: u8 = 1;
/// The reading tables, in name order, from this byte on.
const TABLES: u8 = 3;

/// What decides which sessions are composed from the stream, read before
/// it: sessions whose composition reaches other sessions' readings
/// ([`FALLBACK`]); each message ID with the sessions stating it and the
/// sessions with an event following it ([`MESSAGE_ID`], then [`STATES`] or
/// [`FOLLOWS`]: one message's marks come together, its statements first);
/// and each event ID with its session ([`EVENT_ID`]).
const MARKS: u8 = 253;
const FALLBACK: u8 = 0;
const MESSAGE_ID: u8 = 1;
const EVENT_ID: u8 = 3;
const STATES: u8 = 0;
const FOLLOWS: u8 = 1;
/// The readings composition takes, grouped by the session stating them:
/// its facts' kinds, each message's readings with their values, its event
/// readings and its calls' background outcomes, in that order.
const COMPOSE: u8 = 254;
const FACTS: u8 = 0;
const MESSAGE: u8 = 1;
const EVENT: u8 = 2;
const OUTCOME: u8 = 3;

/// The tables composition writes, and those of their indexes kept while it
/// composes from the stream: those whose first column is the session,
/// message or event a row belongs to, which are also all it reads (its rows
/// in order and the replies to each message). Composition makes those keys in the order it writes, so each
/// such entry lands at its index's end and costs no read; made after the
/// stream, each would scan its whole table again. Their other indexes,
/// keyed by IDs in no order (a message's, an event's), are made after the
/// stream: kept while it writes, each new row would read a page at random.
const SHOWN_TABLES: &[&str] = &["message", "message_route", "block", "tool_call", "event"];
const COMPOSED_IN_ORDER: &[&str] = &["session_pk", "message_pk", "event_pk"];

/// The table an index statement is on, and its first column.
fn index_target(sql: &str) -> Option<(&str, &str)> {
    let (_, on) = sql.split_once(" ON ")?;
    let (table, columns) = on.split_once('(')?;
    let first = columns.split([',', ')']).next()?;
    Some((table.trim().trim_matches('"'), first.trim()))
}

/// Whether an index on a shown table is kept while composition writes it.
fn kept_while_composing(sql: &str) -> bool {
    index_target(sql).is_some_and(|(_, first)| COMPOSED_IN_ORDER.contains(&first))
}

/// Whether an index on a reading table is kept while the load writes it:
/// one whose first column is the input. The load writes the reading tables
/// input by input, chain by chain, so such an index only grows at its end;
/// made after the load, it would scan its whole table again.
fn kept_while_loading(sql: &str) -> bool {
    index_target(sql)
        .is_some_and(|(table, first)| !SHOWN_TABLES.contains(&table) && first == "input_pk")
}

/// The indexes a build's load concerns, as (name, statement, table): every
/// one on a stored table but the one that makes an equal content group one
/// row.
const LOAD_INDEXES: &str =
    "SELECT name,sql,tbl_name FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL
     AND name <> 'content_group_hash'
     AND tbl_name NOT IN (SELECT name FROM sqlite_schema WHERE sql LIKE 'CREATE VIRTUAL TABLE%')";

/// How a worker's run names a content group before the load gives it its
/// key: the worker's number and its number for the group.
fn provisional(worker: u32, seq: u64) -> i64 {
    (i64::from(worker) << 40) | seq as i64
}

/// The keys the load gave the workers' content groups.
#[derive(Default)]
pub(super) struct GroupKeys {
    /// Each worker's first group's key, less its number for that group.
    bases: std::collections::HashMap<u32, i64>,
    /// Groups not stored because one with the same bytes already was (by
    /// another worker, another chain or upkeep), by the key they would have
    /// had: the stored group's key.
    shared: std::collections::HashMap<i64, i64>,
}

impl GroupKeys {
    /// The key of the group a run names `provisional`.
    fn key(&self, provisional: i64) -> Result<i64> {
        let worker = (provisional >> 40) as u32;
        let base = self
            .bases
            .get(&worker)
            .ok_or_else(|| StoreError::Invalid("a run names a group the build lacks".into()))?;
        let key = base + (provisional & ((1 << 40) - 1));
        Ok(self.shared.get(&key).copied().unwrap_or(key))
    }
}

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

/// One phase of a build, as its record shows it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BuildPhase {
    pub name: String,
    pub seconds: f64,
    /// Bytes this process read from and wrote to storage during the phase.
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// Reads the process's control group made from its devices during the
    /// phase, and their bytes, when the system reports them: the mean read
    /// size tells reads in file order (large) from reads at random (a page
    /// or a few).
    pub device_reads: Option<(u64, u64)>,
    /// The process's peak resident memory when the phase ended, and its
    /// resident memory then.
    pub peak_memory: u64,
    pub resident_memory: u64,
    /// What the phase handled: chains, rows, sessions or texts.
    pub items: u64,
}

/// The steps the writer runs in a build, in order, as `status`, the log and
/// the docs name them.
pub const SORTING_TEXTS: &str = "sorting texts";
pub const WRITING_TABLES: &str = "writing tables";
pub const INDEXING: &str = "indexing";
/// Sessions whose every reading one file states, read in key order.
pub const ASSEMBLING_SESSIONS: &str = "assembling sessions";
/// The session tables' indexes that assembling does not fill as it writes.
pub const INDEXING_SESSIONS: &str = "indexing sessions";
/// Sessions whose readings several files state or that copies name, and the
/// events no session holds, read from the index.
pub const JOINING_SESSIONS: &str = "joining sessions across files";
pub const FINISHING_INDEXES: &str = "finishing indexes";
pub const SEARCH_INDEX: &str = "search index";
pub const INSTALLING: &str = "installing";
pub const STEPS: &[&str] = &[
    SORTING_TEXTS,
    WRITING_TABLES,
    INDEXING,
    ASSEMBLING_SESSIONS,
    INDEXING_SESSIONS,
    JOINING_SESSIONS,
    FINISHING_INDEXES,
    SEARCH_INDEX,
    INSTALLING,
];

/// What a writer step counts as it works.
fn counted(step: &str) -> Option<CountUnit> {
    match step {
        SORTING_TEXTS | WRITING_TABLES => Some(CountUnit::Records),
        INDEXING | INDEXING_SESSIONS => Some(CountUnit::Indexes),
        ASSEMBLING_SESSIONS | JOINING_SESSIONS => Some(CountUnit::Sessions),
        _ => None,
    }
}

/// Measures one phase from the process's storage counters.
pub struct PhaseClock {
    name: &'static str,
    unit: Option<CountUnit>,
    started: Instant,
    io: (u64, u64),
    device: Option<(u64, u64)>,
}

impl PhaseClock {
    /// Starts measuring step `name`, whose items are `unit`.
    pub fn start(name: &'static str, unit: Option<CountUnit>) -> Self {
        Self {
            name,
            unit,
            started: Instant::now(),
            io: process_io(),
            device: device_reads(),
        }
    }

    pub fn end(self, items: u64) -> BuildPhase {
        let (read, written) = process_io();
        let device = device_reads()
            .zip(self.device)
            .map(|(now, then)| (now.0.saturating_sub(then.0), now.1.saturating_sub(then.1)));
        let phase = BuildPhase {
            name: self.name.to_owned(),
            seconds: self.started.elapsed().as_secs_f64(),
            bytes_read: read.saturating_sub(self.io.0),
            bytes_written: written.saturating_sub(self.io.1),
            device_reads: device,
            peak_memory: status_bytes("VmHWM:"),
            resident_memory: status_bytes("VmRSS:"),
            items,
        };
        crate::log!(
            info,
            "index: {} took {:.1} s; {}{:.1} MB read{}, {:.1} MB written; peak memory {:.0} MB, resident {:.0} MB",
            phase.name,
            phase.seconds,
            self.unit
                .map(|unit| format!("{} {}; ", phase.items, unit.word()))
                .unwrap_or_default(),
            phase.bytes_read as f64 / 1e6,
            phase
                .device_reads
                .filter(|(reads, _)| *reads > 0)
                .map(|(reads, bytes)| format!(
                    " in {reads} reads of {:.0} KiB",
                    bytes as f64 / reads as f64 / 1024.
                ))
                .unwrap_or_default(),
            phase.bytes_written as f64 / 1e6,
            phase.peak_memory as f64 / 1e6,
            phase.resident_memory as f64 / 1e6,
        );
        phase
    }
}

/// The bytes `conn`'s page cache holds now.
#[allow(unsafe_code)]
fn cache_used(conn: &Connection) -> u64 {
    let (mut now, mut most) = (0, 0);
    // SAFETY: the handle is `conn`'s own, used on its thread for a read of
    // its statistics, which SQLite keeps per connection.
    let status = unsafe {
        rusqlite::ffi::sqlite3_db_status(
            conn.handle(),
            rusqlite::ffi::SQLITE_DBSTATUS_CACHE_USED,
            &mut now,
            &mut most,
            0,
        )
    };
    if status == rusqlite::ffi::SQLITE_OK {
        now.max(0) as u64
    } else {
        0
    }
}

/// Reads and bytes read from devices by this process's control group,
/// summed over its devices (Linux cgroup v2 `io.stat`).
fn device_reads() -> Option<(u64, u64)> {
    let group = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = group.lines().find_map(|line| line.strip_prefix("0::"))?;
    let stat = std::fs::read_to_string(format!("/sys/fs/cgroup{}/io.stat", path.trim())).ok()?;
    let mut total = (0, 0);
    for field in stat.split_whitespace() {
        if let Some(n) = field.strip_prefix("rios=") {
            total.0 += n.parse::<u64>().ok()?;
        } else if let Some(n) = field.strip_prefix("rbytes=") {
            total.1 += n.parse::<u64>().ok()?;
        }
    }
    Some(total)
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

/// A memory figure of this process in bytes, by its field name in
/// `/proc/self/status` (`VmHWM:` the peak resident memory, `VmRSS:` the
/// resident memory now).
fn status_bytes(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(field))
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
    /// The column naming a content group, which a run gives provisionally.
    group: Option<usize>,
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
            group: columns.iter().position(|c| c == "content_group"),
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
            "input_ref" | "text" | "content_group" | "parse_text" => continue,
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

/// The mark that `session` states message `message` ([`STATES`]) or holds
/// an event following it ([`FOLLOWS`]).
fn message_mark(message: &str, how: u8, session: &str) -> Vec<u8> {
    let mut key = vec![MARKS, MESSAGE_ID];
    key.extend_from_slice(message.as_bytes());
    key.push(0);
    key.push(how);
    key.extend_from_slice(session.as_bytes());
    key.push(0);
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
    tables: Vec<Table>,
    sorter: Sorter,
    number: u32,
    groups: u64,
    /// The file the scratch index's content groups are written to.
    scratch: PathBuf,
    archive_db: PathBuf,
    cache: usize,
    /// How many bytes of prepared values a chain keeps (see
    /// [`OpenChain::values`]).
    value_memory: usize,
}

/// A worker's scratch index: its rows in memory, where a chain's writes
/// look them up at random, and its content groups, which are only appended
/// and read back in order, in an attached file. A chain's rows take memory
/// in proportion to the chain, which the engine admits it by.
fn open_scratch(
    groups: &Path,
    archive_db: &Path,
    cache: usize,
) -> Result<(Connection, Vec<Table>)> {
    let conn = Connection::open_in_memory()?;
    conn.pragma_update(None, "temp_store", "memory")?;
    crate::schema::ensure_index(&conn)?;
    // A chain's rows leave the scratch index table by table; the build
    // file's references are those of its own load.
    conn.pragma_update(None, "foreign_keys", false)?;
    let tables = tables(&conn)?;
    let declared: Vec<String> = conn
        .prepare(
            "SELECT sql FROM sqlite_schema WHERE tbl_name='content_group' AND sql IS NOT NULL
             ORDER BY type='index'",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    conn.execute_batch("DROP TABLE main.content_group")?;
    crate::archive::index_build::remove_run(groups)?;
    conn.execute(
        "ATTACH DATABASE ?1 AS g",
        [groups
            .to_str()
            .ok_or_else(|| StoreError::Invalid("a scratch path is not UTF-8".into()))?],
    )?;
    conn.pragma_update(Some("g"), "journal_mode", "off")?;
    conn.pragma_update(Some("g"), "synchronous", "off")?;
    conn.pragma_update(Some("g"), "cache_size", -((cache >> 10) as i64))?;
    for sql in declared {
        // The same table and index in the attached file, which the chain
        // writer's unqualified name then finds.
        let sql = sql
            .replacen("TABLE content_group", "TABLE g.content_group", 1)
            .replacen("INDEX content_group_hash", "INDEX g.content_group_hash", 1);
        conn.execute_batch(&sql)?;
    }
    let uri = format!("file:{}?mode=ro", archive_db.display());
    conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
    crate::source::views(&conn)?;
    Ok((conn, tables))
}

impl BuildWorker {
    /// Creates worker `number` of the build in progress, holding at most
    /// `memory` bytes in its content-group file's cache and its sort
    /// buffer.
    pub fn create(archive: &Archive, number: u64, memory: usize) -> Result<Self> {
        if !archive.index_build.building() {
            return Err(StoreError::Invalid(
                "a build worker belongs to a build in progress".into(),
            ));
        }
        let number = u32::try_from(number)
            .map_err(|_| StoreError::Invalid("too many build workers".into()))?;
        let paths = Paths::of(&archive.root);
        std::fs::create_dir_all(&paths.runs)?;
        let scratch = paths.runs.join(format!("scratch-{number:06}.db"));
        let cache = SCRATCH_CACHE.min(memory / 2).max(16 << 20);
        let archive_db = archive.archive_db();
        let (conn, tables) = open_scratch(&scratch, &archive_db, cache)?;
        let register = crate::archive::open_read(&archive_db)?;
        Ok(Self {
            conn,
            register,
            tables,
            sorter: Sorter::new(
                &paths.runs,
                format!("w{number:06}"),
                memory.saturating_sub(cache).max(16 << 20),
            ),
            number,
            groups: 0,
            scratch,
            archive_db,
            cache,
            value_memory: memory.saturating_sub(cache).max(16 << 20) / 4,
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
        let held: i64 = self.conn.query_row(
            "SELECT page_count*page_size FROM pragma_page_count(), pragma_page_size()",
            [],
            |r| r.get(0),
        )?;
        if held > SCRATCH_KEPT {
            let (conn, tables) = open_scratch(&self.scratch, &self.archive_db, self.cache)?;
            self.conn = conn;
            self.tables = tables;
        }
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
            out: Out {
                sorter: &mut self.sorter,
                tables: &self.tables,
                number: self.number,
                groups: &mut self.groups,
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
            changes: &self.changes,
        };
        self.steps
            .write(&self.tx, &w, step, &mut super::readings::Touched::default())?;
        // What composition takes besides the stored content, kept from
        // what the step prepared rather than read back from the scratch
        // index.
        for (row, prepared) in step.output.messages.iter().zip(&mut step.prepared.messages) {
            let size = prepared.texts.len() * std::mem::size_of::<Option<[u8; 32]>>();
            if size > self.value_room {
                continue;
            }
            self.value_room -= size;
            self.values.insert(
                row.message.id.to_string(),
                (
                    crate::encoding::message_layout(&prepared.block_selection)?,
                    Prepared {
                        hash: prepared.hash,
                        block_count: prepared.block_selection.block_count,
                        words: super::shown::text_words(&row.message),
                        texts: prepared.texts.clone(),
                    },
                ),
            );
        }
        Ok(())
    }

    fn decoded(&mut self, batch: crate::staging::PreparedBatch) -> Result<()> {
        if !batch.batch.blobs.is_empty() {
            for result in self.writer.store_group(vec![batch])? {
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
    // The groups the chain's rows name take the worker's next numbers.
    let mut groups = std::collections::HashMap::new();
    {
        let mut statement = tx.prepare_cached(
            "SELECT g.id,g.hash,g.body,g.id IN (SELECT t.group_id FROM text t
                 WHERE t.id IN (SELECT text_id FROM parse_text))
             FROM content_group g WHERE g.id IN (
                 SELECT t.group_id FROM text t WHERE t.id IN (SELECT text_id FROM parse_text)
                 UNION SELECT content_group FROM message_parse
                 UNION SELECT content_group FROM event_parse)
             ORDER BY g.id",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let local: i64 = row.get(0)?;
            let seq = *out.groups;
            *out.groups += 1;
            groups.insert(local, provisional(out.number, seq));
            let mut key = vec![GROUPS];
            key.extend_from_slice(&out.number.to_be_bytes());
            key.extend_from_slice(&seq.to_be_bytes());
            value.clear();
            value.push(if row.get::<_, bool>(3)? {
                TEXT_GROUP
            } else {
                READING_GROUP
            });
            value.extend_from_slice(row.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?);
            value.extend_from_slice(row.get_ref(2)?.as_blob().map_err(rusqlite::Error::from)?);
            out.sorter.push(&key, &value)?;
        }
    }
    let group = |local: i64| {
        groups
            .get(&local)
            .copied()
            .ok_or_else(|| StoreError::Invalid("a scratch row names a group not moved out".into()))
    };
    {
        let mut statement = tx.prepare_cached(
            "SELECT t.hash,t.normalized_hash,t.group_id,t.start,t.length FROM text t
             WHERE t.id IN (SELECT text_id FROM parse_text)",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let mut key = vec![TEXTS];
            key.extend_from_slice(&row.get::<_, Vec<u8>>(0)?);
            key.push(TEXT);
            value.clear();
            encode_row(
                &[
                    ValueRef::Integer(group(row.get(2)?)?),
                    row.get_ref(3)?,
                    row.get_ref(4)?,
                    row.get_ref(1)?,
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
    compose_records(tx, out, input, &route, chain, values, &group)?;
    for (n, table) in out.tables.iter().enumerate() {
        let stream = TABLES + n as u8;
        let mut statement = tx.prepare_cached(&format!("SELECT * FROM \"{}\"", table.name))?;
        let mut rows = statement.query([])?;
        let mut seq = 0u32;
        while let Some(row) = rows.next()? {
            let named = table.group.map(|at| group(row.get(at)?)).transpose()?;
            let values = (0..table.columns.len())
                .map(|i| {
                    if Some(i) == table.input {
                        Ok(ValueRef::Null)
                    } else if Some(i) == table.group {
                        Ok(ValueRef::Integer(named.expect("a group column")))
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
        "content_group",
    ]) {
        tx.execute(&format!("DELETE FROM \"{table}\""), [])?;
    }
    Ok(())
}

/// Moves the chain's readings out into the composition stream, grouped
/// by the session stating them, each message reading with what composition
/// takes besides its stored content (docs/architecture.md, one path).
fn compose_records(
    tx: &Transaction<'_>,
    out: &mut Out<'_>,
    input: &str,
    route: &str,
    chain: i64,
    values: &mut std::collections::HashMap<String, (Vec<u8>, Prepared)>,
    group: &impl Fn(i64) -> Result<i64>,
) -> Result<()> {
    let mut value = Vec::new();
    let input_pk: i64 = tx.query_row("SELECT pk FROM input_ref WHERE id=?1", [input], |r| {
        r.get(0)
    })?;
    {
        let mut statement =
            tx.prepare_cached("SELECT session_id,json_extract(facts,'$.kind') FROM session_parse")?;
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
        let mut reader = crate::content::Reader::default();
        // In each chain's order, so that its texts are read group by group.
        let mut statement = tx.prepare_cached(
            "SELECT r.message_id,r.session_id,r.source_start,r.source_start+r.source_length,r.seq,r.record_position,
                 r.order_time,r.route,r.completion,r.id_kind='content',r.hash,
                 r.content_group,r.content_start,r.content_length,r.layout
             FROM message_parse r ORDER BY r.seq,r.record_position",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let message: String = row.get(0)?;
            let session: String = row.get(1)?;
            out.sorter
                .push(&message_mark(&message, STATES, &session), &[])?;
            // What a step prepared is the reading's when the reading is that
            // step's statement: the same version and blocks. A reading
            // joined from several statements is read back.
            let prepared = values.remove(&message).filter(|(layout, prepared)| {
                row.get_ref(10)
                    .is_ok_and(|hash| hash.as_blob().is_ok_and(|hash| hash == prepared.hash))
                    && row
                        .get_ref(14)
                        .is_ok_and(|stored| stored.as_blob().is_ok_and(|stored| stored == layout))
            });
            let reading = match prepared {
                Some((_, prepared)) => prepared,
                None => {
                    let stored =
                        super::readings::read_reading(tx, &mut reader, input_pk, chain, &message)?
                            .ok_or_else(|| {
                                StoreError::Invalid("a message reading has no body".into())
                            })?;
                    Prepared::of(super::shown::ReadingValue::of(stored)?)
                }
            };
            if reading.partial() {
                // Its value joins other readings of its version: the
                // session is composed from the index.
                out.sorter.push(&mark_key(FALLBACK, &[&session]), &[])?;
                continue;
            }
            let mut values: Vec<ValueRef<'_>> = (2..11)
                .map(|n| row.get_ref(n))
                .collect::<rusqlite::Result<_>>()?;
            values.push(ValueRef::Integer(group(row.get(11)?)?));
            values.push(row.get_ref(12)?);
            values.push(row.get_ref(13)?);
            values.push(ValueRef::Integer(reading.words));
            for text in &reading.texts {
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
            "SELECT event_id,session_id,route,hash,content_group,content_start,content_length,data_length,
                 seq,source_start,source_start+source_length,after_message_id
             FROM event_parse WHERE session_id IS NOT NULL",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let (event, session): (String, String) = (row.get(0)?, row.get(1)?);
            if let Some(after) = row.get::<_, Option<String>>(11)? {
                out.sorter
                    .push(&message_mark(&after, FOLLOWS, &session), &[])?;
            }
            out.sorter
                .push(&mark_key(EVENT_ID, &[&event, &session]), &[])?;
            let mut values: Vec<ValueRef<'_>> = (2..11)
                .map(|n| row.get_ref(n))
                .collect::<rusqlite::Result<_>>()?;
            values[2] = ValueRef::Integer(group(row.get(4)?)?);
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

/// What composing a message reading takes besides its stored content:
/// whether it holds every block of its version, its words and its blocks'
/// text hashes (see [`super::shown::ReadingValue`]).
struct Prepared {
    hash: [u8; 32],
    block_count: u64,
    words: i64,
    texts: Vec<Option<[u8; 32]>>,
}

impl Prepared {
    fn of(value: super::shown::ReadingValue) -> Self {
        Self {
            hash: value.selection.source_version.0,
            block_count: value.selection.block_count,
            words: value.words,
            texts: value.texts,
        }
    }

    /// Whether the reading holds only some of its version's blocks.
    fn partial(&self) -> bool {
        self.block_count != self.texts.len() as u64
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
                "SELECT session_id FROM session_parse UNION SELECT session_id FROM message_parse
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
    groups: Arc<GroupKeys>,
}

/// Where a build's composition stands between its steps
/// ([`Inner::compose_step`]); it holds no transaction or statement.
pub(super) struct ComposeState {
    step: Step,
    /// Indexes to make once the stream is composed.
    indexes: Vec<String>,
    /// The stream's runs (the composition records and each message
    /// reading's text keys), until composition reaches them.
    unread: Option<(Merger, Merger, Arc<GroupKeys>)>,
    /// The stream being composed: its reader groups sessions into batches
    /// that the composer threads compose, in at most `ahead` bytes.
    composing: Option<(composers::Pool, Arc<Gate>)>,
    ahead: usize,
    composers: usize,
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
    /// Sessions the stream composed, of the sessions it holds: every session
    /// a reading states that is not composed from the index.
    assembled: u64,
    to_assemble: u64,
    /// Indexes made after the stream, of how many.
    indexes_made: u64,
    indexes_total: u64,
    /// Names composed from the index so far, of `fallback`'s length.
    joined: u64,
}

enum Step {
    Links,
    Stream,
    /// The shown tables' indexes the stream does not read.
    Indexes,
    Index {
        next: usize,
    },
    Done,
}

impl ComposeState {
    /// The composition of a loaded build file by `composers` threads; the
    /// stream is read ahead of the writer in at most `ahead` bytes.
    pub(super) fn new(
        loaded: Loaded,
        indexes: Vec<String>,
        index: &Connection,
        ahead: usize,
        composers: usize,
    ) -> Result<Self> {
        let Loaded {
            merged,
            texts,
            mut fallback,
            shared_events,
            groups,
        } = loaded;
        let aliases = Arc::new(super::identity::Aliases::load(index)?);
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
        // The stream holds a session for every ID a reading states (its
        // record, a message, an event or a call's outcome), less those
        // composed from the index, which it passes over.
        let stated: u64 = index.scalar(
            "SELECT count(*) FROM (
                 SELECT DISTINCT session_id FROM session_parse
                 UNION SELECT DISTINCT session_id FROM message_parse
                 UNION SELECT DISTINCT session_id FROM event_parse WHERE session_id IS NOT NULL
                 UNION SELECT DISTINCT session_id FROM call_outcome)",
            [],
        )?;
        let mut passed_over = 0;
        for name in &fallback {
            if index.exists(
                "SELECT EXISTS(SELECT 1 FROM session_parse WHERE session_id=?1)
                     OR EXISTS(SELECT 1 FROM message_parse WHERE session_id=?1)
                     OR EXISTS(SELECT 1 FROM event_parse WHERE session_id=?1)
                     OR EXISTS(SELECT 1 FROM call_outcome WHERE session_id=?1)",
                [name],
            )? {
                passed_over += 1;
            }
        }
        let to_assemble = stated.saturating_sub(passed_over);
        crate::log!(
            info,
            "index: assembling {to_assemble} sessions, then joining {} across files",
            fallback.len(),
        );
        let indexes_total = indexes.len() as u64;
        Ok(Self {
            step: Step::Links,
            indexes,
            unread: Some((merged, texts, groups)),
            composing: None,
            ahead,
            composers,
            stream_done: false,
            fallback: fallback.into_iter().collect(),
            aliases,
            shared_events: Arc::new(shared_events),
            deferred: Vec::new(),
            assembled: 0,
            to_assemble,
            indexes_made: 0,
            indexes_total,
            joined: 0,
        })
    }

    /// The build step composition is in.
    fn build_step(&self) -> &'static str {
        match self.step {
            Step::Links | Step::Stream => ASSEMBLING_SESSIONS,
            Step::Indexes => INDEXING_SESSIONS,
            Step::Index { .. } | Step::Done => JOINING_SESSIONS,
        }
    }

    /// How much of `step`'s work is done, of how much.
    fn count(&self, step: &str) -> (u64, u64) {
        match step {
            ASSEMBLING_SESSIONS => (self.assembled, self.to_assemble),
            INDEXING_SESSIONS => (self.indexes_made, self.indexes_total),
            _ => (self.joined, self.fallback.len() as u64),
        }
    }
}

/// A build the writer advances one bounded step at a time, between the
/// messages it serves: each step commits its own transaction and the job
/// keeps its position, so nothing a step did is done again.
pub(super) struct BuildJob {
    stage: Stage,
    /// The threads that compose the stream's sessions.
    composers: usize,
    announce: Option<crossbeam_channel::Sender<&'static str>>,
    reply: SyncSender<Result<Vec<BuildPhase>>>,
    phases: Vec<BuildPhase>,
    clock: Option<(PhaseClock, u64)>,
    /// The running step's total, when it is known before the step ends.
    total: Option<u64>,
    /// The counted step that ended in this build step, with its final
    /// count, for `status` to record before the step that follows.
    ended: Option<(&'static str, StepCount)>,
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
    /// The running step in the writer's activity, while the writer
    /// advances the build ([`BuildJob::show_step`]).
    step: Option<crate::activity::Doing>,
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
    /// The keys given to the workers' groups; a group stored before with the
    /// same bytes keeps its key, and what names the new one names it.
    groups: GroupKeys,
    holders: Option<Sorter>,
    input: Option<(String, i64)>,
    current_text: Option<(Vec<u8>, i64)>,
    stream: Option<u8>,
    /// Each message reading's text keys, by its key in the composition
    /// stream.
    reading_texts: Option<Sorter>,
    /// Sessions composed from the index, the message and event IDs of
    /// readings upkeep wrote meanwhile, the copy names, the message and the
    /// event ID whose marks are being read, and the event IDs several
    /// sessions state (see [`MARKS`]).
    fallback: BTreeSet<String>,
    upkept_messages: BTreeSet<String>,
    upkept_events: HashSet<String>,
    aliases: Option<super::identity::Aliases>,
    /// The message ID whose marks are being read, with the sessions stating
    /// it so far.
    message: Option<(String, Vec<String>)>,
    last_event: Option<(String, String)>,
    shared_events: HashSet<String>,
}

/// The newest parsed chain of input `i`, as an SQL expression: the chain
/// that decides which content-named messages its input holds
/// (`held_message_parse`).
const NEWEST_READ_CHAIN: &str = "(SELECT max(p.chain) FROM parse_state p
    WHERE p.input_pk=i.pk AND p.state='parsed')";

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
        composers: usize,
        announce: Option<crossbeam_channel::Sender<&'static str>>,
        reply: SyncSender<Result<Vec<BuildPhase>>>,
    ) -> Self {
        Self {
            stage: Stage::Start(runs),
            composers: composers.max(1),
            announce,
            reply,
            phases: Vec::new(),
            clock: None,
            total: None,
            ended: None,
            indexes: Vec::new(),
            after_stream: Vec::new(),
            search: None,
            installing: false,
            step: None,
        }
    }

    /// Shows the running step, since it began, in the writer's activity
    /// until [`BuildJob::hide_step`]; a step started meanwhile replaces it.
    pub(super) fn show_step(&mut self) {
        self.step = self
            .clock
            .as_ref()
            .map(|(clock, _)| crate::activity::doing_since(clock.name, clock.started));
    }

    pub(super) fn hide_step(&mut self) {
        self.step = None;
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
        let total = self.total.take();
        if let Some((clock, items)) = self.clock.take() {
            if let Some(unit) = clock.unit {
                self.ended = Some((
                    clock.name,
                    StepCount {
                        done: items,
                        total,
                        unit,
                    },
                ));
            }
            self.phases.push(clock.end(items));
        }
        if let Some(announce) = &self.announce {
            let _ = announce.send(name);
        }
        self.clock = Some((PhaseClock::start(name, counted(name)), 0));
        if self.step.is_some() {
            self.step = Some(crate::activity::doing(name));
        }
    }

    fn count(&mut self, items: u64) {
        if let Some((_, counted)) = &mut self.clock {
            *counted += items;
        }
        crate::activity::progressed();
    }

    /// Runs `step`, `done` of `total`.
    fn counted_to(&mut self, step: &'static str, (done, total): (u64, u64)) {
        self.phase(step);
        if let Some((_, counted)) = &mut self.clock {
            *counted = done;
        }
        self.total = Some(total);
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
        if !self.location.index_build.building() {
            return Err(StoreError::Invalid("no build is in progress".into()));
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
                // group one row (see `Load::groups`) and those it fills
                // in key order; the build makes the declared indexes the file
                // lacks, those the load fills before it.
                if !runs.is_empty() {
                    for (name, sql, _) in &present {
                        if !kept_while_loading(sql) {
                            self.index.execute(&format!("DROP INDEX \"{name}\""), [])?;
                        }
                    }
                }
                let kept = load_indexes(&self.index)?;
                for (name, sql, table) in declared_load_indexes()? {
                    if kept.iter().any(|(held, ..)| *held == name) {
                        continue;
                    }
                    if kept_while_loading(&sql) {
                        self.index.execute(&sql, [])?;
                    } else if SHOWN_TABLES.contains(&table.as_str()) && !kept_while_composing(&sql)
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
                job.search = Some(crate::text_rows::SearchFill::start(
                    &self.index,
                    self.location.memory_budget().search_fill_memory(),
                )?);
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
                            groups: Arc::default(),
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
                        groups: Arc::new(load.groups),
                    });
                    let holders = Merger::open(
                        holders.finish()?,
                        &paths.runs,
                        self.location.memory_budget().side_merge_memory(),
                    )?;
                    self.size_build_cache(
                        composing.merged.memory() + composing.texts.memory() + holders.memory(),
                        false,
                    )?;
                    Stage::TextHolders(Box::new(TextHolders {
                        merged: holders,
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
                    let composing = holders.composing;
                    drop(holders.merged);
                    self.size_build_cache(
                        composing.merged.memory() + composing.texts.memory(),
                        true,
                    )?;
                    Stage::Indexes(0, composing)
                }
            }
            Stage::Indexes(next, loaded) => {
                job.total = Some(job.indexes.len() as u64);
                match job.indexes.get(next) {
                    Some(sql) => {
                        self.index.execute(sql, [])?;
                        job.count(1);
                        Stage::Indexes(next + 1, loaded)
                    }
                    None => {
                        crate::fault::check("index.build.composing", "")?;
                        job.phase(ASSEMBLING_SESSIONS);
                        let indexes = std::mem::take(&mut job.after_stream);
                        let budget = self.location.memory_budget();
                        let composing = budget.composition_memory();
                        // Beside the run readers and the composition, the
                        // full-text fill follows the blocks composition writes.
                        self.size_build_cache(
                            (loaded.merged.memory()
                                + loaded.texts.memory()
                                + composing
                                + budget.search_fill_memory()) as u64,
                            false,
                        )?;
                        // A batch's decoded pieces, its composer's index and its
                        // composed rows each take about what was decoded: the
                        // stream is read ahead in a third of the share.
                        let state = ComposeState::new(
                            *loaded,
                            indexes,
                            &self.index,
                            composing / 3,
                            job.composers,
                        )?;
                        job.counted_to(ASSEMBLING_SESSIONS, state.count(ASSEMBLING_SESSIONS));
                        Stage::Compose(Box::new(state))
                    }
                }
            }
            Stage::Compose(mut state) => {
                // A part of composition may end within this step: what it
                // did is counted to it before the next part starts.
                let ran = state.build_step();
                let done = self.compose_step(&mut state)?;
                job.counted_to(ran, state.count(ran));
                let now = state.build_step();
                job.counted_to(now, state.count(now));
                if !done {
                    Stage::Compose(state)
                } else {
                    drop(state);
                    crate::archive::index_build::remove_runs(&Paths::of(&self.location.root))?;
                    // The runs are read: what is left of the build is its
                    // file's indexes and full text.
                    self.size_build_cache(0, true)?;
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
                    // Sessions were assembled without their summaries: each is
                    // summarized from its first row once the file is installed.
                    crate::summary::mark_every_session(&tx)?;
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
        let running = job.clock.as_ref().and_then(|(clock, done)| {
            clock.unit.map(|unit| {
                (
                    clock.name,
                    StepCount {
                        done: *done,
                        total: job.total,
                        unit,
                    },
                )
            })
        });
        for (step, count) in ended.into_iter().chain(running) {
            self.location.record_build_count(step, count);
        }
        Ok(false)
    }

    /// Gives the build file's cache what the budget leaves beside `held`
    /// bytes the current step holds of its own and what the process holds
    /// already outside that cache (its other connections' caches, the
    /// program, what the allocator kept).
    /// A step that creates indexes (`sorting`) gives the cache a third of
    /// that: SQLite's sorter keeps records in memory up to the cache's size
    /// beside it, in a buffer that grows by doubling, so up to twice that
    /// while it grows.
    fn size_build_cache(&self, held: impl TryInto<u64>, sorting: bool) -> Result<()> {
        let held = held.try_into().unwrap_or(u64::MAX);
        let outside = status_bytes("VmRSS:").saturating_sub(cache_used(&self.index));
        let mut mib = self
            .location
            .memory_budget()
            .build_cache_beside(held.saturating_add(outside));
        if sorting {
            mib = (mib / 3).max(32);
        }
        self.index
            .pragma_update(None, "cache_size", -(mib * 1024))?;
        Ok(())
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
        // A group's key is never given to another group.
        let next_group: i64 = self.index.scalar(
            "SELECT max(coalesce((SELECT seq FROM sqlite_sequence WHERE name='content_group'),0),
                 coalesce((SELECT max(id) FROM content_group),0))+1",
            [],
        )?;
        // Text groups whose every text another worker or chain also held
        // are named by nothing once the texts are made one.
        self.index.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS build_text_group(id INTEGER PRIMARY KEY);
             DELETE FROM temp.build_text_group;",
        )?;
        // References are checked by the indexes and composition after the
        // load; tables arrive in name order, not in reference order.
        self.index.pragma_update(None, "foreign_keys", false)?;
        let budget = self.location.memory_budget();
        let merged = Merger::open(runs, &paths.runs, budget.run_merge_memory())?;
        // The load's own memory: the run readers and the two sorts of text
        // holders.
        self.size_build_cache(merged.memory() + 2 * budget.side_merge_memory(), false)?;
        Ok(Load {
            merged,
            tables: tables(&self.index)?,
            held: held.into_iter().peekable(),
            next_group,
            groups: GroupKeys::default(),
            holders: Some(Sorter::new(
                &paths.runs,
                "text-holders".into(),
                self.location.memory_budget().side_merge_memory(),
            )),
            input: None,
            current_text: None,
            stream: None,
            reading_texts: Some(Sorter::new(
                &paths.runs,
                "reading-texts".into(),
                self.location.memory_budget().side_merge_memory(),
            )),
            fallback: upkept.sessions,
            upkept_messages: upkept.messages,
            upkept_events: upkept.events,
            aliases: None,
            message: None,
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
                    load.groups
                        .bases
                        .entry(worker)
                        .or_insert(load.next_group - seq);
                    let Some((&kind, rest)) =
                        value.split_first().filter(|(_, rest)| rest.len() >= 32)
                    else {
                        return Err(StoreError::Invalid("a content group has no hash".into()));
                    };
                    let (hash, body) = rest.split_at(32);
                    let stored = t
                        .prepare_cached(
                            "INSERT INTO content_group(id,hash,body) VALUES (?1,?2,?3)
                             ON CONFLICT(hash) DO NOTHING",
                        )?
                        .execute(params![load.next_group, hash, body])?;
                    if stored == 0 {
                        let held: i64 = t
                            .prepare_cached("SELECT id FROM content_group WHERE hash=?1")?
                            .query_row([hash], |r| r.get(0))?;
                        load.groups.shared.insert(load.next_group, held);
                    } else if kind == TEXT_GROUP {
                        t.prepare_cached("INSERT INTO temp.build_text_group(id) VALUES (?1)")?
                            .execute([load.next_group])?;
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
                                let Value::Integer(named) = row[0] else {
                                    return Err(StoreError::Invalid(
                                        "a text names no group".into(),
                                    ));
                                };
                                let group = load.groups.key(named)?;
                                t.prepare_cached(
                                    "INSERT INTO text(hash,normalized_hash,group_id,start,length)
                                     VALUES (?1,?2,?3,?4,?5)",
                                )?
                                .execute(params![hash, row[3], group, row[1], row[2]])?;
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
                        MESSAGE_ID => {
                            let (message, rest) = split_id(rest)?;
                            let (&how, rest) = rest.split_first().ok_or_else(|| {
                                StoreError::Invalid("a build mark of no kind".into())
                            })?;
                            let session = split_id(rest)?.0;
                            // A message one of whose names is a copy's is
                            // shown as its session decides.
                            let named = super::identity::message_session(message)
                                .is_some_and(|named| aliases.involve(&named));
                            if named {
                                load.fallback.insert(session.to_owned());
                            }
                            if load.message.as_ref().is_none_or(|(id, _)| id != message) {
                                load.message = Some((message.to_owned(), Vec::new()));
                            }
                            let (_, stating) = load.message.as_mut().expect("set above");
                            match how {
                                // A message stated under several sessions, or
                                // also by a reading upkeep wrote.
                                STATES => {
                                    if load.upkept_messages.contains(message) {
                                        load.fallback.insert(session.to_owned());
                                    }
                                    if stating.iter().any(|other| other != session) {
                                        load.fallback.extend(stating.iter().cloned());
                                        load.fallback.insert(session.to_owned());
                                    }
                                    stating.push(session.to_owned());
                                }
                                // An event following a message a reading
                                // upkeep wrote states: its session is
                                // composed from the index.
                                FOLLOWS => {
                                    if load.upkept_messages.contains(message) {
                                        load.fallback.insert(session.to_owned());
                                    }
                                }
                                _ => {
                                    return Err(StoreError::Invalid(
                                        "a build mark of no kind".into(),
                                    ));
                                }
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
                    if let Some(at) = table.group {
                        let Value::Integer(named) = row[at] else {
                            return Err(StoreError::Invalid("a reading row names no group".into()));
                        };
                        row[at] = Value::Integer(load.groups.key(named)?);
                    }
                    insert(&t, table, &row)?;
                }
            }
            written += 1;
        };
        if !more {
            // Text groups whose every text another worker also held.
            t.execute(
                "DELETE FROM content_group WHERE id IN (SELECT id FROM temp.build_text_group)
                 AND id NOT IN (SELECT group_id FROM text)",
                [],
            )?;
            t.execute("DELETE FROM temp.build_text_group", [])?;
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
    /// and documents first, from the index, so each session is composed
    /// after its links; then sessions from the stream, grouped as they come;
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
                    // The stream's reader has ended with its runs.
                    self.size_build_cache(0, true)?;
                    state.step = if state.indexes.is_empty() {
                        Step::Index { next: 0 }
                    } else {
                        Step::Indexes
                    };
                }
            }
            Step::Indexes => {
                if let Some(sql) = state.indexes.pop() {
                    self.index.execute(&sql, [])?;
                    state.indexes_made += 1;
                }
                if state.indexes.is_empty() {
                    self.size_build_cache(0, false)?;
                    state.step = Step::Index { next: 0 };
                }
            }
            Step::Index { next } => {
                let mut touched = super::readings::Touched::default();
                // A transaction composes up to its share of sessions.
                let mut end = next;
                while end < state.fallback.len() && end < next + SESSIONS_PER_TRANSACTION {
                    let session = &state.fallback[end];
                    end += 1;
                    touched.whole.insert(session.clone());
                    touched.sessions.insert(session.clone());
                    for row in self
                        .index
                        .prepare_cached(
                            "SELECT session_id,call_id FROM call_outcome WHERE session_id=?1",
                        )?
                        .query_map([session], |r| <(String, String)>::try_from(r))?
                    {
                        touched.calls.insert(row?);
                    }
                }
                state.joined += (end - next) as u64;
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
                state.step = if end < state.fallback.len() {
                    Step::Index { next: end }
                } else {
                    // Every session's lineage is computed once the build's
                    // sessions and links are all written.
                    let tx = self
                        .index
                        .transaction_with_behavior(TransactionBehavior::Immediate)?;
                    tx.execute(
                        "INSERT INTO summary_due(session_pk,from_order,lineage)
                         SELECT pk,X'FF',1 FROM session WHERE true
                         ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1",
                        [],
                    )?;
                    tx.commit()?;
                    Step::Done
                };
            }
            Step::Done => {}
        }
        Ok(matches!(state.step, Step::Done))
    }

    /// Appends sessions the composers composed from the stream, in one
    /// transaction of whole batches.
    fn compose_stream(&mut self, state: &mut ComposeState) -> Result<()> {
        // A session from the stream refers only to rows the build wrote.
        self.index.pragma_update(None, "foreign_keys", false)?;
        let result = self.compose_stream_transaction(state);
        self.index.pragma_update(None, "foreign_keys", true)?;
        result
    }

    fn compose_stream_transaction(&mut self, state: &mut ComposeState) -> Result<()> {
        if state.composing.is_none() {
            self.start_composers(state)?;
        }
        let (pool, gate) = state
            .composing
            .as_mut()
            .ok_or_else(|| StoreError::internal("a build's composers did not start"))?;
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        bump_counter(&tx)?;
        // A transaction ends with a batch, once it took its share of the
        // writer's time.
        let started = std::time::Instant::now();
        while started.elapsed() < STEP_TIME {
            let Some(composed) = pool.next()? else {
                // Every batch of the stream is composed and appended.
                state.stream_done = true;
                break;
            };
            pool.append(&tx, composed.rows)?;
            gate.release(composed.bytes);
            state.deferred.extend(composed.late);
            state.assembled += composed.sessions;
        }
        tx.commit()?;
        Ok(())
    }

    /// Starts reading the stream into batches and the composer threads,
    /// once links and documents are composed.
    fn start_composers(&mut self, state: &mut ComposeState) -> Result<()> {
        let (merged, texts, groups) = state
            .unread
            .take()
            .ok_or_else(|| StoreError::internal("a build's stream was read twice"))?;
        // Every batch is composed under one write counter.
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&tx)?;
        tx.commit()?;
        let shared = composers::Shared::read(
            &self.index,
            Arc::clone(&state.aliases),
            Arc::clone(&state.shared_events),
            counter,
        )?;
        let skipped: BTreeSet<String> = state.fallback.iter().cloned().collect();
        let (batches, gate) = read_stream(
            merged,
            texts,
            skipped,
            Arc::clone(&state.aliases),
            groups,
            state.ahead,
            state.composers,
        );
        let pool = composers::Pool::start(state.composers, shared, batches, &self.index)?;
        state.composing = Some((pool, gate));
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

/// What the stream's reader has decoded and the writer has not yet
/// appended, in bytes, at most a given number: the reader waits while that
/// many are held, unless nothing is (a batch larger than the allowance goes
/// alone), or until the composition is dropped.
pub(super) struct Gate {
    held: std::sync::Mutex<(usize, bool)>,
    room: std::sync::Condvar,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            held: std::sync::Mutex::new((0, false)),
            room: std::sync::Condvar::new(),
        })
    }

    /// Waits until `bytes` more fit in `ahead`; false once closed.
    fn take(&self, bytes: usize, ahead: usize) -> bool {
        let Ok(mut held) = self.held.lock() else {
            return false;
        };
        while !held.1 && held.0 > 0 && held.0 + bytes > ahead {
            match self.room.wait(held) {
                Ok(guard) => held = guard,
                Err(_) => return false,
            }
        }
        held.0 += bytes;
        !held.1
    }

    fn release(&self, bytes: usize) {
        if let Ok(mut held) = self.held.lock() {
            held.0 = held.0.saturating_sub(bytes);
        }
        self.room.notify_all();
    }

    fn close(&self) {
        if let Ok(mut held) = self.held.lock() {
            held.1 = true;
        }
        self.room.notify_all();
    }
}

impl Drop for ComposeState {
    /// A composition dropped before its end lets its reader go.
    fn drop(&mut self) {
        if let Some((_, gate)) = &self.composing {
            gate.close();
        }
    }
}

/// About what a decoded piece holds, from the bytes it was decoded from: a
/// message's decoded value beside its packed blocks.
fn piece_bytes(record: usize) -> usize {
    2 * record + 256
}

/// Reads the composition stream on a thread of its own, merging it with
/// each message reading's text keys and decoding its records, and hands
/// whole sessions to the composers in batches of about
/// [`composers::BATCH_MESSAGES`] messages or a share of `ahead` bytes.
/// Sessions composed from the index are passed over.
fn read_stream(
    mut merged: Merger,
    mut texts: Merger,
    skipped: BTreeSet<String>,
    aliases: Arc<super::identity::Aliases>,
    groups: Arc<GroupKeys>,
    ahead: usize,
    composers: usize,
) -> (
    std::sync::mpsc::Receiver<Result<composers::Batch>>,
    Arc<Gate>,
) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let gate = Gate::new();
    let held = Arc::clone(&gate);
    // Enough batches in flight for every composer and the writer.
    let batch_bytes = (ahead / (2 * composers.max(1) + 1)).max(1 << 20);
    let spawned = std::thread::Builder::new()
        .name("polylogue-compose-stream".into())
        .spawn(move || {
            let mut batch = composers::Batch::default();
            let mut seq = 0;
            let mut ship = |batch: &mut composers::Batch| -> bool {
                let mut full = std::mem::take(batch);
                full.seq = seq;
                seq += 1;
                held.take(full.bytes, ahead) && sender.send(Ok(full)).is_ok()
            };
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
                    batch.pieces.push(Piece::Begin);
                    let mut add = |piece: Piece, bytes: usize| {
                        if matches!(piece, Piece::Message { .. }) {
                            batch.messages += 1;
                        }
                        batch.bytes += bytes;
                        batch.pieces.push(piece);
                        true
                    };
                    read_session(&mut merged, &mut texts, &groups, &session, &mut add)?;
                    batch.pieces.push(Piece::End(session));
                    if (batch.messages >= composers::BATCH_MESSAGES || batch.bytes >= batch_bytes)
                        && !ship(&mut batch)
                    {
                        return Ok(());
                    }
                }
                if !batch.pieces.is_empty() {
                    ship(&mut batch);
                }
                Ok(())
            })();
            if let Err(error) = result {
                let _ = sender.send(Err(error));
            }
        });
    if let Err(error) = spawned {
        let (sender, failed) = std::sync::mpsc::channel();
        let _ = sender.send(Err(error.into()));
        return (failed, gate);
    }
    (receiver, gate)
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
    groups: &GroupKeys,
    session: &str,
    send: &mut impl FnMut(Piece, usize) -> bool,
) -> Result<bool> {
    use super::shown::{EventRow, Given, MessageReading, OutcomeRow};
    let bad = || StoreError::Invalid("a composition record is malformed".into());
    let mut pending: Option<(String, Vec<MessageReading>)> = None;
    // What the pending message's readings were decoded from.
    let mut pending_bytes = 0;
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
            && !send(
                Piece::Message { id, readings },
                piece_bytes(std::mem::take(&mut pending_bytes)),
            )
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
                if !send(Piece::Facts { input, chain, kind }, 0) {
                    return Ok(false);
                }
            }
            MESSAGE => {
                if pending.as_ref().is_some_and(|(id, _)| *id != item)
                    && let Some((id, readings)) = pending.take()
                    && !send(
                        Piece::Message { id, readings },
                        piece_bytes(std::mem::take(&mut pending_bytes)),
                    )
                {
                    return Ok(false);
                }
                pending_bytes += value.len();
                // Columns: start, end, seq, position, order time, route,
                // completion, content-named, hash, the content's group, start
                // and length, words; then each block's text hash.
                let hash = blob(8)?.to_vec();
                if row.len() < 13 {
                    return Err(bad());
                }
                let texts_of = row[13..]
                    .iter()
                    .map(|text| match text {
                        Value::Blob(hash) => Ok(Some(
                            <[u8; 32]>::try_from(hash.as_slice()).map_err(|_| bad())?,
                        )),
                        Value::Null => Ok(None),
                        _ => Err(bad()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let blocks = texts_of.len();
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
                polylogue_model::MessageId::parse(&item).map_err(StoreError::invalid)?;
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
                    stored: crate::content::Stretch {
                        group: groups.key(int(9)?)?,
                        start: int(10)?,
                        length: int(11)?,
                    },
                    given: Some(Given {
                        kind: None,
                        superseded: false,
                        words: int(12)?,
                        texts: texts_of,
                        text_ids: ids,
                    }),
                };
                pending
                    .get_or_insert_with(|| (item, Vec::new()))
                    .1
                    .push(reading);
            }
            EVENT => {
                let bytes = piece_bytes(value.len());
                if !send(
                    Piece::Event(EventRow {
                        input,
                        input_pk: 0,
                        chain,
                        event_id: item,
                        route: route(0)?,
                        hash: blob(1)?.to_vec(),
                        piece: crate::content::Piece {
                            group: groups.key(int(2)?)?,
                            start: int(3)?,
                            length: int(4)?,
                            data_length: Some(int(5)?),
                        },
                        seq: int(6)?,
                        start: int_or_null(7)?,
                        end: int_or_null(8)?,
                    }),
                    bytes,
                ) {
                    return Ok(false);
                }
            }
            OUTCOME => {
                let bytes = piece_bytes(value.len());
                if !send(
                    Piece::Outcome(OutcomeRow {
                        input_pk: 0,
                        chain,
                        position: int(0)?,
                        input,
                        route: route(2)?,
                        call_id: item,
                        body: text(1)?.to_owned(),
                    }),
                    bytes,
                ) {
                    return Ok(false);
                }
            }
            _ => return Err(bad()),
        }
    }
    if let Some((id, readings)) = pending.take() {
        return Ok(send(
            Piece::Message { id, readings },
            piece_bytes(pending_bytes),
        ));
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
