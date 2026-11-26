//! A rebuild's parse workers and the sorted stream they write
//! (docs/architecture.md, one path). The engine reads each chain whole, on
//! several workers; each worker writes a chain's parse into a scratch index
//! with the code an incremental update uses, then moves its rows out into
//! one sorted stream ([`RebuildWorker`], over [`crate::sorted_runs`]) and
//! clears the scratch. The keys of the stream's records decide the order the
//! rebuild loads and assembles them in ([`super::load`], [`super::assemble`]).
use crate::index::file::Paths;
use crate::sorted_runs::Sorter;
use crate::writer::{InputKey, ParseStep, WriterSender};
use crate::{Archive, Result, StoreError};
use rusqlite::types::{Value, ValueRef};
use rusqlite::{Connection, Transaction, TransactionBehavior};
use std::path::{Path, PathBuf};

/// The cache of a worker's content-group file, at most; the rest of a
/// worker's memory is its sort buffer.
pub(super) const SCRATCH_CACHE: usize = 64 << 20;

/// A worker's scratch index past this size after a chain is opened again,
/// so that a large chain's rows leave no memory behind them.
pub(super) const SCRATCH_KEPT: i64 = 64 << 20;

/// Sessions composed per transaction.
pub(super) const SESSIONS_PER_TRANSACTION: usize = 512;

/// Rows the merge writes per transaction, at most.
pub(super) const ROWS_PER_TRANSACTION: u64 = 200_000;

/// The streams of a build, by the first byte of a record's key.
pub(super) const INPUTS: u8 = 0;
/// Content groups, by worker and the worker's number for the group; a
/// group's record is its kind ([`TEXT_GROUP`] or [`READING_GROUP`]), hash and
/// body. Until the load gives it its key, a row names a group by
/// [`provisional`].
pub(super) const GROUPS: u8 = 1;
pub(super) const TEXT_GROUP: u8 = 0;
pub(super) const READING_GROUP: u8 = 1;
/// Texts and the reading text rows that hold them, by hash: a text's own
/// record (`TEXT`) comes before the rows that name it (`TEXT_HOLDER`).
pub(super) const TEXTS: u8 = 2;
pub(super) const TEXT: u8 = 0;
pub(super) const TEXT_HOLDER: u8 = 1;
/// The reading tables, in name order, from this byte on.
pub(super) const TABLES: u8 = 3;

/// What decides which sessions are composed from the stream, read before
/// it: sessions whose composition reaches other sessions' readings
/// ([`FALLBACK`]); each message ID with the sessions stating it and the
/// sessions with an event following it ([`MESSAGE_ID`], then [`STATES`] or
/// [`FOLLOWS`]: one message's marks come together, its statements first);
/// and each event ID with its session ([`EVENT_ID`]).
pub(super) const MARKS: u8 = 253;
pub(super) const FALLBACK: u8 = 0;
pub(super) const MESSAGE_ID: u8 = 1;
pub(super) const EVENT_ID: u8 = 3;
pub(super) const STATES: u8 = 0;
pub(super) const FOLLOWS: u8 = 1;
/// The readings composition takes, grouped by the session stating them:
/// its facts' kinds, each message's readings with their values, its event
/// readings and its calls' background outcomes, in that order.
pub(super) const COMPOSE: u8 = 254;
pub(super) const FACTS: u8 = 0;
pub(super) const MESSAGE: u8 = 1;
pub(super) const EVENT: u8 = 2;
pub(super) const OUTCOME: u8 = 3;

/// The tables composition writes, and those of their indexes kept while it
/// composes from the stream: those whose first column is the session,
/// message or event a row belongs to, which are also all it reads (its rows
/// in order and the replies to each message). Composition makes those keys in the order it writes, so each
/// such entry lands at its index's end and costs no read; made after the
/// stream, each would scan its whole table again. Their other indexes,
/// keyed by IDs in no order (a message's, an event's), are made after the
/// stream: kept while it writes, each new row would read a page at random.
pub(super) const SHOWN_TABLES: &[&str] =
    &["message", "message_route", "block", "tool_call", "event"];
pub(super) const COMPOSED_IN_ORDER: &[&str] = &["session_pk", "message_pk", "event_pk"];

/// The table an index statement is on, and its first column.
pub(super) fn index_target(sql: &str) -> Option<(&str, &str)> {
    let (_, on) = sql.split_once(" ON ")?;
    let (table, columns) = on.split_once('(')?;
    let first = columns.split([',', ')']).next()?;
    Some((table.trim().trim_matches('"'), first.trim()))
}

/// Whether an index on a shown table is kept while composition writes it.
pub(super) fn kept_while_composing(sql: &str) -> bool {
    index_target(sql).is_some_and(|(_, first)| COMPOSED_IN_ORDER.contains(&first))
}

/// Whether an index on a reading table is kept while the load writes it:
/// one whose first column is the input. The load writes the reading tables
/// input by input, chain by chain, so such an index only grows at its end;
/// made after the load, it would scan its whole table again.
pub(super) fn kept_while_loading(sql: &str) -> bool {
    index_target(sql)
        .is_some_and(|(table, first)| !SHOWN_TABLES.contains(&table) && first == "input_pk")
}

/// The indexes a build's load concerns, as (name, statement, table): every
/// one on a stored table but the one that makes an equal content group one
/// row.
pub(super) const LOAD_INDEXES: &str =
    "SELECT name,sql,tbl_name FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL
     AND name <> 'content_group_hash'
     AND tbl_name NOT IN (SELECT name FROM sqlite_schema WHERE sql LIKE 'CREATE VIRTUAL TABLE%')";

/// How a worker's run names a content group before the load gives it its
/// key: the worker's number and its number for the group.
pub(super) fn provisional(worker: u32, seq: u64) -> i64 {
    (i64::from(worker) << 40) | seq as i64
}

/// The keys the load gave the workers' content groups.
#[derive(Default)]
pub(super) struct GroupKeys {
    /// Each worker's first group's key, less its number for that group.
    pub(super) bases: std::collections::HashMap<u32, i64>,
    /// Groups not stored because one with the same bytes already was (by
    /// another worker or another chain), by the key they would have
    /// had: the stored group's key.
    pub(super) shared: std::collections::HashMap<i64, i64>,
}

impl GroupKeys {
    /// The key of the group a run names `provisional`.
    pub(super) fn key(&self, provisional: i64) -> Result<i64> {
        let worker = (provisional >> 40) as u32;
        let base = self
            .bases
            .get(&worker)
            .ok_or_else(|| StoreError::Invalid("a run names a group the build lacks".into()))?;
        let key = base + (provisional & ((1 << 40) - 1));
        Ok(self.shared.get(&key).copied().unwrap_or(key))
    }
}

pub(super) fn load_indexes(conn: &Connection) -> Result<Vec<(String, String, String)>> {
    Ok(conn
        .prepare(LOAD_INDEXES)?
        .query_map([], |r| <(String, String, String)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?)
}

/// [`load_indexes`] as a new build file declares them, so that an index an
/// earlier round dropped is made again from its declaration.
pub(super) fn declared_load_indexes() -> Result<Vec<(String, String, String)>> {
    let declared = Connection::open_in_memory()?;
    crate::schema::ensure_build_index(&declared)?;
    load_indexes(&declared)
}

/// A reading table moved out of the scratch index: its name, columns, the
/// column holding the input key, and the insert that says how rows from
/// several chains combine.
#[derive(Clone)]
pub(super) struct Table {
    pub(super) name: String,
    pub(super) columns: Vec<String>,
    pub(super) input: Option<usize>,
    /// The column naming a content group, which a run gives provisionally.
    pub(super) group: Option<usize>,
    /// The statement that inserts one row into the build file.
    pub(super) insert: String,
}

impl Table {
    pub(super) fn new(name: String, columns: Vec<String>, conflict: &'static str) -> Self {
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
pub(super) fn tables(conn: &Connection) -> Result<Vec<Table>> {
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
pub(super) fn encode_row(values: &[ValueRef<'_>], out: &mut Vec<u8>) {
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

pub(super) fn decode_row(mut bytes: &[u8]) -> Result<Vec<Value>> {
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
pub(super) fn row_key(stream: u8, input: &str, chain: i64, seq: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(input.len() + 14);
    key.push(stream);
    key.extend_from_slice(input.as_bytes());
    key.push(0);
    key.extend_from_slice(&chain.to_be_bytes());
    key.extend_from_slice(&seq.to_be_bytes());
    key
}

/// The key of a session's composition records of `kind`.
pub(super) fn compose_key(session: &str, kind: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(session.len() + 48);
    key.push(COMPOSE);
    key.extend_from_slice(session.as_bytes());
    key.push(0);
    key.push(kind);
    key
}

/// The key of one chain's reading of an item of a session.
pub(super) fn reading_key(session: &str, kind: u8, item: &str, input: &str, chain: i64) -> Vec<u8> {
    let mut key = compose_key(session, kind);
    key.extend_from_slice(item.as_bytes());
    key.push(0);
    key.extend_from_slice(input.as_bytes());
    key.push(0);
    key.extend_from_slice(&chain.to_be_bytes());
    key
}

pub(super) fn mark_key(kind: u8, parts: &[&str]) -> Vec<u8> {
    let mut key = vec![MARKS, kind];
    for part in parts {
        key.extend_from_slice(part.as_bytes());
        key.push(0);
    }
    key
}

/// The mark that `session` states message `message` ([`STATES`]) or holds
/// an event following it ([`FOLLOWS`]).
pub(super) fn message_mark(message: &str, how: u8, session: &str) -> Vec<u8> {
    let mut key = vec![MARKS, MESSAGE_ID];
    key.extend_from_slice(message.as_bytes());
    key.push(0);
    key.push(how);
    key.extend_from_slice(session.as_bytes());
    key.push(0);
    key
}

/// The NUL-ended string at the start of `bytes`, and what follows it.
pub(super) fn split_id(bytes: &[u8]) -> Result<(&str, &[u8])> {
    let bad = || StoreError::Invalid("a build stream key is malformed".into());
    let end = bytes.iter().position(|&b| b == 0).ok_or_else(bad)?;
    let id = std::str::from_utf8(&bytes[..end]).map_err(|_| bad())?;
    Ok((id, &bytes[end + 1..]))
}

/// The input ID at the start of a reading row's key (after its stream byte).
pub(super) fn key_input(key: &[u8]) -> Result<&str> {
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
pub struct RebuildWorker {
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
pub(super) fn open_scratch(
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

impl RebuildWorker {
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
        std::fs::create_dir_all(&paths.sorted)?;
        let scratch = paths.sorted.join(format!("scratch-{number:06}.db"));
        let cache = SCRATCH_CACHE.min(memory / 2).max(16 << 20);
        let archive_db = archive.archive_db();
        let (conn, tables) = open_scratch(&scratch, &archive_db, cache)?;
        let register = crate::archive::open_read(&archive_db)?;
        Ok(Self {
            conn,
            register,
            tables,
            sorter: Sorter::new(
                &paths.sorted,
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
            steps: crate::writer::chain::ChainSteps::new(input, chain, length),
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
    steps: crate::writer::chain::ChainSteps,
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

impl crate::writer::chain::ChainSink for OpenChain<'_> {
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
        let w = crate::writer::chain::ReadingWrite {
            register: self.register,
            changes: &self.changes,
        };
        self.steps.write(
            &self.tx,
            &w,
            step,
            &mut crate::writer::readings::Touched::default(),
        )?;
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
                        words: crate::writer::shown::text_words(&row.message),
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
pub(super) struct Out<'a> {
    sorter: &'a mut Sorter,
    tables: &'a [Table],
    number: u32,
    groups: &'a mut u64,
}

/// Moves the chain's rows from the scratch index into the stream and
/// clears it.
pub(super) fn move_out(
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
pub(super) fn compose_records(
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
        let mut reader = crate::content::Reader::default();
        // In each chain's order, so that its texts are read group by group.
        let mut statement = tx.prepare_cached(
            "SELECT r.message_id,r.session_id,r.source_start,r.source_end,r.seq,r.record_position,
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
                    let stored = crate::writer::readings::read_reading(
                        tx,
                        &mut reader,
                        input_pk,
                        chain,
                        &message,
                    )?
                    .ok_or_else(|| StoreError::Invalid("a message reading has no body".into()))?;
                    Prepared::of(crate::writer::shown::ReadingValue::of(stored)?)
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
                 seq,source_start,source_end,after_message_id
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
/// text hashes (see [`crate::writer::shown::ReadingValue`]).
pub(super) struct Prepared {
    hash: [u8; 32],
    block_count: u64,
    words: i64,
    texts: Vec<Option<[u8; 32]>>,
}

impl Prepared {
    fn of(value: crate::writer::shown::ReadingValue) -> Self {
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
