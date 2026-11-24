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
use crate::archive::index_build::Paths;
use crate::sorted_runs::{Merger, Sorter};
use rusqlite::types::{Value, ValueRef};
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

/// Measures one phase from the process's storage counters.
pub struct PhaseClock {
    name: &'static str,
    started: Instant,
    io: (u64, u64),
}

impl PhaseClock {
    pub fn start(name: &'static str) -> Self {
        Self {
            name,
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
            "index build: {} took {:.1} s; {} items; {:.1} MB read, {:.1} MB written; peak memory {:.0} MB",
            phase.name,
            phase.seconds,
            phase.items,
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
    if found.len() + TABLES as usize > 255 {
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
    texts: crate::text_rows::TextIds,
    blobs: Arc<crate::blobs::Blobs>,
    tables: Vec<Table>,
    sorter: Sorter,
    number: u32,
    groups: u64,
    scratch: PathBuf,
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
        let texts = crate::text_rows::TextIds::new(&conn, archive.memory_budget().keys() / 16)?;
        Ok(Self {
            conn,
            register,
            texts,
            blobs: Arc::new(archive.blobs()?),
            tables,
            sorter: Sorter::new(&paths.runs, format!("w{number:06}"), memory.max(32 << 20)),
            number,
            groups: 0,
            scratch,
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
            changes,
            register: &self.register,
            texts: &self.texts,
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
            },
            text_transaction: self.texts.transaction(),
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
    changes: crate::change_summary::ChangeSummary,
    register: &'w Connection,
    texts: &'w crate::text_rows::TextIds,
    tx: Transaction<'w>,
    input: String,
    chain: i64,
    writer: &'w WriterSender,
    blobs: &'w Arc<crate::blobs::Blobs>,
    out: Out<'w>,
    text_transaction: crate::text_rows::TextTransaction<'w>,
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
            texts: self.texts,
            counter: 1,
            primary: false,
            changes: &self.changes,
        };
        self.steps
            .write(&self.tx, &w, step, &mut super::readings::Touched::default())
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
        move_out(&self.tx, &mut self.out, &self.input, self.chain)?;
        self.tx.commit()?;
        self.text_transaction.commit();
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
fn move_out(tx: &Transaction<'_>, out: &mut Out<'_>, input: &str, chain: i64) -> Result<()> {
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
    // Text groups take the worker's next numbers.
    let mut groups = std::collections::HashMap::new();
    {
        let mut statement = tx.prepare_cached("SELECT id,body FROM text_group ORDER BY id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let local: i64 = row.get(0)?;
            let seq = *out.groups;
            *out.groups += 1;
            groups.insert(local, seq);
            let mut key = vec![GROUPS];
            key.extend_from_slice(&out.number.to_be_bytes());
            key.extend_from_slice(&seq.to_be_bytes());
            out.sorter.push(&key, &row.get::<_, Vec<u8>>(1)?)?;
        }
    }
    {
        let mut statement = tx.prepare_cached(
            "SELECT hash,normalized_hash,group_id,start,length FROM text
             WHERE id IN (SELECT text_id FROM parse_text)",
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
                ],
                &mut value,
            );
            out.sorter.push(&key, &value)?;
        }
    }
    let mut seq = 0u32;
    {
        let mut statement = tx.prepare_cached(
            "SELECT p.chain,p.document,p.item,p.idx,t.hash FROM parse_text p
             JOIN text t ON t.id=p.text_id ORDER BY p.document,p.item,p.idx",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let mut key = vec![TEXTS];
            key.extend_from_slice(&row.get::<_, Vec<u8>>(4)?);
            key.push(TEXT_HOLDER);
            key.extend_from_slice(&row_key(0, input, chain, seq)[1..]);
            seq += 1;
            value.clear();
            encode_row(
                &[
                    ValueRef::Null,
                    row.get_ref(0)?,
                    row.get_ref(1)?,
                    row.get_ref(2)?,
                    row.get_ref(3)?,
                    ValueRef::Null,
                ],
                &mut value,
            );
            out.sorter.push(&key, &value)?;
        }
    }
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

/// A build the writer advances one bounded step at a time, between the
/// messages it serves: each step commits its own transaction and the job
/// keeps its position, so nothing a step did is done again.
pub(super) struct BuildJob {
    stage: Stage,
    announce: Option<std::sync::mpsc::Sender<&'static str>>,
    reply: SyncSender<Result<Vec<BuildPhase>>>,
    phases: Vec<BuildPhase>,
    clock: Option<(PhaseClock, u64)>,
    /// The indexes dropped for the load, recreated after it.
    indexes: Vec<(String, String)>,
}

enum Stage {
    Start(Vec<PathBuf>),
    Load(Box<Load>),
    TextHolders(Box<TextHolders>),
    Indexes(usize),
    /// The sessions to compose, and the next one.
    Compose(Vec<String>, usize),
    FullText(crate::text_rows::SearchBuild),
    Deferred(Vec<String>),
    Install,
}

/// The merge of the workers' runs, where it stands.
struct Load {
    merged: Merger,
    tables: Vec<Table>,
    held: std::iter::Peekable<std::vec::IntoIter<(Vec<u8>, i64)>>,
    next_text: i64,
    next_group: i64,
    bases: std::collections::HashMap<u32, i64>,
    holders: Option<Sorter>,
    input: Option<(String, i64)>,
    current_text: Option<(Vec<u8>, i64)>,
    stream: Option<u8>,
}

/// The reading text rows, joined to their texts, being loaded.
struct TextHolders {
    merged: Merger,
    input: Option<(String, i64)>,
    table: Table,
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
            indexes: Vec::new(),
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
            self.phases.push(clock.end(items));
        }
        if let Some(announce) = &self.announce {
            let _ = announce.send(name);
        }
        self.clock = Some((PhaseClock::start(name), 0));
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
                job.phase("merge texts");
                self.index.flush_prepared_statement_cache();
                job.indexes = self
                    .index
                    .prepare(
                        "SELECT name,sql FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL
                         AND tbl_name NOT IN (SELECT name FROM sqlite_schema WHERE sql LIKE 'CREATE VIRTUAL TABLE%')",
                    )?
                    .query_map([], |r| <(String, String)>::try_from(r))?
                    .collect::<rusqlite::Result<_>>()?;
                for (name, _) in &job.indexes {
                    self.index.execute(&format!("DROP INDEX \"{name}\""), [])?;
                }
                Stage::Load(Box::new(self.start_load(runs)?))
            }
            Stage::Load(mut load) => {
                if self.load_step(job, &mut load)? {
                    Stage::Load(load)
                } else {
                    crate::fault::check("index.build.loading", "")?;
                    let paths = Paths::of(&self.location.root);
                    let holders = load.holders.take().expect("the text holders' sorter");
                    Stage::TextHolders(Box::new(TextHolders {
                        merged: Merger::open(holders.finish()?, &paths.runs)?,
                        input: None,
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
                    crate::archive::index_build::remove_runs(&Paths::of(&self.location.root))?;
                    job.phase("indexes");
                    Stage::Indexes(0)
                }
            }
            Stage::Indexes(next) => match job.indexes.get(next) {
                Some((_, sql)) => {
                    self.index.execute(sql, [])?;
                    job.count(1);
                    Stage::Indexes(next + 1)
                }
                None => {
                    job.phase("compose");
                    Stage::Compose(self.stated_sessions()?, 0)
                }
            },
            Stage::Compose(sessions, next) if next == 0 || next < sessions.len() => {
                self.compose_chunk(&sessions, next)?;
                let after = (next + SESSIONS_PER_TRANSACTION).min(sessions.len());
                job.count((after - next) as u64);
                Stage::Compose(sessions, after.max(1))
            }
            Stage::Compose(..) => {
                job.phase("full text");
                // The search entries are written from every text's marks;
                // entries queued by upkeep writes during the build would
                // repeat them.
                self.index.execute("DELETE FROM search_pending", [])?;
                Stage::FullText(crate::text_rows::SearchBuild::new())
            }
            Stage::FullText(mut search) => {
                if search.step(&self.index)? {
                    let started = std::time::Instant::now();
                    self.index.execute(
                        "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
                        [],
                    )?;
                    crate::timing::add(crate::timing::Stage::TrigramIndex, started.elapsed());
                    let mut deferred = crate::schema::built_at_end_sql()?;
                    deferred.reverse();
                    Stage::Deferred(deferred)
                } else {
                    Stage::FullText(search)
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
                    job.phase("install");
                    Stage::Install
                }
            },
            Stage::Install => {
                self.install_build()?;
                self.texts = crate::text_rows::TextIds::new(
                    &self.index,
                    self.location.memory_budget().keys() / 3,
                )?;
                self.search.ready();
                self.commits
                    .committed(crate::completion::Committed::CONTENT);
                return Ok(true);
            }
        };
        Ok(false)
    }

    /// Opens the merge of every worker's runs.
    fn start_load(&mut self, runs: Vec<PathBuf>) -> Result<Load> {
        let paths = Paths::of(&self.location.root);
        std::fs::create_dir_all(&paths.runs)?;
        // Texts upkeep wrote into the build file while the build ran keep
        // their rows; a run's equal text takes that row.
        let held: Vec<(Vec<u8>, i64)> = self
            .index
            .prepare("SELECT hash,id FROM text ORDER BY hash")?
            .query_map([], |r| <(Vec<u8>, i64)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        // References are checked by the indexes and composition after the
        // load; tables arrive in name order, not in reference order.
        self.index.pragma_update(None, "foreign_keys", false)?;
        Ok(Load {
            merged: Merger::open(runs, &paths.runs)?,
            tables: tables(&self.index)?,
            held: held.into_iter().peekable(),
            next_text: self
                .index
                .scalar("SELECT coalesce(max(id),0)+1 FROM text", [])?,
            next_group: self
                .index
                .scalar("SELECT coalesce(max(id),0)+1 FROM text_group", [])?,
            bases: std::collections::HashMap::new(),
            holders: Some(Sorter::new(
                &paths.runs,
                "text-holders".into(),
                (self.location.memory_budget().total() / 8) as usize,
            )),
            input: None,
            current_text: None,
            stream: None,
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
            let Some((key, value)) = load.merged.next()? else {
                break false;
            };
            let (&kind, rest) = key
                .split_first()
                .ok_or_else(|| StoreError::Invalid("a build stream holds an empty key".into()))?;
            if load.stream != Some(kind) {
                job.phase(match kind {
                    INPUTS | GROUPS | TEXTS => "merge texts",
                    _ => "load",
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
                    t.prepare_cached("INSERT INTO text_group(id,body) VALUES (?1,?2)")?
                        .execute(params![load.next_group, value])?;
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
                                t.prepare_cached(
                                    "INSERT INTO text(id,hash,normalized_hash,indexed,group_id,start,length)
                                     VALUES (?1,?2,?3,0,?4,?5,?6)",
                                )?
                                .execute(params![
                                    load.next_text,
                                    hash,
                                    row[4],
                                    base + seq,
                                    row[2],
                                    row[3]
                                ])?;
                                load.current_text = Some((hash.to_vec(), load.next_text));
                                load.next_text += 1;
                            }
                        }
                        Some(&TEXT_HOLDER) => {
                            let (_, id) = load.current_text.as_ref().ok_or_else(|| {
                                StoreError::Invalid("a reading names a text the build lacks".into())
                            })?;
                            let mut row = decode_row(value)?;
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

    /// Composes every session the readings state once, in key order, with
    /// the composition upkeep uses; the sessions composed.
    /// The sessions the readings state, in key order.
    fn stated_sessions(&self) -> Result<Vec<String>> {
        Ok(self
            .index
            .prepare(
                "SELECT session_id FROM session_record UNION SELECT session_id FROM message_parse
                 UNION SELECT session_id FROM event_parse WHERE session_id IS NOT NULL
                 UNION SELECT alias FROM session_alias
                 UNION SELECT parent_id FROM link_parse UNION SELECT child_id FROM link_parse
                 UNION SELECT session_id FROM call_outcome ORDER BY 1",
            )?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Composes the next transaction's worth of sessions, from `next`, with
    /// the composition upkeep uses; the first step also composes what no
    /// session holds (input events, documents).
    fn compose_chunk(&mut self, sessions: &[String], next: usize) -> Result<()> {
        let first = next == 0;
        let chunk = &sessions[next..sessions.len().min(next + SESSIONS_PER_TRANSACTION)];
        {
            let text_transaction = self.texts.transaction();
            let tx = self
                .index
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            let counter = bump_counter(&tx)?;
            let changes = crate::change_summary::ChangeSummary::start(&tx)?;
            let mut touched = super::readings::Touched::default();
            touched.whole.extend(chunk.iter().cloned());
            touched.sessions.extend(chunk.iter().cloned());
            for row in tx
                .prepare_cached(
                    "SELECT session_id,call_id FROM call_outcome WHERE session_id>=?1 AND session_id<=?2",
                )?
                .query_map(
                    params![chunk.first(), chunk.last()],
                    |r| <(String, String)>::try_from(r),
                )?
            {
                touched.calls.insert(row?);
            }
            for row in tx
                .prepare_cached(
                    "SELECT fields,data,kind FROM event_parse WHERE session_id>=?1 AND session_id<=?2
                     ORDER BY at,input_pk,chain,seq",
                )?
                .query_map(params![chunk.first(), chunk.last()], |r| {
                    <(Vec<u8>, Vec<u8>, String)>::try_from(r)
                })?
            {
                let (fields, data, kind) = row?;
                if polylogue_model::live::is_live_event(&kind) {
                    let mut event: Event = crate::encoding::unpack_json(&fields)?;
                    event.data = crate::encoding::unpack_json(&data)?;
                    touched.live.push(event);
                }
            }
            if first {
                // What no session holds: input events and documents.
                for row in tx
                    .prepare("SELECT event_id FROM event_parse WHERE session_id IS NULL")?
                    .query_map([], |r| r.get::<_, String>(0))?
                {
                    touched.events.insert((None, row?));
                }
                for row in tx
                    .prepare("SELECT DISTINCT document_id FROM document_parse")?
                    .query_map([], |r| r.get::<_, String>(0))?
                {
                    touched.documents.insert(row?);
                }
            }
            rows::finish(&tx, touched, &changes, &self.texts, true, counter)?;
            tx.commit()?;
            text_transaction.commit();
        }
        Ok(())
    }
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
