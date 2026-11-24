//! The batch build of a private index (docs/architecture.md, one path).
//! The engine prepares each chain's reading on its own, in parallel, into
//! run files ([`BuildRun`]): the same reading rows upkeep writes, each run in
//! a small file of its own whose lookups stay in memory. The writer then
//! loads the runs into the build file ([`Inner::build`]): texts first, made
//! one per hash by a merge of the runs' hash-ordered texts, then every
//! run's reading rows in key order with no secondary index to keep; it
//! creates the indexes, composes every session once with the composition
//! upkeep uses ([`super::shown`]), writes the search entries and installs the
//! file. Nothing the build writes is looked up to decide what to write, and a
//! build that does not finish is discarded and starts again from the stored
//! bytes ([`crate::archive::index_build`]).
use super::*;
use crate::archive::index_build::Paths;
use std::collections::{BinaryHeap, VecDeque};
use std::os::unix::fs::FileExt as _;
use std::path::PathBuf;
use std::time::Instant;

/// Run files close once their file reaches this size, so the reading rows
/// and texts a run looks up stay in its page cache.
const RUN_BYTES: u64 = 512 << 20;

/// Each run connection's page cache; the rest of a run stays in the
/// operating system's page cache.
const RUN_CACHE_KIB: i64 = 32 << 10;

/// Texts read from a run per query while the runs' texts are merged.
const MERGE_BATCH: i64 = 4096;

/// Sessions composed per transaction.
const SESSIONS_PER_TRANSACTION: usize = 512;

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
        eprintln!(
            "polylogue: index build: {} took {:.1} s; {} items; {:.1} MB read, {:.1} MB written; peak memory {:.0} MB",
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

/// One worker's run file: whole chains' readings, written as upkeep writes
/// them into an index of their own.
pub struct BuildRun {
    conn: Connection,
    register: Connection,
    texts: crate::text_rows::TextIds,
    blobs: Arc<crate::blobs::Blobs>,
    path: PathBuf,
    chains: u64,
}

impl BuildRun {
    /// Creates run `number` of the build in progress.
    pub fn create(archive: &Archive, number: u64) -> Result<Self> {
        if archive.is_primary() || !archive.index_build.building() {
            return Err(StoreError::Invalid(
                "runs belong to a standalone build in progress".into(),
            ));
        }
        let paths = Paths::of(&archive.root);
        std::fs::create_dir_all(&paths.runs)?;
        let path = paths.runs.join(format!("{number:06}.db"));
        crate::archive::index_build::remove_run(&path)?;
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "off")?;
        conn.pragma_update(None, "synchronous", "off")?;
        conn.pragma_update(None, "temp_store", "memory")?;
        conn.pragma_update(None, "cache_size", -RUN_CACHE_KIB)?;
        crate::schema::ensure_index(&conn)?;
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
            path,
            chains: 0,
        })
    }

    /// Writes a chain's reading, read from its first record to `length`.
    /// Bytes its parse decoded are stored first, through the writer.
    pub fn write(
        &mut self,
        writer: &WriterSender,
        input: &InputKey,
        chain: i64,
        length: u64,
        mut reading: ChainParse,
    ) -> Result<()> {
        crate::fault::check("index.build.prepare", &input.id)?;
        reading.finish()?;
        let decoded = reading.take_decoded();
        if !decoded.is_empty() {
            for result in writer.store_group(decoded, Arc::clone(&self.blobs))? {
                result?;
            }
        }
        let text_transaction = self.texts.transaction();
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        super::chain::write_reading(
            &tx,
            super::chain::ReadingWrite {
                register: &self.register,
                texts: &self.texts,
                counter: 1,
                primary: false,
                changes: &changes,
            },
            input,
            chain,
            length,
            &reading,
            &mut super::readings::Touched::default(),
        )?;
        drop(changes);
        tx.commit()?;
        text_transaction.commit();
        self.chains += 1;
        Ok(())
    }

    /// Whether the run has grown to the size at which it closes.
    pub fn full(&self) -> Result<bool> {
        let pages: i64 = self.conn.scalar("PRAGMA page_count", [])?;
        let size: i64 = self.conn.scalar("PRAGMA page_size", [])?;
        Ok((pages * size) as u64 >= RUN_BYTES)
    }

    /// Closes the run; its file, for the writer to load, when it holds a
    /// chain.
    pub fn finish(self) -> Result<Option<PathBuf>> {
        let Self {
            conn, path, chains, ..
        } = self;
        conn.close().map_err(|(_, error)| error)?;
        if chains == 0 {
            crate::archive::index_build::remove_run(&path)?;
            return Ok(None);
        }
        Ok(Some(path))
    }
}

/// Whether and how a run's table is copied into the build file. A
/// reading's rows each carry their input and chain, and conflict with no
/// other chain's; a repository lookup is the last one read; rows keyed by
/// content across chains are written once. Input and text rows are merged
/// before the copy, and a table with neither an input nor a content key is
/// not part of a reading.
fn copy(table: &str, columns: &[String]) -> Option<&'static str> {
    match table {
        "input_ref" | "text" | "text_group" => None,
        "text_fragment" | "wire_declaration" => Some("OR IGNORE"),
        "repo_lookup" => Some("OR REPLACE"),
        _ if columns.iter().any(|column| column == "input_pk") => Some(""),
        _ => None,
    }
}

/// A run's texts in hash order, read a batch at a time.
struct RunTexts {
    conn: Connection,
    buffer: VecDeque<StoredText>,
    after: Option<Vec<u8>>,
    done: bool,
}

struct StoredText {
    hash: Vec<u8>,
    normalized: Vec<u8>,
    group: i64,
    start: i64,
    length: i64,
    id: i64,
}

impl RunTexts {
    fn open(path: &std::path::Path) -> Result<Self> {
        Ok(Self {
            conn: crate::archive::open_read(path)?,
            buffer: VecDeque::new(),
            after: None,
            done: false,
        })
    }

    fn peek(&mut self) -> Result<Option<&StoredText>> {
        if self.buffer.is_empty() && !self.done {
            let rows = self
                .conn
                .prepare_cached(
                    "SELECT hash,normalized_hash,group_id,start,length,id FROM text
                     WHERE ?1 IS NULL OR hash>?1 ORDER BY hash LIMIT ?2",
                )?
                .query_map(params![self.after, MERGE_BATCH], |r| {
                    Ok(StoredText {
                        hash: r.get(0)?,
                        normalized: r.get(1)?,
                        group: r.get(2)?,
                        start: r.get(3)?,
                        length: r.get(4)?,
                        id: r.get(5)?,
                    })
                })?
                .collect::<rusqlite::Result<VecDeque<_>>>()?;
            self.done = (rows.len() as i64) < MERGE_BATCH;
            self.after = rows.back().map(|text| text.hash.clone());
            self.buffer = rows;
        }
        Ok(self.buffer.front())
    }
}

impl Inner {
    /// Loads the runs into the build file, composes every session, writes
    /// the search entries and the deferred indexes, and installs the file.
    /// Each phase's name is sent to `announce` as it starts.
    pub(super) fn build(
        &mut self,
        runs: Vec<PathBuf>,
        announce: Option<std::sync::mpsc::Sender<&'static str>>,
    ) -> Result<Vec<BuildPhase>> {
        let start = |name: &'static str| {
            if let Some(announce) = &announce {
                let _ = announce.send(name);
            }
            PhaseClock::start(name)
        };
        if self.sinex.is_some() || !self.location.index_build.building() {
            return Err(StoreError::Invalid(
                "no standalone build is in progress".into(),
            ));
        }
        let mut phases = Vec::new();
        self.index.flush_prepared_statement_cache();
        let clock = start("merge texts");
        let indexes: Vec<(String, String)> = self
            .index
            .prepare(
                "SELECT name,sql FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL
                 AND tbl_name NOT IN (SELECT name FROM sqlite_schema WHERE sql LIKE 'CREATE VIRTUAL TABLE%')",
            )?
            .query_map([], |r| <(String, String)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        for (name, _) in &indexes {
            self.index.execute(&format!("DROP INDEX \"{name}\""), [])?;
        }
        let texts = self.merge_texts(&runs)?;
        phases.push(clock.end(texts));
        let clock = start("load");
        let mut rows = 0;
        for path in &runs {
            self.stop.check()?;
            rows += self.load_run(path)?;
        }
        phases.push(clock.end(rows));
        let clock = start("indexes");
        for (_, sql) in &indexes {
            self.stop.check()?;
            self.index.execute(sql, [])?;
        }
        phases.push(clock.end(indexes.len() as u64));
        let clock = start("compose");
        let sessions = self.compose_all()?;
        phases.push(clock.end(sessions));
        let clock = start("full text");
        self.build_deferred()?;
        phases.push(clock.end(0));
        let clock = start("install");
        self.install_build()?;
        self.texts =
            crate::text_rows::TextIds::new(&self.index, self.location.memory_budget().keys() / 3)?;
        self.search.ready();
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        phases.push(clock.end(0));
        Ok(phases)
    }

    /// Every run's texts, made one row per text; the texts written.
    fn merge_texts(&mut self, runs: &[PathBuf]) -> Result<u64> {
        let held: bool = self.index.exists("SELECT EXISTS(SELECT 1 FROM text)", [])?;
        if held {
            return Err(StoreError::Invalid(
                "the build file holds texts before its runs are loaded".into(),
            ));
        }
        // Each run's groups keep their order under a base above the runs
        // before it. A run is attached outside the transactions that read it.
        let mut bases = Vec::with_capacity(runs.len());
        let mut base: i64 = self
            .index
            .scalar("SELECT coalesce(max(id),0) FROM text_group", [])?;
        for path in runs {
            bases.push(base);
            self.attach_run(path)?;
            let tx = self
                .index
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute(
                "INSERT INTO text_group(id,body) SELECT id+?1,body FROM run.text_group ORDER BY id",
                [base],
            )?;
            base += tx.scalar::<i64>("SELECT coalesce(max(id),0) FROM run.text_group", [])?;
            tx.commit()?;
            self.index.execute("DETACH DATABASE run", [])?;
        }
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let written = merge_texts(&tx, runs, &bases)?;
        tx.execute(
            "DELETE FROM text_group WHERE id NOT IN (SELECT group_id FROM text)",
            [],
        )?;
        tx.commit()?;
        Ok(written)
    }

    /// Copies one run's rows, with its input and text keys made the build
    /// file's, in the run's key order.
    fn load_run(&mut self, path: &std::path::Path) -> Result<u64> {
        self.attach_run(path)?;
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Tables are copied in name order, so a row may come before the row
        // it refers to; references are checked when the run commits.
        tx.execute_batch(
            "PRAGMA defer_foreign_keys = ON;
             CREATE TEMP TABLE input_map(local INTEGER PRIMARY KEY, main INTEGER NOT NULL);
             CREATE TEMP TABLE text_map(local INTEGER PRIMARY KEY, main INTEGER NOT NULL);
             INSERT INTO input_ref(id,format,route) SELECT id,format,route FROM run.input_ref ORDER BY pk
             ON CONFLICT(id) DO UPDATE SET format=excluded.format,route=excluded.route
             WHERE format IS NOT excluded.format OR route IS NOT excluded.route;
             INSERT INTO temp.input_map SELECT r.pk,m.pk FROM run.input_ref r JOIN main.input_ref m ON m.id=r.id;",
        )?;
        let map = std::fs::read(map_path(path))?;
        {
            let mut insert = tx.prepare("INSERT INTO temp.text_map VALUES (?1,?2)")?;
            for (local, slot) in map.as_chunks::<8>().0.iter().enumerate() {
                let main = i64::from_le_bytes(*slot);
                if main != 0 {
                    insert.execute(params![local as i64, main])?;
                }
            }
        }
        let tables: Vec<String> = tx
            .prepare(
                "SELECT name FROM run.sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'
                 AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name",
            )?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut loaded = 0;
        for table in tables {
            let columns: Vec<String> = tx
                .prepare(&format!(
                    "SELECT name FROM pragma_table_info('{table}','run') ORDER BY cid"
                ))?
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            let Some(conflict) = copy(&table, &columns) else {
                continue;
            };
            if !tx.exists(&format!("SELECT EXISTS(SELECT 1 FROM run.\"{table}\")"), [])? {
                continue;
            }
            let mut from = format!("run.\"{table}\" r");
            let values: Vec<String> = columns
                .iter()
                .map(|column| match column.as_str() {
                    "input_pk" => "im.main".to_owned(),
                    "text_id" => "tm.main".to_owned(),
                    other => format!("r.\"{other}\""),
                })
                .collect();
            if columns.iter().any(|c| c == "input_pk") {
                from.push_str(" CROSS JOIN temp.input_map im ON im.local=r.input_pk");
            }
            if columns.iter().any(|c| c == "text_id") {
                from.push_str(" CROSS JOIN temp.text_map tm ON tm.local=r.text_id");
            }
            let names = columns
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(",");
            let copied = tx.execute(
                &format!(
                    "INSERT {conflict} INTO main.\"{table}\"({names}) SELECT {} FROM {from}",
                    values.join(","),
                ),
                [],
            )?;
            let held: i64 = tx.scalar(&format!("SELECT count(*) FROM run.\"{table}\""), [])?;
            if conflict.is_empty() && copied as i64 != held {
                return Err(StoreError::Invalid(format!(
                    "run {} has {table} rows naming no input or text",
                    path.display()
                )));
            }
            loaded += copied as u64;
        }
        tx.execute_batch("DROP TABLE temp.input_map; DROP TABLE temp.text_map;")?;
        tx.commit()?;
        self.index.execute("DETACH DATABASE run", [])?;
        crate::archive::index_build::remove_run(path)?;
        crate::fault::check("index.build.after_run", "")?;
        Ok(loaded)
    }

    /// Attaches a run file as `run`, read-only.
    fn attach_run(&self, path: &std::path::Path) -> Result<()> {
        let uri = format!("file:{}?mode=ro", path.display());
        self.index.execute("ATTACH DATABASE ?1 AS run", [uri])?;
        Ok(())
    }

    /// Composes every session the readings state once, in key order, with
    /// the composition upkeep uses; the sessions composed.
    fn compose_all(&mut self) -> Result<u64> {
        let sessions: Vec<String> = self
            .index
            .prepare(
                "SELECT session_id FROM session_record UNION SELECT session_id FROM message_parse
                 UNION SELECT session_id FROM event_parse WHERE session_id IS NOT NULL
                 UNION SELECT alias FROM session_alias
                 UNION SELECT parent_id FROM link_parse UNION SELECT child_id FROM link_parse
                 UNION SELECT session_id FROM call_outcome ORDER BY 1",
            )?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut first = true;
        let count = sessions.len() as u64;
        let mut chunks = sessions.chunks(SESSIONS_PER_TRANSACTION).peekable();
        while first || chunks.peek().is_some() {
            self.stop.check()?;
            let chunk = chunks.next().unwrap_or_default();
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
                first = false;
            }
            rows::finish(&tx, touched, &changes, &self.texts, true, counter)?;
            tx.commit()?;
            text_transaction.commit();
        }
        Ok(count)
    }
}

/// The path of the file mapping a run's text identities to the build file's.
fn map_path(run: &std::path::Path) -> PathBuf {
    run.with_extension("texts")
}

/// Merges the runs' texts by hash: each distinct text gets the next
/// identity, in hash order, and the group its first run stored it in. Each
/// run's identities are mapped to the build file's in a file of eight bytes
/// per identity. Returns the texts written.
fn merge_texts(tx: &Transaction<'_>, runs: &[PathBuf], bases: &[i64]) -> Result<u64> {
    let mut sources = runs
        .iter()
        .map(|path| RunTexts::open(path))
        .collect::<Result<Vec<_>>>()?;
    let maps = runs
        .iter()
        .map(|path| std::fs::File::create(map_path(path)))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (n, source) in sources.iter_mut().enumerate() {
        if let Some(text) = source.peek()? {
            heap.push(std::cmp::Reverse((text.hash.clone(), n)));
        }
    }
    let mut next: i64 = tx.scalar("SELECT coalesce(max(id),0)+1 FROM text", [])?;
    let mut current: Option<(Vec<u8>, i64)> = None;
    let mut insert = tx.prepare(
        "INSERT INTO text(id,hash,normalized_hash,indexed,group_id,start,length) VALUES (?1,?2,?3,0,?4,?5,?6)",
    )?;
    let mut written = 0;
    while let Some(std::cmp::Reverse((hash, n))) = heap.pop() {
        let text = sources[n].buffer.pop_front().expect("a peeked text");
        let id = match &current {
            Some((held, id)) if *held == hash => *id,
            _ => {
                let id = next;
                next += 1;
                insert.execute(params![
                    id,
                    text.hash,
                    text.normalized,
                    text.group + bases[n],
                    text.start,
                    text.length
                ])?;
                written += 1;
                current = Some((hash, id));
                id
            }
        };
        let slot = u64::try_from(text.id)
            .map_err(|_| StoreError::stored(0, "a run's text has a negative identity"))?;
        maps[n].write_all_at(&id.to_le_bytes(), slot * 8)?;
        if let Some(text) = sources[n].peek()? {
            heap.push(std::cmp::Reverse((text.hash.clone(), n)));
        }
    }
    Ok(written)
}
