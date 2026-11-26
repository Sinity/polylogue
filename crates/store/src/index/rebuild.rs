//! The index rebuild (docs/architecture.md, one path): a private
//! `index-rebuild/build.db` written by its own connection on its own
//! thread, from the archive as it stood at the rebuild's start. The index
//! writer never writes it; it keeps writing the installed `index.db` when
//! there is a current one, and is stopped and started again on the new file
//! only for the install.
//!
//! A rebuild runs its stages in order ([`RebuildStage`]). The engine parses
//! every stored chain on several workers ([`RebuildWorker`]), each into
//! sorted files of its own; [`Rebuild::finish`] then keys the texts, loads
//! the parse tables, assembles every session, makes the indexes, settles
//! full text and installs the file. Each stage leaves a [`StageRecord`] in
//! the log and in `status`. A rebuild that fails is discarded and starts
//! again from the stored bytes; one that is stopped is started again by the
//! next open ([`crate::index::file`]).
use crate::archive::Archive;
use crate::archive::build_progress::{CountUnit, StepCount};
use crate::archive::writer::Position;
use crate::index::file::Paths;
use crate::{ConnExt as _, Result, StoreError};
use rusqlite::{Connection, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

mod assemble;
mod load;
mod worker;
pub use worker::{OpenChain, RebuildWorker};

/// A stage of a rebuild, in the order the stages run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RebuildStage {
    /// Every stored chain read whole by the parse workers into sorted files.
    Parsing,
    /// Inputs, content groups and texts loaded, each text made one row.
    KeyingTexts,
    /// The parse tables loaded in key order, with their indexes.
    LoadingParses,
    /// Every session assembled from its parses.
    Assembling,
    /// The indexes made after assembly, full-text membership and the
    /// archive's alternate text.
    Indexing,
    /// The full-text files finished.
    FullText,
    /// The file synced and installed as `index.db`.
    Installing,
}

impl RebuildStage {
    /// Every stage, in the order a rebuild runs them.
    pub const ALL: [Self; 7] = [
        Self::Parsing,
        Self::KeyingTexts,
        Self::LoadingParses,
        Self::Assembling,
        Self::Indexing,
        Self::FullText,
        Self::Installing,
    ];

    /// The stage as `status`, the log and the docs name it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::KeyingTexts => "keying texts",
            Self::LoadingParses => "loading parses",
            Self::Assembling => "assembling",
            Self::Indexing => "indexing",
            Self::FullText => "full text",
            Self::Installing => "installing",
        }
    }

    /// What the stage counts as it works, if anything.
    pub fn unit(self) -> Option<CountUnit> {
        match self {
            Self::Parsing => Some(CountUnit::Bytes),
            Self::KeyingTexts | Self::LoadingParses => Some(CountUnit::Records),
            Self::Assembling => Some(CountUnit::Sessions),
            Self::Indexing | Self::FullText | Self::Installing => None,
        }
    }
}

/// What one stage of a rebuild did: its own record, logged when it ends and
/// kept for `status` until the rebuild is installed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StageRecord {
    pub stage: RebuildStage,
    /// Wall-clock start and end, in Unix milliseconds.
    pub started_ms: i64,
    pub ended_ms: i64,
    pub seconds: f64,
    /// What the stage handled, in its unit ([`RebuildStage::unit`]): bytes
    /// parsed, records loaded, sessions assembled; zero for a stage that
    /// counts nothing.
    pub items: u64,
    /// Bytes this process read from and wrote to storage during the stage.
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// Reads the process's control group made from its devices during the
    /// stage, and their bytes, when the system reports them: the mean read
    /// size tells reads in file order (large) from reads at random (a page
    /// or a few).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_reads: Option<(u64, u64)>,
    /// The process's peak resident memory during the stage, in bytes; when
    /// `peak_is_stage` is false the system refused to reset the peak and
    /// this is the process's lifetime peak.
    pub peak_memory: u64,
    pub peak_is_stage: bool,
    /// The process's peak resident memory over its life so far.
    pub process_peak_memory: u64,
}

/// The stage running, and what its record is measured from.
struct Running {
    stage: RebuildStage,
    started_ms: i64,
    started: Instant,
    io: (u64, u64),
    device: Option<(u64, u64)>,
    peak: crate::peak_memory::Stage,
    items: u64,
    total: Option<u64>,
}

/// A rebuild in progress: the build file's own connection, the archive
/// position it is built from, and its stages so far.
pub struct Rebuild {
    archive: Archive,
    conn: Connection,
    position: Position,
    stages: Stages,
}

/// A rebuild's stages: the one running and the records of those ended.
struct Stages {
    archive: Archive,
    clock: polylogue_model::WallClock,
    running: Option<Running>,
    records: Vec<StageRecord>,
}

impl Rebuild {
    /// Starts the rebuild the archive's index files say is being written
    /// (`Rebuilding::Writing`), at the archive's present record position,
    /// with its first stage, parsing, running.
    pub fn start(archive: &Archive, clock: polylogue_model::WallClock) -> Result<Self> {
        if !archive.index_build.building() {
            return Err(StoreError::internal("no index rebuild is being written"));
        }
        let (_, position) = crate::index::file::archive_identity(&crate::archive::open_read(
            &archive.archive_db(),
        )?)?;
        let conn = crate::archive::index_build::open_writer(archive)?;
        let uri = format!("file:{}?mode=ro", archive.archive_db().display());
        conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        crate::source::views(&conn)?;
        let stop = archive.stop().clone();
        conn.progress_handler(1_000, Some(move || stop.is_cancelled()))?;
        let mut rebuild = Self {
            archive: archive.clone(),
            conn,
            position,
            stages: Stages {
                archive: archive.clone(),
                clock,
                running: None,
                records: Vec::new(),
            },
        };
        archive.record_build_plan(&RebuildStage::ALL.map(RebuildStage::name));
        rebuild.stage(RebuildStage::Parsing);
        Ok(rebuild)
    }

    /// The archive position the rebuild covers: what the archive had
    /// committed when it started. What it commits later reaches the index
    /// by incremental updates once the rebuild is installed.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Installs an export's assets into the build file, before its
    /// sessions are assembled: messages that name an asset find it then.
    pub fn import_assets(&mut self, staged: crate::export_items::ExportItems) -> Result<()> {
        let archive = crate::archive::open_read(&self.archive.archive_db())?;
        crate::assets::install(&mut self.conn, &archive, staged)
    }

    /// Runs every stage after parsing, from the parse workers' sorted files
    /// `runs` (`parsed` bytes of stored chains read), assembling on `assemblers`
    /// threads, and installs the rebuilt index through `writer`. Returns
    /// every stage's record. A rebuild that fails before its file is ready
    /// is discarded and started again, empty, for the same reason; one that
    /// fails while it is installed stays ready, for the index writer's next
    /// commit event or the next start to install.
    pub fn finish(
        mut self,
        runs: Vec<PathBuf>,
        parsed: u64,
        assemblers: usize,
        writer: &crate::WriterSender,
    ) -> Result<Vec<StageRecord>> {
        self.stages.set_count(parsed);
        let mut search = None;
        let built = self.build(runs, assemblers, &mut search);
        // The full-text fill's threads read the build file: they end first.
        drop(search);
        let Self {
            archive,
            conn,
            mut stages,
            ..
        } = self;
        let readied = built.and_then(|()| {
            stages.start(RebuildStage::Installing);
            conn.close().map_err(|(_, error)| StoreError::from(error))?;
            crate::archive::index_build::ready(&archive)
        });
        if let Err(error) = readied {
            if archive.stop().is_cancelled() {
                return Err(StoreError::Stopped);
            }
            crate::archive::index_build::restart(&archive, &error.to_string()).map_err(
                |restart| {
                    StoreError::Invalid(format!(
                        "{error}; the failed rebuild could not be discarded: {restart}"
                    ))
                },
            )?;
            return Err(error);
        }
        writer.install()?;
        stages.end();
        Ok(stages.records)
    }

    fn build(
        &mut self,
        runs: Vec<PathBuf>,
        assemblers: usize,
        search: &mut Option<crate::text_rows::SearchFill>,
    ) -> Result<()> {
        self.stage(RebuildStage::KeyingTexts);
        let (indexes, after_stream) = self.prepare_indexes()?;
        *search = Some(crate::text_rows::SearchFill::start(
            &self.conn,
            self.archive.memory_budget().search_fill_memory(),
        )?);
        let loaded = self.load(runs)?;
        self.stage(RebuildStage::LoadingParses);
        self.conn.pragma_update(None, "foreign_keys", true)?;
        self.size_cache(
            (loaded.merged.memory() + loaded.texts.memory()) as u64,
            true,
        )?;
        for sql in &indexes {
            self.archive.stop().check()?;
            self.conn.execute(sql, [])?;
        }
        self.stage(RebuildStage::Assembling);
        self.assemble(loaded, after_stream, assemblers)?;
        crate::archive::index_build::remove_runs(&Paths::of(&self.archive.root))?;
        self.stage(RebuildStage::Indexing);
        // The runs are read: what is left of the rebuild is its file's
        // indexes and full text.
        self.size_cache(0, true)?;
        {
            // Assembly wrote assembled rows only: the search membership of
            // every text comes from the blocks now, and from here on
            // membership and the trigram index follow the assembled rows as
            // they do in an installed index.
            let tx = self
                .conn
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            crate::writer::shown::mark_searched_texts(&tx)?;
            tx.execute_batch(crate::schema::SEARCH_TRIGGERS)?;
            // Sessions were assembled without their summaries: each is
            // summarized from its first row once the file is installed.
            crate::summary::mark_every_session(&tx)?;
            tx.commit()?;
        }
        search
            .as_mut()
            .ok_or_else(|| StoreError::internal("a rebuild has no full-text fill"))?
            .settle(&self.conn)?;
        let started = Instant::now();
        self.conn.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
            [],
        )?;
        crate::timing::add(crate::timing::Stage::TrigramIndex, started.elapsed());
        for sql in crate::schema::built_at_end_sql()? {
            self.archive.stop().check()?;
            let started = Instant::now();
            self.conn.execute(&sql, [])?;
            crate::timing::add(crate::timing::Stage::SecondaryIndexes, started.elapsed());
            crate::fault::check("index.deferred", &self.archive.root.to_string_lossy())?;
        }
        self.take_in_archive()?;
        self.stage(RebuildStage::FullText);
        let search = search
            .as_mut()
            .ok_or_else(|| StoreError::internal("a rebuild has no full-text fill"))?;
        while !search.wait(std::time::Duration::from_millis(250))? {
            self.archive.stop().check()?;
        }
        Ok(())
    }

    /// Drops the build file's indexes the load would have to keep in no
    /// order, and returns the statements that make them again: those made
    /// after the load, and those on assembled tables made after the
    /// sessions are appended (see [`worker::SHOWN_TABLES`]).
    fn prepare_indexes(&mut self) -> Result<(Vec<String>, Vec<String>)> {
        use worker::{
            COMPOSED_IN_ORDER, SHOWN_TABLES, declared_load_indexes, index_target,
            kept_while_composing, kept_while_loading, load_indexes,
        };
        let (mut indexes, mut after_stream) = (Vec::new(), Vec::new());
        // A load drops every index but the one that makes an equal group one
        // row and those it fills in key order.
        for (name, sql, _) in load_indexes(&self.conn)? {
            if !kept_while_loading(&sql) {
                self.conn.execute(&format!("DROP INDEX \"{name}\""), [])?;
            }
        }
        let kept = load_indexes(&self.conn)?;
        for (name, sql, table) in declared_load_indexes()? {
            if kept.iter().any(|(held, ..)| *held == name) {
                continue;
            }
            if kept_while_loading(&sql) {
                self.conn.execute(&sql, [])?;
            } else if SHOWN_TABLES.contains(&table.as_str()) && !kept_while_composing(&sql) {
                after_stream.push(sql);
            } else {
                indexes.push(sql);
            }
        }
        // Of the indexes made at the end, those assembly fills in order are
        // made before it.
        for sql in crate::schema::built_at_end_sql()? {
            if index_target(&sql).is_some_and(|(table, first)| {
                SHOWN_TABLES.contains(&table) && COMPOSED_IN_ORDER.contains(&first)
            }) {
                indexes.push(sql);
            }
        }
        Ok((indexes, after_stream))
    }

    /// Takes in what the archive committed up to the rebuild's position
    /// that is not a chain's bytes: the text of alternates. The index writer
    /// follows the archive from that position once the file is installed.
    fn take_in_archive(&mut self) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::alternates::project(&tx)?;
        crate::meta_set(
            &tx,
            crate::index::file::ARCHIVE_POSITION,
            self.position.0.to_string(),
        )?;
        tx.commit()?;
        Ok(())
    }

    fn stage(&mut self, stage: RebuildStage) {
        self.stages.start(stage);
    }

    fn count(&mut self, items: u64) {
        self.stages.count(items);
    }

    fn total(&mut self, total: u64) {
        self.stages.total(total);
    }

    /// Gives the build file's cache what the budget leaves beside `held`
    /// bytes the stage holds of its own and what the process holds already
    /// outside that cache (its other connections' caches, the program, what
    /// the allocator kept). A stage that creates indexes (`sorting`) gives
    /// the cache a third of that: SQLite's sorter keeps records in memory up
    /// to the cache's size beside it, in a buffer that grows by doubling, so
    /// up to twice that while it grows.
    fn size_cache(&self, held: impl TryInto<u64>, sorting: bool) -> Result<()> {
        let held = held.try_into().unwrap_or(u64::MAX);
        let outside =
            crate::peak_memory::status_bytes("VmRSS:").saturating_sub(cache_used(&self.conn));
        let mut mib = self
            .archive
            .memory_budget()
            .build_cache_beside(held.saturating_add(outside));
        if sorting {
            mib = (mib / 3).max(32);
        }
        self.conn.pragma_update(None, "cache_size", -(mib * 1024))?;
        Ok(())
    }
}

impl Stages {
    /// Ends the stage running, if any, and starts `stage`; a stage already
    /// running goes on.
    fn start(&mut self, stage: RebuildStage) {
        if self
            .running
            .as_ref()
            .is_some_and(|running| running.stage == stage)
        {
            return;
        }
        self.end();
        let started_ms = self.clock.now_ms();
        if let Some(since) = polylogue_model::Timestamp::from_unix_ms(started_ms) {
            self.archive.record_build_step(stage.name(), since);
        }
        self.running = Some(Running {
            stage,
            started_ms,
            started: Instant::now(),
            io: process_io(),
            device: device_reads(),
            peak: crate::peak_memory::Stage::start(),
            items: 0,
            total: None,
        });
        self.report_count();
    }

    /// Ends the stage running: logs its record and keeps it.
    fn end(&mut self) {
        let Some(running) = self.running.take() else {
            return;
        };
        let (read, written) = process_io();
        let device = device_reads()
            .zip(running.device)
            .map(|(now, then)| (now.0.saturating_sub(then.0), now.1.saturating_sub(then.1)));
        let (peak_memory, peak_is_stage) = running.peak.peak();
        let record = StageRecord {
            stage: running.stage,
            started_ms: running.started_ms,
            ended_ms: self.clock.now_ms().max(running.started_ms),
            seconds: running.started.elapsed().as_secs_f64(),
            items: running.items,
            bytes_read: read.saturating_sub(running.io.0),
            bytes_written: written.saturating_sub(running.io.1),
            device_reads: device,
            peak_memory,
            peak_is_stage,
            process_peak_memory: crate::peak_memory::lifetime(),
        };
        crate::log!(
            info,
            "index: {} took {:.1} s; {}{:.1} MB read{}, {:.1} MB written; peak memory {:.0} MB{}, process peak {:.0} MB",
            record.stage.name(),
            record.seconds,
            running
                .stage
                .unit()
                .map(|unit| format!("{} {}; ", record.items, unit.word()))
                .unwrap_or_default(),
            record.bytes_read as f64 / 1e6,
            record
                .device_reads
                .filter(|(reads, _)| *reads > 0)
                .map(|(reads, bytes)| format!(
                    " in {reads} reads of {:.0} KiB",
                    bytes as f64 / reads as f64 / 1024.
                ))
                .unwrap_or_default(),
            record.bytes_written as f64 / 1e6,
            record.peak_memory as f64 / 1e6,
            if record.peak_is_stage {
                ""
            } else {
                " (the process's; its peak could not be reset)"
            },
            record.process_peak_memory as f64 / 1e6,
        );
        self.archive.record_stage(record.clone());
        self.records.push(record);
    }

    /// Adds `items` to the running stage's count.
    fn count(&mut self, items: u64) {
        if let Some(running) = &mut self.running {
            running.items += items;
        }
        crate::activity::progressed();
        self.report_count();
    }

    /// Sets the running stage's count.
    fn set_count(&mut self, items: u64) {
        if let Some(running) = &mut self.running {
            running.items = items;
        }
        self.report_count();
    }

    /// Sets the running stage's total, when it is known before it ends.
    fn total(&mut self, total: u64) {
        if let Some(running) = &mut self.running {
            running.total = Some(total);
        }
        self.report_count();
    }

    fn report_count(&self) {
        let Some(running) = &self.running else {
            return;
        };
        // Parsing is counted by the parse workers as they go.
        if running.stage == RebuildStage::Parsing {
            return;
        }
        if let Some(unit) = running.stage.unit() {
            self.archive.record_build_count(
                running.stage.name(),
                StepCount {
                    done: running.items,
                    total: running.total,
                    unit,
                },
            );
        }
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
