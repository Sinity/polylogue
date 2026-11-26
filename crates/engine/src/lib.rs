//! The loop. One loop runs in the daemon:
//!
//! ```text
//! collect -> store -> parse -> derive -> wait
//! ```
//!
//! Each step finds its work by comparing what exists with what should
//! exist. There is no to-do list and no repair path; `status` calls the
//! same comparison.
//!
//! Parser working state lives with active input ranges. Continuation is restored
//! from stored checkpoints when changed input starts; history-sized lookups use
//! the disk-backed parser context.

pub mod collect;
mod compressed;
mod local_files;
pub mod pool;
pub mod read;
pub mod repo_lookup;
mod watch;

use databases::Databases;
use plan::{InputState, Plan};
use polylogue_model::LockExt;
use polylogue_model::batch::StoppedAttempt;
use polylogue_sources::{ORIGINS, OriginSpec, Support};
use polylogue_store::{
    Archive, Change, FileFacts, InputKey, KnownInput, Observation, ReadPool, WriteReport, Writer,
};
use pool::{Pool, Priority};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod build;
mod build_log;
mod capture_read;
mod chain_names;
mod collectors;
mod databases;
mod failure_log;
mod import;
pub use import::python_polylogue::{ClassCount, PythonImportReport, RefusedRow};
mod parser_checkpoint;
mod pass;
mod plan;
mod prefix_copies;

/// An origin has a reading only when its declared parser or generic row-dump
/// reader supplies one. Unparsed provider bytes have no parser version to invent.
fn has_reading(origin: &OriginSpec) -> bool {
    origin.support == Support::Parsed
        || (origin.support == Support::NoParser && origin.capture.is_row_dump())
}

/// One view of the built-in registry is shared by reparse planning and read freshness.
pub fn reading_contracts() -> Vec<polylogue_store::ReadingContract> {
    ORIGINS
        .iter()
        .map(|origin| {
            if origin.support == Support::Kept {
                polylogue_store::ReadingContract::Kept {
                    origin: origin.name.to_owned(),
                }
            } else if !has_reading(origin) {
                polylogue_store::ReadingContract::RawOnly {
                    origin: origin.name.to_owned(),
                }
            } else {
                polylogue_store::ReadingContract::Expected {
                    origin: origin.name.to_owned(),
                    version: origin.parse_version(),
                }
            }
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Store(polylogue_store::StoreError),
    /// A chain's bytes cannot be read under its reading version: it stands
    /// until either changes, and is the only error recorded against a chain.
    #[error("{0}")]
    Chain(polylogue_model::ChainFailure),
    #[error(transparent)]
    Blobs(#[from] hashed_blobs::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Watch(#[from] notify::Error),
    /// The archive's stop request was made while a pass ran
    /// ([`polylogue_store::Archive::stop`]).
    #[error("the loop was asked to stop")]
    Stopped,
}

impl polylogue_model::Caused for EngineError {
    fn cause(&self) -> polylogue_model::Cause {
        use polylogue_model::Cause;
        match self {
            Self::Store(error) => error.cause(),
            Self::Chain(failure) => failure.cause(),
            Self::Blobs(error) => polylogue_store::StoreError::blob_cause(error),
            Self::Io(error) => polylogue_store::StoreError::io_cause(error),
            Self::Watch(error) => match &error.kind {
                notify::ErrorKind::Io(error) => polylogue_store::StoreError::io_cause(error),
                notify::ErrorKind::MaxFilesWatch => Cause::Environment,
                _ => Cause::Bug,
            },
            Self::Stopped => Cause::Stopping,
        }
    }

    fn code(&self) -> polylogue_model::Code {
        use polylogue_model::Code;
        match self {
            Self::Store(error) => error.code(),
            Self::Chain(failure) => failure.code(),
            Self::Blobs(hashed_blobs::Error::Corrupt { .. } | hashed_blobs::Error::Refused(_)) => {
                Code::BlobCorrupt
            }
            Self::Blobs(_) | Self::Io(_) | Self::Watch(_) => Code::Io,
            Self::Stopped => Code::ShuttingDown,
        }
    }
}

impl EngineError {
    /// The source file changed since its read was planned in a way the read
    /// cannot take as growth ([`read::read_source_range`]).
    pub(crate) fn source_changed(&self) -> bool {
        let store = match self {
            Self::Store(error) => Some(error),
            Self::Io(error) => error
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<polylogue_store::StoreError>()),
            _ => None,
        };
        matches!(store, Some(polylogue_store::StoreError::SourceChanged(_)))
    }
}

impl From<polylogue_model::ChainFailure> for EngineError {
    fn from(failure: polylogue_model::ChainFailure) -> Self {
        Self::Chain(failure)
    }
}

impl From<polylogue_model::Reason> for EngineError {
    fn from(reason: polylogue_model::Reason) -> Self {
        Self::Store(reason.into())
    }
}

impl EngineError {
    /// A broken invariant in the engine: reported as a bug, never retried.
    pub fn internal(message: impl ToString) -> Self {
        Self::Store(polylogue_store::StoreError::internal(message))
    }
}

impl From<polylogue_store::StoreError> for EngineError {
    fn from(error: polylogue_store::StoreError) -> Self {
        match error {
            polylogue_store::StoreError::Stopped => Self::Stopped,
            error => Self::Store(error),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// The one wall clock supplied by the binary boundary.
    pub clock: polylogue_model::WallClock,
    /// The home directory origins' locations are relative to.
    pub home: PathBuf,
    /// Tools' own location variables (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`)
    /// as the running user set them. An origin whose root names one of
    /// these reads from it instead of its default under `home`.
    pub tool_dirs: BTreeMap<String, PathBuf>,
    /// The process allowance shared by caches, queued work and parse workers.
    pub memory_budget: polylogue_store::archive::MemoryBudget,
    /// Parse workers, bounded by CPU and memory allowances.
    pub workers: usize,
    /// Combined SQLite page-cache target for client and loop readers.
    /// Mapped reader pages are disabled. This is not a total RSS limit.
    pub reader_memory_bytes: u64,
    /// Bytes in the parse pool at once. The pool takes new work
    /// only while the bytes in flight stay under this.
    pub bytes_in_flight: u64,
    /// `/etc/machine-id`: paths from two machines never collide.
    pub machine: String,
    pub host: String,
    /// The range one job reads at most (default 16 MiB), so a large file
    /// is stored and parsed in steps (a document origin's file is read whole
    /// in one job); an export's items commit in batches
    /// of about this many of its bytes. Not a limit: a record longer than
    /// this doubles the range for its input until it fits.
    pub chunk: u64,
    /// While a pass runs, its spools and recently written files are looked
    /// at this often, and what changed there goes ahead of the pass's
    /// remaining work (two seconds).
    pub live_every: Duration,
    /// Whether the loop reads capture sources: provider locations, spools,
    /// tool databases and collectors. Without them it parses and derives
    /// only what the archive already stores, and imports what it is given.
    pub reads_sources: bool,
}

impl Config {
    /// Apply one process budget while preserving smaller explicit working budgets.
    pub fn with_memory_budget(mut self, budget: polylogue_store::archive::MemoryBudget) -> Self {
        self.memory_budget = budget;
        self.workers = self
            .workers
            .min(budget.worker_count(self.workers, self.workers));
        self.reader_memory_bytes = self.reader_memory_bytes.min(budget.readers());
        self.bytes_in_flight = self.bytes_in_flight.min(budget.in_flight());
        self.chunk = self.chunk.min(16 << 20);
        self
    }

    /// One reader slot is kept for the loop; client reads have `workers`
    /// slots. All slots share one page-cache target, split again between index,
    /// archive and optional embeddings by the store. Integer division leaves unused bytes.
    /// The target first sets aside the handles of chains being read: one
    /// per parse worker and one per client slot.
    pub fn reader_memory_per_connection(&self) -> polylogue_store::Result<u64> {
        if self.workers == 0 {
            return Err(polylogue_model::Reason::NotPositive {
                field: "workers".into(),
            }
            .into());
        }
        let overflow = || polylogue_store::StoreError::internal("reader slot count overflow");
        let workers = u64::try_from(self.workers).map_err(|_| overflow())?;
        let slots = workers.checked_add(1).ok_or_else(overflow)?;
        let chains = workers
            .checked_mul(2 * polylogue_store::blobs::CHAIN_CACHE_KIB * 1024)
            .ok_or_else(overflow)?;
        let bytes = self.reader_memory_bytes.saturating_sub(chains) / slots;
        if bytes < 3 * 1024 {
            return Err(polylogue_model::Reason::BadConfiguration {
                problem: "the reader memory budget needs at least 3 KiB per connection".into(),
            }
            .into());
        }
        Ok(bytes)
    }

    /// The directory an origin's locations are relative to.
    pub fn root(&self, root: polylogue_sources::origin::Root) -> PathBuf {
        match root.env.and_then(|var| self.tool_dirs.get(var)) {
            Some(dir) if dir.is_absolute() => dir.clone(),
            Some(dir) => self.home.join(dir),
            None => self.home.join(root.default),
        }
    }

    /// Polylogue's own state directory (`$XDG_STATE_HOME/polylogue`): the
    /// spools, and what their writers keep beside them.
    pub fn state_dir(&self) -> PathBuf {
        self.root(polylogue_sources::origin::Root::STATE)
            .join(polylogue_sources::origin::STATE_DIR)
    }

    /// Where Polylogue's own writer puts the files of the spool origin
    /// `origin`, which the loop then collects.
    ///
    /// # Panics
    /// When `origin` is not a spool origin: the names are constants.
    pub fn spool_dir(&self, origin: &str) -> PathBuf {
        let spec = polylogue_sources::origin::by_name(origin)
            .unwrap_or_else(|| panic!("no origin {origin}"));
        let dir = spec
            .spool_dir()
            .unwrap_or_else(|| panic!("{origin} is not a spool origin"));
        self.root(spec.root).join(dir)
    }

    /// Construct from facts captured once by the binary. Location overrides
    /// are resolved to absolute paths before they cross this boundary.
    pub fn new(
        home: PathBuf,
        tool_dirs: BTreeMap<String, PathBuf>,
        machine: String,
        host: String,
        workers: usize,
        clock: polylogue_model::WallClock,
    ) -> Result<Self, EngineError> {
        if !home.is_absolute() || tool_dirs.values().any(|path| !path.is_absolute()) {
            return Err(polylogue_model::Reason::BadConfiguration {
                problem: "configured locations must be absolute".into(),
            }
            .into());
        }
        if workers == 0 || machine.trim().is_empty() || host.trim().is_empty() {
            return Err(polylogue_model::Reason::BadConfiguration {
                problem: "resource and host identity inputs must be nonempty".into(),
            }
            .into());
        }
        // The working shares at most; the budget the process is given
        // (`with_memory_budget`, which the engine applies) lowers them and
        // the worker count, so a default never caps a larger machine.
        Ok(Self {
            clock,
            home,
            tool_dirs,
            workers,
            memory_budget: Default::default(),
            reader_memory_bytes: 128 << 20,
            bytes_in_flight: 512 << 20,
            machine,
            host,
            chunk: 16 << 20,
            live_every: Duration::from_secs(2),
            reads_sources: true,
        })
    }
}

/// What one pass of the loop found and did, for `status`.
#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct Pass {
    /// Historical effects accepted during stored-input replay.
    pub authored: UserStateReport,
    /// Whole-location scans, excluding changed paths and live-file checks.
    pub full_scans: u64,
    /// Input visits, summed across turns by run_pass.
    pub inputs_seen: u64,
    pub inputs_changed: u64,
    /// Bytes read from input files (new bytes plus any parsed again).
    pub bytes_read: u64,
    /// Bytes the pass added to the archive: a blob or chunk it already
    /// held counts nothing.
    pub bytes_stored: u64,
    pub segment_syncs: u64,
    pub segment_bytes_written: u64,
    pub blobs_appended: u64,
    pub blobs_deduplicated: u64,
    pub messages_written: u64,
    /// Older chains whose reading was recorded from a newer chain of the
    /// same input whose bytes begin alike, instead of parsed again.
    pub readings_copied: u64,
    /// Bytes of those older chains the copies read: each copy's position,
    /// before which the chain is not parsed.
    pub bytes_copied: u64,
    /// Sessions whose derived products the pass recomputed.
    pub sessions_derived: u64,
    /// An import left previously queued derived values to the background worker.
    #[serde(default)]
    pub derivation_pending: bool,
    pub seconds: f64,
    pub unreadable_roots: Vec<PathBuf>,
    /// What failed in the pass, each with what it happened to and its cause.
    pub errors: Vec<polylogue_model::Failure>,
    /// Capture work is still queued.
    pub capture_pending: bool,
}

impl Pass {
    /// Adds a later pass's work to this one; unreadable roots and errors
    /// accumulate.
    pub fn add(&mut self, next: Pass) {
        self.authored.applied += next.authored.applied;
        self.authored.present += next.authored.present;
        self.authored.pending += next.authored.pending;
        self.full_scans += next.full_scans;
        self.inputs_seen += next.inputs_seen;
        self.inputs_changed += next.inputs_changed;
        self.bytes_read += next.bytes_read;
        self.bytes_stored += next.bytes_stored;
        self.segment_syncs += next.segment_syncs;
        self.segment_bytes_written += next.segment_bytes_written;
        self.blobs_appended += next.blobs_appended;
        self.blobs_deduplicated += next.blobs_deduplicated;
        self.messages_written += next.messages_written;
        self.readings_copied += next.readings_copied;
        self.bytes_copied += next.bytes_copied;
        self.sessions_derived += next.sessions_derived;
        self.derivation_pending = next.derivation_pending;
        self.seconds += next.seconds;
        for root in next.unreadable_roots {
            if !self.unreadable_roots.contains(&root) {
                self.unreadable_roots.push(root);
            }
        }
        self.errors.extend(next.errors);
        self.capture_pending = next.capture_pending;
    }
}

/// How long the loop runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// Until it is told to stop: the daemon's service.
    UntilStopped,
    /// Until a pass ends with no work pending and nothing queued
    /// (`daemon --until-idle`).
    UntilIdle,
}

/// What a loop did from its start to its end, for `daemon --until-idle`.
#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RunSummary {
    /// Passes the loop ran.
    pub passes: u64,
    /// Imports it took while running.
    pub imports: u64,
    /// The passes' work added together; `seconds` is the time spent in passes.
    pub work: Pass,
}

/// What the running loop reports, for `status`. A client that waits for a
/// change to be read waits until the change shows and the loop is not busy:
/// rows become visible to readers before the pass that wrote them has
/// derived from them.
#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct LoopStatus {
    pub full_scans: u64,
    /// A pass or import is running, or a capture continues
    /// in the next pass.
    pub busy: bool,
    /// Passes finished since the loop started.
    pub passes: u64,
    pub last_pass: Pass,
    /// When `last_pass` ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<polylogue_model::Timestamp>,
}

/// The running loop's [`LoopStatus`] and a count of the changes to what
/// `status` reports, so a client waits for the next change instead of
/// polling. The loop counts its own changes; the daemon counts the others
/// (a status snapshot read after a commit) with [`Self::changed`].
#[derive(Debug, Default)]
pub struct SharedLoopStatus {
    state: Mutex<(LoopStatus, u64)>,
    signal: std::sync::Condvar,
}

impl SharedLoopStatus {
    pub fn new(status: LoopStatus) -> Self {
        Self {
            state: Mutex::new((status, 0)),
            signal: std::sync::Condvar::new(),
        }
    }

    pub fn get(&self) -> LoopStatus {
        self.state.locked().0.clone()
    }

    /// Changes the loop's status and counts the change.
    pub fn update(&self, change: impl FnOnce(&mut LoopStatus)) {
        let mut state = self.state.locked();
        change(&mut state.0);
        state.1 += 1;
        self.signal.notify_all();
    }

    /// Counts a change to something else `status` reports.
    pub fn changed(&self) {
        self.state.locked().1 += 1;
        self.signal.notify_all();
    }

    /// The changes counted since the loop started.
    pub fn changes(&self) -> u64 {
        self.state.locked().1
    }

    /// Waits until the count of changes passes `seen` or `until` comes, and
    /// returns the count then.
    pub fn wait_past(&self, seen: u64, until: Instant) -> u64 {
        let mut state = self.state.locked();
        while state.1 <= seen {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            state = polylogue_model::unpoisoned(self.signal.wait_timeout(state, left)).0;
        }
        state.1
    }

    /// Waits until `ready` holds for the loop's status or `until` comes;
    /// the status that held, or `None` when the time ran out.
    pub fn wait_for(
        &self,
        until: Instant,
        mut ready: impl FnMut(&LoopStatus) -> bool,
    ) -> Option<LoopStatus> {
        let mut state = self.state.locked();
        loop {
            if ready(&state.0) {
                return Some(state.0.clone());
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            state = polylogue_model::unpoisoned(self.signal.wait_timeout(state, left)).0;
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct ImportOptions {
    pub origin: Option<String>,
    /// Recognize and classify every member, storing nothing.
    pub dry_run: bool,
    /// Computed by the accepting channel; imported bytes never supply it.
    #[serde(skip)]
    pub authorship: Option<polylogue_model::authorship::Authorship>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ImportRefusal {
    pub path: PathBuf,
    pub reason: String,
    pub structures_tried: Vec<String>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ImportClassification {
    pub path: PathBuf,
    pub origin: Option<String>,
    pub class: String,
    pub reason: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ImportPackage {
    pub path: PathBuf,
    pub sessions: u64,
    pub documents: u64,
    pub assets: u64,
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ImportReport {
    pub packages: Vec<ImportPackage>,
    pub refused: Vec<ImportRefusal>,
    pub members: Vec<ImportClassification>,
    pub imported: Vec<PathBuf>,
    /// Files whose bytes are already in the archive.
    pub already_stored: Vec<PathBuf>,
    /// The user state Polylogue exports carried.
    pub user_state: UserStateReport,
    pub pass: Pass,
    /// What became of each vector of a Python Polylogue vector store.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_import: Option<PythonImportReport>,
    /// When what the import stored can be read.
    #[serde(default)]
    pub readable: Readable,
}

/// When an import's inputs can be read. An import parses and derives what
/// it stored before it answers, so its rows are readable at once, except
/// while an index is built: the build reads them, and they are readable
/// once it is installed.
#[derive(
    Debug, Default, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(tag = "when", rename_all = "snake_case")]
pub enum Readable {
    #[default]
    Now,
    AfterBuild {
        build: polylogue_store::archive::index_build::IndexBuild,
    },
}

/// Historical authored records restored through the recorded import ingress.
#[derive(
    Debug, Default, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct UserStateReport {
    /// Accepted and saved, awaiting the confirmed stream.
    pub pending: u64,
    /// Written into this archive.
    pub applied: u64,
    /// Already here, from an earlier import or the archive's own history.
    pub present: u64,
}

/// Runtime import failures retain typed store errors for the accepting
/// transport's shared error classifier, including changes of access scope.
#[derive(Debug)]
pub enum ImportFailure {
    Parent(polylogue_store::StoreError),
    Import(EngineError),
}

/// Messages the loop thread waits on.
pub enum LoopMsg {
    /// Something changed under a watched location.
    Changed,
    Import {
        paths: Vec<PathBuf>,
        /// The declared structure the paths are read as, or
        /// `python-vectors` for a Python Polylogue vector store.
        format: Option<String>,
        dry_run: bool,
        authorship: polylogue_model::authorship::Authorship,
        parent: Option<Arc<polylogue_store::writer::ParentLifetime>>,
        reply: SyncSender<Result<ImportReport, ImportFailure>>,
    },
    /// Ends the loop at its next turn: a pass in progress yields as it does
    /// for an import, then [`Engine::run`] returns. The archive's stop
    /// request sends it to wake a loop waiting between passes; that request
    /// also ends the pass's reads and parsing at once.
    Stop,
}

struct Requests {
    receiver: crossbeam_channel::Receiver<LoopMsg>,
    pending: Option<LoopMsg>,
    yielding: Arc<std::sync::atomic::AtomicBool>,
    serving: bool,
}

pub struct Engine {
    requests: Option<Requests>,
    /// Imports answered by a build while it ran, for the loop's summary.
    imports_during_build: u64,
    /// The paths an import request stored new bytes for and their sizes,
    /// kept across its retry after a build installed under it.
    import_stored: Vec<(PathBuf, u64)>,
    /// What is failing, logged when it starts and when it clears.
    failure_log: failure_log::FailureLog,
    /// The files the current pass planned to read.
    planned: HashSet<PathBuf>,
    collectors: collectors::Collectors,
    pub config: Config,
    pub archive: Archive,
    writer: Writer,
    reads: ReadPool,
    /// The existing archive scope that committed the cached confirmed state.
    read_scope: Option<polylogue_store::ReadScopeToken>,
    blobs: Arc<polylogue_store::blobs::Blobs>,
    inputs: HashMap<PathBuf, InputState>,
    /// Rejected writes awaiting checkpoint reload after their plans finish.
    /// Paths excluded from every nested live scan until their readings commit.
    reparsing: HashMap<PathBuf, usize>,
    /// The executable and parser settings for stopped-work selection.
    stopped_attempt: StoppedAttempt,
    watched_loaded: bool,
    readings_pending: bool,
    stored_planning: Option<pass::StoredPlanning>,
    captures: Databases,
    /// Rollout captures waiting for a parse worker, the live ones first
    /// ([`compressed::RolloutCapture`]).
    rollout_captures: std::collections::VecDeque<compressed::RolloutCapture>,
    /// The index build's current step and when it began ([`build_log`]).
    build_step: Option<(&'static str, Instant)>,
    /// What a build's storing turn found, and whether it walked the sources
    /// to find it: the turn after the build looks at these again rather than
    /// walking a second time.
    found_before_build: Option<(bool, Vec<collect::Found>)>,
    wake: Option<crossbeam_channel::Sender<LoopMsg>>,
    watcher: Option<watch::Roots>,
    databases: BTreeMap<PathBuf, &'static OriginSpec>,
    live_checked: Instant,
    pass_open: bool,
    collectors_at: Option<Instant>,
    capture_checked: Option<Instant>,
    lookups: repo_lookup::Background,
    /// How many bytes two stored chains share from their starts; each pair
    /// is compared once per pass.
    shared_prefixes: std::cell::RefCell<prefix_copies::SharedPrefixes>,
    /// When repository lookups were last refreshed; at most once a minute.
    lookups_at: Option<Instant>,
}

impl Engine {
    /// A handle for user writes (notes, deliveries, settings), which go
    /// through the same writer as the loop's batches.
    pub fn writer(&self) -> polylogue_store::WriterSender {
        self.writer.sender()
    }

    /// Commits the rows the writer holds and reports how many of each it wrote.
    fn flush_writes(&mut self) -> Result<WriteReport, EngineError> {
        Ok(self.writer.flush_rows()?)
    }

    /// Starts the writer on an archive the caller has locked.
    pub fn new(
        config: Config,
        archive: Archive,
        stopped_attempt: StoppedAttempt,
    ) -> Result<Self, EngineError> {
        let budget = config.memory_budget;
        let config = config.with_memory_budget(budget);
        let archive = archive.with_memory_budget(config.memory_budget);
        let writer = Writer::start(&archive)?;
        Self::with_writer(config, archive, writer, stopped_attempt)
    }

    fn with_writer(
        config: Config,
        archive: Archive,
        writer: Writer,
        stopped_attempt: StoppedAttempt,
    ) -> Result<Self, EngineError> {
        let reader_memory = config.reader_memory_per_connection()?;
        let reads = ReadPool::with_budget(&archive, 1, reader_memory)?;
        let blobs = Arc::new(archive.blobs()?);
        Ok(Self {
            config,
            archive,
            writer,
            reads,
            read_scope: None,
            blobs,
            inputs: HashMap::new(),
            reparsing: HashMap::new(),
            stopped_attempt,
            watched_loaded: false,
            readings_pending: false,
            stored_planning: None,
            captures: Databases::default(),
            rollout_captures: Default::default(),
            build_step: None,
            found_before_build: None,
            wake: None,
            watcher: None,
            databases: BTreeMap::new(),
            live_checked: Instant::now(),
            pass_open: false,
            collectors_at: None,
            capture_checked: None,
            lookups: repo_lookup::Background::default(),
            shared_prefixes: Default::default(),
            lookups_at: None,
            collectors: collectors::Collectors::default(),
            requests: None,
            imports_during_build: 0,
            import_stored: Vec::new(),
            failure_log: Default::default(),
            planned: HashSet::new(),
        })
    }

    /// Derives changed sessions.
    fn derive(&self) -> Result<u64, EngineError> {
        match self.writer.derive() {
            Err(polylogue_store::StoreError::WriterGone) if self.archive.stop().is_cancelled() => {
                Err(EngineError::Stopped)
            }
            result => Ok(result?),
        }
    }

    fn location_id(&self, path: &Path) -> String {
        location_id(&self.config.machine, path)
    }

    /// Cached confirmed state keeps its original token until invalidation.
    /// A reset between the caller's entry check and this call must not bind
    /// old state to a newly obtained token.
    fn cached_scope(&mut self) -> Result<polylogue_store::ReadScopeToken, EngineError> {
        let scope = match &self.read_scope {
            Some(scope) => scope.clone(),
            None => self.archive.scope_token()?,
        };
        scope.check()?;
        if self.read_scope.is_none() {
            self.read_scope = Some(scope.clone());
        }
        Ok(scope)
    }

    fn load(&mut self) -> Result<(), EngineError> {
        if self
            .read_scope
            .as_ref()
            .is_some_and(|scope| scope.check().is_err())
        {
            self.discard_confirmed_state();
        }
        let scope = self.cached_scope()?;
        // Kept replay plans from stored chains, not machine locations.
        // Leave discovery unloaded so enabling sources can still restore it.
        if !self.config.reads_sources || self.watched_loaded {
            return Ok(());
        }
        let reader = self.reads.reader()?;

        let known: Vec<KnownInput> = reader.known_watched_inputs(&self.config.machine)?;
        drop(reader);
        for mut k in known {
            let path_bytes = std::mem::take(&mut k.path_bytes);
            use std::os::unix::ffi::OsStringExt;
            let path = PathBuf::from(std::ffi::OsString::from_vec(path_bytes));
            self.adopt_watched_input(&path, k);
        }
        scope.check()?;
        self.watched_loaded = true;
        Ok(())
    }

    /// A watched location resolves identity before adopting its chain progress.
    /// Database refresh uses its selected input ID instead of resolving the path.
    fn adopt_watched_input(&mut self, path: &Path, known: KnownInput) {
        let Some(origin) = ORIGINS.iter().find(|origin| origin.name == known.format) else {
            return;
        };
        let progress = known
            .chain
            .map(|chain| polylogue_store::read::ChainProgress {
                chain,
                imported: origin.route.is_imported(),
                length: known.stored,
                parsed_upto: known.parsed_upto,
                parsed: known.parsed,
                waiting: known.waiting,
            });
        self.inputs
            .entry(path.to_path_buf())
            .or_insert_with(|| InputState::new(known.input_id.clone(), origin))
            .adopt_confirmed(&known.input_id, origin, known.facts, progress.as_ref());
    }

    fn observation(&self, path: &Path, facts: Option<FileFacts>) -> Observation {
        Observation {
            machine: self.config.machine.clone(),
            host: self.config.host.clone(),
            path: path.display().to_string(),
            path_bytes: path.as_os_str().as_encoded_bytes().to_vec(),
            facts,
            seen_at_ms: self.config.clock.now().as_millisecond(),
            import: None,
            container: None,
            database: None,
        }
    }
}

/// A location's input ID ([`polylogue_sources::InputIdentity::Location`]):
/// the first 32 hex digits of `blake3(machine \0 real path bytes)`, stable
/// for one file on one machine. The real path has its directory's symlinks
/// resolved, so a file reached through a linked directory (`~/.claude`
/// linked to `$CLAUDE_CONFIG_DIR`) is one input; the file itself need not
/// exist, so a deleted file keeps its ID.
pub(crate) fn location_id(machine: &str, path: &Path) -> String {
    let real = match (path.parent(), path.file_name()) {
        (Some(directory), Some(name)) => std::fs::canonicalize(directory)
            .map(|directory| directory.join(name))
            .unwrap_or_else(|_| path.to_owned()),
        _ => path.to_owned(),
    };
    let mut h = blake3::Hasher::new();
    h.update(machine.as_bytes());
    h.update(b"\0");
    h.update(real.as_os_str().as_encoded_bytes());
    h.finalize().to_hex()[..32].to_owned()
}

#[cfg(test)]
fn test_config() -> Config {
    Config::new(
        PathBuf::from("/synthetic/home"),
        BTreeMap::new(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        2,
        polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
    )
    .unwrap()
}
