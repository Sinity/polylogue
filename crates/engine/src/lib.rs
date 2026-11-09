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
//! Parser state lives in memory. After a restart it is rebuilt lazily: an
//! input's state is rebuilt by parsing its bytes again only when that input
//! next changes, never all at start-up.

pub mod collect;
mod compressed;
mod local_files;
pub mod pool;
pub mod read;
pub mod repo_lookup;
mod watch;

use databases::Databases;
use plan::{InputState, Plan};
use polylogue_model::batch::StoppedAttempt;
use polylogue_sources::{ORIGINS, OriginSpec, Support};
use polylogue_store::{
    Archive, Change, FileFacts, InputKey, KnownInput, Observation, ReadPool, WriteReport, Writer,
};
use pool::{Pool, Priority};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod acquisition;
mod collectors;
mod databases;
mod import;
pub use import::older_archive::{ClassCount, OlderArchiveReport, RefusedRow};
pub use import::predecessor::PredecessorReport;
mod parser_checkpoint;
mod pass;
mod plan;
mod reparse;
mod writes;

/// An origin has a reading only when its declared parser or generic row-dump
/// reader supplies one. Raw provider bytes have no parser version to invent.
fn has_reading(origin: &OriginSpec) -> bool {
    origin.support == Support::Parsed
        || (origin.support == Support::NoParser
            && origin.capture == polylogue_sources::Capture::RowDump)
}

/// One native registry projection is shared by reparse planning and read currency.
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
                    version: origin.reading_version(),
                }
            }
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
    #[error("parser: {0}")]
    Parse(String),
    #[error(transparent)]
    Blobs(#[from] hashed_blobs::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Watch(#[from] notify::Error),
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
    /// Parse workers; defaults to the number of cores.
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
    /// The range one job reads at most (default 64 MiB), so a large file
    /// is stored and parsed in steps; an export's items commit in batches
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
    /// One reader slot is kept for the loop; client reads have `workers`
    /// slots. All slots share one page-cache target, split again between index,
    /// archive and optional embeddings by the store. Integer division leaves unused bytes.
    pub fn reader_memory_per_connection(&self) -> polylogue_store::Result<u64> {
        if self.workers == 0 {
            return Err(polylogue_store::StoreError::Invalid(
                "workers must be positive".into(),
            ));
        }
        let slots = u64::try_from(self.workers)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| {
                polylogue_store::StoreError::Invalid("reader slot count overflow".into())
            })?;
        let bytes = self.reader_memory_bytes / slots;
        if bytes < 3 * 1024 {
            return Err(polylogue_store::StoreError::Invalid(
                "reader memory budget needs at least 3 KiB per connection".into(),
            ));
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
            return Err(EngineError::Parse(
                "configured locations must be absolute".into(),
            ));
        }
        if workers == 0 || machine.trim().is_empty() || host.trim().is_empty() {
            return Err(EngineError::Parse(
                "resource and host identity inputs must be nonempty".into(),
            ));
        }
        Ok(Self {
            clock,
            home,
            tool_dirs,
            workers,
            reader_memory_bytes: 128 << 20,
            bytes_in_flight: 512 << 20,
            machine,
            host,
            chunk: 64 << 20,
            live_every: Duration::from_secs(2),
            reads_sources: true,
        })
    }
}

/// What one pass of the loop found and did, for `status`.
#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct Pass {
    /// Historical effects accepted during ordinary stored-input replay.
    pub authored: UserStateReport,
    /// Whole-location scans, excluding changed paths and live-file checks.
    pub full_scans: u64,
    /// Input visits, summed across turns by run_pass.
    pub inputs_seen: u64,
    pub inputs_changed: u64,
    /// Bytes read from input files (new bytes plus any parsed again).
    pub bytes_read: u64,
    /// New bytes stored.
    pub bytes_stored: u64,
    pub segment_syncs: u64,
    pub segment_bytes_written: u64,
    pub blobs_appended: u64,
    pub blobs_deduplicated: u64,
    pub messages_written: u64,
    /// Sessions whose derived products the pass recomputed.
    pub sessions_derived: u64,
    pub seconds: f64,
    pub unreadable_roots: Vec<PathBuf>,
    pub errors: Vec<String>,
    /// Capture work is queued or waiting for primary-mode confirmation.
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
        self.sessions_derived += next.sessions_derived;
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
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize)]
pub struct ImportOptions {
    pub format: Option<String>,
    pub explain: bool,
    /// An archive in the older format to import, in place of paths.
    pub from_archive: Option<PathBuf>,
    /// With `from_archive`: read and check it, write nothing.
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
    /// Files whose exact bytes are already in the archive.
    pub already_stored: Vec<PathBuf>,
    /// The user state Polylogue exports carried.
    pub user_state: UserStateReport,
    pub pass: Pass,
    /// What became of each row of an archive in the older format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub older_archive: Option<OlderArchiveReport>,
    /// What became of each row of one of the Python predecessor's stores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predecessor: Option<PredecessorReport>,
}

/// Historical authored records restored through the frozen import ingress.
#[derive(
    Debug, Default, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct UserStateReport {
    /// Durably accepted and awaiting the confirmed primary stream.
    pub pending: u64,
    /// Written into this archive.
    pub applied: u64,
    /// Already here, from an earlier import or the archive's own history.
    pub present: u64,
}

/// Runtime import failures retain parent cancellation and deadline errors for
/// the accepting transport's shared error classifier.
#[derive(Debug)]
pub enum ImportFailure {
    Parent(polylogue_store::StoreError),
    Import(String),
}

/// Messages the loop thread waits on.
pub enum LoopMsg {
    /// Something changed under a watched location.
    Changed,
    Import {
        paths: Vec<PathBuf>,
        /// The declared structure the paths are read as, or the predecessor
        /// store they are.
        format: Option<String>,
        /// An archive in the older format to import, in place of paths.
        from_archive: Option<PathBuf>,
        dry_run: bool,
        authorship: polylogue_model::authorship::Authorship,
        parent: Option<Arc<polylogue_store::writer::ParentLifetime>>,
        reply: SyncSender<Result<ImportReport, ImportFailure>>,
    },

    Stop,
}

pub struct Engine {
    collectors: collectors::Collectors,
    pub config: Config,
    pub archive: Archive,
    writer: Writer,
    reads: ReadPool,
    /// The existing archive scope that admitted the cached confirmed state.
    read_scope: Option<polylogue_store::ReadScopeToken>,
    blobs: Arc<polylogue_store::blobs::Blobs>,
    inputs: HashMap<PathBuf, InputState>,
    /// Rejected writes awaiting checkpoint reload after their plans finish.
    /// Paths excluded from every nested live scan until their readings commit.
    reparsing: HashMap<PathBuf, usize>,
    /// Actual executable and parser settings for stopped-work selection.
    stopped_attempt: StoppedAttempt,
    loaded: bool,
    captures: Databases,
    wake: Option<Sender<LoopMsg>>,
    watcher: Option<watch::Roots>,
    databases: BTreeMap<PathBuf, &'static str>,
    live_checked: Instant,
    pass_open: bool,
    collectors_at: Option<Instant>,
    capture_checked: Option<Instant>,
    lookups: repo_lookup::Cache,
    /// When repository lookups were last refreshed; at most once a minute.
    lookups_at: Option<Instant>,
}

impl Engine {
    /// A handle for user writes (notes, deliveries, settings), which go
    /// through the same writer as the loop's batches.
    pub fn writer(&self) -> Result<polylogue_store::WriterSender, EngineError> {
        Ok(self.writer.sender()?)
    }

    /// Starts the writer on an archive the caller has locked.
    pub fn new(
        config: Config,
        archive: Archive,
        stopped_attempt: StoppedAttempt,
    ) -> Result<Self, EngineError> {
        let writer = Writer::start(&archive)?;
        Self::with_writer(config, archive, writer, stopped_attempt)
    }

    /// Starts the writer on an archive Sinex holds (primary mode), opened
    /// with [`Archive::open_primary`].
    pub fn on_sinex(
        config: Config,
        archive: Archive,
        sinex: Arc<dyn polylogue_store::primary::Sinex>,
        stopped_attempt: StoppedAttempt,
    ) -> Result<Self, EngineError> {
        let writer = Writer::start_on_sinex(&archive, sinex)?;
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
            loaded: false,
            captures: Databases::default(),
            wake: None,
            watcher: None,
            databases: BTreeMap::new(),
            live_checked: Instant::now(),
            pass_open: false,
            collectors_at: None,
            capture_checked: None,
            lookups: repo_lookup::Cache::default(),
            lookups_at: None,
            collectors: collectors::Collectors::default(),
        })
    }

    /// Derives changed sessions. Primary facts are emitted with their source reading.
    fn derive(&self) -> Result<u64, EngineError> {
        Ok(self.writer.derive()?)
    }

    /// A location's input id: stable for one path on one machine.
    fn location_id(&self, path: &Path) -> String {
        let mut h = blake3::Hasher::new();
        h.update(self.config.machine.as_bytes());
        h.update(b"\0");
        h.update(path.as_os_str().as_encoded_bytes());
        h.finalize().to_hex()[..32].to_owned()
    }

    /// Cached confirmed state keeps its original token until invalidation.
    /// A reset between the caller's entry check and this call must not bind
    /// old state to a newly acquired token.
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
        if self.loaded {
            return Ok(());
        }
        let reader = self.reads.reader()?;

        let known: Vec<KnownInput> = reader.known_watched_inputs(&self.config.machine)?;
        drop(reader);
        for mut k in known {
            let Some(path_bytes) = k.path_bytes.take() else {
                // Historical display text cannot identify a native location.
                continue;
            };
            use std::os::unix::ffi::OsStringExt;
            let path = PathBuf::from(std::ffi::OsString::from_vec(path_bytes));
            self.adopt_watched_input(&path, k);
        }
        scope.check()?;
        self.loaded = true;
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
                arrival_rank: origin.arrival.rank(),
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
        }
    }
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
