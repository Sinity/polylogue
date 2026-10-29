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
use polylogue_sources::{Capture, ORIGINS, OriginSpec, Support};
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

mod collectors;
mod databases;
mod import;
mod pass;
mod plan;
mod reparse;
mod writes;

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
    /// is stored and parsed in steps. Not a limit: a record longer than
    /// this doubles the range for its input until it fits.
    pub chunk: u64,
    /// The running loop scans every watched location at least this often,
    /// even when no change is reported (four hours).
    pub rescan_every: Duration,
    /// While a pass runs, its spools and recently written files are looked
    /// at this often, and what changed there goes ahead of the pass's
    /// remaining work (two seconds).
    pub live_every: Duration,
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

    /// The running machine's settings. `POLYLOGUE_HOME` replaces the home
    /// directory for tests and measurements.
    pub fn from_env() -> Self {
        let read = |p: &str| {
            std::fs::read_to_string(p)
                .map(|s| s.trim().to_owned())
                .unwrap_or_else(|_| "unknown".into())
        };
        // With `POLYLOGUE_HOME` (tests, measurements) the tools' own
        // variables belong to the real home and are not read.
        let isolated = std::env::var_os("POLYLOGUE_HOME").is_some();
        let tool_dirs = polylogue_sources::ORIGINS
            .iter()
            .filter_map(|o| o.root.env)
            .filter(|_| !isolated)
            .filter_map(|var| {
                let dir = std::env::var_os(var).filter(|d| !d.is_empty())?;
                Some((var.to_owned(), PathBuf::from(dir)))
            })
            .collect();
        Self {
            home: std::env::var_os("POLYLOGUE_HOME")
                .or_else(|| std::env::var_os("HOME"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/")),
            tool_dirs,
            workers: std::thread::available_parallelism().map_or(4, |n| n.get()),
            reader_memory_bytes: 128 << 20,
            bytes_in_flight: 512 << 20,
            machine: read("/etc/machine-id"),
            host: read("/proc/sys/kernel/hostname"),
            chunk: 64 << 20,
            rescan_every: Duration::from_secs(4 * 3600),
            live_every: Duration::from_secs(2),
        }
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
    /// A database reader has queued or pending work.
    pub capture_pending: bool,
}

/// What the running loop reports, for `status`. A client that waits for a
/// change to be read waits until the change shows and the loop is not busy:
/// rows become visible to readers before the pass that wrote them has
/// derived from them.
#[derive(Debug, Default, Clone, Serialize)]
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
    /// Computed by the accepting channel; imported bytes never supply it.
    #[serde(skip)]
    pub authorship: Option<polylogue_model::authorship::Authorship>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportRefusal {
    pub path: PathBuf,
    pub reason: String,
    pub structures_tried: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportClassification {
    pub path: PathBuf,
    pub origin: Option<String>,
    pub class: String,
    pub reason: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct ImportPackage {
    pub path: PathBuf,
    pub sessions: u64,
    pub documents: u64,
    pub assets: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
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
}

/// Authored requests replayed from canonical bundles.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct UserStateReport {
    /// Durably accepted and awaiting the confirmed primary stream.
    pub pending: u64,
    /// Written into this archive.
    pub applied: u64,
    /// Already here, from an earlier import or the archive's own history.
    pub present: u64,
}

/// Messages the loop thread waits on.
pub enum LoopMsg {
    /// Something changed under a watched location.
    Changed,
    Import {
        paths: Vec<PathBuf>,
        authorship: polylogue_model::authorship::Authorship,
        reply: SyncSender<Result<ImportReport, String>>,
    },

    Stop,
}

pub struct Engine {
    collectors: collectors::Collectors,
    pub config: Config,
    pub archive: Archive,
    writer: Writer,
    reads: ReadPool,
    blobs: Arc<polylogue_store::blobs::Blobs>,
    inputs: HashMap<PathBuf, InputState>,
    /// Rejected writes awaiting checkpoint reload after their plans finish.
    /// Paths excluded from every nested live scan until their readings commit.
    reparsing: HashMap<PathBuf, usize>,
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

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

impl Engine {
    /// A handle for user writes (notes, deliveries, settings), which go
    /// through the same writer as the loop's batches.
    pub fn writer(&self) -> Result<polylogue_store::WriterSender, EngineError> {
        Ok(self.writer.sender()?)
    }

    /// Starts the writer on an archive the caller has locked.
    pub fn new(config: Config, archive: Archive) -> Result<Self, EngineError> {
        let writer = Writer::start(&archive)?;
        Self::with_writer(config, archive, writer)
    }

    /// Starts the writer on an archive Sinex holds (primary mode), opened
    /// with [`Archive::open_primary`].
    pub fn on_sinex(
        config: Config,
        archive: Archive,
        sinex: Arc<dyn polylogue_store::primary::Sinex>,
    ) -> Result<Self, EngineError> {
        let writer = Writer::start_on_sinex(&archive, sinex)?;
        Self::with_writer(config, archive, writer)
    }

    fn with_writer(config: Config, archive: Archive, writer: Writer) -> Result<Self, EngineError> {
        let reader_memory = config.reader_memory_per_connection()?;
        let reads = ReadPool::with_budget(&archive, 1, reader_memory)?;
        let blobs = Arc::new(archive.blobs()?);
        Ok(Self {
            config,
            archive,
            writer,
            reads,
            blobs,
            inputs: HashMap::new(),
            reparsing: HashMap::new(),
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

    fn load(&mut self) -> Result<(), EngineError> {
        if self.loaded {
            return Ok(());
        }
        let reader = self.reads.reader()?;

        let mut known: Vec<KnownInput> = reader.known_watched_inputs(&self.config.machine)?;
        let mut database_paths: HashMap<String, String> = self
            .inputs
            .iter()
            .filter(|(path, state)| state.origin.is_row_dump(path))
            .map(|(path, state)| (state.input_id.clone(), path.display().to_string()))
            .collect();
        for input in &mut known {
            if ORIGINS
                .iter()
                .any(|origin| origin.name == input.format && origin.capture == Capture::RowDump)
            {
                if let Some(path) = database_paths.get(&input.input_id) {
                    input.path = path.clone();
                    input.facts = None;
                    continue;
                }
                if let Some((machine, path)) =
                    reader.tool_database_source(&input.input_id, &self.blobs)?
                    && machine == self.config.machine
                {
                    database_paths.insert(input.input_id.clone(), path.clone());
                    input.path = path;
                    input.facts = None;
                }
            }
        }
        drop(reader);
        self.inputs
            .retain(|path, state| !state.origin.is_row_dump(path));
        for k in known {
            let Some(origin) = ORIGINS.iter().find(|o| o.name == k.format) else {
                continue;
            };
            if !origin.is_row_dump(Path::new(&k.path))
                && self.inputs.contains_key(Path::new(&k.path))
            {
                continue;
            }
            self.inputs.insert(
                PathBuf::from(&k.path),
                InputState {
                    input_id: k.input_id,
                    origin,
                    facts: k.facts,
                    stored: k.stored,
                    parsed_upto: k.parsed_upto,
                    parsed: k.parsed,
                    waiting: k.waiting,
                    parser_state: None,
                    tail: None,
                    chain: k.chain,
                    failed: None,
                    failed_history: HashMap::new(),
                },
            );
        }
        self.loaded = true;
        Ok(())
    }

    fn observation(&self, path: &Path, facts: Option<FileFacts>) -> Observation {
        Observation {
            machine: self.config.machine.clone(),
            host: self.config.host.clone(),
            path: path.display().to_string(),
            facts,
            seen_at_ms: now_ms(),
            import: None,
            container: None,
        }
    }
}
