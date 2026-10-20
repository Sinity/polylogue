//! The convergence loop (architecture 5). One loop runs in the daemon:
//!
//! ```text
//! collect -> store -> parse -> derive -> excise -> publish (Sinex primary) -> wait
//! ```
//!
//! Each step finds its work by comparing what exists with what should
//! exist. There is no to-do list and no repair path; `status` calls the
//! same comparison.
//!
//! Parser state lives in memory. After a restart it is rebuilt lazily: an
//! input's state is rebuilt by parsing its bytes again only when that input
//! next changes, never all at start-up (architecture 2, measured facts).

pub mod ai_studio_drive;
pub mod collect;
pub mod pool;
pub mod read;
pub mod repo_lookup;

use polylogue_model::Json;
use polylogue_sources::{Capture, ORIGINS, OriginSpec, Support};
pub use polylogue_store::excise::{ExciseRequest, Excised, Scope as ExciseScope};
use polylogue_store::tool_databases::Exporter;
use polylogue_store::{
    Archive, Change, FileFacts, InputKey, KnownInput, Observation, ReadPool, WriteReport, Writer,
};
use pool::{Done, Job, Pool, Priority};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod import;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
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
    /// Combined SQLite page-cache target for client and convergence readers.
    /// Mapped reader pages are disabled. This is not a total RSS limit.
    pub reader_memory_bytes: u64,
    /// Bytes admitted into the parse pool at once. The pool takes new work
    /// only while the bytes in flight stay under this.
    pub bytes_in_flight: u64,
    /// `/etc/machine-id`: paths from two machines never collide.
    pub machine: String,
    pub host: String,
    /// The range one job reads at most (default 64 MiB), so a large file
    /// is stored and parsed in steps. Not a limit: a record longer than
    /// this doubles the range for its input until it fits.
    pub chunk: u64,
    /// Further directories stored blobs are read from: in `primary` mode,
    /// Sinex's content store, which holds the archive's bytes.
    pub blob_roots: Vec<PathBuf>,
}

impl Config {
    /// One reader slot is reserved for convergence; client reads have `workers`
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
            blob_roots: Vec::new(),
        }
    }
}

/// What one pass of the loop found and did, for `status`.
#[derive(Debug, Default, Clone, Serialize)]
pub struct Pass {
    pub inputs_seen: u64,
    pub inputs_changed: u64,
    /// Bytes read from input files (new bytes plus any parsed again).
    pub bytes_read: u64,
    /// New bytes stored.
    pub bytes_stored: u64,
    pub messages_written: u64,
    /// Sessions whose derived products the pass recomputed.
    pub sessions_derived: u64,
    pub seconds: f64,
    pub unreadable_roots: Vec<PathBuf>,
    pub errors: Vec<String>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct ImportReport {
    pub imported: Vec<PathBuf>,
    /// Files whose exact bytes are already in the archive.
    pub already_stored: Vec<PathBuf>,
    /// Files no parsed origin recognizes.
    pub unrecognized: Vec<PathBuf>,
    pub pass: Pass,
}

/// Messages the loop thread waits on.
pub enum LoopMsg {
    /// Something changed under a watched location.
    Changed,
    Import {
        paths: Vec<PathBuf>,
        reply: SyncSender<Result<ImportReport, String>>,
    },
    Excise {
        request: ExciseRequest,
        reply: SyncSender<Result<Excised, String>>,
    },
    Stop,
}

/// Parser state stays in memory for inputs written within this long.
const KEEP_STATE_NS: i128 = 6 * 3600 * 1_000_000_000;

/// The longest a burst of file events delays the next pass.
const COALESCE: Duration = Duration::from_secs(2);

/// What the loop knows about one input between passes. Parser state
/// (`carry`) is `None` until the input is next parsed after a restart.
struct InputState {
    input_id: String,
    origin: &'static OriginSpec,
    facts: Option<FileFacts>,
    stored: u64,
    parsed_upto: u64,
    parser_version: Option<u32>,
    /// A successful parse reached source EOF with an incomplete last record.
    waiting: bool,
    carry: Option<Json>,
    tail: Option<Vec<u8>>,
    /// The revision chain whose bytes are stored, when known from the
    /// archive; `None` once this run stored a new chain.
    chain: Option<i64>,
    /// The file facts and parser version a parse failed at: the input is
    /// tried again only when its bytes or its parser change (law 63).
    failed: Option<(FileFacts, u32)>,
}

/// One input's work in this pass, advanced job by job.
struct Plan {
    path: PathBuf,
    origin: &'static OriginSpec,
    input: InputKey,
    change: Change,
    observation: Option<Observation>,
    facts: Option<FileFacts>,
    size: u64,
    piece_pos: u64,
    parse_pos: u64,
    piece_started: bool,
    carry: Option<Json>,
    tail: Option<Vec<u8>>,
    span: u64,
    chunk: u64,
    priority: Priority,
    /// A location the loop keeps watching (not a one-off import).
    tracked: bool,
    failed: bool,
    /// The bytes after `parse_pos` are an incomplete last record.
    waiting: bool,
    /// Where the input's bytes come from.
    source: pool::Bytes,
}

impl Plan {
    fn finished(&self) -> bool {
        self.failed
            || (self.piece_pos >= self.size && (self.waiting || self.parse_pos >= self.size))
    }

    fn next_job(&mut self) -> Job {
        let start = self.parse_pos;
        let end = self.size.min(start + self.span);
        let piece_from = (end > self.piece_pos).then_some(self.piece_pos);
        let (change, observation, continues, length) = match piece_from {
            Some(_) if !self.piece_started => {
                (self.change.clone(), self.observation.clone(), false, end)
            }
            Some(_) => (self.change.clone(), None, true, end),
            None => (Change::Unchanged, None, false, self.piece_pos),
        };
        Job {
            path: self.path.clone(),
            origin: self.origin,
            input: self.input.clone(),
            change,
            observation,
            continues,
            start,
            piece_from,
            end,
            at_end: end == self.size,
            length,
            carry: self.carry.take(),
            priority: self.priority,
            source: self.source.clone(),
        }
    }

    fn apply(&mut self, d: Done, parsed: bool) {
        if d.error.is_some() {
            self.failed = true;
            self.carry = d.carry;
            return;
        }
        if d.end > self.piece_pos {
            self.piece_pos = d.end;
            self.piece_started = true;
        }
        self.tail = Some(d.tail);
        self.carry = d.carry;
        if !parsed {
            self.parse_pos = d.end;
        } else if d.parsed_upto > d.start {
            self.parse_pos = d.parsed_upto;
            self.span = self.chunk;
        } else if d.end < self.size {
            // One record is longer than the range read: read more of it.
            self.span = (d.end - d.start).saturating_mul(2);
        } else {
            // Only an incomplete last record remains; it is parsed when the
            // file grows, from `parse_pos`, which stays before it.
            self.waiting = true;
        }
    }
}

pub struct Engine {
    pub config: Config,
    pub archive: Archive,
    writer: Writer,
    reads: ReadPool,
    blobs: Arc<hashed_blobs::Store>,
    inputs: HashMap<PathBuf, InputState>,
    loaded: bool,
    exporter: Exporter,
    /// Paths on this machine an excision removed: never collected again.
    refused: HashSet<PathBuf>,
    lookups: repo_lookup::Cache,
    /// When repository lookups were last refreshed; at most once a minute.
    lookups_at: Option<Instant>,
}

/// Whether `origin` reads the input at `path` whole, as one document.
fn whole(origin: &OriginSpec, path: &Path) -> bool {
    (origin.open)(path).is_some_and(|p| p.is_document())
}

/// The range a plan's first job reads: a document whole, lines by chunk.
fn span(origin: &OriginSpec, path: &Path, chunk: u64) -> u64 {
    if whole(origin, path) { u64::MAX } else { chunk }
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
        let reader_memory = config.reader_memory_per_connection()?;
        let writer = Writer::start(&archive)?;
        let reads = ReadPool::with_budget(&archive, 1, reader_memory)?;
        let blobs = Arc::new(hashed_blobs::Store::open(
            archive.root.clone(),
            config.blob_roots.clone(),
            0o600,
        )?);
        Ok(Self {
            config,
            archive,
            writer,
            reads,
            blobs,
            inputs: HashMap::new(),
            loaded: false,
            exporter: Exporter::default(),
            refused: HashSet::new(),
            lookups: repo_lookup::Cache::default(),
            lookups_at: None,
        })
    }

    /// Runs an excision through the writer, stops collecting the paths it
    /// removed, and parses the inputs whose rows it cleared again from
    /// their overwritten stored bytes ([`polylogue_store::excise`]).
    pub fn excise(&mut self, request: ExciseRequest) -> Result<Excised, EngineError> {
        self.load()?;
        let done = self.writer.excise(request, Arc::clone(&self.blobs))?;
        if done.dry_run {
            return Ok(done);
        }
        let mut plans = Vec::new();
        let known = if done.inputs_reparsed.is_empty() {
            Vec::new()
        } else {
            self.reads.reader()?.known_inputs(&self.config.machine)?
        };
        for k in known {
            if !done.inputs_reparsed.contains(&k.input_id) {
                continue;
            }
            let path = PathBuf::from(&k.path);
            if let Some(st) = self.inputs.get_mut(&path) {
                st.chain = k.chain;
                st.stored = k.stored;
                st.parsed_upto = k.parsed_upto;
                st.waiting = k.waiting;
                st.carry = None;
                st.tail = None;
                st.failed = None;
            }
            if let Some(plan) = self.stored_plan(&path)? {
                plans.push(plan);
            }
        }
        // An input that fails here stays behind its stored bytes, so the
        // next pass parses it again.
        self.execute(plans, &mut Pass::default())?;
        self.writer.derive()?;
        let prefix = format!("{}:", self.config.machine);
        for p in &done.refused_paths {
            if let Some(path) = p.strip_prefix(&prefix) {
                let path = PathBuf::from(path);
                self.inputs.remove(&path);
                self.refused.insert(path);
            }
        }
        Ok(done)
    }

    /// Brings a tool database's export up to date when due; returns the
    /// export file, or `None` when there is none yet.
    fn export(
        &mut self,
        db: &Path,
        origin: &'static OriginSpec,
        pass: &mut Pass,
    ) -> Option<PathBuf> {
        let name = polylogue_sources::origin::export_name(db)?;
        let out = self.archive.exports_dir().join(origin.name).join(name);
        if let Err(e) = self.exporter.export(db, &out) {
            pass.errors.push(format!("export {}: {e}", db.display()));
        }
        out.is_file().then_some(out)
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
        for row in reader.query("SELECT refused_json AS r FROM a.excision", &[])? {
            let refused: serde_json::Value = row["r"]
                .as_str()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();
            for l in refused["locations"].as_array().into_iter().flatten() {
                if l["machine"].as_str() == Some(self.config.machine.as_str())
                    && let Some(path) = l["path"].as_str()
                {
                    self.refused.insert(PathBuf::from(path));
                }
            }
        }
        let known: Vec<KnownInput> = reader.known_inputs(&self.config.machine)?;
        drop(reader);
        for k in known {
            if self.refused.contains(Path::new(&k.path)) {
                continue;
            }
            let Some(origin) = ORIGINS.iter().find(|o| o.name == k.format) else {
                continue;
            };
            self.inputs.insert(
                PathBuf::from(&k.path),
                InputState {
                    input_id: k.input_id,
                    origin,
                    facts: k.facts,
                    stored: k.stored,
                    parsed_upto: k.parsed_upto,
                    parser_version: k.parser_version,
                    waiting: k.waiting,
                    carry: None,
                    tail: None,
                    chain: k.chain,
                    failed: None,
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
        }
    }

    /// Writes the next Drive revision of each AI Studio prompt into the
    /// spool, when the operator turned the collector on. The written files
    /// change a watched location, so another pass follows on its own.
    /// Returns the error to report on the pass, if the sync failed.
    /// Writes the repository lookups for the working directories the index
    /// holds (spec sources/README.md, the repo-lookup record), at most once
    /// a minute. The files it writes are collected in this same pass.
    fn refresh_lookups(&mut self) -> Result<(), String> {
        if self
            .lookups_at
            .is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(60))
        {
            return Ok(());
        }
        self.lookups_at = Some(Instant::now());
        let rows = self
            .reads
            .reader()
            .and_then(|r| {
                r.query(
                    "SELECT DISTINCT cwd, cwd_json FROM session WHERE cwd IS NOT NULL",
                    &[],
                )
            })
            .map_err(|e| format!("repository lookups: {e}"))?;
        let dir = self
            .config
            .spool_dir(polylogue_sources::repo_lookup::ORIGIN);
        let read_at = polylogue_model::Timestamp::from_unix_ms(now_ms())
            .map(|t| t.to_string())
            .unwrap_or_default();
        repo_lookup::refresh(
            repo_lookup::working_directories(&rows),
            &dir,
            &mut self.lookups,
            &read_at,
        )
        .map(|_| ())
        .map_err(|e| format!("repository lookups in {}: {e}", dir.display()))
    }

    fn sync_drive(&self) -> Option<String> {
        let on = self
            .reads
            .reader()
            .ok()?
            .setting(ai_studio_drive::SETTING, &self.config.machine)
            .ok()??;
        if on != "on" {
            return None;
        }
        let collector = ai_studio_drive::Collector {
            endpoints: ai_studio_drive::Endpoints::default(),
            token_path: self.config.state_dir().join("token.json"),
            spool: self.config.spool_dir(polylogue_sources::ai_studio::ORIGIN),
        };
        collector
            .sync()
            .err()
            .map(|e| format!("ai-studio collector: {e}"))
    }

    /// Runs one pass of the loop to completion.
    pub fn converge(&mut self) -> Result<Pass, EngineError> {
        let started = Instant::now();
        self.load()?;
        let drive_error = self.sync_drive();
        let lookup_error = self.refresh_lookups().err();
        let (found, unreadable) = collect::scan(&self.config);
        let mut pass = Pass {
            unreadable_roots: unreadable,
            errors: drive_error.into_iter().chain(lookup_error).collect(),
            ..Pass::default()
        };
        let mut plans = Vec::new();
        let mut seen = HashSet::new();
        for f in found {
            // A tool database is never stored as bytes: its logical export
            // is the input (architecture 4.2).
            let f = if f.origin.capture == Capture::LogicalExport {
                let Some(export) = self.export(&f.path, f.origin, &mut pass) else {
                    continue;
                };
                collect::Found {
                    path: export,
                    origin: f.origin,
                }
            } else {
                f
            };
            if self.refused.contains(&f.path) {
                continue;
            }
            seen.insert(f.path.clone());
            let Ok(meta) = std::fs::metadata(&f.path) else {
                continue;
            };
            pass.inputs_seen += 1;
            let now = FileFacts::of(&meta);
            if !self.inputs.contains_key(&f.path) {
                if !collect::recognize(&f) {
                    continue;
                }
                let id = self.location_id(&f.path);
                self.inputs.insert(
                    f.path.clone(),
                    InputState {
                        input_id: id,
                        origin: f.origin,
                        facts: None,
                        stored: 0,
                        parsed_upto: 0,
                        parser_version: None,
                        waiting: false,
                        carry: None,
                        tail: None,
                        chain: None,
                        failed: None,
                    },
                );
            }
            if let Some(plan) = self.plan(&f.path, now)? {
                plans.push(plan);
            }
        }
        // Paths seen before and gone now: an observation, never a delete.
        let gone: Vec<PathBuf> = self
            .inputs
            .iter()
            .filter(|(p, s)| s.facts.is_some() && !seen.contains(*p) && !p.exists())
            .map(|(p, _)| p.clone())
            .collect();
        for path in gone {
            let st = self.inputs.get_mut(&path).expect("known input");
            st.facts = None;
            let batch = polylogue_store::Batch {
                input: InputKey {
                    id: st.input_id.clone(),
                    format: st.origin.name.into(),
                    route_rank: 0,
                },
                change: Change::Disappeared,
                observation: Some(Observation {
                    machine: self.config.machine.clone(),
                    host: self.config.host.clone(),
                    path: path.display().to_string(),
                    facts: None,
                    seen_at_ms: now_ms(),
                }),
                continues: false,
                piece: None,
                blobs: vec![],
                length: st.stored,
                parse: None,
                release: Default::default(),
            };
            self.writer.send(batch)?;
        }
        // Inputs whose files are gone or moved are parsed from what the
        // archive stored, when the index is behind it.
        let unseen: Vec<PathBuf> = self
            .inputs
            .keys()
            .filter(|p| !seen.contains(*p))
            .cloned()
            .collect();
        for path in unseen {
            if let Some(plan) = self.stored_plan(&path)? {
                plans.push(plan);
            }
        }
        pass.inputs_changed = plans.len() as u64;
        if plans.is_empty() {
            let report = self.writer.flush()?;
            pass.errors.extend(report.errors);
        }
        self.execute(plans, &mut pass)?;
        pass.sessions_derived = self.writer.derive()?;
        pass.seconds = started.elapsed().as_secs_f64();
        Ok(pass)
    }

    /// Decides what to read for one tracked path.
    fn plan(&mut self, path: &Path, now: FileFacts) -> Result<Option<Plan>, EngineError> {
        let obs = self.observation(path, Some(now));
        let st = self.inputs.get_mut(path).expect("tracked input");
        if st.failed == Some((now, st.origin.parser_version)) {
            return Ok(None);
        }
        let parsed = st.origin.support == Support::Parsed;
        let mut change = polylogue_store::blobs::classify(st.facts, Some(now));
        if change == Change::Unchanged && st.stored < now.size {
            // Stored only in part (a stop in the middle of a large file).
            change = Change::Appended { from: st.stored };
        }
        if let Change::Appended { from } = change
            && let Some(tail) = &st.tail
            && !read::tail_intact(path, from, tail)?
        {
            change = Change::Rewritten;
        }
        let version_changed = parsed && st.parser_version != Some(st.origin.parser_version);
        let (piece_pos, parse_pos, carry, priority) = match &change {
            Change::First | Change::Rewritten => {
                st.stored = 0;
                st.parsed_upto = 0;
                st.chain = None;
                (0, 0, None, Priority::Bulk)
            }
            Change::Appended { .. } => {
                let resumable = st.carry.is_some() || st.parsed_upto == 0;
                if resumable && !version_changed {
                    (st.stored, st.parsed_upto, st.carry.take(), Priority::Live)
                } else {
                    (st.stored, 0, None, Priority::Live)
                }
            }
            Change::Unchanged | Change::Disappeared => return self.stored_plan(path),
        };
        // A document is parsed whole: any change parses it from its start.
        let (parse_pos, carry) = if whole(st.origin, path) {
            (0, None)
        } else {
            (parse_pos, carry)
        };
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin: st.origin,
            input: InputKey {
                id: st.input_id.clone(),
                format: st.origin.name.into(),
                route_rank: 0,
            },
            change,
            observation: Some(obs),
            facts: Some(now),
            size: now.size,
            piece_pos,
            parse_pos,
            piece_started: false,
            carry,
            tail: None,
            span: span(st.origin, path, self.config.chunk),
            chunk: self.config.chunk,
            priority,
            tracked: true,
            failed: false,
            waiting: false,
            source: pool::Bytes::File,
        }))
    }

    /// A parse-only plan over the bytes the archive stored for an input,
    /// when the index is behind them: a new parser version, a fresh index,
    /// or a file that moved or went. The same loop builds the index from
    /// stored bytes as from files, so a rebuild needs no command of its own.
    fn stored_plan(&mut self, path: &Path) -> Result<Option<Plan>, EngineError> {
        let Some(st) = self.inputs.get(path) else {
            return Ok(None);
        };
        let parsed = st.origin.support == Support::Parsed;
        let version_changed = st.parser_version != Some(st.origin.parser_version);
        let behind = (!st.waiting && st.parsed_upto < st.stored) || version_changed;
        let Some(chain) = st.chain else {
            return Ok(None);
        };
        if !parsed || st.stored == 0 || !behind {
            return Ok(None);
        }
        let (input_id, origin, stored, facts) =
            (st.input_id.clone(), st.origin, st.stored, st.facts);
        let bytes = self
            .reads
            .reader()?
            .stored_chain(&input_id, chain, stored)?;
        let route_rank = self
            .reads
            .reader()?
            .retained_input(&input_id)?
            .ok_or_else(|| {
                polylogue_store::StoreError::Invalid("retained revision disappeared".into())
            })?
            .route_rank;
        let source = match bytes.single_file_blob() {
            Some(hash) => pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::open(
                &self.blobs,
                hash,
                stored,
            )?))),
            None => pool::Bytes::Stored(Arc::new(bytes)),
        };
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin,
            input: InputKey {
                id: input_id,
                format: origin.name.into(),
                route_rank,
            },
            change: Change::Unchanged,
            observation: None,
            facts,
            size: stored,
            piece_pos: stored,
            parse_pos: 0,
            piece_started: false,
            carry: None,
            tail: None,
            span: span(origin, path, self.config.chunk),
            chunk: self.config.chunk,
            priority: Priority::Bulk,
            tracked: true,
            failed: false,
            waiting: false,
            source,
        }))
    }

    /// Runs plans on the parse pool, one job per input at a time, until
    /// every input is stored and parsed up to its size; then waits for the
    /// writer to commit.
    fn execute(&mut self, plans: Vec<Plan>, pass: &mut Pass) -> Result<(), EngineError> {
        if plans.is_empty() {
            return Ok(());
        }
        let (done_tx, done_rx) = channel();
        let pool = Pool::start(
            self.config.workers,
            self.config.bytes_in_flight,
            self.writer.sender()?,
            Arc::clone(&self.blobs),
            done_tx,
        );
        let mut running: HashMap<PathBuf, Plan> = HashMap::new();
        let mut outstanding = 0usize;
        let mut plans = plans;
        // Live work first; large bulk inputs first among the bulk, so the
        // longest jobs do not start last.
        plans.sort_by_key(|p| (p.priority == Priority::Bulk, std::cmp::Reverse(p.size)));
        for mut plan in plans {
            if plan.finished() {
                self.settle(plan);
                continue;
            }
            let job = plan.next_job();
            pass.bytes_read += job.end - job.start;
            pool.submit(job);
            outstanding += 1;
            running.insert(plan.path.clone(), plan);
        }
        while outstanding > 0 {
            let Ok(d) = done_rx.recv() else {
                break;
            };
            outstanding -= 1;
            let Some(mut plan) = running.remove(&d.path) else {
                continue;
            };
            pass.bytes_stored += d.stored;
            if let Some(e) = &d.error {
                pass.errors.push(e.clone());
            }
            let parsed = plan.origin.support == Support::Parsed;
            plan.apply(d, parsed);
            if plan.finished() {
                self.settle(plan);
            } else {
                let job = plan.next_job();
                pass.bytes_read += job.end - job.start;
                pool.submit(job);
                outstanding += 1;
                running.insert(plan.path.clone(), plan);
            }
        }
        drop(pool);
        let report: WriteReport = self.writer.flush()?;
        pass.messages_written += report.messages;
        pass.errors.extend(report.errors);
        Ok(())
    }

    /// Records what a finished plan leaves for the next pass. Parser state
    /// is kept in memory only for files written recently; an older file's
    /// state is rebuilt by parsing it again if it ever grows.
    fn settle(&mut self, plan: Plan) {
        if !plan.tracked {
            return;
        }
        if let Some(st) = self.inputs.get_mut(&plan.path) {
            st.stored = plan.piece_pos.max(st.stored);
            if plan.failed {
                // Tried again when the bytes or the parser change; the facts
                // stay old, so a change is still seen as one.
                st.carry = None;
                st.waiting = false;
                st.failed = plan.facts.map(|f| (f, plan.origin.parser_version));
                return;
            }
            st.failed = None;
            let recent = plan
                .facts
                .is_some_and(|f| i128::from(now_ms()) * 1_000_000 - f.mtime_ns < KEEP_STATE_NS);
            st.facts = plan.facts;
            st.parsed_upto = plan.parse_pos.min(plan.size);
            st.parser_version = Some(plan.origin.parser_version);
            st.waiting = plan.waiting;
            st.carry = if recent { plan.carry } else { None };
            st.tail = plan.tail;
        }
    }

    /// Runs passes until stopped: after any change under a watched
    /// location, and at least once a minute.
    pub fn run(
        &mut self,
        tx: Sender<LoopMsg>,
        rx: &Receiver<LoopMsg>,
        status: &Arc<Mutex<Pass>>,
        ready: impl FnOnce(),
    ) -> Result<(), EngineError> {
        use notify::Watcher as _;
        let wake = tx.clone();
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if res.as_ref().is_ok_and(collect::requires_scan) {
                    let _ = wake.send(LoopMsg::Changed);
                }
            })?;
        for root in collect::roots(&self.config) {
            watcher.watch(&root, notify::RecursiveMode::Recursive)?;
        }
        drop(tx);
        ready();
        loop {
            match self.converge() {
                Ok(pass) => {
                    if let Ok(mut s) = status.lock() {
                        *s = pass;
                    }
                }
                Err(e) => {
                    if let Ok(mut s) = status.lock() {
                        s.errors = vec![e.to_string()];
                    }
                }
            }
            let mut msg = rx.recv_timeout(Duration::from_secs(60));
            let burst = Instant::now();
            loop {
                match msg {
                    Ok(LoopMsg::Stop) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
                    Ok(LoopMsg::Import { paths, reply }) => {
                        let _ = reply.send(self.import(&paths).map_err(|e| e.to_string()));
                    }
                    Ok(LoopMsg::Excise { request, reply }) => {
                        let _ = reply.send(self.excise(request).map_err(|e| e.to_string()));
                    }
                    Ok(LoopMsg::Changed) | Err(RecvTimeoutError::Timeout) => {}
                }
                // Coalesce a burst of file events into one pass.
                if burst.elapsed() > COALESCE {
                    break;
                }
                msg = rx.recv_timeout(Duration::from_millis(200));
                if matches!(msg, Err(RecvTimeoutError::Timeout)) {
                    break;
                }
            }
        }
    }
}
