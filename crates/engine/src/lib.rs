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

pub mod collect;
pub mod derive;
pub mod pool;
pub mod publish;
pub mod stage;

use polylogue_model::Json;
use polylogue_sources::{Capture, Framing, ORIGINS, OriginSpec, Support};
use polylogue_store::export::Exporter;
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

pub struct Config {
    /// The home directory origins' locations are relative to.
    pub home: PathBuf,
    /// Tools' own location variables (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`)
    /// as the running user set them. An origin whose root names one of
    /// these reads from it instead of its default under `home`.
    pub tool_dirs: BTreeMap<String, PathBuf>,
    /// Parse workers; defaults to the number of cores.
    pub workers: usize,
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
}

impl Config {
    /// The running machine's settings. `POLYLOGUE_HOME` replaces the home
    /// directory for tests and measurements.
    /// The directory an origin's locations are relative to.
    pub fn root(&self, root: polylogue_sources::origin::Root) -> PathBuf {
        match root.env.and_then(|var| self.tool_dirs.get(var)) {
            Some(dir) if dir.is_absolute() => dir.clone(),
            Some(dir) => self.home.join(dir),
            None => self.home.join(root.default),
        }
    }

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
            bytes_in_flight: 512 << 20,
            machine: read("/etc/machine-id"),
            host: read("/proc/sys/kernel/hostname"),
            chunk: 64 << 20,
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
    carry: Option<Json>,
    tail: Option<Vec<u8>>,
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
    /// The input's bytes when it is not a file (a zip member).
    source: Option<Arc<Vec<u8>>>,
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
}

/// The bytes a zip file starts with.
const ZIP_MAGIC: &[u8] = b"PK\x03\x04";

/// Bytes of imported inputs held in memory and run through the pool at
/// once. A member larger than this runs alone.
const IMPORT_GROUP: u64 = 1 << 30;

/// The origin that reads a zip member, or the one that keeps it as bytes.
fn recognize_member(at: &Path, bytes: &[u8]) -> &'static OriginSpec {
    let head = &bytes[..bytes.len().min(64)];
    recognize_import(at, head)
        .unwrap_or_else(|| polylogue_sources::origin::by_name(KEPT).expect("declared origin"))
}

/// The origin an imported file belongs to. Several recognizers can accept
/// one file (Claude Code takes any JSON-lines transcript, since an exported
/// transcript can sit anywhere); the one whose own layout the path ends in
/// is preferred, so a Grok CLI history or a hook carrier copied with its
/// directories keeps its origin.
fn recognize_import(path: &Path, head: &[u8]) -> Option<&'static OriginSpec> {
    let mut found = ORIGINS
        .iter()
        .filter(|o| o.support == Support::Parsed && (o.recognize)(path, head));
    let first = found.next()?;
    let in_layout = |o: &OriginSpec| o.locations.iter().any(|l| collect::ends_in(path, l));
    if in_layout(first) {
        return Some(first);
    }
    Some(found.find(|o| in_layout(o)).unwrap_or(first))
}

/// The origin of export members no parser reads: kept as bytes.
const KEPT: &str = "export-file";

/// The range a plan's first job reads: a document whole, lines by chunk.
fn span(origin: &OriginSpec, chunk: u64) -> u64 {
    match origin.framing {
        Framing::Lines => chunk,
        Framing::Document => u64::MAX,
    }
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
        let reads = ReadPool::open(&archive)?;
        let blobs = Arc::new(hashed_blobs::Store::open(
            archive.root.clone(),
            vec![],
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
        })
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
        let known: Vec<KnownInput> = self.reads.reader()?.known_inputs(&self.config.machine)?;
        for k in known {
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
                    carry: None,
                    tail: None,
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

    /// Runs one pass of the loop to completion.
    pub fn converge(&mut self) -> Result<Pass, EngineError> {
        let started = Instant::now();
        self.load()?;
        let (found, unreadable) = collect::scan(&self.config);
        let mut pass = Pass {
            unreadable_roots: unreadable,
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
                        carry: None,
                        tail: None,
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
            };
            self.writer.send(batch)?;
        }
        pass.inputs_changed = plans.len() as u64;
        if plans.is_empty() {
            let report = self.writer.flush()?;
            pass.errors.extend(report.errors);
        }
        self.execute(plans, &mut pass)?;
        pass.seconds = started.elapsed().as_secs_f64();
        Ok(pass)
    }

    /// Decides what to read for one tracked path.
    fn plan(&mut self, path: &Path, now: FileFacts) -> Result<Option<Plan>, EngineError> {
        let obs = self.observation(path, Some(now));
        let st = self.inputs.get_mut(path).expect("tracked input");
        let parsed = st.origin.support == Support::Parsed;
        let mut change = polylogue_store::blobs::classify(st.facts, Some(now));
        if change == Change::Unchanged && st.stored < now.size {
            // Stored only in part (a stop in the middle of a large file).
            change = Change::Appended { from: st.stored };
        }
        if let Change::Appended { from } = change
            && let Some(tail) = &st.tail
            && !stage::tail_intact(path, from, tail)?
        {
            change = Change::Rewritten;
        }
        let version_changed = parsed && st.parser_version != Some(st.origin.parser_version);
        let (piece_pos, parse_pos, carry, priority) = match &change {
            Change::First | Change::Rewritten => {
                st.stored = 0;
                st.parsed_upto = 0;
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
            Change::Unchanged if version_changed && st.stored > 0 => {
                (st.stored, 0, None, Priority::Bulk)
            }
            Change::Unchanged | Change::Disappeared => return Ok(None),
        };
        // A document is parsed whole: any change parses it from its start.
        let (parse_pos, carry) = match st.origin.framing {
            Framing::Lines => (parse_pos, carry),
            Framing::Document => (0, None),
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
            span: span(st.origin, self.config.chunk),
            chunk: self.config.chunk,
            priority,
            tracked: true,
            failed: false,
            waiting: false,
            source: None,
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
                // Facts stay old, so the next pass looks again.
                st.carry = None;
                return;
            }
            let recent = plan
                .facts
                .is_some_and(|f| i128::from(now_ms()) * 1_000_000 - f.mtime_ns < KEEP_STATE_NS);
            st.facts = plan.facts;
            st.parsed_upto = plan.parse_pos.min(plan.size);
            st.parser_version = Some(plan.origin.parser_version);
            st.carry = if recent { plan.carry } else { None };
            st.tail = plan.tail;
        }
    }

    /// Brings in export files by explicit request (architecture 5.1). An
    /// imported file is an input named by the BLAKE3 of its bytes, so
    /// importing the same file again stores nothing. A zip the origins do
    /// not read whole is opened and each member imported on its own: the
    /// ones an origin recognizes are parsed, the rest (images, audio, the
    /// export's own HTML) are kept as bytes. Members are read one at a time
    /// and imported in groups of about [`IMPORT_GROUP`] bytes, so an export
    /// larger than memory still imports.
    pub fn import(&mut self, paths: &[PathBuf]) -> Result<ImportReport, EngineError> {
        let started = Instant::now();
        let mut report = ImportReport::default();
        let mut group = Vec::new();
        let mut group_bytes = 0u64;
        for path in paths {
            let mut head = [0u8; 64];
            let n = std::io::Read::read(&mut std::fs::File::open(path)?, &mut head)?;
            let head = &head[..n];
            let whole = recognize_import(path, head);
            if whole.is_none() && head.starts_with(ZIP_MAGIC) {
                let mut archive = zip::ZipArchive::new(std::fs::File::open(path)?)
                    .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                for i in 0..archive.len() {
                    let mut member = archive
                        .by_index(i)
                        .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                    if !member.is_file() {
                        continue;
                    }
                    let name = member.name().to_owned();
                    let mut bytes = Vec::with_capacity(member.size() as usize);
                    std::io::Read::read_to_end(&mut member, &mut bytes)?;
                    let at = PathBuf::from(format!("{}!/{name}", path.display()));
                    let origin = recognize_member(&at, &bytes);
                    if let Some(plan) =
                        self.import_plan(&at, origin, Arc::new(bytes), &mut report)?
                    {
                        group_bytes += plan.size;
                        group.push(plan);
                    }
                    if group_bytes >= IMPORT_GROUP {
                        self.execute(std::mem::take(&mut group), &mut report.pass)?;
                        group_bytes = 0;
                    }
                }
                continue;
            }
            let Some(origin) = whole else {
                report.unrecognized.push(path.clone());
                continue;
            };
            let bytes = Arc::new(std::fs::read(path)?);
            if let Some(plan) = self.import_plan(path, origin, bytes, &mut report)? {
                group_bytes += plan.size;
                group.push(plan);
            }
            if group_bytes >= IMPORT_GROUP {
                self.execute(std::mem::take(&mut group), &mut report.pass)?;
                group_bytes = 0;
            }
        }
        self.execute(group, &mut report.pass)?;
        report.pass.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    /// One imported input's plan, or `None` when its exact bytes are
    /// already stored.
    fn import_plan(
        &mut self,
        path: &Path,
        origin: &'static OriginSpec,
        bytes: Arc<Vec<u8>>,
        report: &mut ImportReport,
    ) -> Result<Option<Plan>, EngineError> {
        let id = blake3::hash(&bytes).to_hex().to_string();
        if self.reads.reader()?.has_input(&id)? {
            report.already_stored.push(path.to_path_buf());
            return Ok(None);
        }
        let facts = std::fs::metadata(path).ok().map(|m| FileFacts::of(&m));
        report.imported.push(path.to_path_buf());
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin,
            input: InputKey {
                id,
                format: origin.name.into(),
                route_rank: 2,
            },
            change: Change::First,
            observation: Some(self.observation(path, facts)),
            facts,
            size: bytes.len() as u64,
            piece_pos: 0,
            parse_pos: 0,
            piece_started: false,
            carry: None,
            tail: None,
            span: span(origin, self.config.chunk),
            chunk: self.config.chunk,
            priority: Priority::Live,
            tracked: false,
            failed: false,
            waiting: false,
            source: Some(bytes),
        }))
    }

    /// Runs passes until stopped: after any change under a watched
    /// location, and at least once a minute.
    pub fn run(
        &mut self,
        tx: Sender<LoopMsg>,
        rx: &Receiver<LoopMsg>,
        status: &Arc<Mutex<Pass>>,
    ) -> Result<(), EngineError> {
        use notify::Watcher as _;
        let wake = tx.clone();
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if res.is_ok() {
                    let _ = wake.send(LoopMsg::Changed);
                }
            })?;
        for root in collect::roots(&self.config) {
            watcher.watch(&root, notify::RecursiveMode::Recursive)?;
        }
        drop(tx);
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

#[cfg(test)]
mod tests;
