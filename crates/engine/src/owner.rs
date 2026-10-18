//! The single writer actor: imports inputs, advances their parser state and
//! builds their Messages. Capture, parsing and Message encoding run on worker
//! threads; the actor only applies their results.
use crate::{
    canonical_message,
    converge::{self, PendingInstall, Recovery, Turn},
    source_turn::{
        self, Budgets, CommitGuard, CommittedContinuation, Computation, SourceContext, Window,
    },
};
use polylogue_model::BlobHash;
use polylogue_sources::{claude_code, codex, frame::StepBudget};
use polylogue_store::parser_state::Phase;
use polylogue_store::{
    Error as StoreError, ImportObservation, InputFormat, PreparedBlob, ReadHandle, StoredInput,
    StoredSource, Writer,
    parser_state::{Basis, Budget, Head, ParserReader},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
// Work limits, not format limits. A record or cell over the limit stays
// pending and the diagnostic records the size it needs.
mod history_capture;
use history_capture::TurnCaptureJob;
const COMMANDS: usize = 16;
const REQUEST_BYTES: usize = 65_536;
const WINDOW_BYTES: u64 = 65_536;
const DETAIL_BYTES: usize = 4096;
const CONTEXT: &str =
    r#"{"provider":"codex","format":"codex-rollout","context":"immutable-import-v1"}"#;
#[derive(Debug)]
pub enum Error {
    Store(StoreError),
    Source(source_turn::Error),
    Canonical(canonical_message::Error),
    InvalidEvidence(String),
    Busy,
    Stopped,
    RequestTooLarge { required: usize, maximum: usize },
    InvalidRequest(&'static str),
    WorkerPanicked,
    UnknownInput,
}
impl From<StoreError> for Error {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}
impl From<source_turn::Error> for Error {
    fn from(e: source_turn::Error) -> Self {
        Self::Source(e)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => std::fmt::Display::fmt(e, f),
            Self::Source(e) => std::fmt::Display::fmt(e, f),
            Self::InvalidEvidence(e) => f.write_str(e),
            Self::Canonical(e) => std::fmt::Display::fmt(e, f),
            Self::Busy => f.write_str("owner is busy; retry the unchanged request"),
            Self::Stopped => f.write_str("owner stopped"),
            Self::RequestTooLarge { required, maximum } => write!(
                f,
                "request needs {required} bytes; maximum queued request is {maximum}"
            ),
            Self::InvalidRequest(e) => f.write_str(e),
            Self::WorkerPanicked => f.write_str("owner worker panicked"),
            Self::UnknownInput => f.write_str("unknown immutable input"),
        }
    }
}
impl std::error::Error for Error {}
pub(crate) type Result<T> = std::result::Result<T, Error>;
type Reply<T> = SyncSender<Result<T>>;
#[derive(Clone, Debug)]
pub struct Observation {
    pub id: String,
    pub machine: String,
    pub host: String,
    pub path: String,
    pub seen_at: i64,
}
impl Observation {
    fn borrowed(&self) -> ImportObservation<'_> {
        ImportObservation {
            id: &self.id,
            machine: &self.machine,
            host: &self.host,
            path: &self.path,
            seen_at: self.seen_at,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ImportRequest {
    pub format: InputFormat,
    pub context: Vec<String>,
    pub path: PathBuf,
    pub observation: Observation,
    pub metadata_allowance: usize,
}
#[derive(Debug)]
pub struct AuthoredReply {
    pub revision: polylogue_model::authored::Revision,
    pub recovery: Option<polylogue_model::RecoveryRecord>,
}
#[derive(Debug)]
pub struct Progress {
    pub input: StoredInput,
    pub parser: Option<Head>,
    pub failure: Option<String>,
    pub recovery_error: Option<String>,
    pub active: bool,
    pub source_rows: polylogue_model::settlement::SourceRows,
}
#[derive(Debug)]
pub struct ConfiguredImport {
    pub input: StoredInput,
    pub acquired: bool,
}
enum ImportReply {
    Ordinary(Reply<StoredInput>),
    Configured(Reply<ConfiguredImport>),
}
impl ImportReply {
    fn send(self, result: Result<StoredInput>, acquired: bool) {
        match self {
            Self::Ordinary(reply) => {
                let _ = reply.send(result);
            }
            Self::Configured(reply) => {
                let _ = reply.send(result.map(|input| ConfiguredImport { input, acquired }));
            }
        }
    }
}
enum PreparedCapture {
    Blob(PreparedBlob),
    History(polylogue_store::history_pages::PreparedHistory),
    Claude(polylogue_store::observed::PreparedClaude),
}
enum Message {
    Note(
        polylogue_model::authored::WriteNote,
        Instant,
        Reply<AuthoredReply>,
    ),
    Import(
        ImportRequest,
        Option<(polylogue_store::configured_codex::Candidate, BlobHash, u64)>,
        ImportReply,
    ),
    Claude(
        ImportRequest,
        polylogue_store::configured_claude::Location,
        u64,
        Reply<ConfiguredImport>,
    ),
    CheckAcquisition(Reply<()>),
    Progress(String, Reply<Progress>),
    Retry(String, Reply<()>),
    Captured(Result<PreparedCapture>),
    CaptureYield(
        polylogue_store::history_pages::turns::Progress,
        SyncSender<std::result::Result<(), StoreError>>,
    ),
    TurnCaptured(Result<PreparedCapture>),
    Computed(Result<Computation>),
    CanonicalComputed(Result<CanonicalTurn>),
    Wake,
}
struct Inner {
    sender: SyncSender<Message>,
    accepting: Mutex<bool>,
    cancelled: Arc<AtomicBool>,
}
#[derive(Clone)]
pub struct Handle {
    inner: Arc<Inner>,
}
impl Handle {
    pub fn try_note(
        &self,
        request: polylogue_model::authored::WriteNote,
        deadline: Instant,
    ) -> Result<Receiver<Result<AuthoredReply>>> {
        request
            .validate()
            .map_err(|e| Error::Store(StoreError::Authored(e)))?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Note(request, deadline, reply))?;
        Ok(rx)
    }
    fn send(&self, message: Message) -> Result<()> {
        let accepting = self.inner.accepting.lock().map_err(|_| Error::Stopped)?;
        if !*accepting || self.inner.cancelled.load(Ordering::Acquire) {
            return Err(Error::Stopped);
        }
        self.inner.sender.try_send(message).map_err(|e| match e {
            TrySendError::Full(_) => Error::Busy,
            TrySendError::Disconnected(_) => Error::Stopped,
        })
    }
    pub fn try_import(&self, request: ImportRequest) -> Result<Receiver<Result<StoredInput>>> {
        if request.format == InputFormat::ClaudeCodeToolResult {
            return Err(Error::InvalidRequest(
                "Claude companions require the configured main acquisition path",
            ));
        }
        self.validate_import(&request, false)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Import(request, None, ImportReply::Ordinary(reply)))?;
        Ok(rx)
    }
    pub fn try_configured_import(
        &self,
        request: ImportRequest,
        candidate: polylogue_store::configured_codex::Candidate,
        expected: BlobHash,
        expected_length: u64,
    ) -> Result<Receiver<Result<ConfiguredImport>>> {
        if candidate.path() != request.path || candidate.format() != request.format {
            return Err(Error::InvalidRequest(
                "configured candidate does not match import",
            ));
        }
        self.validate_import(&request, true)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Import(
            request,
            Some((candidate, expected, expected_length)),
            ImportReply::Configured(reply),
        ))?;
        Ok(rx)
    }
    pub fn try_claude_acquisition(
        &self,
        path: PathBuf,
        observation: Observation,
        quantum_bytes: u64,
    ) -> Result<Receiver<Result<ConfiguredImport>>> {
        let location = polylogue_store::configured_claude::Location::explicit_main(path)?;
        self.try_claude_location(location, observation, quantum_bytes)
    }
    pub fn try_claude_location(
        &self,
        location: polylogue_store::configured_claude::Location,
        observation: Observation,
        quantum_bytes: u64,
    ) -> Result<Receiver<Result<ConfiguredImport>>> {
        if !(65536..=8 * 1024 * 1024).contains(&quantum_bytes) {
            return Err(Error::InvalidRequest(
                "Claude quantum must be between 65536 and 8388608 bytes",
            ));
        }
        let request = ImportRequest {
            format: InputFormat::ClaudeCodeJsonl,
            context: location.context().to_vec(),
            path: location.path().to_path_buf(),
            observation,
            metadata_allowance: 65536,
        };
        self.validate_import(&request, false)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Claude(request, location, quantum_bytes, reply))?;
        Ok(rx)
    }
    pub fn try_history_import(
        &self,
        request: ImportRequest,
    ) -> Result<Receiver<Result<ConfiguredImport>>> {
        if request.format != InputFormat::CodexHistorySqlite {
            return Err(Error::InvalidRequest("expected physical history request"));
        }
        self.validate_import(&request, true)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Import(
            request,
            None,
            ImportReply::Configured(reply),
        ))?;
        Ok(rx)
    }
    pub fn try_check_import_acquisition(&self) -> Result<Receiver<Result<()>>> {
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::CheckAcquisition(reply))?;
        Ok(rx)
    }
    fn validate_import(&self, request: &ImportRequest, configured: bool) -> Result<()> {
        let o = &request.observation;
        let required = request
            .path
            .as_os_str()
            .len()
            .saturating_add(o.id.len())
            .saturating_add(o.machine.len())
            .saturating_add(o.host.len())
            .saturating_add(o.path.len())
            .saturating_add(
                request
                    .context
                    .iter()
                    .map(|v| v.len().saturating_add(8))
                    .fold(0usize, usize::saturating_add),
            );
        let maximum = if configured { 131072 } else { REQUEST_BYTES };
        if required > maximum || request.metadata_allowance > maximum {
            return Err(Error::RequestTooLarge {
                required: required.max(request.metadata_allowance),
                maximum,
            });
        }
        // Reject excessive declarations before allocating the borrowed slice.
        let context: Vec<&str> = request.context.iter().map(String::as_str).collect();
        request.format.validate_context(&context)?;
        if !request.path.is_absolute() || request.path.as_os_str() != std::ffi::OsStr::new(&o.path)
        {
            return Err(Error::InvalidRequest(
                "capture path must be absolute and exactly match supplied observation.path",
            ));
        }
        Ok(())
    }
    pub fn try_progress(&self, input_id: &str) -> Result<Receiver<Result<Progress>>> {
        check_id(input_id)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Progress(input_id.into(), reply))?;
        Ok(rx)
    }
    pub fn try_retry(&self, input_id: &str) -> Result<Receiver<Result<()>>> {
        check_id(input_id)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Retry(input_id.into(), reply))?;
        Ok(rx)
    }
}
fn check_id(id: &str) -> Result<()> {
    if id.len() > REQUEST_BYTES {
        return Err(Error::RequestTooLarge {
            required: id.len(),
            maximum: REQUEST_BYTES,
        });
    }
    Ok(())
}
pub struct Owner {
    handle: Handle,
    join: Option<JoinHandle<Result<()>>>,
}
impl Owner {
    pub fn start(writer: Writer, cancelled: Arc<AtomicBool>) -> Result<Self> {
        let (sender, rx) = mpsc::sync_channel(COMMANDS);
        let inner = Arc::new(Inner {
            sender,
            accepting: Mutex::new(true),
            cancelled,
        });
        let actor = Arc::clone(&inner);
        let join = thread::Builder::new()
            .name("polylogue-owner".into())
            .spawn(move || Actor::new(writer, actor).run(rx))
            .map_err(StoreError::Io)?;
        Ok(Self {
            handle: Handle { inner },
            join: Some(join),
        })
    }
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }
    fn stop(&self) {
        self.handle.inner.cancelled.store(true, Ordering::Release);
        if let Ok(mut accepting) = self.handle.inner.accepting.lock() {
            *accepting = false;
        }
        let _ = self.handle.inner.sender.try_send(Message::Wake);
    }
    pub fn shutdown(mut self) -> Result<()> {
        self.stop();
        self.join
            .take()
            .ok_or(Error::Stopped)?
            .join()
            .map_err(|_| Error::WorkerPanicked)?
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.stop();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
struct CaptureJob {
    expected: Option<BlobHash>,
    format: InputFormat,
    context: Vec<String>,
    observation: Observation,
    allowance: usize,
    reply: ImportReply,
    join: JoinHandle<()>,
}
struct ComputeJob {
    input: StoredSource,
    head: Head,
    join: JoinHandle<()>,
}
// Keyset order only schedules work; Message order comes from the source
// ordinals in the provider's PendingMessage.
#[derive(Clone)]
pub(crate) enum CanonicalPosition {
    Audit(Box<crate::source_rows::Audit>),
    Codex(usize),
    ToolsAfter(Option<String>),
    // Publish Session operations before recording a contested Message
    // diagnostic. The input stays unfinished either way.
    ToolsBeforeFailure(Option<String>, String),
    ClaudeAfter(Option<crate::state::claude::PendingCursor>),
    ClaudeMessage(
        crate::state::claude::PendingCursor,
        claude_code::keyed::PendingMessage,
    ),
}
impl CanonicalPosition {
    pub(crate) fn first(format: InputFormat) -> Self {
        match format {
            InputFormat::CodexRollout => Self::Codex(0),
            InputFormat::CodexHistorySqlite | InputFormat::CodexPromptHistory => {
                Self::ToolsAfter(None)
            }
            InputFormat::ClaudeCodeJsonl => Self::ClaudeAfter(None),
            InputFormat::ClaudeCodeToolResult => unreachable!("companions are never scheduled"),
        }
    }
    fn next(self) -> Result<Self> {
        match self {
            Self::Codex(n) => Ok(Self::Codex(n.checked_add(1).ok_or(StoreError::Range)?)),
            Self::ClaudeMessage(cursor, _) => Ok(Self::ClaudeAfter(Some(cursor))),
            Self::ToolsAfter(_)
            | Self::ToolsBeforeFailure(_, _)
            | Self::Audit(_)
            | Self::ClaudeAfter(_) => Err(Error::InvalidRequest("no current Claude Message")),
        }
    }
    pub(crate) fn logical_key(&self) -> Result<Option<[u8; 32]>> {
        match self {
            Self::Codex(n) => canonical_key(*n).map(Some),
            Self::ClaudeMessage(_, p) => Ok(Some(claude_key(p.message.as_str()))),
            Self::ToolsAfter(_)
            | Self::ToolsBeforeFailure(_, _)
            | Self::Audit(_)
            | Self::ClaudeAfter(_) => Ok(None),
        }
    }
}
enum CanonicalTurn {
    Selected(polylogue_store::contributions::CompatibilityTurn),
    Audited(crate::source_rows::Turn),
    Action(canonical_message::Action),
    Pending(
        crate::state::claude::PendingCursor,
        claude_code::keyed::PendingMessage,
    ),
    Operation(String, polylogue_model::tool_item::Operation, BlobHash),
    Done,
}
struct CanonicalJob {
    input: StoredSource,
    head: Head,
    position: CanonicalPosition,
    handle: Option<canonical_message::Handle>,
    join: JoinHandle<Option<CanonicalReaders>>,
}
struct CanonicalReaders {
    values: ReadHandle,
    parser: ParserReader,
}
enum Active {
    Selecting(
        StoredSource,
        Head,
        CanonicalPosition,
        polylogue_store::contributions::CompatibleSelection,
    ),
    Begin(StoredSource),
    Recover(StoredSource, Head),
    Compute(StoredSource, Head, Option<CommittedContinuation>),
    Install(StoredSource, Head, PendingInstall, CommitGuard),
    HistoryInstall(StoredSource, Head, PendingInstall),
    Canonical(
        StoredSource,
        Head,
        CanonicalPosition,
        Option<canonical_message::Handle>,
    ),
}
struct Actor {
    inner: Arc<Inner>,
    capture: Option<CaptureJob>,
    turn_capture: Option<TurnCaptureJob>,
    waiting_capture: Option<(
        ImportRequest,
        Option<(polylogue_store::configured_codex::Candidate, BlobHash, u64)>,
        ImportReply,
    )>,
    history_budget: polylogue_store::history_pages::turns::Budget,
    compute: Option<ComputeJob>,
    canonical: Option<CanonicalJob>,
    // One worker holds this reader at a time. Reusing the idle connections
    // avoids reparsing the SQLite schema on every turn; each store operation
    // still opens a fresh view and no transaction spans actor turns.
    canonical_reader: Option<CanonicalReaders>,
    active: Option<Active>,
    // A hint for new or retried inputs. The stored catalog scan still finds
    // them if hints are merged or the process restarts.
    changed_input: Option<String>,
    background_due: bool,
    after: String,
    scanning: bool,
    scan_again: bool,
    recovery_error: Option<String>,
    stopping: bool,
    fatal: Option<Error>,
    // Fields drop in declaration order. Close the exercised idle read
    // connections before SQLite's last writer performs WAL cleanup.
    writer: Writer,
}
impl Actor {
    fn new(writer: Writer, inner: Arc<Inner>) -> Self {
        Self {
            writer,
            inner,
            capture: None,
            turn_capture: None,
            waiting_capture: None,
            history_budget: polylogue_store::history_pages::turns::Budget::default(),
            compute: None,
            canonical: None,
            canonical_reader: None,
            active: None,
            changed_input: None,
            background_due: false,
            after: String::new(),
            scanning: true,
            scan_again: false,
            recovery_error: None,
            stopping: false,
            fatal: None,
        }
    }
    fn begin_stop(&mut self) {
        self.stopping = true;
        if let Ok(mut accepting) = self.inner.accepting.lock() {
            *accepting = false;
        }
        self.inner.cancelled.store(true, Ordering::Release);
        self.active = None;
        if let Some(history) = &mut self.turn_capture {
            history.grant.take();
        }
        if let Some((_, _, reply)) = self.waiting_capture.take() {
            reply.send(Err(Error::Stopped), false);
        }
    }
    fn observe_stop(&mut self) {
        if self.inner.cancelled.load(Ordering::Acquire) && !self.stopping {
            self.begin_stop();
        }
    }
    fn stop_for_completion<T>(&mut self, result: &Result<T>) -> bool {
        self.observe_stop();
        // A worker cancelled by shutdown is not a parser defect, even if the
        // shutdown arrived after dispatch.
        if matches!(
            result,
            Err(Error::Stopped
                | Error::Store(StoreError::Cancelled)
                | Error::Source(source_turn::Error::Store(StoreError::Cancelled))
                | Error::Canonical(canonical_message::Error::Store(StoreError::Cancelled))
                | Error::Canonical(canonical_message::Error::Cancelled))
        ) {
            self.begin_stop();
        }
        self.stopping
    }
    fn run(mut self, rx: Receiver<Message>) -> Result<()> {
        let mut commands_since_grant = 0;
        loop {
            self.observe_stop();
            let immediate = !self.stopping
                && (self
                    .turn_capture
                    .as_ref()
                    .is_some_and(|h| h.grant.is_some())
                    || (self.compute.is_none()
                        && self.canonical.is_none()
                        && (self.active.is_some() || self.scanning)));
            let mut mailbox_empty = false;
            let message = match rx.try_recv() {
                Ok(m) => Some(m),
                Err(TryRecvError::Disconnected) => {
                    self.begin_stop();
                    None
                }
                Err(TryRecvError::Empty) if immediate => {
                    mailbox_empty = true;
                    None
                }
                Err(TryRecvError::Empty) => match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(m) => Some(m),
                    Err(RecvTimeoutError::Timeout) => {
                        mailbox_empty = true;
                        None
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        self.begin_stop();
                        None
                    }
                },
            };
            if let Some(message) = message {
                commands_since_grant += 1;
                if let Err(e) = self.message(message) {
                    self.fatal = Some(e);
                    self.begin_stop();
                }
            }
            if !self.stopping {
                if let Err(e) = self.tick() {
                    self.fatal = Some(e);
                    self.begin_stop();
                }
            }
            if !self.stopping && (mailbox_empty || commands_since_grant >= COMMANDS) {
                if let Err(e) = self.advance_capture_turn() {
                    self.fatal = Some(e);
                    self.begin_stop();
                }
                commands_since_grant = 0;
            }
            if self.stopping
                && self.turn_capture.is_none()
                && self.capture.is_none()
                && self.compute.is_none()
                && self.canonical.is_none()
            {
                // Accepting closed under its mutex before this drain, so every
                // accepted request gets a reply.
                while let Ok(message) = rx.try_recv() {
                    let _ = self.message(message);
                }
                return self.fatal.map_or(Ok(()), Err);
            }
        }
    }
    fn finish_capture(&mut self, job: CaptureJob, result: Result<PreparedCapture>) -> Result<()> {
        job.join.join().map_err(|_| Error::WorkerPanicked)?;
        if self.stop_for_completion(&result) {
            let _ = job.reply.send(Err(Error::Stopped), false);
            return Ok(());
        }
        let mut acquired = true;
        let mut claude_installation_started = false;
        let result = result.and_then(|proof| {
            let proof = match proof {
                PreparedCapture::History(p) => {
                    if let Some(input) = self
                        .writer
                        .known_history_snapshot(p.hash(), &job.observation.borrowed())?
                    {
                        acquired = false;
                        return Ok(input);
                    }
                    return self
                        .writer
                        .store_history_pages(p, &job.observation.borrowed(), job.allowance)
                        .map_err(Error::Store);
                }
                PreparedCapture::Claude(prepared) => {
                    let before = self.writer.claude_input_at(
                        std::path::Path::new(&job.observation.path),
                        &job.observation.machine,
                    )?;
                    let source = self.writer.store_claude_main(
                        prepared.main,
                        &job.observation.machine,
                        &job.observation.host,
                        &job.observation.id,
                        job.observation.seen_at,
                    )?;
                    // A later companion failure must not hide the stored main file.
                    claude_installation_started = true;
                    for (index, (path, body)) in prepared.bodies.into_iter().enumerate() {
                        let mut observation = job.observation.clone();
                        observation.id = format!("{}:claude-body:{index}", observation.id);
                        observation.path = format!("copy:{}", path.display());
                        self.writer.store_claude_body(
                            &source.input.id,
                            body,
                            &observation.borrowed(),
                        )?;
                    }
                    let after = self
                        .writer
                        .stored_source(&source.input.id)?
                        .ok_or(Error::UnknownInput)?;
                    acquired = before.as_ref().is_none_or(|old| {
                        old.revision != after.revision || old.companions != after.companions
                    });
                    return Ok(after.input);
                }
                PreparedCapture::Blob(p) => p,
            };
            if job.expected.is_some_and(|hash| hash != proof.hash()) {
                return Err(Error::InvalidEvidence(
                    "configured source changed after discovery; retry next pass".into(),
                ));
            }
            let hash = self.writer.store_captured_blob(proof)?;
            Ok(self.writer.store_immutable_input(
                hash,
                job.format,
                &job.context.iter().map(String::as_str).collect::<Vec<_>>(),
                &job.observation.borrowed(),
                job.allowance,
            )?)
        });
        if (result.is_ok() && acquired) || (result.is_err() && claude_installation_started) {
            if let Ok(input) = &result {
                self.changed_input = Some(input.id.clone());
            }
            self.scan_again = true;
            if !self.scanning
                && self.active.is_none()
                && self.compute.is_none()
                && self.canonical.is_none()
            {
                self.restart_scan();
            }
        }
        let _ = job.reply.send(result, acquired);
        Ok(())
    }
    fn active_id(&self) -> Option<&str> {
        if let Some(job) = &self.canonical {
            return Some(&job.input.input.id);
        }
        if let Some(job) = &self.compute {
            return Some(&job.input.input.id);
        }
        match &self.active {
            Some(
                Active::Begin(i)
                | Active::Recover(i, _)
                | Active::Compute(i, _, _)
                | Active::HistoryInstall(i, _, _)
                | Active::Install(i, _, _, _)
                | Active::Canonical(i, _, _, _)
                | Active::Selecting(i, _, _, _),
            ) => Some(&i.input.id),
            None => None,
        }
    }
    fn message(&mut self, message: Message) -> Result<()> {
        // recv_timeout may have blocked since the loop last checked for stop.
        self.observe_stop();
        match message {
            Message::Note(request, deadline, reply) => self.note(request, deadline, reply),
            Message::Import(request, configured, reply) => {
                self.import(request, configured, reply)?
            }
            Message::Claude(request, location, quantum_bytes, reply) => {
                if self.stopping {
                    let _ = reply.send(Err(Error::Stopped));
                } else {
                    self.start_claude(
                        request,
                        location,
                        quantum_bytes,
                        ImportReply::Configured(reply),
                    )?;
                }
            }
            Message::CheckAcquisition(reply) => {
                let result = if self.stopping {
                    Err(Error::Stopped)
                } else {
                    self.writer.check_import_acquisition().map_err(Error::Store)
                };
                let _ = reply.send(result);
            }
            Message::Progress(id, reply) => {
                let result = if self.stopping {
                    Err(Error::Stopped)
                } else {
                    self.progress(&id)
                };
                let _ = reply.send(result);
            }
            Message::Retry(id, reply) => {
                let result = if self.stopping {
                    Err(Error::Stopped)
                } else if self.active_id() == Some(id.as_str()) {
                    Err(Error::Busy)
                } else {
                    self.retry(&id)
                };
                let _ = reply.send(result);
            }
            Message::Captured(result) => {
                let Some(job) = self.capture.take() else {
                    return Err(Error::InvalidRequest("unexpected capture completion"));
                };
                self.finish_capture(job, result)?;
            }
            Message::CaptureYield(progress, grant) => {
                let history = self
                    .turn_capture
                    .as_mut()
                    .ok_or(Error::InvalidRequest("unexpected acquisition yield"))?;
                history.progress = Some(progress);
                history.regular_served = false;
                if self.stopping {
                    drop(grant);
                } else {
                    history.grant = Some(grant);
                }
            }
            Message::TurnCaptured(result) => {
                let history = self
                    .turn_capture
                    .take()
                    .ok_or(Error::InvalidRequest("unexpected acquisition completion"))?;
                self.finish_capture(history.job, result)?;
            }
            Message::Computed(result) => self.computed(result)?,
            Message::CanonicalComputed(result) => self.canonical_computed(result)?,
            Message::Wake => {}
        }
        Ok(())
    }
    fn note(
        &mut self,
        request: polylogue_model::authored::WriteNote,
        deadline: Instant,
        reply: Reply<AuthoredReply>,
    ) {
        let result = if self.stopping {
            Err(Error::Stopped)
        } else if Instant::now() >= deadline {
            Err(StoreError::Authored(polylogue_model::Failure::new(
                polylogue_model::ErrorKind::Timeout,
                "queued authored write deadline passed; no effect committed",
            ))
            .into())
        } else {
            self.writer
                .authored_recovery()
                .and_then(|recovery| {
                    self.writer
                        .write_note_until(&request, deadline)
                        .map(|revision| AuthoredReply { revision, recovery })
                })
                .map_err(Error::Store)
        };
        let _ = reply.send(result);
    }
    fn import(
        &mut self,
        request: ImportRequest,
        configured: Option<(polylogue_store::configured_codex::Candidate, BlobHash, u64)>,
        reply: ImportReply,
    ) -> Result<()> {
        if self.stopping {
            let _ = reply.send(Err(Error::Stopped), false);
            return Ok(());
        }
        let replay = self.writer.replay_immutable_import(
            request.format,
            &request
                .context
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            &request.observation.borrowed(),
            request.metadata_allowance,
        );
        match replay {
            Ok(Some(input)) => {
                if let Some((candidate, hash, _)) = &configured {
                    if let Err(error) = self.writer.check_import_acquisition() {
                        reply.send(Err(error.into()), false);
                        return Ok(());
                    }
                    if input.id
                        != polylogue_model::InputId::imported(hash, candidate.format().name(), &[])
                            .as_str()
                    {
                        reply.send(
                            Err(StoreError::ClientRequestConflict {
                                id: request.observation.id.clone(),
                            }
                            .into()),
                            false,
                        );
                        return Ok(());
                    }
                }
                self.scan_again = true;
                if !self.scanning
                    && self.active.is_none()
                    && self.compute.is_none()
                    && self.canonical.is_none()
                {
                    self.restart_scan();
                }
                let _ = reply.send(Ok(input), false);
                return Ok(());
            }
            Err(e) => {
                let _ = reply.send(Err(e.into()), false);
                return Ok(());
            }
            Ok(None) => {}
        }
        if let Some((candidate, hash, _)) = &configured {
            match self.writer.known_configured_snapshot(
                candidate,
                *hash,
                &request.observation.machine,
                &request.observation.host,
                &request.observation.path,
            ) {
                Ok(Some(input)) => {
                    reply.send(Ok(input), false);
                    return Ok(());
                }
                Err(error) => {
                    reply.send(Err(error.into()), false);
                    return Ok(());
                }
                Ok(None) => {}
            }
        }
        if request.format == InputFormat::CodexHistorySqlite {
            return self.start_history(request, reply);
        }
        if self.capture.is_none()
            && self
                .turn_capture
                .as_ref()
                .is_some_and(|h| h.grant.is_none() || h.regular_served)
        {
            if self.waiting_capture.is_none() {
                self.waiting_capture = Some((request, configured, reply));
            } else {
                reply.send(Err(Error::Busy), false);
            }
            return Ok(());
        }
        if self.capture.is_some() {
            let _ = reply.send(Err(Error::Busy), false);
            return Ok(());
        }
        if let Err(e) = self.writer.check_import_acquisition() {
            let _ = reply.send(Err(e.into()), false);
            return Ok(());
        }
        let capture = match self.writer.blob_capture() {
            Ok(c) => c,
            Err(e) => {
                let _ = reply.send(Err(e.into()), false);
                return Ok(());
            }
        };
        if let Some(history) = &mut self.turn_capture {
            history.regular_served = true;
        }
        let cancelled = Arc::clone(&self.inner.cancelled);
        let sender = self.inner.sender.clone();
        let path = request.path;
        let expected = configured.as_ref().map(|(_, hash, _)| *hash);
        let join = thread::Builder::new()
            .name("polylogue-capture".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Some((candidate, _, length)) = configured {
                        capture
                            .capture_codex_candidate(&candidate, length, &cancelled)
                            .map(PreparedCapture::Blob)
                    } else {
                        capture
                            .capture_regular_file(&path, &cancelled)
                            .map(PreparedCapture::Blob)
                    }
                }))
                .map_err(|_| Error::WorkerPanicked)
                .and_then(|r| r.map_err(Error::Store));
                let _ = sender.send(Message::Captured(result));
            });
        match join {
            Ok(join) => {
                self.capture = Some(CaptureJob {
                    expected,
                    format: request.format,
                    context: request.context,
                    observation: request.observation,
                    allowance: request.metadata_allowance,
                    reply,
                    join,
                })
            }
            Err(e) => {
                let _ = reply.send(Err(StoreError::Io(e).into()), false);
            }
        }
        Ok(())
    }
    fn computed(&mut self, result: Result<Computation>) -> Result<()> {
        let Some(job) = self.compute.take() else {
            return Err(Error::InvalidRequest("unexpected source completion"));
        };
        job.join.join().map_err(|_| Error::WorkerPanicked)?;
        if self.stop_for_completion(&result) {
            return Ok(());
        }
        match result {
            Ok(Computation::Pending(pending)) => {
                let (install, guard) = pending.into_install();
                match PendingInstall::begin(
                    &mut self.writer,
                    &install.predecessor,
                    install.progress,
                    install.changes,
                ) {
                    Ok(p) => self.active = Some(Active::Install(job.input, job.head, p, guard)),
                    Err(e) => self.fail(&job.head, e.to_string())?,
                }
            }
            Ok(Computation::History { progress, changes }) => {
                match PendingInstall::begin(&mut self.writer, &job.head, progress, changes) {
                    Ok(install) => {
                        self.active = Some(Active::HistoryInstall(job.input, job.head, install))
                    }
                    Err(e) => self.fail(&job.head, e.to_string())?,
                }
            }
            Ok(Computation::Idle { .. }) => {}
            Err(e) => self.fail(&job.head, e.to_string())?,
        }
        Ok(())
    }
    fn canonical_computed(&mut self, result: Result<CanonicalTurn>) -> Result<()> {
        let job = self
            .canonical
            .take()
            .ok_or(Error::InvalidRequest("unexpected canonical completion"))?;
        self.canonical_reader = job.join.join().map_err(|_| Error::WorkerPanicked)?;
        if self.stop_for_completion(&result) {
            return Ok(());
        }
        match result {
            Ok(CanonicalTurn::Operation(id, operation, digest)) => {
                match self
                    .writer
                    .publish_tool_item(&job.head.token, &operation, digest)
                {
                    Ok(_) => {
                        self.active = Some(Active::Canonical(
                            job.input,
                            job.head,
                            match job.position {
                                CanonicalPosition::ToolsBeforeFailure(_, reason) => {
                                    CanonicalPosition::ToolsBeforeFailure(Some(id), reason)
                                }
                                _ => CanonicalPosition::ToolsAfter(Some(id)),
                            },
                            None,
                        ))
                    }
                    Err(e) => self.fail(&job.head, e.to_string())?,
                }
            }
            Ok(CanonicalTurn::Action(action)) => {
                let operation = action.operation();
                match canonical_message::apply(&mut self.writer, action) {
                    Ok(canonical_message::Applied::Prepared(handle)) => {
                        self.active = Some(Active::Canonical(
                            job.input,
                            job.head,
                            job.position,
                            Some(handle),
                        ));
                    }
                    Ok(canonical_message::Applied::Progress) => {
                        self.active = Some(Active::Canonical(
                            job.input,
                            job.head,
                            job.position,
                            job.handle,
                        ));
                    }
                    Ok(
                        canonical_message::Applied::Source(source)
                        | canonical_message::Applied::Existing(source),
                    ) => {
                        match self
                            .writer
                            .begin_compatible_source_selection(&source, 65536)
                        {
                            Ok(selection) => {
                                self.active = Some(Active::Selecting(
                                    job.input,
                                    job.head,
                                    job.position,
                                    selection,
                                ))
                            }
                            Err(e) => self.fail(&job.head, format!("canonical selection: {e}"))?,
                        }
                    }
                    Err(e) => self.fail(&job.head, format!("canonical {operation}: {e}"))?,
                }
            }
            Ok(CanonicalTurn::Selected(turn)) => {
                use polylogue_store::contributions::CompatibilityTurn;
                match turn {
                    CompatibilityTurn::Progress(selection) => {
                        self.active = Some(Active::Selecting(
                            job.input,
                            job.head,
                            job.position,
                            selection,
                        ))
                    }
                    CompatibilityTurn::Certified(selection, certificate) => {
                        match self.writer.certify_body_compatibility(certificate) {
                            Ok(()) => {
                                self.active = Some(Active::Selecting(
                                    job.input,
                                    job.head,
                                    job.position,
                                    selection,
                                ))
                            }
                            Err(e) => {
                                self.fail(&job.head, format!("canonical compatibility: {e}"))?
                            }
                        }
                    }
                    CompatibilityTurn::Ready(selection) => {
                        match self
                            .writer
                            .install_compatible_source_selection(selection, 65536)
                        {
                            Ok(_) => {
                                self.active = Some(Active::Canonical(
                                    job.input,
                                    job.head,
                                    job.position.next()?,
                                    None,
                                ))
                            }
                            Err(e) => self.fail(&job.head, format!("canonical selection: {e}"))?,
                        }
                    }
                    CompatibilityTurn::Contested => {
                        let reason = "canonical Message has incompatible or unsupported input candidates; merge remains pending".to_owned();
                        if job.input.format == InputFormat::CodexRollout {
                            self.active = Some(Active::Canonical(
                                job.input,
                                job.head,
                                CanonicalPosition::ToolsBeforeFailure(None, reason),
                                None,
                            ));
                        } else {
                            self.fail(&job.head, reason)?;
                        }
                    }
                }
            }
            Ok(CanonicalTurn::Pending(cursor, pending)) => {
                self.active = Some(Active::Canonical(
                    job.input,
                    job.head,
                    CanonicalPosition::ClaudeMessage(cursor, pending),
                    None,
                ));
            }
            Ok(CanonicalTurn::Done)
                if matches!(job.position, CanonicalPosition::ToolsBeforeFailure(_, _)) =>
            {
                let CanonicalPosition::ToolsBeforeFailure(_, reason) = job.position else {
                    unreachable!()
                };
                self.fail(&job.head, reason)?;
            }
            Ok(CanonicalTurn::Done) if job.head.status == Phase::WaitingTail => {
                // Complete Claude prefix rows may publish while its
                // unframed suffix waits. The source stays pending.
            }
            Ok(CanonicalTurn::Done) => {
                self.active = Some(Active::Canonical(
                    job.input,
                    job.head,
                    CanonicalPosition::Audit(Box::new(crate::source_rows::Audit::new())),
                    None,
                ));
            }
            Ok(CanonicalTurn::Audited(crate::source_rows::Turn::Progress(audit))) => {
                self.active = Some(Active::Canonical(
                    job.input,
                    job.head,
                    CanonicalPosition::Audit(Box::new(audit)),
                    None,
                ));
            }
            Ok(CanonicalTurn::Audited(crate::source_rows::Turn::Complete(receipt))) => {
                receipt.validate(
                    &job.head.token,
                    &crate::source_rows::computed_from(&job.input),
                )?;
                let bytes = serde_json::to_vec(&receipt)
                    .map_err(|e| Error::InvalidEvidence(e.to_string()))?;
                self.writer.publish_source_rows(
                    &job.head,
                    &bytes,
                    matches!(
                        receipt.rows.state,
                        polylogue_model::settlement::SourceRowState::Settled
                    ),
                )?;
            }
            Err(e) => {
                let changed = matches!(
                    &e,
                    Error::Store(StoreError::StaleParserState)
                        | Error::Canonical(canonical_message::Error::Store(
                            StoreError::StaleParserState
                        ))
                );
                if !changed || !self.reselect_obsolete_canonical(&job.input, &job.head)? {
                    self.fail(&job.head, format!("canonical compute: {e}"))?;
                }
            }
        }
        Ok(())
    }
    fn progress(&mut self, id: &str) -> Result<Progress> {
        let input = self.writer.stored_source(id)?.ok_or(Error::UnknownInput)?;
        let parser = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?;
        let failure = match &parser {
            Some(head) => self.writer.parser_failure(&head.token)?,
            None => None,
        };
        let obligation = match parser.as_ref().map(|h| h.status) {
            None if input.format == InputFormat::CodexPromptHistory => {
                "prompt_reference_conversation_evidence"
            }
            None => "source_parser_admission",
            Some(Phase::WaitingTail) => "unfinished_source_record",
            Some(Phase::Parsing) => "source_record_interpretation",
            _ => "canonical_source_row_effects_and_audit",
        };
        let mut source_rows = crate::source_rows::pending(&input, obligation);
        if let Some(head) = &parser {
            if head.acquired_upto != input.input.length
                || head.token.basis != parser_identity(&input)
                || head.parser_version != parser_version(input.format)
            {
                source_rows =
                    crate::source_rows::pending(&input, "current_interpretation_admission");
            } else if let Some(reason) = &failure {
                // A failed install may still hold unsealed work. Report it as
                // unavailable, never as a waiting tail.
                source_rows.state = polylogue_model::settlement::SourceRowState::Unavailable {
                    usable_result: self.writer.source_has_tool_items(&input.input)?,
                    notes: vec![reason.clone()],
                    record_errors: head.counts.undecodable,
                };
            } else if self.writer.parser_work(&head.token)?.is_none() {
                let snapshot = self.writer.parser_snapshot(&head.token)?;
                if let Some(receipt) = crate::source_rows::read(&snapshot, &input)? {
                    source_rows = receipt.rows;
                }
            }
        }
        Ok(Progress {
            input: input.input,
            parser,
            failure,
            recovery_error: self.recovery_error.clone(),
            active: self.active_id() == Some(id),
            source_rows,
        })
    }
    fn retry(&mut self, id: &str) -> Result<()> {
        let input = self.writer.stored_source(id)?.ok_or(Error::UnknownInput)?;
        if input.format == InputFormat::CodexPromptHistory {
            return Err(Error::InvalidRequest(
                "prompt references have no conversation parser to retry",
            ));
        }
        if let Some(head) = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?
        {
            self.writer
                .record_parser_failure(&head.token, None, DETAIL_BYTES)?;
        }
        self.changed_input = Some(id.to_owned());
        self.scan_again = true;
        self.recovery_error = None;
        if !self.scanning
            && self.active.is_none()
            && self.compute.is_none()
            && self.canonical.is_none()
        {
            self.restart_scan();
        }
        Ok(())
    }
    fn restart_scan(&mut self) {
        self.after.clear();
        self.scanning = true;
        self.scan_again = false;
        self.recovery_error = None;
    }
    fn fail(&mut self, head: &Head, details: String) -> Result<()> {
        self.observe_stop();
        if self.stopping {
            return Ok(());
        }
        let details = diagnostic(details);
        self.writer
            .record_parser_failure(&head.token, Some(&details), DETAIL_BYTES)?;
        Ok(())
    }
    // An append can replace the source while its old audit is running. Discard
    // that attempt only when the parser state was rejected as stale and the
    // stored source really changed; other invariant failures still fail. The
    // next turn picks up the current parser state.
    fn reselect_obsolete_canonical(
        &mut self,
        previous: &StoredSource,
        predecessor: &Head,
    ) -> Result<bool> {
        let Some(input) = self.writer.stored_source(&previous.input.id)? else {
            return Ok(false);
        };
        let head = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?;
        let source_changed = input.revision != previous.revision
            || input.input.chain != previous.input.chain
            || input.input.length != previous.input.length
            || parser_identity(&input) != parser_identity(&previous);
        let predecessor_changed = head.as_ref().is_some_and(|head| {
            head.token != predecessor.token
                || head.acquired_upto != predecessor.acquired_upto
                || head.status != predecessor.status
                || head.finalization_pending != predecessor.finalization_pending
        });
        if !source_changed && !predecessor_changed {
            return Ok(false);
        }
        self.active = Some(Active::Begin(input));
        Ok(true)
    }
    fn take_canonical_reader(&mut self) -> Result<CanonicalReaders> {
        match self.canonical_reader.take() {
            Some(reader) => Ok(reader),
            None => Ok(CanonicalReaders {
                values: self.writer.read_handle()?,
                parser: self.writer.parser_reader()?,
            }),
        }
    }
    fn tick(&mut self) -> Result<()> {
        if self.compute.is_some() || self.canonical.is_some() {
            return Ok(());
        }
        let active = match self.active.take() {
            Some(a) => a,
            None => match self.next_input()? {
                Some(a) => a,
                None => return Ok(()),
            },
        };
        match active {
            Active::Begin(input) => self.begin(input)?,
            Active::Recover(input, head) => self.recover(input, head)?,
            Active::Compute(input, head, continuation) => {
                self.start_compute(input, head, continuation)?
            }
            Active::Selecting(input, head, position, selection) => {
                self.start_selection(input, head, position, selection)?
            }
            Active::Canonical(input, head, position, handle) => {
                self.start_canonical(input, head, position, handle)?
            }
            Active::HistoryInstall(input, head, install) => {
                self.install_history(input, head, install)?
            }
            Active::Install(input, head, install, guard) => {
                self.install(input, head, install, guard)?
            }
        }
        Ok(())
    }
    /// Picks the next input: a recent-change hint or the next catalog entry.
    fn next_input(&mut self) -> Result<Option<Active>> {
        if !self.scanning {
            return Ok(None);
        }
        // Alternate recent-change hints with the catalog scan. The
        // current input finishes first so its in-memory continuation
        // survives; new work need not wait for the whole catalog.
        let changed = if self.background_due {
            None
        } else {
            self.changed_input.take()
        };
        let next = if let Some(id) = changed {
            self.background_due = true;
            self.writer.stored_source(&id)
        } else {
            self.background_due = false;
            let next = self.writer.next_stored_source(&self.after);
            if let Ok(Some(input)) = &next {
                self.after = input.input.id.clone();
            }
            next
        };
        match next {
            Ok(Some(input)) => {
                // Prompt references are not parsed as conversations.
                if input.format == InputFormat::CodexPromptHistory {
                    return Ok(None);
                }
                Ok(Some(Active::Begin(input)))
            }
            Ok(None) => {
                self.scanning = false;
                if self.scan_again {
                    self.restart_scan();
                }
                Ok(None)
            }
            Err(e) => {
                self.recovery_error = Some(diagnostic(e.to_string()));
                self.scanning = false;
                Ok(None)
            }
        }
    }
    fn begin(&mut self, input: StoredSource) -> Result<()> {
        let digest = parser_identity(&input);
        let context_json = parser_context_json(&input)?;
        match self.writer.begin_parser(&Basis {
            input_pk: input.input.input_pk,
            chain: input.input.chain,
            parser_version: parser_version(input.format),
            digest,
            acquired_upto: input.input.length,
            companions_json: Some(&context_json),
            excisions_seen: 0,
        }) {
            Ok(head) => {
                if self.writer.parser_failure(&head.token)?.is_none() {
                    self.active = Some(Active::Recover(input, head));
                }
            }
            Err(e) => {
                self.recovery_error = Some(diagnostic(e.to_string()));
                self.scanning = false;
            }
        }
        Ok(())
    }
    fn recover(&mut self, input: StoredSource, head: Head) -> Result<()> {
        let head = match converge::recover_turn(&mut self.writer, &head.token, install_budget()) {
            Ok(Recovery::Yielded) => {
                self.active = Some(Active::Recover(input, head));
                return Ok(());
            }
            Ok(Recovery::Reparse) => head,
            Ok(Recovery::Published(head)) => head,
            Err(e) => return self.fail(&head, e.to_string()),
        };
        if !caught_up(&head) {
            self.active = Some(Active::Compute(input, head, None));
        } else if can_finalize(&input, &head)
            && crate::source_rows::read(&self.writer.parser_snapshot(&head.token)?, &input)?
                .is_none()
        {
            self.active = Some(Active::Canonical(
                input.clone(),
                head,
                CanonicalPosition::first(input.format),
                None,
            ));
        }
        Ok(())
    }
    fn start_compute(
        &mut self,
        input: StoredSource,
        head: Head,
        continuation: Option<CommittedContinuation>,
    ) -> Result<()> {
        let snapshot = match self.writer.parser_snapshot(&head.token) {
            Ok(s) => s,
            Err(e) => {
                self.fail(&head, e.to_string())?;
                return Ok(());
            }
        };
        let reader = match self.writer.read_handle() {
            Ok(r) => r,
            Err(e) => {
                self.fail(&head, e.to_string())?;
                return Ok(());
            }
        };
        let offset = source_turn::read_offset(&head, continuation.as_ref())?;
        let length = head
            .acquired_upto
            .checked_sub(offset)
            .ok_or(StoreError::Range)?
            .min(WINDOW_BYTES);
        let hash = input.binding;
        let input_id = input.input.id.clone();
        let input_chain = input.input.chain;
        let input_for_history = input.clone();
        let history = input.format == InputFormat::CodexHistorySqlite;
        let context = if history {
            SourceContext::Codex
        } else {
            parser_context(&input)?
        };
        let cancelled = Arc::clone(&self.inner.cancelled);
        let sender = self.inner.sender.clone();
        let join = thread::Builder::new()
            .name("polylogue-source".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Computation> {
                        if cancelled.load(Ordering::Acquire) {
                            return Err(StoreError::Cancelled.into());
                        }
                        if history {
                            return compute_history(&reader, &input_for_history, &cancelled);
                        }
                        let bytes = reader.read_source_range_checked(
                            &input_id,
                            input_chain,
                            hash,
                            offset,
                            length,
                            || {
                                if cancelled.load(Ordering::Acquire) {
                                    Err(StoreError::Cancelled)
                                } else {
                                    Ok(())
                                }
                            },
                        )?;
                        if cancelled.load(Ordering::Acquire) {
                            return Err(StoreError::Cancelled.into());
                        }
                        for body in input_for_history
                            .companions
                            .iter()
                            .filter(|body| !body.inline_duplicate)
                        {
                            reader.claude_body(body)?;
                        }
                        Ok(source_turn::compute_with_bodies(
                            &snapshot,
                            continuation,
                            Window {
                                offset,
                                bytes: &bytes,
                            },
                            context,
                            source_budgets(),
                            &input_for_history.companions,
                        )?)
                    },
                ))
                .map_err(|_| Error::WorkerPanicked)
                .and_then(|r| r);
                let _ = sender.send(Message::Computed(result));
            });
        match join {
            Ok(join) => self.compute = Some(ComputeJob { input, head, join }),
            Err(e) => self.fail(&head, e.to_string())?,
        }
        Ok(())
    }
    fn start_selection(
        &mut self,
        input: StoredSource,
        head: Head,
        position: CanonicalPosition,
        selection: polylogue_store::contributions::CompatibleSelection,
    ) -> Result<()> {
        let reader = self.take_canonical_reader()?;
        let cancelled = Arc::clone(&self.inner.cancelled);
        let sender = self.inner.sender.clone();
        let join = thread::Builder::new()
            .name("polylogue-compatible-source".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    selection.advance(&reader.values, 65536, &cancelled)
                }))
                .map_err(|_| Error::WorkerPanicked)
                .and_then(|r| r.map_err(Error::Store))
                .map(CanonicalTurn::Selected);
                let _ = sender.send(Message::CanonicalComputed(result));
                Some(reader)
            });
        match join {
            Ok(join) => {
                self.canonical = Some(CanonicalJob {
                    input,
                    head,
                    position,
                    handle: None,
                    join,
                })
            }
            Err(e) => self.fail(&head, e.to_string())?,
        }
        Ok(())
    }
    fn start_canonical(
        &mut self,
        input: StoredSource,
        head: Head,
        position: CanonicalPosition,
        handle: Option<canonical_message::Handle>,
    ) -> Result<()> {
        let CanonicalReaders {
            values: reader,
            parser,
        } = self.take_canonical_reader()?;
        let snapshot = parser.snapshot(&head.token)?;
        let state = handle
            .as_ref()
            .map(|h| canonical_message::inspect(&mut self.writer, h, 65536))
            .transpose();
        let state = match state {
            Ok(s) => s,
            Err(e) => {
                self.fail(&head, format!("canonical inspect: {e}"))?;
                return Ok(());
            }
        };
        let logical_key = position.logical_key()?;
        let current = if state.is_none() {
            logical_key
                .as_ref()
                .map(|key| {
                    self.writer
                        .current_source_projection(&head.token, key, 65536)
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        let input_for_audit = input.clone();
        let source_binding = input.binding;
        let context = input.context.clone();
        let work_position = position.clone();
        let cancelled = Arc::clone(&self.inner.cancelled);
        let sender = self.inner.sender.clone();
        let join = thread::Builder::new()
            .name("polylogue-canonical".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<CanonicalTurn> {
                        if cancelled.load(Ordering::Acquire) {
                            return Err(StoreError::Cancelled.into());
                        }
                        if let CanonicalPosition::Audit(audit) = work_position {
                            return Ok(CanonicalTurn::Audited(audit.advance(
                                &snapshot,
                                &input_for_audit,
                                source_budgets().reads,
                            )?));
                        }
                        let tools_after = match &work_position {
                            CanonicalPosition::ToolsAfter(after)
                            | CanonicalPosition::ToolsBeforeFailure(after, _) => {
                                Some(after.clone())
                            }
                            CanonicalPosition::Codex(ordinal) => {
                                let attempt = crate::state::codex::Attempt::new(
                                    &snapshot,
                                    source_budgets().reads,
                                );
                                let resume = attempt.resume().map_err(Error::InvalidEvidence)?;
                                (*ordinal >= resume.message_count()).then_some(None)
                            }
                            _ => None,
                        };
                        if let Some(after) = tools_after {
                            use polylogue_sources::codex::keyed::{Cell, Key, ReadView};
                            let attempt = crate::state::codex::Attempt::new(
                                &snapshot,
                                source_budgets().reads,
                            );
                            return match attempt
                                .next(&Key::Operation(after.unwrap_or_default()))
                                .map_err(Error::InvalidEvidence)?
                            {
                                Some((Key::Operation(id), Cell::Operation(operation))) => {
                                    let digest = operation_digest(
                                        &reader,
                                        &input_for_audit,
                                        &operation,
                                        &cancelled,
                                    )?;
                                    Ok(CanonicalTurn::Operation(id, operation, digest))
                                }
                                None => Ok(CanonicalTurn::Done),
                                _ => Err(Error::InvalidRequest("invalid native operation cell")),
                            };
                        }
                        let action = if let Some(state) = state {
                            canonical_message::compute(
                                &snapshot,
                                &reader,
                                &state,
                                canonical_budgets(),
                                &cancelled,
                            )
                            .map_err(Error::Canonical)?
                        } else if let CanonicalPosition::ClaudeAfter(after) = &work_position {
                            let view =
                                crate::state::claude::View::new(&snapshot, source_budgets().reads)?;
                            return Ok(match view.next_pending(after.as_ref())? {
                                Some((cursor, pending)) => CanonicalTurn::Pending(cursor, pending),
                                None => CanonicalTurn::Done,
                            });
                        } else if let CanonicalPosition::ClaudeMessage(_, pending) = &work_position
                        {
                            canonical_message::start_claude(
                                &snapshot,
                                &canonical_message::ClaudeRequest {
                                    pending: pending.clone(),
                                    source_binding,
                                    context,
                                    revision: input_for_audit.revision,
                                    logical_key: logical_key.ok_or(Error::InvalidRequest(
                                        "missing Claude logical key",
                                    ))?,
                                },
                                current,
                                source_budgets().reads,
                                65536,
                            )
                            .map_err(Error::Canonical)?
                        } else {
                            let CanonicalPosition::Codex(ordinal) = work_position else {
                                return Err(Error::InvalidRequest("invalid canonical position"));
                            };
                            let attempt = crate::state::codex::Attempt::new(
                                &snapshot,
                                source_budgets().reads,
                            );
                            let resume = attempt.resume();
                            if let Some(e) = attempt.take_failure() {
                                return Err(e.into());
                            }
                            let resume = resume.map_err(|e| {
                                Error::Canonical(canonical_message::Error::Source(e))
                            })?;
                            if ordinal >= resume.message_count() {
                                return Ok(CanonicalTurn::Done);
                            }
                            canonical_message::start(
                                &snapshot,
                                &canonical_message::Request {
                                    message: ordinal,
                                    revision: input_for_audit.revision,
                                    logical_key: logical_key.ok_or(Error::InvalidRequest(
                                        "missing Codex logical key",
                                    ))?,
                                    source_binding,
                                },
                                current,
                                source_budgets().reads,
                                65536,
                            )
                            .map_err(Error::Canonical)?
                        };
                        Ok(CanonicalTurn::Action(action))
                    },
                ))
                .map_err(|_| Error::WorkerPanicked)
                .and_then(|r| r);
                // Release the snapshot before the actor applies the
                // result. Reuse keeps connections, not a read view.
                let (result, readers) = match snapshot.into_reader() {
                    Ok(parser) => (
                        result,
                        Some(CanonicalReaders {
                            values: reader,
                            parser,
                        }),
                    ),
                    Err(error) => (Err(error.into()), None),
                };
                let _ = sender.send(Message::CanonicalComputed(result));
                readers
            });
        match join {
            Ok(join) => {
                self.canonical = Some(CanonicalJob {
                    input,
                    head,
                    position,
                    handle,
                    join,
                })
            }
            Err(e) => self.fail(&head, e.to_string())?,
        }
        Ok(())
    }
    fn install_history(
        &mut self,
        input: StoredSource,
        head: Head,
        mut install: PendingInstall,
    ) -> Result<()> {
        match install.turn(&mut self.writer, install_budget()) {
            Ok(Turn::Yielded) => self.active = Some(Active::HistoryInstall(input, head, install)),
            Ok(Turn::Published(published)) => {
                self.active = Some(Active::Canonical(
                    input,
                    published,
                    CanonicalPosition::ToolsAfter(None),
                    None,
                ))
            }
            Err(e) => self.fail(&head, e.to_string())?,
        }
        Ok(())
    }
    fn install(
        &mut self,
        input: StoredSource,
        head: Head,
        mut install: PendingInstall,
        guard: CommitGuard,
    ) -> Result<()> {
        match install.turn(&mut self.writer, install_budget()) {
            Ok(Turn::Yielded) => self.active = Some(Active::Install(input, head, install, guard)),
            Ok(Turn::Published(published)) => match guard.published(&published) {
                Ok(continuation) => {
                    if !caught_up(&published) {
                        self.active = Some(Active::Compute(input, published, Some(continuation)));
                    } else if can_finalize(&input, &published) {
                        self.active = Some(Active::Canonical(
                            input.clone(),
                            published,
                            CanonicalPosition::first(input.format),
                            None,
                        ));
                    }
                }
                Err(e) => self.fail(&published, e.to_string())?,
            },
            Err(e) => self.fail(&head, e.to_string())?,
        }
        Ok(())
    }
}
fn operation_digest(
    reader: &polylogue_store::ReadHandle,
    input: &StoredSource,
    operation: &polylogue_model::tool_item::Operation,
    cancelled: &AtomicBool,
) -> Result<BlobHash> {
    use polylogue_model::{tool_item::ToolOrigin, version_evidence::Framer};
    if let [ToolOrigin::History { row_hash, .. }] = operation.origins.as_slice() {
        return Ok(BlobHash::from_bytes(*row_hash));
    }
    let mut units = Vec::new();
    for origin in &operation.origins {
        if cancelled.load(Ordering::Acquire) {
            return Err(StoreError::Cancelled.into());
        }
        let ToolOrigin::Rollout {
            start,
            end,
            raw_hash,
            tag,
        } = origin
        else {
            return Err(Error::InvalidRequest(
                "mixed representation source candidate",
            ));
        };
        let bytes = reader.read_source_range_checked(
            &input.input.id,
            input.input.chain,
            input.binding,
            *start,
            end - start,
            || {
                if cancelled.load(Ordering::Acquire) {
                    Err(StoreError::Cancelled)
                } else {
                    Ok(())
                }
            },
        )?;
        let canonical =
            polylogue_model::identity::canonical_record(&bytes, &["timestamp", "ordinal"])
                .map_err(|e| Error::InvalidEvidence(e.to_string()))?;
        if blake3::hash(&canonical).as_bytes() != raw_hash {
            return Err(Error::InvalidEvidence(
                "operation raw evidence hash differs".into(),
            ));
        }
        units.push((tag.as_str(), canonical));
    }
    let mut frame = if units.len() == 1 {
        Framer::single_whole_record(units[0].1.len() as u64, u64::MAX)
    } else {
        Framer::compound(units.len() as u64, u64::MAX)
            .map_err(|e| Error::InvalidEvidence(e.to_string()))?
    };
    for (tag, bytes) in &units {
        if units.len() != 1 {
            frame
                .begin_json(tag.len() as u64, bytes.len() as u64)
                .map_err(|e| Error::InvalidEvidence(e.to_string()))?;
            frame
                .tag(tag)
                .map_err(|e| Error::InvalidEvidence(e.to_string()))?;
        }
        frame
            .json(bytes)
            .map_err(|e| Error::InvalidEvidence(e.to_string()))?;
    }
    frame
        .finish()
        .map_err(|e| Error::InvalidEvidence(e.to_string()))
}

fn compute_history(
    reader: &polylogue_store::ReadHandle,
    input: &StoredSource,
    cancelled: &AtomicBool,
) -> Result<Computation> {
    use polylogue_sources::codex::keyed::{Cell, Key};
    use polylogue_store::parser_state::{Accounting, CellChange, Phase, Progress};
    let rows = reader.codex_history_rows(input, cancelled)?;
    let mut counts = Accounting::default();
    let mut changes = Vec::new();
    for row in rows {
        counts.framed += 1;
        match codex::history::map(&row) {
            Ok(Some(operation)) => {
                let key = Key::Operation(format!(
                    "1:{}",
                    operation
                        .id()
                        .map_err(|e| Error::InvalidEvidence(e.to_string()))?
                ));
                changes.push(CellChange {
                    namespace: format!("codex.{}", key.family()),
                    key: key.ordered_bytes(),
                    value: Some(
                        serde_json::to_vec(&Cell::Operation(operation))
                            .map_err(|e| Error::InvalidEvidence(e.to_string()))?,
                    ),
                });
                counts.used += 1;
            }
            Ok(None) => counts.unrecognized += 1,
            Err(_) => counts.undecodable += 1,
        }
    }
    Ok(Computation::History {
        progress: Progress {
            parsed_upto: input.input.length,
            inspected_upto: input.input.length,
            phase: Phase::Finalizing,
            finalization_pending: true,
            counts,
        },
        changes,
    })
}

fn can_finalize(input: &StoredSource, head: &Head) -> bool {
    head.status == Phase::Finalizing
        || (input.observed()
            && input.format == InputFormat::ClaudeCodeJsonl
            && head.status == Phase::WaitingTail)
}
fn caught_up(head: &Head) -> bool {
    matches!(head.status, Phase::WaitingTail | Phase::Done)
        || (head.status == Phase::Finalizing && head.parsed_upto == head.acquired_upto)
}
pub(crate) fn parser_version(format: InputFormat) -> u32 {
    match format {
        InputFormat::CodexRollout => codex::PARSER_VERSION,
        InputFormat::CodexHistorySqlite => codex::history::PARSER_VERSION,
        InputFormat::CodexPromptHistory => 1,
        InputFormat::ClaudeCodeJsonl => claude_code::PARSER_VERSION,
        InputFormat::ClaudeCodeToolResult => 0,
    }
}
pub(crate) fn parser_context(input: &StoredSource) -> Result<SourceContext> {
    let parts = input.context.iter().map(String::as_str).collect::<Vec<_>>();
    input.format.validate_context(&parts)?;
    match (input.format, parts.as_slice()) {
        (InputFormat::CodexRollout, []) => Ok(SourceContext::Codex),
        (InputFormat::ClaudeCodeJsonl, ["claude-main", stem]) => {
            Ok(SourceContext::Claude(claude_code::FileCtx::main(stem)))
        }
        (InputFormat::ClaudeCodeJsonl, ["claude-subagent", agent]) => {
            Ok(SourceContext::Claude(claude_code::FileCtx::Subagent {
                folder_owner: String::new(),
                agent: (*agent).into(),
                meta: None,
            }))
        }
        (InputFormat::ClaudeCodeJsonl, ["claude-subagent", agent, owner]) => {
            Ok(SourceContext::Claude(claude_code::FileCtx::Subagent {
                folder_owner: (*owner).into(),
                agent: (*agent).into(),
                meta: None,
            }))
        }
        _ => Err(Error::InvalidRequest("unsupported source context")),
    }
}
fn parser_context_json(input: &StoredSource) -> Result<String> {
    if input.format == InputFormat::CodexRollout {
        return Ok(CONTEXT.into());
    }
    #[derive(serde::Serialize)]
    struct Context<'a> {
        format: &'static str,
        context: &'a [String],
        companions: &'a [polylogue_store::claude_bodies::Companion],
    }
    serde_json::to_string(&Context {
        format: input.format.name(),
        context: &input.context,
        companions: &input.companions,
    })
    .map_err(|_| Error::InvalidRequest("source context serialization failed"))
}
pub(crate) fn parser_identity(input: &StoredSource) -> [u8; 32] {
    if input.format == InputFormat::CodexRollout {
        return basis(&input.binding);
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue.claude.immutable.parser-basis.v1\0");
    hash.update(&parser_version(input.format).to_be_bytes());
    hash.update(&crate::source_rows::RECIPE.to_be_bytes());
    hash.update(input.binding.bytes());
    // The context is part of the input's identity; length-prefix each part.
    hash.update(&(input.context.len() as u64).to_be_bytes());
    for item in &input.context {
        hash.update(&(item.len() as u64).to_be_bytes());
        hash.update(item.as_bytes());
    }
    if !input.companions.is_empty() {
        hash.update(
            &serde_json::to_vec(&input.companions).expect("companion manifest is serializable"),
        );
    }
    *hash.finalize().as_bytes()
}
fn basis(hash: &BlobHash) -> [u8; 32] {
    let mut bytes = b"polylogue.codex.immutable.parser-basis.v1\0".to_vec();
    bytes.extend_from_slice(&codex::PARSER_VERSION.to_be_bytes());
    bytes.extend_from_slice(&crate::source_rows::RECIPE.to_be_bytes());
    bytes.extend_from_slice(&crate::state::codex::STATE_LAYOUT_VERSION.to_be_bytes());
    bytes.extend_from_slice(hash.bytes());
    bytes.extend_from_slice(CONTEXT.as_bytes());
    *BlobHash::of(&bytes).bytes()
}
fn source_budgets() -> Budgets {
    Budgets {
        framing: StepBudget {
            bytes: WINDOW_BYTES as usize,
            // Let tiny ignored records share one 64 KiB read instead of a
            // worker turn and install every 32 lines.
            records: 1024,
            record_bytes: 1024 * 1024,
        },
        reads: Budget {
            cells: 65536,
            encoded_bytes: 16 * 1024 * 1024,
        },
        encoding: Budget {
            cells: 65536,
            encoded_bytes: 16 * 1024 * 1024,
        },
    }
}
fn install_budget() -> Budget {
    Budget {
        // Small cells share a transaction. The actor checks cancellation and
        // serves its mailbox between transactions.
        cells: 64,
        encoded_bytes: 2 * 1024 * 1024,
    }
}
fn diagnostic(mut text: String) -> String {
    if text.len() > DETAIL_BYTES {
        let marker = " [diagnostic truncated]";
        let mut end = DETAIL_BYTES - marker.len();
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(marker);
    }
    text
}

pub(crate) fn claude_key(message: &str) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue.claude.logical-message.v1\0");
    hash.update(&(message.len() as u64).to_be_bytes());
    hash.update(message.as_bytes());
    *hash.finalize().as_bytes()
}

pub(crate) fn canonical_key(ordinal: usize) -> Result<[u8; 32]> {
    let ordinal = u64::try_from(ordinal).map_err(|_| StoreError::Range)?;
    let mut bytes = b"polylogue.codex.logical-message.v1\0".to_vec();
    bytes.extend_from_slice(&ordinal.to_be_bytes());
    Ok(*BlobHash::of(&bytes).bytes())
}
fn canonical_budgets() -> canonical_message::Budgets {
    canonical_message::Budgets {
        read: source_budgets().reads,
        metadata: 65536,
        value_chunk: 65536,
        hash_bytes: 65536,
    }
}

#[cfg(test)]
mod tests;
