//! Single Writer actor for retained sources and provider-state progress
//! plus compatible source Message materialization through one shared core.
//! Capture, provider interpretation and canonical encoding run outside the actor.
//! Search, contested merges and complete history coverage remain pending.
use crate::{
    canonical_message,
    converge::{self, PendingInstall, Recovery, Turn},
    source_turn::{
        self, Budgets, CommitGuard, CommittedContinuation, Computation, SourceContext, Window,
    },
};
use polylogue_model::BlobHash;
use polylogue_sources::{claude_code, codex, frame::StepBudget};
use polylogue_store::{
    Error as StoreError, ImportObservation, InputFormat, PreparedBlob, RetainedInput,
    RetainedSource, Writer,
    parser_state::{Basis, Budget, Head},
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
// These are admission limits, not format limits or a hard RSS bound. A rejected
// record/cell remains owed and its required capacity is kept in the diagnostic.
// Configurable larger admissions and full canonical convergence remain open.
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
    pub recovery: Option<polylogue_model::RecoveryQualification>,
}
#[derive(Debug)]
pub struct Progress {
    pub input: RetainedInput,
    pub parser: Option<Head>,
    pub failure: Option<String>,
    pub recovery_error: Option<String>,
    pub active: bool,
    pub source_rows: polylogue_model::settlement::SourceRows,
}
enum Message {
    Note(
        polylogue_model::authored::WriteNote,
        Instant,
        Reply<AuthoredReply>,
    ),
    Import(ImportRequest, Reply<RetainedInput>),
    Progress(String, Reply<Progress>),
    Retry(String, Reply<()>),
    Captured(Result<PreparedBlob>),
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
    pub fn try_import(&self, request: ImportRequest) -> Result<Receiver<Result<RetainedInput>>> {
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
        if required > REQUEST_BYTES || request.metadata_allowance > REQUEST_BYTES {
            return Err(Error::RequestTooLarge {
                required: required.max(request.metadata_allowance),
                maximum: REQUEST_BYTES,
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
        let (reply, rx) = mpsc::sync_channel(1);
        self.send(Message::Import(request, reply))?;
        Ok(rx)
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
    format: InputFormat,
    context: Vec<String>,
    observation: Observation,
    allowance: usize,
    reply: Reply<RetainedInput>,
    join: JoinHandle<()>,
}
struct ComputeJob {
    input: RetainedSource,
    head: Head,
    join: JoinHandle<()>,
}
// Keyset order schedules obligations only; source ordinals remain in the
// provider's PendingMessage and do not derive from this traversal.
#[derive(Clone)]
pub(crate) enum CanonicalPosition {
    Audit(Box<crate::source_rows::Audit>),
    Codex(usize),
    ToolsAfter(Option<String>),
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
            InputFormat::CodexHistorySqlite => Self::ToolsAfter(None),
            InputFormat::ClaudeCodeJsonl => Self::ClaudeAfter(None),
        }
    }
    fn next(self) -> Result<Self> {
        match self {
            Self::Codex(n) => Ok(Self::Codex(n.checked_add(1).ok_or(StoreError::Range)?)),
            Self::ClaudeMessage(cursor, _) => Ok(Self::ClaudeAfter(Some(cursor))),
            Self::ToolsAfter(_) | Self::Audit(_) | Self::ClaudeAfter(_) => {
                Err(Error::InvalidRequest("no current Claude Message"))
            }
        }
    }
    pub(crate) fn logical_key(&self) -> Result<Option<[u8; 32]>> {
        match self {
            Self::Codex(n) => canonical_key(*n).map(Some),
            Self::ClaudeMessage(_, p) => {
                let mut h = blake3::Hasher::new();
                h.update(b"polylogue.claude.logical-message.v1\0");
                h.update(&(p.message.as_str().len() as u64).to_be_bytes());
                h.update(p.message.as_str().as_bytes());
                Ok(Some(*h.finalize().as_bytes()))
            }
            Self::ToolsAfter(_) | Self::Audit(_) | Self::ClaudeAfter(_) => Ok(None),
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
    input: RetainedSource,
    head: Head,
    position: CanonicalPosition,
    handle: Option<canonical_message::Handle>,
    join: JoinHandle<()>,
}
enum Active {
    Selecting(
        RetainedSource,
        Head,
        CanonicalPosition,
        polylogue_store::contributions::CompatibleSelection,
    ),
    Begin(RetainedSource),
    Recover(RetainedSource, Head),
    Compute(RetainedSource, Head, Option<CommittedContinuation>),
    Install(RetainedSource, Head, PendingInstall, CommitGuard),
    HistoryInstall(RetainedSource, Head, PendingInstall),
    Canonical(
        RetainedSource,
        Head,
        CanonicalPosition,
        Option<canonical_message::Handle>,
    ),
}
struct Actor {
    writer: Writer,
    inner: Arc<Inner>,
    capture: Option<CaptureJob>,
    compute: Option<ComputeJob>,
    canonical: Option<CanonicalJob>,
    active: Option<Active>,
    after: String,
    scanning: bool,
    scan_again: bool,
    recovery_error: Option<String>,
    stopping: bool,
    fatal: Option<Error>,
}
impl Actor {
    fn new(writer: Writer, inner: Arc<Inner>) -> Self {
        Self {
            writer,
            inner,
            capture: None,
            compute: None,
            canonical: None,
            active: None,
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
    }
    fn run(mut self, rx: Receiver<Message>) -> Result<()> {
        loop {
            if self.inner.cancelled.load(Ordering::Acquire) && !self.stopping {
                self.begin_stop();
            }
            let immediate = !self.stopping
                && self.compute.is_none()
                && self.canonical.is_none()
                && (self.active.is_some() || self.scanning);
            let message = match rx.try_recv() {
                Ok(m) => Some(m),
                Err(TryRecvError::Disconnected) => {
                    self.begin_stop();
                    None
                }
                Err(TryRecvError::Empty) if immediate => None,
                Err(TryRecvError::Empty) => match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(m) => Some(m),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => {
                        self.begin_stop();
                        None
                    }
                },
            };
            if let Some(message) = message {
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
            if self.stopping
                && self.capture.is_none()
                && self.compute.is_none()
                && self.canonical.is_none()
            {
                // Closing accepting under its mutex precedes this drain; no
                // request accepted before shutdown can lose its explicit reply.
                while let Ok(message) = rx.try_recv() {
                    let _ = self.message(message);
                }
                return self.fatal.map_or(Ok(()), Err);
            }
        }
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
        match message {
            Message::Note(request, deadline, reply) => {
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
            Message::Import(request, reply) => {
                if self.stopping {
                    let _ = reply.send(Err(Error::Stopped));
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
                        self.scan_again = true;
                        if !self.scanning
                            && self.active.is_none()
                            && self.compute.is_none()
                            && self.canonical.is_none()
                        {
                            self.restart_scan();
                        }
                        let _ = reply.send(Ok(input));
                        return Ok(());
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e.into()));
                        return Ok(());
                    }
                    Ok(None) => {}
                }
                if self.capture.is_some() {
                    let _ = reply.send(Err(Error::Busy));
                    return Ok(());
                }
                if let Err(e) = self.writer.check_import_acquisition() {
                    let _ = reply.send(Err(e.into()));
                    return Ok(());
                }
                let capture = match self.writer.blob_capture() {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = reply.send(Err(e.into()));
                        return Ok(());
                    }
                };
                let cancelled = Arc::clone(&self.inner.cancelled);
                let sender = self.inner.sender.clone();
                let path = request.path;
                let history = request.format == InputFormat::CodexHistorySqlite;
                let join = thread::Builder::new()
                    .name("polylogue-capture".into())
                    .spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if history {
                                capture.capture_history_snapshot(&path, &cancelled)
                            } else {
                                capture.capture_regular_file(&path, &cancelled)
                            }
                        }))
                        .map_err(|_| Error::WorkerPanicked)
                        .and_then(|r| r.map_err(Error::Store));
                        let _ = sender.send(Message::Captured(result));
                    });
                match join {
                    Ok(join) => {
                        self.capture = Some(CaptureJob {
                            format: request.format,
                            context: request.context,
                            observation: request.observation,
                            allowance: request.metadata_allowance,
                            reply,
                            join,
                        })
                    }
                    Err(e) => {
                        let _ = reply.send(Err(StoreError::Io(e).into()));
                    }
                }
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
                job.join.join().map_err(|_| Error::WorkerPanicked)?;
                if self.stopping {
                    let _ = job.reply.send(Err(Error::Stopped));
                    return Ok(());
                }
                let result = result.and_then(|proof| {
                    let hash = self.writer.retain_captured_blob(proof)?;
                    Ok(self.writer.retain_immutable_input(
                        hash,
                        job.format,
                        &job.context.iter().map(String::as_str).collect::<Vec<_>>(),
                        &job.observation.borrowed(),
                        job.allowance,
                    )?)
                });
                if result.is_ok() {
                    self.scan_again = true;
                    if !self.scanning
                        && self.active.is_none()
                        && self.compute.is_none()
                        && self.canonical.is_none()
                    {
                        self.restart_scan();
                    }
                }
                let _ = job.reply.send(result);
            }
            Message::Computed(result) => {
                let Some(job) = self.compute.take() else {
                    return Err(Error::InvalidRequest("unexpected source completion"));
                };
                job.join.join().map_err(|_| Error::WorkerPanicked)?;
                if self.stopping {
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
                            Ok(p) => {
                                self.active = Some(Active::Install(job.input, job.head, p, guard))
                            }
                            Err(e) => self.fail(&job.head, e.to_string())?,
                        }
                    }
                    Ok(Computation::History { progress, changes }) => {
                        match PendingInstall::begin(&mut self.writer, &job.head, progress, changes)
                        {
                            Ok(install) => {
                                self.active =
                                    Some(Active::HistoryInstall(job.input, job.head, install))
                            }
                            Err(e) => self.fail(&job.head, e.to_string())?,
                        }
                    }
                    Ok(Computation::Idle { .. }) => {}
                    Err(e) => self.fail(&job.head, e.to_string())?,
                }
            }
            Message::CanonicalComputed(result) => {
                let job = self
                    .canonical
                    .take()
                    .ok_or(Error::InvalidRequest("unexpected canonical completion"))?;
                job.join.join().map_err(|_| Error::WorkerPanicked)?;
                if self.stopping {
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
                                    CanonicalPosition::ToolsAfter(Some(id)),
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
                                    Err(e) => {
                                        self.fail(&job.head, format!("canonical selection: {e}"))?
                                    }
                                }
                            }
                            Err(e) => {
                                self.fail(&job.head, format!("canonical {operation}: {e}"))?
                            }
                        }
                    }
                    Ok(CanonicalTurn::Selected(turn)) => {
                        use polylogue_store::contributions::CompatibilityTurn;
                        match turn {
                            CompatibilityTurn::Progress(selection) => self.active=Some(Active::Selecting(job.input,job.head,job.position,selection)),
                            CompatibilityTurn::Certified(selection,certificate) => {
                                match self.writer.certify_body_compatibility(certificate) {
                                    Ok(())=>self.active=Some(Active::Selecting(job.input,job.head,job.position,selection)),
                                    Err(e)=>self.fail(&job.head,format!("canonical compatibility: {e}"))?,
                                }
                            },
                            CompatibilityTurn::Ready(selection) => {
                                match self.writer.install_compatible_source_selection(selection,65536) {
                                    Ok(_)=>self.active=Some(Active::Canonical(job.input,job.head,job.position.next()?,None)),
                                    Err(e)=>self.fail(&job.head,format!("canonical selection: {e}"))?,
                                }
                            },
                            CompatibilityTurn::Contested=>self.fail(&job.head,"canonical Message has incompatible or unsupported input candidates; merge remains pending".into())?,
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
                    Ok(CanonicalTurn::Done) if job.head.status == "waiting_tail" => {
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
                        receipt
                            .validate(&job.head.token, &crate::source_rows::basis(&job.input))?;
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
                    Err(e) => self.fail(&job.head, format!("canonical compute: {e}"))?,
                }
            }
            Message::Wake => {}
        }
        Ok(())
    }
    fn progress(&mut self, id: &str) -> Result<Progress> {
        let input = self
            .writer
            .retained_source(id)?
            .ok_or(Error::UnknownInput)?;
        let parser = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?;
        let failure = match &parser {
            Some(head) => self.writer.parser_failure(&head.token)?,
            None => None,
        };
        let obligation = match parser.as_ref().map(|h| h.status.as_str()) {
            None => "source_parser_admission",
            Some("waiting_tail") => "unfinished_source_record",
            Some("parsing") => "source_record_interpretation",
            _ => "canonical_source_row_effects_and_audit",
        };
        let mut source_rows = crate::source_rows::pending(&input, obligation);
        if let Some(head) = &parser {
            if head.acquired_upto != input.input.length
                || head.token.basis != parser_basis(&input)
                || head.parser_version != parser_version(input.format)
            {
                source_rows =
                    crate::source_rows::pending(&input, "current_interpretation_admission");
            } else if let Some(reason) = &failure {
                // A failed admitted install may still own unsealed work. Its
                // retained diagnostic is unavailability, never a waiting tail.
                source_rows.state = polylogue_model::settlement::SourceRowState::Unavailable {
                    usable_result: false,
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
        let input = self
            .writer
            .retained_source(id)?
            .ok_or(Error::UnknownInput)?;
        if let Some(head) = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?
        {
            self.writer
                .record_parser_failure(&head.token, None, DETAIL_BYTES)?;
        }
        self.scan_again = true;
        self.recovery_error = None;
        if self.active.is_none() && self.compute.is_none() && self.canonical.is_none() {
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
        let details = diagnostic(details);
        self.writer
            .record_parser_failure(&head.token, Some(&details), DETAIL_BYTES)?;
        Ok(())
    }
    fn tick(&mut self) -> Result<()> {
        if self.compute.is_some() || self.canonical.is_some() {
            return Ok(());
        }
        let active = match self.active.take() {
            Some(a) => a,
            None => {
                if !self.scanning {
                    return Ok(());
                }
                match self.writer.next_retained_source(&self.after) {
                    Ok(Some(input)) => {
                        self.after = input.input.id.clone();
                        Active::Begin(input)
                    }
                    Ok(None) => {
                        self.scanning = false;
                        if self.scan_again {
                            self.restart_scan();
                        }
                        return Ok(());
                    }
                    Err(e) => {
                        self.recovery_error = Some(diagnostic(e.to_string()));
                        self.scanning = false;
                        return Ok(());
                    }
                }
            }
        };
        match active {
            Active::Begin(input) => {
                let digest = parser_basis(&input);
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
            }
            Active::Recover(input, head) => {
                match converge::recover_turn(&mut self.writer, &head.token, install_budget()) {
                    Ok(Recovery::Yielded) => self.active = Some(Active::Recover(input, head)),
                    Ok(Recovery::Reparse) => {
                        if !settled(&head) {
                            self.active = Some(Active::Compute(input, head, None));
                        } else if can_finalize(&input, &head)
                            && crate::source_rows::read(
                                &self.writer.parser_snapshot(&head.token)?,
                                &input,
                            )?
                            .is_none()
                        {
                            self.active = Some(Active::Canonical(
                                input.clone(),
                                head,
                                CanonicalPosition::first(input.format),
                                None,
                            ));
                        }
                    }
                    Ok(Recovery::Published(head)) => {
                        if !settled(&head) {
                            self.active = Some(Active::Compute(input, head, None));
                        } else if can_finalize(&input, &head)
                            && crate::source_rows::read(
                                &self.writer.parser_snapshot(&head.token)?,
                                &input,
                            )?
                            .is_none()
                        {
                            self.active = Some(Active::Canonical(
                                input.clone(),
                                head,
                                CanonicalPosition::first(input.format),
                                None,
                            ));
                        }
                    }
                    Err(e) => self.fail(&head, e.to_string())?,
                }
            }
            Active::Compute(input, head, continuation) => {
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
                                    return compute_history(
                                        &reader,
                                        &input_for_history,
                                        &cancelled,
                                    );
                                }
                                let bytes = reader.read_source_range(
                                    &input_id,
                                    input_chain,
                                    hash,
                                    offset,
                                    length,
                                )?;
                                if cancelled.load(Ordering::Acquire) {
                                    return Err(StoreError::Cancelled.into());
                                }
                                Ok(source_turn::compute(
                                    &snapshot,
                                    continuation,
                                    Window {
                                        offset,
                                        bytes: &bytes,
                                    },
                                    context,
                                    source_budgets(),
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
            }
            Active::Selecting(input, head, position, selection) => {
                let reader = self.writer.read_handle()?;
                let cancelled = Arc::clone(&self.inner.cancelled);
                let sender = self.inner.sender.clone();
                let join = thread::Builder::new()
                    .name("polylogue-compatible-source".into())
                    .spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            selection.advance(&reader, 65536, &cancelled)
                        }))
                        .map_err(|_| Error::WorkerPanicked)
                        .and_then(|r| r.map_err(Error::Store))
                        .map(CanonicalTurn::Selected);
                        let _ = sender.send(Message::CanonicalComputed(result));
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
            }
            Active::Canonical(input, head, position, handle) => {
                let snapshot = self.writer.parser_snapshot(&head.token)?;
                let reader = self.writer.read_handle()?;
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
                                    CanonicalPosition::ToolsAfter(after) => Some(after.clone()),
                                    CanonicalPosition::Codex(ordinal) => {
                                        let attempt = crate::state::codex::Attempt::new(
                                            &snapshot,
                                            source_budgets().reads,
                                        );
                                        let resume =
                                            attempt.resume().map_err(Error::InvalidEvidence)?;
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
                                        _ => Err(Error::InvalidRequest(
                                            "invalid native operation cell",
                                        )),
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
                                } else if let CanonicalPosition::ClaudeAfter(after) = &work_position
                                {
                                    let view = crate::state::claude::View::new(
                                        &snapshot,
                                        source_budgets().reads,
                                    )?;
                                    return Ok(match view.next_pending(after.as_ref())? {
                                        Some((cursor, pending)) => {
                                            CanonicalTurn::Pending(cursor, pending)
                                        }
                                        None => CanonicalTurn::Done,
                                    });
                                } else if let CanonicalPosition::ClaudeMessage(_, pending) =
                                    &work_position
                                {
                                    canonical_message::start_claude(
                                        &snapshot,
                                        &canonical_message::ClaudeRequest {
                                            pending: pending.clone(),
                                            source_binding,
                                            context,
                                            revision: input_for_audit.revision,
                                            logical_key: logical_key.ok_or(
                                                Error::InvalidRequest("missing Claude logical key"),
                                            )?,
                                        },
                                        current,
                                        source_budgets().reads,
                                        65536,
                                    )
                                    .map_err(Error::Canonical)?
                                } else {
                                    let CanonicalPosition::Codex(ordinal) = work_position else {
                                        return Err(Error::InvalidRequest(
                                            "invalid canonical position",
                                        ));
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
                                            logical_key: logical_key.ok_or(
                                                Error::InvalidRequest("missing Codex logical key"),
                                            )?,
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
                        let _ = sender.send(Message::CanonicalComputed(result));
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
            }
            Active::HistoryInstall(input, head, mut install) => {
                match install.turn(&mut self.writer, install_budget()) {
                    Ok(Turn::Yielded) => {
                        self.active = Some(Active::HistoryInstall(input, head, install))
                    }
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
            }
            Active::Install(input, head, mut install, guard) => {
                match install.turn(&mut self.writer, install_budget()) {
                    Ok(Turn::Yielded) => {
                        self.active = Some(Active::Install(input, head, install, guard))
                    }
                    Ok(Turn::Published(published)) => match guard.published(&published) {
                        Ok(continuation) => {
                            if !settled(&published) {
                                self.active =
                                    Some(Active::Compute(input, published, Some(continuation)));
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
            }
        }
        Ok(())
    }
}
fn operation_digest(
    reader: &polylogue_store::ReadHandle,
    input: &RetainedSource,
    operation: &polylogue_model::tool_item::Operation,
    cancelled: &AtomicBool,
) -> Result<BlobHash> {
    use polylogue_model::{tool_item::ToolOrigin, version_evidence::Framer};
    if let [ToolOrigin::History { raw_hash, .. }] = operation.origins.as_slice() {
        return Ok(BlobHash::from_bytes(*raw_hash));
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
        let bytes = reader.read_source_range(
            &input.input.id,
            input.input.chain,
            input.binding,
            *start,
            end - start,
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
    input: &RetainedSource,
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

fn can_finalize(input: &RetainedSource, head: &Head) -> bool {
    head.status == "finalizing"
        || (input.observed()
            && input.format == InputFormat::ClaudeCodeJsonl
            && head.status == "waiting_tail")
}
fn settled(head: &Head) -> bool {
    matches!(head.status.as_str(), "waiting_tail" | "finalizing" | "done")
}
pub(crate) fn parser_version(format: InputFormat) -> u32 {
    match format {
        InputFormat::CodexRollout => codex::PARSER_VERSION,
        InputFormat::CodexHistorySqlite => codex::history::PARSER_VERSION,
        InputFormat::ClaudeCodeJsonl => claude_code::PARSER_VERSION,
    }
}
pub(crate) fn parser_context(input: &RetainedSource) -> Result<SourceContext> {
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
        _ => Err(Error::InvalidRequest("unsupported retained source context")),
    }
}
fn parser_context_json(input: &RetainedSource) -> Result<String> {
    if input.format == InputFormat::CodexRollout {
        return Ok(CONTEXT.into());
    }
    #[derive(serde::Serialize)]
    struct Context<'a> {
        format: &'static str,
        context: &'a [String],
    }
    serde_json::to_string(&Context {
        format: input.format.name(),
        context: &input.context,
    })
    .map_err(|_| Error::InvalidRequest("source context serialization failed"))
}
pub(crate) fn parser_basis(input: &RetainedSource) -> [u8; 32] {
    if input.format == InputFormat::CodexRollout {
        return basis(&input.binding);
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue.claude.immutable.parser-basis.v1\0");
    hash.update(&parser_version(input.format).to_be_bytes());
    hash.update(&crate::source_rows::RECIPE.to_be_bytes());
    hash.update(input.binding.bytes());
    // The admitted portable context is source identity; frame each component.
    hash.update(&(input.context.len() as u64).to_be_bytes());
    for item in &input.context {
        hash.update(&(item.len() as u64).to_be_bytes());
        hash.update(item.as_bytes());
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
            records: 32,
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
        cells: 2,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::Path, time::Instant};

    fn request(path: &Path, id: &str) -> ImportRequest {
        ImportRequest {
            format: InputFormat::CodexRollout,
            context: vec![],
            path: path.to_owned(),
            observation: Observation {
                id: id.into(),
                machine: "synthetic-machine".into(),
                host: "synthetic-host".into(),
                path: path.to_str().unwrap().into(),
                seen_at: 0,
            },
            metadata_allowance: 4096,
        }
    }
    fn owner(root: &Path) -> Owner {
        Owner::start(
            Writer::open(root).unwrap(),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
    }
    fn import(handle: &Handle, request: ImportRequest) -> RetainedInput {
        handle
            .try_import(request)
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap()
    }
    fn progress(handle: &Handle, id: &str) -> Progress {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            match handle.try_progress(id) {
                Ok(reply) => {
                    return reply
                        .recv_timeout(Duration::from_secs(10))
                        .unwrap()
                        .unwrap();
                }
                Err(Error::Busy) if Instant::now() < end => thread::yield_now(),
                other => panic!("progress admission: {:?}", other.err()),
            }
        }
    }
    fn until_settled(handle: &Handle, id: &str) -> Progress {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            let p = progress(handle, id);
            assert!(p.failure.is_none(), "{:?}", p.failure);
            assert!(p.recovery_error.is_none(), "{:?}", p.recovery_error);
            if p.parser.as_ref().is_some_and(settled) {
                return p;
            }
            assert!(
                Instant::now() < end,
                "provider did not reach stable frontier"
            );
            thread::yield_now();
        }
    }
    fn raw() -> &'static [u8] {
        b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"synthetic-session\",\"source\":\"cli\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"id\":\"u\",\"role\":\"user\",\"content\":[{\"text\":\"hello\"}]}}\n"
    }
    #[test]
    fn authored_reply_does_not_open_derived_status_after_commit() {
        use polylogue_model::authored::{NoteKind, Target, WriteNote};
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let writer = Writer::open(root).unwrap();
        let reader = writer.read_handle().unwrap();
        // Hold the real connections while making only a new derived-tier open
        // fail. This is an isolated fixture, never a live archive operation.
        std::fs::rename(root.join("embeddings.db"), root.join("held-embeddings.db")).unwrap();
        std::fs::create_dir(root.join("embeddings.db")).unwrap();
        assert!(writer.status().is_err());
        let service = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
        let request = WriteNote {
            request_id: "01900000-0000-7000-8000-000000000501".into(),
            author: "agent".into(),
            machine: "synthetic-machine".into(),
            created_at: None,
            kind: Some(NoteKind::Correction),
            value: "Correction survives a derived status outage.".into(),
            targets: Some(vec![Target {
                reference: "message:future".into(),
                seen_hash: None,
            }]),
            relations: vec![],
        };
        let committed = service
            .handle()
            .try_note(request.clone(), Instant::now() + Duration::from_secs(30))
            .unwrap()
            .recv_timeout(Duration::from_secs(30))
            .unwrap()
            .unwrap();
        assert!(committed.recovery.is_none());
        let retry = service
            .handle()
            .try_note(request, Instant::now() + Duration::from_secs(30))
            .unwrap()
            .recv_timeout(Duration::from_secs(30))
            .unwrap()
            .unwrap();
        assert_eq!(committed.revision, retry.revision);
        assert_eq!(
            reader
                .authored_snapshot()
                .unwrap()
                .family(&committed.revision.reference())
                .unwrap()
                .history
                .len(),
            1
        );
        std::fs::remove_dir(root.join("embeddings.db")).unwrap();
        std::fs::rename(root.join("held-embeddings.db"), root.join("embeddings.db")).unwrap();
        service.shutdown().unwrap();
    }
    #[test]
    fn declared_claude_context_replays_without_provider_and_remains_pending() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("untrusted-download-name.jsonl");
        let raw=b"{\"type\":\"user\",\"sessionId\":\"declared-main\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"retained prompt\"}}\n";
        std::fs::write(&path, raw).unwrap();
        let mut request = request(&path, "claude-observation");
        request.format = InputFormat::ClaudeCodeJsonl;
        request.context = vec!["claude-main".into(), "declared-main".into()];
        let root = dir.path().join("archive");
        let service = owner(&root);
        let retained = import(&service.handle(), request.clone());
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let p = progress(&service.handle(), &retained.id);
            assert_eq!(p.source_rows.basis.input, p.input.id);
            if let Some(head) = p.parser {
                if head.parsed_upto == retained.length {
                    assert_eq!(head.counts.used, 1);
                    assert_eq!(
                        head.finalization_pending,
                        !matches!(
                            p.source_rows.state,
                            polylogue_model::settlement::SourceRowState::Settled
                        )
                    );
                    assert_ne!(head.status, "done");
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "Claude parser did not publish its checkpoint"
            );
            thread::yield_now();
        }
        std::fs::remove_file(&path).unwrap();
        assert_eq!(import(&service.handle(), request.clone()), retained);
        let mut wrong = request.clone();
        wrong.context[1] = "another-main".into();
        assert!(matches!(
            service
                .handle()
                .try_import(wrong)
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap(),
            Err(Error::Store(StoreError::ClientRequestConflict { .. }))
        ));
        service.shutdown().unwrap();
        let service = owner(&root);
        assert_eq!(import(&service.handle(), request), retained);
        assert_eq!(
            progress(&service.handle(), &retained.id)
                .source_rows
                .basis
                .input,
            retained.id
        );
        service.shutdown().unwrap();
    }
    #[test]
    fn claude_actor_selects_each_pending_locator_and_replays_after_provider_loss() {
        use polylogue_sources::{claude_code::keyed::MemoryReadView, frame::drive_jsonl};
        let raw = concat!(
            "{\"type\":\"user\",\"sessionId\":\"declared-main\",\"uuid\":\"u\",\"promptSource\":\"typed\",\"message\":{\"role\":\"user\",\"content\":\"retained question\"}}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"declared-main\",\"uuid\":\"a\",\"parentUuid\":\"u\",\"message\":{\"id\":\"answer\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"retained answer\"}],\"stop_reason\":\"end_turn\"}}\n"
        );
        let framed = drive_jsonl(
            claude_code::KeyedRecords::new(
                claude_code::FileCtx::main("declared-main"),
                MemoryReadView::default(),
            )
            .unwrap(),
            raw.as_bytes(),
            0,
        );
        let ids = framed
            .output
            .unwrap()
            .delta
            .pending_messages
            .into_iter()
            .map(|p| p.message)
            .collect::<Vec<_>>();
        assert_eq!(ids.len(), 2);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opaque-download.jsonl");
        std::fs::write(&path, raw).unwrap();
        let root = dir.path().join("archive");
        let writer = Writer::open(&root).unwrap();
        let reader = writer.read_handle().unwrap();
        let service = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
        let mut r = request(&path, "claude-selected");
        r.format = InputFormat::ClaudeCodeJsonl;
        r.context = vec!["claude-main".into(), "declared-main".into()];
        let retained = import(&service.handle(), r.clone());
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let p = progress(&service.handle(), &retained.id);
            assert!(p.failure.is_none(), "{:?}", p.failure);
            assert_eq!(p.source_rows.basis.input, p.input.id);
            let selected = ids
                .iter()
                .map(|id| reader.selected_message(id).unwrap())
                .collect::<Vec<_>>();
            if selected.iter().all(Option::is_some) {
                for (view, text) in selected
                    .into_iter()
                    .zip(["retained question", "retained answer"])
                {
                    let view = view.unwrap();
                    let block = view
                        .block(polylogue_model::BlockSlot::Item(0))
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        block.text_range(0, 64).unwrap().unwrap().bytes,
                        text.as_bytes()
                    );
                }
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Claude pending locators did not become selected"
            );
            thread::yield_now();
        }
        std::fs::remove_file(&path).unwrap();
        assert_eq!(import(&service.handle(), r.clone()), retained);
        service.shutdown().unwrap();
        drop(reader);
        let service = owner(&root);
        assert_eq!(import(&service.handle(), r), retained);
        assert_eq!(
            progress(&service.handle(), &retained.id)
                .source_rows
                .basis
                .input,
            retained.id
        );
        service.shutdown().unwrap();
    }
    #[test]
    fn actor_selects_a_source_message_but_never_certifies_history_done() {
        use polylogue_sources::{
            codex::keyed::{KeyedRecords, MemoryReadView, Resume},
            frame::drive_jsonl,
        };
        let framed = drive_jsonl(
            KeyedRecords::new(MemoryReadView::default(), Resume::default()),
            raw(),
            0,
        );
        let delta = framed.output.unwrap();
        let mut memory = MemoryReadView::default();
        memory.apply(&delta);
        let message = memory.projected_message(0).unwrap().unwrap().id;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic-canonical.jsonl");
        std::fs::write(&path, raw()).unwrap();
        let writer = Writer::open(&dir.path().join("archive")).unwrap();
        let reader = writer.read_handle().unwrap();
        let service = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
        let retained = import(
            &service.handle(),
            request(&path, "synthetic-canonical-request"),
        );
        std::fs::remove_file(&path).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let p = progress(&service.handle(), &retained.id);
            assert!(p.failure.is_none(), "{:?}", p.failure);
            assert_eq!(p.source_rows.basis.input, p.input.id);
            if let Some(view) = reader.selected_message(&message).unwrap() {
                let block = view
                    .block(polylogue_model::BlockSlot::Item(0))
                    .unwrap()
                    .unwrap();
                let range = block.text_range(0, 64).unwrap().unwrap();
                assert_eq!(range.bytes, b"hello");
                assert_eq!(
                    p.parser.unwrap().finalization_pending,
                    !matches!(
                        p.source_rows.state,
                        polylogue_model::settlement::SourceRowState::Settled
                    )
                );
                break;
            }
            assert!(
                Instant::now() < deadline,
                "canonical source message did not become readable"
            );
            thread::yield_now();
        }
        // Receipt replay still works after provider loss while the actor serves
        // selected-message readers; retention is independent of finalization.
        assert_eq!(
            import(
                &service.handle(),
                request(&path, "synthetic-canonical-request")
            )
            .id,
            retained.id
        );
        service.shutdown().unwrap();
        assert!(reader.selected_message(&message).unwrap().is_some());
    }
    #[test]
    fn retained_receipt_replays_before_missing_source_and_recovers_provider_state() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let path = dir.path().join("synthetic.jsonl");
        std::fs::write(&path, raw()).unwrap();
        let service = owner(&root);
        let retained = import(&service.handle(), request(&path, "synthetic-request"));
        std::fs::remove_file(&path).unwrap();
        let replay = import(&service.handle(), request(&path, "synthetic-request"));
        assert_eq!(retained.id, replay.id);
        let mut changed = request(&path, "synthetic-request");
        changed.observation.seen_at = 1;
        assert!(matches!(
            service
                .handle()
                .try_import(changed)
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap(),
            Err(Error::Store(_))
        ));
        service.shutdown().unwrap();
        let service = owner(&root);
        let p = until_settled(&service.handle(), &retained.id);
        let head = p.parser.unwrap();
        assert_eq!(head.counts.framed, 2);
        assert_eq!(head.parsed_upto, raw().len() as u64);
        assert!(matches!(head.status.as_str(), "finalizing" | "done"));
        assert_eq!(
            head.finalization_pending,
            !matches!(
                p.source_rows.state,
                polylogue_model::settlement::SourceRowState::Settled
            )
        );
        assert_eq!(p.source_rows.basis.input, p.input.id);
        service.shutdown().unwrap();
    }
    #[test]
    fn partial_tail_is_stable_across_restart_and_explicit_retry() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let path = dir.path().join("synthetic-tail.jsonl");
        let mut bytes = raw().to_vec();
        bytes.extend_from_slice(b"{\"type\":");
        std::fs::write(&path, &bytes).unwrap();
        let service = owner(&root);
        let retained = import(&service.handle(), request(&path, "synthetic-tail-request"));
        let before = until_settled(&service.handle(), &retained.id)
            .parser
            .unwrap();
        assert_eq!(before.status, "waiting_tail");
        assert_eq!(before.parsed_upto, raw().len() as u64);
        assert_eq!(before.inspected_upto, bytes.len() as u64);
        service.shutdown().unwrap();
        let service = owner(&root);
        service
            .handle()
            .try_retry(&retained.id)
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        for _ in 0..8 {
            let p = progress(&service.handle(), &retained.id);
            let head = p.parser.unwrap();
            assert_eq!(head.token, before.token);
            assert_eq!(head.counts, before.counts);
            assert_eq!(
                head.finalization_pending,
                !matches!(
                    p.source_rows.state,
                    polylogue_model::settlement::SourceRowState::Settled
                )
            );
            assert_eq!(p.source_rows.basis.input, p.input.id);
        }
        service.shutdown().unwrap();
    }
    #[test]
    fn oversized_logical_record_is_an_inspectable_admission_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic-large.jsonl");
        let mut bytes = b"{\"type\":\"synthetic\",\"payload\":\"".to_vec();
        bytes.resize(1024 * 1024 + 64, b'x');
        bytes.extend_from_slice(b"\"}\n");
        std::fs::write(&path, &bytes).unwrap();
        let service = owner(&dir.path().join("archive"));
        let retained = import(&service.handle(), request(&path, "synthetic-large-request"));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let p = progress(&service.handle(), &retained.id);
            if let Some(failure) = p.failure {
                assert_eq!(
                    failure,
                    format!(
                        "source framing requires {} bytes (allocation failed: false)",
                        bytes.len() - 1
                    )
                );
                let head = p.parser.unwrap();
                assert_eq!(head.parsed_upto, 0);
                assert_eq!(head.counts.framed, 0);
                assert_ne!(head.status, "done");
                assert_eq!(p.source_rows.basis.input, p.input.id);
                break;
            }
            assert!(Instant::now() < deadline, "missing admission failure");
            thread::yield_now();
        }
        service.shutdown().unwrap();
    }
    #[test]
    fn queue_capacity_request_bytes_and_shutdown_are_explicit() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        let inner = Arc::new(Inner {
            sender,
            accepting: Mutex::new(true),
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        let handle = Handle { inner };
        let _first = handle.try_progress("synthetic").unwrap();
        assert!(matches!(handle.try_progress("synthetic"), Err(Error::Busy)));
        assert!(matches!(
            handle.try_progress(&"x".repeat(REQUEST_BYTES + 1)),
            Err(Error::RequestTooLarge { .. })
        ));
        let mut mismatched = request(Path::new("/synthetic/path"), "synthetic");
        mismatched.observation.path = "/synthetic/other".into();
        assert!(matches!(
            handle.try_import(mismatched),
            Err(Error::InvalidRequest(_))
        ));
        *handle.inner.accepting.lock().unwrap() = false;
        assert!(matches!(handle.try_retry("synthetic"), Err(Error::Stopped)));
        let text = diagnostic("ą".repeat(DETAIL_BYTES));
        assert!(text.len() <= DETAIL_BYTES);
        assert!(text.ends_with("[diagnostic truncated]"));
    }
    #[test]
    fn shutdown_replies_to_accepted_import_instead_of_losing_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic.jsonl");
        std::fs::write(&path, raw()).unwrap();
        let service = owner(&dir.path().join("archive"));
        let reply = service
            .handle()
            .try_import(request(&path, "synthetic-stop"))
            .unwrap();
        service.shutdown().unwrap();
        assert!(matches!(
            reply.recv_timeout(Duration::from_secs(10)),
            Ok(Ok(_)) | Ok(Err(Error::Stopped))
        ));
    }
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
