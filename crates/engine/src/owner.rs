//! Single Writer actor for retained Codex immutable imports and provider-state
//! progress. Capture and raw-window computation run on separate workers. This
//! does not publish canonical Messages, search, or complete history coverage.
use crate::{
    converge::{self, PendingInstall, Recovery, Turn},
    source_turn::{
        self, Budgets, CommitGuard, CommittedContinuation, Computation, SourceContext, Window,
    },
};
use polylogue_model::BlobHash;
use polylogue_sources::{codex, frame::StepBudget};
use polylogue_store::{
    Error as StoreError, ImmutableImport, ImportObservation, InputFormat, PreparedBlob,
    RetainedInput, Writer,
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
    time::Duration,
};
// These are admission limits, not format limits or a hard RSS bound. A rejected
// record/cell remains owed and its required capacity is kept in the diagnostic.
// Configurable larger admissions and canonical publication are separate work.
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
type Result<T> = std::result::Result<T, Error>;
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
#[derive(Debug)]
pub struct ImportRequest {
    pub path: PathBuf,
    pub observation: Observation,
    pub metadata_allowance: usize,
}
#[derive(Debug)]
pub struct Progress {
    pub input: RetainedInput,
    pub parser: Option<Head>,
    pub failure: Option<String>,
    pub recovery_error: Option<String>,
    pub active: bool,
    pub canonical_complete: bool,
}
enum Message {
    Import(ImportRequest, Reply<RetainedInput>),
    Progress(String, Reply<Progress>),
    Retry(String, Reply<()>),
    Captured(Result<PreparedBlob>),
    Computed(Result<Computation>),
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
            .saturating_add(o.path.len());
        if required > REQUEST_BYTES || request.metadata_allowance > REQUEST_BYTES {
            return Err(Error::RequestTooLarge {
                required: required.max(request.metadata_allowance),
                maximum: REQUEST_BYTES,
            });
        }
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
    observation: Observation,
    allowance: usize,
    reply: Reply<RetainedInput>,
    join: JoinHandle<()>,
}
struct ComputeJob {
    input: ImmutableImport,
    head: Head,
    join: JoinHandle<()>,
}
enum Active {
    Begin(ImmutableImport),
    Recover(ImmutableImport, Head),
    Compute(ImmutableImport, Head, Option<CommittedContinuation>),
    Install(ImmutableImport, Head, PendingInstall, CommitGuard),
}
struct Actor {
    writer: Writer,
    inner: Arc<Inner>,
    capture: Option<CaptureJob>,
    compute: Option<ComputeJob>,
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
            if self.stopping && self.capture.is_none() && self.compute.is_none() {
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
        if let Some(job) = &self.compute {
            return Some(&job.input.input.id);
        }
        match &self.active {
            Some(
                Active::Begin(i)
                | Active::Recover(i, _)
                | Active::Compute(i, _, _)
                | Active::Install(i, _, _, _),
            ) => Some(&i.input.id),
            None => None,
        }
    }
    fn message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::Import(request, reply) => {
                if self.stopping {
                    let _ = reply.send(Err(Error::Stopped));
                    return Ok(());
                }
                let replay = self.writer.replay_immutable_import(
                    InputFormat::CodexRollout,
                    &[],
                    &request.observation.borrowed(),
                    request.metadata_allowance,
                );
                match replay {
                    Ok(Some(input)) => {
                        self.scan_again = true;
                        if !self.scanning && self.active.is_none() && self.compute.is_none() {
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
                let join = thread::Builder::new()
                    .name("polylogue-capture".into())
                    .spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            capture.capture_regular_file(&path, &cancelled)
                        }))
                        .map_err(|_| Error::WorkerPanicked)
                        .and_then(|r| r.map_err(Error::Store));
                        let _ = sender.send(Message::Captured(result));
                    });
                match join {
                    Ok(join) => {
                        self.capture = Some(CaptureJob {
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
                        InputFormat::CodexRollout,
                        &[],
                        &job.observation.borrowed(),
                        job.allowance,
                    )?)
                });
                if result.is_ok() {
                    self.scan_again = true;
                    if !self.scanning && self.active.is_none() && self.compute.is_none() {
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
                    Ok(Computation::Idle { .. }) => {}
                    Err(e) => self.fail(&job.head, e.to_string())?,
                }
            }
            Message::Wake => {}
        }
        Ok(())
    }
    fn progress(&mut self, id: &str) -> Result<Progress> {
        let input = self
            .writer
            .immutable_import(id)?
            .ok_or(Error::UnknownInput)?;
        let parser = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?;
        let failure = match &parser {
            Some(head) => self.writer.parser_failure(&head.token)?,
            None if input.format != InputFormat::CodexRollout => Some(
                "source context for this provider is not registered by this Codex owner".into(),
            ),
            None => None,
        };
        Ok(Progress {
            input: input.input,
            parser,
            failure,
            recovery_error: self.recovery_error.clone(),
            active: self.active_id() == Some(id),
            canonical_complete: false,
        })
    }
    fn retry(&mut self, id: &str) -> Result<()> {
        let input = self
            .writer
            .immutable_import(id)?
            .ok_or(Error::UnknownInput)?;
        if input.format != InputFormat::CodexRollout {
            return Err(Error::InvalidRequest(
                "this owner only interprets Codex immutable imports",
            ));
        }
        if let Some(head) = self
            .writer
            .parser_head(input.input.input_pk, input.input.chain)?
        {
            self.writer
                .record_parser_failure(&head.token, None, DETAIL_BYTES)?;
        }
        self.scan_again = true;
        self.recovery_error = None;
        if self.active.is_none() && self.compute.is_none() {
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
        if self.compute.is_some() {
            return Ok(());
        }
        let active = match self.active.take() {
            Some(a) => a,
            None => {
                if !self.scanning {
                    return Ok(());
                }
                match self.writer.next_immutable_import(&self.after) {
                    Ok(Some(input)) => {
                        self.after = input.input.id.clone();
                        if input.format != InputFormat::CodexRollout {
                            return Ok(());
                        }
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
                let digest = basis(&input.hash);
                match self.writer.begin_parser(&Basis {
                    input_pk: input.input.input_pk,
                    chain: input.input.chain,
                    parser_version: codex::PARSER_VERSION,
                    digest,
                    acquired_upto: input.input.length,
                    companions_json: Some(CONTEXT),
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
                        }
                    }
                    Ok(Recovery::Published(head)) => {
                        if !settled(&head) {
                            self.active = Some(Active::Compute(input, head, None));
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
                let hash = input.hash;
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
                                let bytes = reader.read_blob_range(hash, offset, length)?;
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
                                    SourceContext::Codex,
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
fn settled(head: &Head) -> bool {
    matches!(head.status.as_str(), "waiting_tail" | "finalizing" | "done")
}
fn basis(hash: &BlobHash) -> [u8; 32] {
    let mut bytes = b"polylogue.codex.immutable.parser-basis.v1\0".to_vec();
    bytes.extend_from_slice(&codex::PARSER_VERSION.to_be_bytes());
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
        assert_eq!(head.status, "finalizing");
        assert!(head.finalization_pending);
        assert!(!p.canonical_complete);
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
            assert!(head.finalization_pending);
            assert!(!p.canonical_complete);
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
                assert!(
                    failure.contains("capacity") || failure.contains("record"),
                    "{failure}"
                );
                let head = p.parser.unwrap();
                assert_eq!(head.parsed_upto, 0);
                assert_eq!(head.counts.framed, 0);
                assert_ne!(head.status, "done");
                assert!(!p.canonical_complete);
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
