//! One blocking socket client for the CLI, the MCP proxy, the hook and the
//! browser host. A request is one JSON line; its reply is server-sent-event
//! frames: zero or more events, then one `answer`. A reply that ends
//! without its answer is reported as the daemon having gone away.
use command_surface::{Answer, SseDecoder, SseEvent};
use polylogue_store::Archive;
use serde_json::Value;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Waits for a listening daemon or the end of the archive owner's startup.
/// The watch is registered before connecting, so publication between the
/// first failed connection and this wait cannot be lost.
pub(crate) fn wait_for_socket(path: &Path, until: Option<Instant>) -> io::Result<()> {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    let watch = crate::socket::Watch::new(path)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let archive = Archive::at(parent);
    loop {
        let last_error = match UnixStream::connect(path) {
            Ok(_) => return Ok(()),
            Err(error) if crate::cli::is_down(&error) && archive.lock_held() => error.kind(),
            Err(error) => return Err(error),
        };
        loop {
            let timeout = match until {
                Some(until) => {
                    let left = until.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            format!(
                                "the daemon did not start before the deadline \
                                 (last connection error: {last_error})"
                            ),
                        ));
                    }
                    Some(Timespec::try_from(left).map_err(io::Error::other)?)
                }
                None => None,
            };
            let mut events = [PollFd::new(&watch, PollFlags::IN)];
            match poll(&mut events, timeout.as_ref()) {
                Ok(_) | Err(rustix::io::Errno::INTR) => {}
                Err(error) => return Err(error.into()),
            }
            if watch.changed()? {
                break;
            }
        }
    }
}

static THIS_PROCESS: OnceLock<command_surface::CallerEvidence> = OnceLock::new();

/// Gathers the facts about this process once, from `main`: its harness
/// variables, harness ancestors, terminal and unit.
pub fn remember_this_process(environment: &crate::config::EnvSnapshot) {
    let variable = |name: &str| environment.get_os(name).map(Into::into);
    let mut caller = command_surface::gather_current(variable);
    crate::hook::hermes_session(&mut caller, variable);
    let _ = THIS_PROCESS.set(caller);
}

/// The facts about this process every request it sends carries.
pub fn this_process() -> command_surface::CallerEvidence {
    THIS_PROCESS.get().cloned().unwrap_or_default()
}

/// A new request for `operation` with these arguments and the default deadline.
pub fn request(
    operation: &str,
    arguments: Value,
    request_id: String,
) -> command_surface::RunnableRequest {
    command_surface::Request {
        operation: operation.into(),
        arguments,
        request_id,
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    }
}

/// How long a client waits for the answer: the request's read deadline
/// with the protocol's grace for a request that reads, none for one that
/// writes, within the parent's bound.
pub(crate) fn connection_deadline(
    request: &command_surface::RunnableRequest,
    parent_deadline_ms: Option<i64>,
) -> Option<Duration> {
    use crate::daemon::handler::Bound;
    let command = crate::commands::command(&request.operation);
    let class = command.map_or(command_surface::LatencyClass::Local, |command| {
        command.latency
    });
    let policy = command.and_then(|command| {
        crate::daemon::handler::policy(
            command,
            &request.arguments,
            polylogue_model::authorship::Channel::Cli,
        )
        .ok()
    });
    let deadline = match policy {
        // A request that reads and then writes may wait on the writer after its
        // read finished; the write is the user's, so the client waits for the
        // daemon's answer, which reports a read that overran its deadline itself.
        Some(policy) if policy.changes_archive => command_surface::Deadline::None,
        Some(policy)
            if policy.bound == Bound::CallerOnly
                && request.deadline == command_surface::Deadline::Default =>
        {
            command_surface::Deadline::None
        }
        Some(_)
            if request.deadline == command_surface::Deadline::Default
                && crate::commands::deadlines().get(class).is_none() =>
        {
            command_surface::Deadline::Milliseconds(
                crate::daemon::execution::DEFAULT_READ_DEADLINE_MS,
            )
        }
        _ => request.deadline,
    };
    let budget = deadline.connection_timeout(crate::commands::deadlines().get(class));
    match (
        budget,
        parent_deadline_ms.map(|deadline| crate::daemon::execution::remaining(Some(deadline))),
    ) {
        (Some(request), Some(parent)) => Some(request.min(parent)),
        (request, parent) => request.or(parent),
    }
}

/// Lets another thread end a call in progress by hanging up its
/// connection: the MCP proxy, when its client cancels a tool call. The
/// daemon's hang-up watcher then cancels the request's reads.
#[derive(Default)]
pub(crate) struct HangUp {
    connection: Mutex<Option<UnixStream>>,
    done: std::sync::atomic::AtomicBool,
}

impl HangUp {
    pub(crate) fn hang_up(&self) {
        self.done.store(true, std::sync::atomic::Ordering::Release);
        if let Some(connection) = &*self.connection.lock().unwrap_or_else(|e| e.into_inner()) {
            let _ = connection.shutdown(std::net::Shutdown::Both);
        }
    }

    fn hung_up(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::Acquire)
    }

    fn watch(&self, connection: &UnixStream) -> io::Result<()> {
        *self.connection.lock().unwrap_or_else(|e| e.into_inner()) = Some(connection.try_clone()?);
        if self.hung_up() {
            let _ = connection.shutdown(std::net::Shutdown::Both);
        }
        Ok(())
    }
}

fn cancelled() -> Answer<Value> {
    Answer::error(crate::protocol::failure(
        polylogue_model::Code::Cancelled,
        "the request was cancelled",
    ))
}

/// What one call sends and how long it waits.
pub(crate) struct Exchange<'a> {
    pub request: &'a command_surface::RunnableRequest,
    pub caller: command_surface::CallerEvidence,
    pub parent_deadline_ms: Option<i64>,
    /// Bounds connecting and the whole reply; `None` waits for the daemon.
    pub budget: Option<Duration>,
    pub hang_up: Option<&'a HangUp>,
}

impl<'a> Exchange<'a> {
    pub(crate) fn new(request: &'a command_surface::RunnableRequest) -> Self {
        Self {
            request,
            caller: this_process(),
            parent_deadline_ms: None,
            budget: connection_deadline(request, None),
            hang_up: None,
        }
    }

    /// Sends the request on `socket` and reads its reply: each event goes to
    /// `event`, which ends the call early by returning `true`; then the
    /// answer. `None` when `event` ended it.
    pub(crate) fn on(
        &self,
        socket: UnixStream,
        event: impl FnMut(SseEvent) -> io::Result<bool>,
    ) -> io::Result<Option<Answer<Value>>> {
        let until = self
            .budget
            .and_then(|budget| Instant::now().checked_add(budget));
        if let Some(hang_up) = self.hang_up {
            hang_up.watch(&socket)?;
        }
        let outgoing = crate::protocol::wire::Outgoing {
            request: self.request,
            caller: self.caller.clone(),
            parent_deadline_ms: self.parent_deadline_ms,
        };
        let mut line = serde_json::to_vec(&outgoing).map_err(io::Error::other)?;
        line.push(b'\n');
        let mut socket = socket;
        socket.set_write_timeout(self.budget)?;
        socket.write_all(&line)?;
        let reply = receive(&mut socket, until, event);
        // After the caller hangs up, the connection ends without its answer
        // because of the caller, not the daemon.
        let hung_up = self.hang_up.is_some_and(HangUp::hung_up);
        let ended = |answer: &Answer<Value>| {
            answer.failure().and_then(|failure| failure.code.as_deref()) == Some("not_running")
        };
        match reply {
            Err(_) if hung_up => Ok(Some(cancelled())),
            Ok(Some(answer)) if hung_up && ended(&answer) => Ok(Some(cancelled())),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "the daemon did not answer before the deadline",
                ))
            }
            reply => reply,
        }
    }

    /// Connects to `socket` at once, without waiting for a daemon that is
    /// still starting, and takes the one answer.
    pub(crate) fn call(&self, socket: &Path) -> io::Result<Answer<Value>> {
        let connection = UnixStream::connect(socket)?;
        self.on(connection, refuse_events)
            .map(|answer| answer.expect("a call without events runs to its answer"))
    }

    /// Connects once the archive's daemon listens, waiting for one still
    /// opening the archive within the budget.
    pub(crate) fn connect(&self, archive: &Archive) -> io::Result<UnixStream> {
        let deadline = self.budget.map(crate::daemon::execution::deadline);
        crate::cli::when_ready(archive, deadline, || UnixStream::connect(archive.socket()))
    }

    /// [`Exchange::connect`], then the one answer; a deadline that passes,
    /// starting or answering, is the deadline failure.
    pub(crate) fn call_ready(&self, archive: &Archive) -> io::Result<Answer<Value>> {
        let answer = self.connect(archive).and_then(|connection| {
            self.on(connection, refuse_events)
                .map(|answer| answer.expect("a call without events runs to its answer"))
        });
        match answer {
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                Ok(Answer::error(crate::protocol::failure(
                    polylogue_model::Code::DeadlineExceeded,
                    error.to_string(),
                )))
            }
            answer => answer,
        }
    }
}

fn refuse_events(_: SseEvent) -> io::Result<bool> {
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "the daemon sent events to a call that takes one answer",
    ))
}

/// Reads one reply's frames until its answer.
pub(crate) fn receive(
    input: &mut impl Receive,
    until: Option<Instant>,
    mut event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> io::Result<Option<Answer<Value>>> {
    let mut decoder = SseDecoder::new();
    let mut bytes = [0; 8192];
    let mut first = true;
    loop {
        input.wait_until(until)?;
        let count = input.read(&mut bytes)?;
        if count == 0 {
            return Ok(Some(gone()));
        }
        if std::mem::take(&mut first) {
            // A daemon from a build before framed replies answers with one
            // JSON line; anything else that is not a frame is no answer.
            if bytes[0] == b'{' {
                return Ok(Some(Answer::error(crate::protocol::protocol_mismatch(
                    "the request",
                    "an unframed answer",
                ))));
            }
            if !matches!(bytes[0], b'e' | b'd' | b'i' | b'r' | b':') {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the daemon's reply is not a framed answer",
                ));
            }
        }
        let frames = decoder
            .push(&bytes[..count])
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.message))?;
        for frame in frames {
            if frame.event == "answer" {
                return serde_json::from_str(&frame.data)
                    .map(Some)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error));
            }
            if event(frame)? {
                return Ok(None);
            }
        }
    }
}

/// The answer of a reply that ended before it.
fn gone() -> Answer<Value> {
    Answer::error(crate::protocol::failure(
        polylogue_model::Code::NotRunning,
        "the daemon went away before it answered; its service log says why",
    ))
}

/// A reply's input, which bounds each read by the call's deadline.
pub(crate) trait Receive: Read {
    fn wait_until(&mut self, until: Option<Instant>) -> io::Result<()>;
}

impl Receive for UnixStream {
    fn wait_until(&mut self, until: Option<Instant>) -> io::Result<()> {
        let Some(until) = until else {
            return self.set_read_timeout(None);
        };
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        self.set_read_timeout(Some(left))
    }
}

#[cfg(test)]
impl Receive for &[u8] {
    fn wait_until(&mut self, _: Option<Instant>) -> io::Result<()> {
        Ok(())
    }
}

/// Sends one request to a running daemon, without waiting for one that is
/// still starting. A deadline that passes is the deadline failure.
pub fn call(
    socket: &Path,
    request: &command_surface::RunnableRequest,
) -> io::Result<Answer<Value>> {
    match Exchange::new(request).call(socket) {
        Err(error) if error.kind() == io::ErrorKind::TimedOut => Ok(Answer::error(
            crate::protocol::failure(polylogue_model::Code::DeadlineExceeded, error.to_string()),
        )),
        answer => answer,
    }
}

/// A hook uses the existing daemon only; its budget covers the entire exchange.
pub(crate) fn call_with_budget(
    socket: &Path,
    request: &command_surface::RunnableRequest,
    budget: Duration,
) -> io::Result<Answer<Value>> {
    Exchange {
        budget: Some(budget),
        ..Exchange::new(request)
    }
    .call(socket)
}

/// A ready call from async code, the MCP server's: the exchange runs on a
/// blocking thread, and the token's cancellation hangs up its connection.
pub(crate) async fn call_ready_as_bounded(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
    cancelled: &tokio_util::sync::CancellationToken,
    caller: command_surface::CallerEvidence,
    parent_deadline_ms: Option<i64>,
) -> io::Result<Answer<Value>> {
    let hang_up = std::sync::Arc::new(HangUp::default());
    let mut work = tokio::task::spawn_blocking({
        let (archive, request, hang_up) = (archive.clone(), request.clone(), hang_up.clone());
        move || {
            Exchange {
                caller,
                parent_deadline_ms,
                budget: connection_deadline(&request, parent_deadline_ms),
                hang_up: Some(&hang_up),
                ..Exchange::new(&request)
            }
            .call_ready(&archive)
        }
    });
    tokio::select! {
        answer = &mut work => return answer.map_err(io::Error::other)?,
        () = cancelled.cancelled() => hang_up.hang_up(),
    }
    work.await.map_err(io::Error::other)?
}

/// Sends one request once the archive's daemon listens. A deadline that
/// passes is the deadline failure.
pub(crate) fn call_ready(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
) -> io::Result<Answer<Value>> {
    Exchange::new(request).call_ready(archive)
}

/// One call whose reply may stream: events go to `event` as they arrive.
pub(crate) fn call_streaming(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
    budget: Option<Duration>,
    event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> io::Result<Option<Answer<Value>>> {
    let exchange = Exchange {
        budget,
        ..Exchange::new(request)
    };
    let connection = exchange.connect(archive)?;
    exchange.on(connection, event)
}

#[cfg(test)]
mod tests;
