//! Unix socket framing and listener lifecycle. Every request is answered
//! through [`answer`](super::handler::answer); every reply is one or more
//! server-sent-event frames, ending with exactly one `answer` frame.
//!
//! One thread blocks on the listener and wakes only for a connection or the
//! stop signal. Each connection has one thread that reads its requests and
//! answers each on that thread; one shared watcher notices hang-ups. While a
//! request runs, its events are sent without waiting for the client and
//! spill to the staging directory when the client is slow; once it has
//! answered, and holds no reader, the rest is sent at the client's pace.

use super::output::{Output, Sink, Spill, answer_frame};
use super::service::{self, Daemon};
use crate::operation::Caller;
use polylogue_model::LockExt as _;
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The largest piece written between two checks of the access scope.
const SEND_CHUNK: usize = 64 * 1024;

/// The liveness check readiness and the systemd watchdog send. The client's
/// thread answers it with no reader and no cache, so a long vector load or a
/// busy reader pool never fails it.
pub(crate) const PING: &str = "ping";

/// A connection as a [`Sink`]: without waiting while a request runs, and at
/// the client's pace once it has answered. A hang-up or a stopping daemon
/// ends a blocked send at once.
struct Connection<'s> {
    stream: &'s mut UnixStream,
    waiting: bool,
}

impl Connection<'_> {
    fn set_waiting(&mut self, waiting: bool) -> std::io::Result<()> {
        if self.waiting != waiting {
            self.stream.set_nonblocking(!waiting)?;
            self.waiting = waiting;
        }
        Ok(())
    }
}

impl Sink for Connection<'_> {
    fn try_send(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.set_waiting(false)?;
        match self.stream.write(bytes) {
            Ok(sent) => Ok(sent),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(0),
            Err(error) => Err(error),
        }
    }

    fn send(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.set_waiting(true)?;
        for piece in bytes.chunks(SEND_CHUNK) {
            self.stream.write_all(piece)?;
        }
        Ok(())
    }
}

/// One reply on a connection; `sent` says whether all of it went out, so
/// the connection can take its next request.
struct Reply<'s> {
    spill: Option<Spill<Connection<'s>>>,
    sent: bool,
}

impl<'s> Reply<'s> {
    fn new(stream: &'s mut UnixStream, staging: PathBuf) -> Self {
        Self {
            spill: Some(Spill::new(
                Connection {
                    stream,
                    waiting: true,
                },
                staging,
            )),
            sent: false,
        }
    }
}

fn answered() -> std::io::Error {
    std::io::Error::other("the reply has already answered")
}

impl Output for Reply<'_> {
    fn events(&mut self, frames: &[u8]) -> std::io::Result<()> {
        self.spill.as_mut().ok_or_else(answered)?.write(frames)
    }

    fn answer(
        &mut self,
        answer: command_surface::Answer<serde_json::Value>,
        _: Option<command_surface::FailureKind>,
    ) -> std::io::Result<()> {
        let mut spill = self.spill.take().ok_or_else(answered)?;
        spill.write(&answer_frame(&answer)?)?;
        spill.finish()?.set_waiting(true)?;
        self.sent = true;
        Ok(())
    }
}

/// A reply with nothing but its final answer, for a line that names no
/// operation the daemon can run.
fn write_answer(
    stream: &mut UnixStream,
    answer: &command_surface::Answer<serde_json::Value>,
) -> bool {
    answer_frame(answer)
        .and_then(|frame| stream.write_all(&frame))
        .is_ok()
}

/// What every connection of one server shares.
struct Shared {
    daemon: Arc<Daemon>,
    hangups: super::hangup::Watcher,
    clients: Clients,
}

/// The connections still being served, for the drain at shutdown.
#[derive(Default)]
struct Clients {
    open: Mutex<Vec<(u64, UnixStream)>>,
    next: std::sync::atomic::AtomicU64,
    closed: Condvar,
    panicked: std::sync::atomic::AtomicBool,
}

/// How long complete requests may still answer at shutdown before their
/// connections are hung up.
const DRAIN_RESPONSES: Duration = Duration::from_millis(4_500);
/// The whole socket drain, within the daemon's 5 s stop
/// (docs/operations.md).
const DRAIN_BUDGET: Duration = Duration::from_secs(5);

impl Clients {
    /// Stops new input, lets complete requests answer for `responses`, then
    /// hangs up the rest, which cancels their reads, and waits for them
    /// until `budget` has passed since the start. Returns how many
    /// connections were still open then; their threads end on their own.
    fn drain(&self, responses: Duration, budget: Duration) -> usize {
        let start = Instant::now();
        // Stop receiving new input while complete queued requests can still
        // receive their refusal and an accepted write can send its stored result.
        for (_, stream) in self.open.locked().iter() {
            let _ = stream.shutdown(Shutdown::Read);
        }
        let open = self.wait_closed(self.open.locked(), start + responses);
        // Unresponsive peers cannot hold shutdown past the response window:
        // the hang-up ends blocked sends and cancels their reads.
        for (_, stream) in open.iter() {
            let _ = stream.shutdown(Shutdown::Both);
        }
        self.wait_closed(open, start + budget).len()
    }

    fn wait_closed<'a>(
        &'a self,
        mut open: std::sync::MutexGuard<'a, Vec<(u64, UnixStream)>>,
        until: Instant,
    ) -> std::sync::MutexGuard<'a, Vec<(u64, UnixStream)>> {
        while !open.is_empty() {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            open = polylogue_model::unpoisoned(self.closed.wait_timeout(open, left)).0;
        }
        open
    }
}

/// The caller's facts as the daemon observes them from the socket peer. The
/// request contributes only an MCP call's tool call ID, which joins a
/// matching hook event.
pub(crate) fn peer(
    stream: &UnixStream,
    supplied: &command_surface::CallerEvidence,
) -> std::io::Result<(command_surface::CallerEvidence, command_surface::Via)> {
    use std::os::unix::ffi::OsStringExt as _;
    let pid = rustix::net::sockopt::socket_peercred(stream)?
        .pid
        .as_raw_pid()
        .unsigned_abs();
    let root = PathBuf::from(format!("/proc/{pid}"));
    let mut details = command_surface::gather_peer(pid)?;
    let environment = std::fs::read(root.join("environ"))?;
    let variable = |name: &str| {
        environment.split(|byte| *byte == 0).find_map(|entry| {
            let value = entry.strip_prefix(name.as_bytes())?.strip_prefix(b"=")?;
            Some(std::ffi::OsString::from_vec(value.to_vec()))
        })
    };
    crate::hook::hermes_session(&mut details, variable);
    let mut arguments = std::fs::read(root.join("cmdline"))?;
    // Each argument ends with a NUL.
    if arguments.last() == Some(&0) {
        arguments.pop();
    }
    let words = arguments
        .split(|byte| *byte == 0)
        .map(|word| std::ffi::OsString::from_vec(word.to_vec()))
        .collect::<Vec<_>>();
    let via = via_of_command_line(&words);
    if via == command_surface::Via::Mcp && supplied.tool_call_id.is_some() {
        details.tool_call_id = supplied.tool_call_id.clone();
    }
    Ok((details, via))
}

/// The socket peer's working directory, against which relative paths in its
/// requests resolve.
fn working_directory(stream: &UnixStream) -> std::io::Result<PathBuf> {
    let credentials = rustix::net::sockopt::socket_peercred(stream)?;
    std::fs::read_link(format!("/proc/{}/cwd", credentials.pid.as_raw_pid()))
}

/// How a peer process reached the daemon, from its command line as the
/// command line's own scan reads it.
fn via_of_command_line(arguments: &[std::ffi::OsString]) -> command_surface::Via {
    use crate::cli::invocation::Shape;
    use command_surface::Via;
    match crate::cli::invocation::scan(arguments, &crate::config::EnvSnapshot::from_values([]))
        .shape
    {
        Shape::Hook => Via::Hook,
        Shape::NativeHost => Via::Extension,
        Shape::Command(Some(command)) if command.name == "mcp" => Via::Mcp,
        Shape::Command(Some(command)) if command.name == "internal browser-host" => Via::Extension,
        _ => Via::Cli,
    }
}

fn serve_client(stream: UnixStream, shared: &Shared) {
    let d = &shared.daemon;
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    // The connection's requests end when it hangs up or the daemon stops.
    let cancellation = d.archive.stop().child();
    let Ok(_watched) = shared.hangups.watch(&stream, cancellation.clone()) else {
        return;
    };
    let mut out = stream;
    let mut requests = BufReader::new(read_half);
    let mut buffer = Vec::new();
    loop {
        buffer.clear();
        match requests.read_until(b'\n', &mut buffer) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        // An undecodable line is a bad request; the connection stays open.
        let Ok(line) = std::str::from_utf8(&buffer) else {
            let answer = command_surface::Answer::error(crate::protocol::failure(
                polylogue_model::Code::BadRequest,
                "a request line must be UTF-8 JSON",
            ));
            if !write_answer(&mut out, &answer) {
                return;
            }
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        let incoming = match crate::protocol::wire::incoming(line) {
            Ok(incoming) => incoming,
            Err(error) => {
                let answer = command_surface::Answer::error(crate::protocol::failure(
                    polylogue_model::Code::BadRequest,
                    error.to_string(),
                ));
                if !write_answer(&mut out, &answer) {
                    return;
                }
                continue;
            }
        };
        if incoming.request.operation == PING {
            if !write_answer(
                &mut out,
                &command_surface::Answer::ok(serde_json::json!({})),
            ) {
                return;
            }
            continue;
        }
        let crate::protocol::wire::Incoming {
            request,
            caller,
            parent_deadline_ms,
        } = incoming;
        let (evidence, via) =
            peer(&out, &caller).unwrap_or_else(|_| (Default::default(), command_surface::Via::Cli));
        let mut caller = Caller::from_peer(&d.callers, &evidence, via, cancellation.child());
        caller.cwd = working_directory(&out).ok();
        caller.outer_deadline = parent_deadline_ms.and_then(|deadline| {
            Instant::now().checked_add(super::execution::remaining(Some(deadline)))
        });
        let mut reply = Reply::new(&mut out, d.archive.staging_dir());
        super::handler::answer(d, &request, &caller, &mut reply);
        if !reply.sent {
            return;
        }
    }
}

/// The loop holds a file open for each blob it is storing, and a first build
/// stores thousands at once. Services start with a soft limit of 1,024 open
/// files; the hard limit is the real bound, so the soft one is raised to it.
fn raise_open_file_limit() {
    use rustix::process::{Resource, getrlimit, setrlimit};
    let limit = getrlimit(Resource::Nofile);
    if limit.current != limit.maximum {
        let raised = rustix::process::Rlimit {
            current: limit.maximum,
            maximum: limit.maximum,
        };
        if let Err(e) = setrlimit(Resource::Nofile, raised) {
            polylogue_store::log!(warning, "raise the open file limit: {e}");
        }
    }
}

/// How `polylogue daemon` runs: whether its loop reads capture sources,
/// and whether it stops once its loop is idle.
#[derive(Debug, Clone, Copy)]
pub struct Run {
    pub reads_sources: bool,
    pub lifetime: polylogue_engine::Lifetime,
}

/// Serves the archive until a signal or, for a run until idle, until the
/// loop has no work; then returns what the loop did in a run until idle.
/// Each transport is dropped before the application workers and archive lock.
/// Why `serve` failed: before the daemon served its socket, or while it ran.
#[derive(Debug)]
pub enum Failed {
    Start(std::io::Error),
    Run(std::io::Error),
}
impl Failed {
    pub fn error(&self) -> &std::io::Error {
        match self {
            Self::Start(error) | Self::Run(error) => error,
        }
    }
}
impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error().fmt(f)
    }
}

pub fn serve(
    startup: &crate::config::Startup,
    run: Run,
) -> Result<Option<polylogue_engine::RunSummary>, Failed> {
    raise_open_file_limit();
    let supervisor =
        super::lifecycle::Supervisor::start(&startup.lifecycle).map_err(Failed::Start)?;
    let root = &startup.config.archive;
    let mut engine = startup.engine.clone();
    engine.reads_sources = run.reads_sources;
    let runtime = service::start(
        Arc::clone(&startup.config),
        engine,
        startup.mode.clone(),
        run.lifetime,
        supervisor.progress(),
    )
    .map_err(Failed::Start)?;
    let idle = runtime.idle();
    let mut steps = supervisor
        .progress()
        .steps(super::lifecycle::Phase::Opening);
    steps.next("binding the socket");
    let server = Server::start(Arc::clone(&runtime.daemon)).map_err(Failed::Start)?;
    polylogue_store::log!(
        info,
        "serving {} on {}{}",
        root.display(),
        runtime.daemon.archive.socket().display(),
        if run.reads_sources {
            ""
        } else {
            ", reading no sources"
        }
    );
    steps.next("starting the web reader");
    let web = match crate::web::start(Arc::clone(&runtime.daemon)) {
        Ok(web) => {
            polylogue_store::log!(info, "web reader on http://127.0.0.1:{}/", web.port);
            Some(web)
        }
        Err(error) => {
            polylogue_store::log!(warning, "web reader not started: {error}");
            None
        }
    };
    steps.finish();
    supervisor.run(runtime, server, web).map_err(Failed::Run)?;
    let summary = idle.borrow().clone();
    Ok(summary)
}

/// Owns the socket listener and every accepted client thread.
pub struct Server {
    stop: Option<UnixStream>,
    thread: Option<JoinHandle<std::io::Result<()>>>,
    socket: Option<PathBuf>,
    listener_stopped: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl Server {
    pub fn start(daemon: Arc<Daemon>) -> std::io::Result<Self> {
        let listener = bind(&daemon)?;
        let (stop, stopped) = UnixStream::pair()?;
        let (stopped_sender, listener_stopped) = tokio::sync::oneshot::channel();
        let shared = Arc::new(Shared {
            daemon: Arc::clone(&daemon),
            hangups: super::hangup::Watcher::start()?,
            clients: Clients::default(),
        });
        let mut server = Self {
            stop: Some(stop),
            thread: None,
            socket: Some(daemon.archive.socket()),
            listener_stopped: Some(listener_stopped),
        };
        let accept = Accept { listener, stopped };
        server.thread = Some(polylogue_model::spawn_named(
            "polylogue-socket",
            move || listen(accept, shared, stopped_sender),
        )?);
        Ok(server)
    }

    /// Asks the listener to stop without waiting for it.
    pub(crate) fn request_stop(&mut self) {
        if let Some(mut stop) = self.stop.take() {
            let _ = stop.write_all(&[1]);
        }
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.request_stop();
        self.join()
    }

    fn join(&mut self) -> std::io::Result<()> {
        let result = self.thread.take().map_or(Ok(()), |thread| {
            thread
                .join()
                .map_err(|_| std::io::Error::other("socket listener panicked"))?
        });
        let cleanup =
            self.socket
                .take()
                .map_or(Ok(()), |socket| match std::fs::remove_file(socket) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    result => result,
                });
        result.and(cleanup)
    }

    pub(crate) fn take_listener_stopped(
        &mut self,
    ) -> std::io::Result<tokio::sync::oneshot::Receiver<()>> {
        self.listener_stopped
            .take()
            .ok_or_else(|| std::io::Error::other("socket lifecycle already observed"))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            polylogue_store::log!(error, "socket shutdown: {error}");
        }
    }
}

/// Binds the archive's socket. A socket file nobody answers on is left
/// from a stopped daemon; the archive lock held by `daemon` proves no
/// other daemon owns this archive.
pub fn bind(daemon: &Daemon) -> std::io::Result<UnixListener> {
    crate::socket::bind(&daemon.archive.socket())
}

/// The listener and the stop signal the accept thread blocks on.
struct Accept {
    listener: UnixListener,
    stopped: UnixStream,
}

impl Accept {
    /// Blocks until a connection arrives or the stop signal is written;
    /// `None` on stop.
    fn next(&self) -> std::io::Result<Option<UnixStream>> {
        use rustix::event::{PollFd, PollFlags, poll};
        loop {
            let mut events = [
                PollFd::new(&self.listener, PollFlags::IN),
                PollFd::new(&self.stopped, PollFlags::IN),
            ];
            match poll(&mut events, None) {
                Ok(_) => {}
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => return Err(error.into()),
            }
            if !events[1].revents().is_empty() {
                return Ok(None);
            }
            match self.listener.accept() {
                Ok((stream, _)) => return Ok(Some(stream)),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(error) => return Err(error),
            }
        }
    }
}

fn listen(
    accept: Accept,
    shared: Arc<Shared>,
    stopped_sender: tokio::sync::oneshot::Sender<()>,
) -> std::io::Result<()> {
    let mut failure = accept.listener.set_nonblocking(true).err();
    while failure.is_none() {
        let stream = match accept.next() {
            Ok(Some(stream)) => stream,
            Ok(None) => break,
            Err(error) => {
                failure = Some(error);
                break;
            }
        };
        if let Err(error) = start_client(&shared, stream) {
            failure = Some(error);
        }
    }
    drop(accept);
    let _ = stopped_sender.send(());
    let clients = &shared.clients;
    let left = clients.drain(DRAIN_RESPONSES, DRAIN_BUDGET);
    if left > 0 {
        polylogue_store::log!(
            warning,
            "socket drain left {left} connection(s) whose requests ignored cancellation"
        );
    }
    if clients.panicked.load(std::sync::atomic::Ordering::Acquire) {
        failure.get_or_insert_with(|| std::io::Error::other("socket client panicked"));
    }
    failure.map_or(Ok(()), Err)
}

/// Serves one connection on a thread of its own; the thread removes the
/// connection from the open set when it ends, however it ends.
fn start_client(shared: &Arc<Shared>, stream: UnixStream) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    let id = shared
        .clients
        .next
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    shared.clients.open.locked().push((id, stream.try_clone()?));
    struct Close<'a>(&'a Clients, u64);
    impl Drop for Close<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0
                    .panicked
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            self.0.open.locked().retain(|(id, _)| *id != self.1);
            self.0.closed.notify_all();
        }
    }
    let worker = Arc::clone(shared);
    let started = polylogue_model::spawn_named("polylogue-client", move || {
        let _close = Close(&worker.clients, id);
        serve_client(stream, &worker);
    });
    if let Err(error) = started {
        shared.clients.open.locked().retain(|(open, _)| *open != id);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
