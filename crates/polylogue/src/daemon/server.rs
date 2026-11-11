//! Unix socket framing and listener lifecycle. Application operations live
//! in `service`; ordinary requests receive one response, finite feeds a batch.
//!
//! One thread blocks on the listener and wakes only for a connection or the
//! stop signal. Each connection has one thread that reads its requests,
//! hands their work to the executor and sends the answers; one shared
//! watcher notices hang-ups. Sending follows one rule: a send waits for the
//! client until the request's deadline, or until the client hangs up, and
//! never while holding a reader or an executor place.

use super::service::{self, Daemon, error};
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The largest piece written between two checks of the access scope.
const SEND_CHUNK: usize = 64 * 1024;

/// Writes `bytes` under the send rule: each piece waits for the client until
/// `deadline` (a send past it still takes what the socket accepts at once),
/// and the access scope is checked before each piece, outside the scope
/// fence. A hang-up or a stopping daemon ends a blocked write at once.
fn send(
    output: &mut UnixStream,
    bytes: &[u8],
    scopes: &crate::protocol::ReadScopes,
    deadline: Option<Instant>,
) -> Result<(), SendError> {
    for piece in bytes.chunks(SEND_CHUNK) {
        scopes.check().map_err(SendError::Scope)?;
        let wait = deadline.map(|deadline| {
            deadline
                .saturating_duration_since(Instant::now())
                .max(Duration::from_millis(1))
        });
        output.set_write_timeout(wait).map_err(SendError::Io)?;
        output.write_all(piece).map_err(SendError::Io)?;
    }
    Ok(())
}

enum SendError {
    Scope(polylogue_store::StoreError),
    Io(std::io::Error),
}

/// Sends one finite answer. Serialization happens before the first byte;
/// an answer whose scope changed before anything was sent is replaced by the
/// safe failure, and one that changes mid-answer ends the connection, since a
/// truncated line is never a complete answer.
fn send_answer(
    output: &mut UnixStream,
    answer: &impl serde::Serialize,
    scopes: &crate::protocol::ReadScopes,
    deadline: Option<Instant>,
) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(answer).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    let mut scopes = scopes;
    let unscoped = crate::protocol::ReadScopes::default();
    if let Err(error) = scopes.check() {
        // The safe failure carries nothing from the old scope.
        let mut failure =
            serde_json::to_vec(&service::store_error(&error)).map_err(std::io::Error::other)?;
        failure.push(b'\n');
        bytes = failure;
        scopes = &unscoped;
    }
    match send(output, &bytes, scopes, deadline) {
        Ok(()) => Ok(()),
        Err(SendError::Scope(error)) => Err(std::io::Error::other(error)),
        Err(SendError::Io(error)) => Err(error),
    }
}

/// The liveness check readiness and the systemd watchdog send. The client's
/// thread answers it with no reader and no cache, so a long vector load or a
/// busy reader pool never fails it.
pub(crate) const PING: &str = "ping";

fn write_answer(out: &mut UnixStream, answer: &command_surface::Answer<serde_json::Value>) -> bool {
    send_answer(out, answer, &Default::default(), None).is_ok()
}

/// What every connection of one server shares.
struct Shared {
    daemon: Arc<Daemon>,
    operations: Arc<command_surface::OperationTable>,
    /// Drives the operation table's futures; the work itself runs on the
    /// executor.
    runtime: tokio::runtime::Runtime,
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

impl Clients {
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(u64, UnixStream)>> {
        self.open
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn serve_client(stream: UnixStream, shared: &Shared) {
    let d = &shared.daemon;
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let cancellation = polylogue_store::read::limit::ReadCancellation::default();
    let Ok(_watched) = shared.hangups.watch(&stream, cancellation.clone()) else {
        return;
    };
    let mut out = stream;
    for line in BufReader::new(read_half).lines() {
        let Ok(line) = line else {
            return;
        };
        if line.trim().is_empty() {
            continue;
        }
        let incoming = match crate::protocol::wire::incoming(&line) {
            Some(Ok(incoming)) => incoming,
            Some(Err(error)) => {
                let answer = command_surface::Answer::error(crate::protocol::failure(
                    "bad_request",
                    error.to_string(),
                ));
                if !write_answer(&mut out, &answer) {
                    return;
                }
                continue;
            }
            None => {
                let answer = error("bad_request", "a request names its operation");
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
            mut request,
            caller,
            parent_deadline_ms,
        } = incoming;
        let (evidence, channel) =
            crate::protocol::caller::peer(&out, &caller).unwrap_or_else(|_| {
                (
                    Default::default(),
                    polylogue_model::authorship::Channel::Cli,
                )
            });
        let mut accepted =
            crate::protocol::caller::accept(&d.callers, &evidence, channel, cancellation.clone());
        accepted.parent = parent_deadline_ms.map(|deadline| {
            crate::operation::ParentLifetime::capture(
                super::execution::limit(Some(deadline), cancellation.clone()).deadline(),
                cancellation.clone(),
            )
        });
        accepted.invocation = Some(
            d.invocation(&accepted.authorship)
                .with_accepted_request(Some(Arc::new(request.clone()))),
        );
        accepted.working_directory = crate::protocol::caller::working_directory(&out).ok();
        let caller = crate::commands::Caller {
            surface: match channel {
                polylogue_model::authorship::Channel::Mcp => crate::commands::Surface::Mcp,
                polylogue_model::authorship::Channel::Hook => crate::commands::Surface::Hook,
                _ => crate::commands::Surface::Cli,
            },
            cwd: accepted.working_directory.clone(),
        };
        let Some(command) = crate::commands::command(&request.operation) else {
            let answer = error(
                "bad_request",
                format!("unknown operation {}", request.operation),
            );
            if !write_answer(&mut out, &answer) {
                return;
            }
            continue;
        };
        if let crate::commands::Run::Feed { registration, .. } = command.run {
            let limit = stream_limit(&request, command, true, &accepted, &cancellation);
            let arguments = request.arguments.clone();
            if !send_stream(
                &mut out,
                d,
                super::executor::Class::Ordinary,
                &accepted,
                limit,
                Box::new(move |d, _, context, output| {
                    registration.produce(&arguments, d, context, output)
                }),
            ) {
                return;
            }
            continue;
        }
        let Some(registration) = command.registration() else {
            let answer = error(
                "bad_request",
                format!("{} runs in the calling process", command.name),
            );
            if !write_answer(&mut out, &answer) {
                return;
            }
            continue;
        };
        let policy = match registration.admit(&request.arguments, command, &caller) {
            Ok(policy) => policy,
            Err(failure) => {
                if !write_answer(&mut out, &failure.answer()) {
                    return;
                }
                continue;
            }
        };
        // A write keeps its result instead of inheriting a read deadline.
        if !policy.read_phase {
            request.deadline = command_surface::Deadline::None;
        }
        let read_scope = if policy.read_only {
            match d.reads.scope_token() {
                Ok(scope) => Some(scope),
                Err(error) => {
                    if !write_answer(&mut out, &service::store_error(&error)) {
                        return;
                    }
                    continue;
                }
            }
        } else {
            None
        };
        accepted.read_scope = read_scope.clone();
        if policy.streaming {
            let limit = stream_limit(
                &request,
                command,
                policy.default_deadline,
                &accepted,
                &cancellation,
            );
            let arguments = request.arguments.clone();
            let produce_caller = caller.clone();
            if !send_stream(
                &mut out,
                d,
                policy.class,
                &accepted,
                limit,
                Box::new(move |d, invocation, context, output| {
                    registration.produce(
                        &arguments,
                        command,
                        d,
                        invocation,
                        &produce_caller,
                        context,
                        output,
                    )
                }),
            ) {
                return;
            }
            continue;
        }
        let deadline = stream_limit(&request, command, true, &accepted, &cancellation).deadline();
        let via = if channel == polylogue_model::authorship::Channel::Mcp {
            command_surface::Via::Mcp
        } else {
            command_surface::Via::Cli
        };
        let trusted = command_surface::Caller::authenticated(accepted.scope(), accepted)
            .with_access(command_surface::Access::Write);
        let answer = shared.runtime.block_on(shared.operations.call_as(
            request,
            via,
            trusted,
            Default::default(),
        ));
        let publication = crate::protocol::ReadScopes::single(read_scope);
        let answer = crate::protocol::scope_answer(answer, &publication);
        if send_answer(&mut out, &answer, &publication, deadline).is_err() {
            return;
        }
    }
}

/// The read limit of a stream: the request's deadline (the default one when
/// it names none and its row has one), within the caller's own bound.
fn stream_limit(
    request: &command_surface::RunnableRequest,
    command: &crate::commands::Command,
    default_deadline: bool,
    accepted: &crate::protocol::caller::Accepted,
    cancellation: &polylogue_store::read::limit::ReadCancellation,
) -> polylogue_store::read::limit::ReadLimit {
    use polylogue_store::read::limit::ReadLimit;
    let declared = crate::commands::deadlines().get(command.latency);
    let duration = match request.deadline {
        command_surface::Deadline::Default if !default_deadline => None,
        command_surface::Deadline::Default if declared.is_none() => {
            Some(super::execution::timeout())
        }
        deadline => deadline.duration(declared),
    };
    let mut limit = duration.map_or_else(
        || ReadLimit::without_deadline(cancellation.clone()),
        |duration| ReadLimit::new(duration, cancellation.clone()),
    );
    if let Some(parent) = &accepted.parent {
        let parent_limit = parent.read_limit(cancellation.clone());
        if parent_limit
            .deadline()
            .is_some_and(|parent| limit.deadline().is_none_or(|request| parent < request))
        {
            limit = parent_limit;
        }
    }
    limit
}

/// Starts a stream on the executor and sends its events as they come,
/// outside the producer's reader; `false` when the connection is unusable.
fn send_stream(
    out: &mut UnixStream,
    d: &Arc<Daemon>,
    class: super::executor::Class,
    accepted: &crate::protocol::caller::Accepted,
    limit: polylogue_store::read::limit::ReadLimit,
    produce: super::streaming::Produce,
) -> bool {
    let deadline = limit.deadline();
    let invocation = accepted
        .invocation
        .clone()
        .expect("an accepted request captures its invocation");
    let scopes = crate::protocol::ReadScopes::single(d.reads.scope_token().ok());
    let input = match super::streaming::start(d, class, limit, invocation, produce) {
        Ok(input) => input,
        Err(failure) => return crate::protocol::stream::error(out, failure).is_ok(),
    };
    loop {
        let next = match input.next_blocking() {
            Ok(next) => next,
            Err(error) => {
                return crate::protocol::stream::error(
                    out,
                    crate::daemon::Failure::new("io", error),
                )
                .is_ok();
            }
        };
        match next {
            super::stream_buffer::Next::Bytes(bytes) => {
                if send(out, &bytes, &scopes, deadline).is_err() {
                    return false;
                }
            }
            super::stream_buffer::Next::End => return true,
            super::stream_buffer::Next::Failed(failure) => {
                return crate::protocol::stream::error(out, failure).is_ok();
            }
        }
    }
}

/// Runs the daemon until its listener stops. Each transport is dropped
/// before the application workers and archive lock.
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
            eprintln!("polylogue: raise the open file limit: {e}");
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
pub fn serve(
    startup: &crate::config::Startup,
    run: Run,
) -> std::io::Result<Option<polylogue_engine::RunSummary>> {
    raise_open_file_limit();
    let supervisor = super::lifecycle::Supervisor::start(&startup.lifecycle)?;
    let root = &startup.config.archive;
    let mut engine = startup.engine.clone();
    engine.reads_sources = run.reads_sources;
    let runtime = service::start(
        Arc::clone(&startup.config),
        engine,
        startup.mode.clone(),
        run.lifetime,
        startup.stopped_attempt,
    )?;
    let idle = runtime.idle();
    let server = Server::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations))?;
    eprintln!(
        "polylogue: serving {} on {}{}",
        root.display(),
        runtime.daemon.archive.socket().display(),
        if run.reads_sources {
            ""
        } else {
            ", reading no sources"
        }
    );
    let web = match crate::web::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations))
    {
        Ok(web) => {
            eprintln!("polylogue: web reader on http://127.0.0.1:{}/", web.port);
            Some(web)
        }
        Err(error) => {
            eprintln!("polylogue: web reader not started: {error}");
            None
        }
    };
    supervisor.run(runtime, server, web)?;
    let summary = idle.borrow().clone();
    Ok(summary)
}

/// Owns the socket listener and every accepted client thread.
pub struct Server {
    stop: Option<UnixStream>,
    thread: Option<JoinHandle<std::io::Result<()>>>,
    socket: Option<PathBuf>,
    quiescing: Option<tokio::sync::oneshot::Receiver<()>>,
    #[cfg(test)]
    wakeups: Arc<std::sync::atomic::AtomicU64>,
}

impl Server {
    pub fn start(
        daemon: Arc<Daemon>,
        operations: Arc<command_surface::OperationTable>,
    ) -> std::io::Result<Self> {
        let listener = bind(&daemon)?;
        let (stop, stopped) = UnixStream::pair()?;
        let (quiesce, quiescing) = tokio::sync::oneshot::channel();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("polylogue-socket-timers")
            .enable_time()
            .build()?;
        let shared = Arc::new(Shared {
            daemon: Arc::clone(&daemon),
            operations,
            runtime,
            hangups: super::hangup::Watcher::start()?,
            clients: Clients::default(),
        });
        #[cfg(test)]
        let wakeups = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut server = Self {
            stop: Some(stop),
            thread: None,
            socket: Some(daemon.archive.socket()),
            quiescing: Some(quiescing),
            #[cfg(test)]
            wakeups: Arc::clone(&wakeups),
        };
        let accept = Accept {
            listener,
            stopped,
            #[cfg(test)]
            wakeups,
        };
        server.thread = Some(
            std::thread::Builder::new()
                .name("polylogue-socket".into())
                .spawn(move || listen(accept, shared, quiesce))?,
        );
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

    pub(crate) fn take_quiescing(&mut self) -> std::io::Result<tokio::sync::oneshot::Receiver<()>> {
        self.quiescing
            .take()
            .ok_or_else(|| std::io::Error::other("socket lifecycle already observed"))
    }

    /// How often the listener thread has woken.
    #[cfg(test)]
    pub(crate) fn wakeups(&self) -> u64 {
        self.wakeups.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: socket shutdown: {error}");
        }
    }
}

/// Binds the archive's socket. A socket file nobody answers on is left
/// from a stopped daemon; the archive lock held by `daemon` proves no
/// other daemon owns this archive.
pub fn bind(daemon: &Daemon) -> std::io::Result<UnixListener> {
    let socket = daemon.archive.socket();
    if socket.exists() {
        std::fs::remove_file(&socket)?;
    }
    UnixListener::bind(&socket)
}

/// The listener and the stop signal the accept thread blocks on.
struct Accept {
    listener: UnixListener,
    stopped: UnixStream,
    #[cfg(test)]
    wakeups: Arc<std::sync::atomic::AtomicU64>,
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
            #[cfg(test)]
            self.wakeups
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    quiesce: tokio::sync::oneshot::Sender<()>,
) -> std::io::Result<()> {
    let mut failure = None;
    if let Err(error) = accept.listener.set_nonblocking(true) {
        failure = Some(error);
    }
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
    let _ = quiesce.send(());
    // Stop receiving new input while complete queued requests can still receive
    // their refusal and an accepted write can publish its durable receipt.
    let clients = &shared.clients;
    for (_, stream) in clients.lock().iter() {
        let _ = stream.shutdown(Shutdown::Read);
    }
    let drain_until = Instant::now() + Duration::from_secs(5);
    let mut open = clients.lock();
    while !open.is_empty() {
        let left = drain_until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        open = clients
            .closed
            .wait_timeout(open, left)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .0;
    }
    // Unresponsive peers cannot hold shutdown past the response drain window:
    // the shutdown ends blocked sends and the hang-up cancels their reads.
    for (_, stream) in open.iter() {
        let _ = stream.shutdown(Shutdown::Both);
    }
    while !open.is_empty() {
        open = clients
            .closed
            .wait(open)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    drop(open);
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
    shared.clients.lock().push((id, stream.try_clone()?));
    struct Close<'a>(&'a Clients, u64);
    impl Drop for Close<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0
                    .panicked
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            self.0.lock().retain(|(id, _)| *id != self.1);
            self.0.closed.notify_all();
        }
    }
    let worker = Arc::clone(shared);
    let started = std::thread::Builder::new()
        .name("polylogue-client".into())
        .spawn(move || {
            let _close = Close(&worker.clients, id);
            serve_client(stream, &worker);
        });
    if let Err(error) = started {
        shared.clients.lock().retain(|(open, _)| *open != id);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
