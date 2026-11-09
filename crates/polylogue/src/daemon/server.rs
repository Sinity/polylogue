//! Unix socket framing and listener lifecycle. Application operations live
//! in `service`; ordinary requests receive one response, finite feeds a batch.

use super::service::{self, Daemon, error};
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

/// Serialize before entering the native epoch fence; fence only each bounded write.
fn write_scoped_json(
    output: &mut UnixStream,
    answer: &impl serde::Serialize,
    scopes: &crate::protocol::ReadScopes,
) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(answer).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    output.set_write_timeout(Some(Duration::from_millis(10)))?;
    let until = std::time::Instant::now() + Duration::from_secs(1);
    let mut offset = 0;
    while offset < bytes.len() {
        let end = bytes.len().min(offset + 64 * 1024);
        let published = scopes.with_current(|| Ok(output.write(&bytes[offset..end])));
        let published = match published {
            Ok(result) => result,
            Err(error) if offset == 0 => {
                // No old response bytes escaped; publish the safe failure without its old scope.
                let mut error = serde_json::to_vec(&service::store_error(&error))
                    .map_err(std::io::Error::other)?;
                error.push(b'\n');
                return output.write_all(&error);
            }
            // A truncated JSON line cannot be accepted as a completed response.
            Err(error) => return Err(std::io::Error::other(error)),
        };
        match published {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(written) => offset += written,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) && std::time::Instant::now() < until => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// The liveness check readiness and the systemd watchdog send. The client's
/// thread answers it with no reader and no cache, so a long vector load or a
/// busy reader pool never fails it.
pub(crate) const PING: &str = "ping";

fn write_answer(out: &mut UnixStream, answer: &command_surface::Answer<serde_json::Value>) -> bool {
    writeln!(
        out,
        "{}",
        serde_json::to_string(answer).expect("answer serializes")
    )
    .is_ok()
}

fn serve_client(stream: UnixStream, d: &Arc<Daemon>, operations: &command_surface::OperationTable) {
    if stream
        .set_write_timeout(Some(Duration::from_secs(1)))
        .is_err()
    {
        return;
    }
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let cancellation = polylogue_store::read::limit::ReadCancellation::default();
    let Ok(_watch) = super::disconnect::Watch::start(&stream, cancellation.clone()) else {
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
            write_stream(&mut out, d, &accepted, &limit, |d, context, output| {
                registration.produce(&request.arguments, d, context, output)
            });
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
            let invocation = accepted
                .invocation
                .clone()
                .expect("an accepted request captures its invocation");
            write_stream(&mut out, d, &accepted, &limit, |d, context, output| {
                registration.produce(
                    &request.arguments,
                    command,
                    d,
                    &invocation,
                    &caller,
                    context,
                    output,
                )
            });
            continue;
        }
        let via = if channel == polylogue_model::authorship::Channel::Mcp {
            command_surface::Via::Mcp
        } else {
            command_surface::Via::Cli
        };
        let trusted = command_surface::Caller::authenticated(accepted.scope(), accepted)
            .with_access(command_surface::Access::Write);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("operation runtime starts");
        let answer =
            runtime.block_on(operations.call_as(request, via, trusted, Default::default()));
        let publication = crate::protocol::ReadScopes::single(read_scope);
        let answer = crate::protocol::scope_answer(answer, &publication);
        if write_scoped_json(&mut out, &answer, &publication).is_err() {
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

/// Writes a stream's events to the socket under its read limit; a failure
/// after the stream began ends it with the error answer.
fn write_stream(
    out: &mut UnixStream,
    d: &Daemon,
    accepted: &crate::protocol::caller::Accepted,
    limit: &polylogue_store::read::limit::ReadLimit,
    produce: impl FnOnce(
        &Daemon,
        &Arc<crate::operation::Context>,
        &mut dyn std::io::Write,
    ) -> Result<(), crate::daemon::Failure>,
) {
    let _ = out.set_write_timeout(Some(Duration::from_millis(10)));
    let produced = limit.scope(|| {
        limit.check()?;
        let invocation = accepted
            .invocation
            .clone()
            .expect("an accepted request captures its invocation");
        let context = d.resolve_operation_context(invocation)?;
        let scope = d.reads.scope_token()?;
        let mut writer = GuardedWriter::new(out, limit.clone(), scope);
        let result = produce(d, &context, &mut writer);
        match writer.failure() {
            Some(failure) => Err(failure),
            None => result,
        }
    });
    if let Err(failure) = produced {
        let _ = crate::protocol::stream::error(out, failure);
    }
    let _ = out.set_write_timeout(Some(Duration::from_secs(1)));
}

/// Data writes yield regularly so revoked scopes and cancellations release readers.
pub(crate) struct GuardedWriter<'a> {
    output: &'a mut dyn Write,
    limit: polylogue_store::read::limit::ReadLimit,
    failure: Option<crate::daemon::Failure>,
    scope: polylogue_store::ReadScopeToken,
}
impl<'a> GuardedWriter<'a> {
    pub fn new(
        output: &'a mut dyn Write,
        limit: polylogue_store::read::limit::ReadLimit,
        scope: polylogue_store::ReadScopeToken,
    ) -> Self {
        Self {
            output,
            limit,
            failure: None,
            scope,
        }
    }
    /// Actual transport writes are individually bounded; never fence their retry loop.
    fn activate<T>(
        &mut self,
        publish: impl FnOnce(&mut dyn Write) -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        if let Err(error) = self.limit.check() {
            self.failure = Some(crate::daemon::Failure::store(&error));
            return Err(std::io::Error::other(error));
        }
        match self.scope.with_current(|| Ok(publish(self.output))) {
            Ok(result) => result,
            Err(error) => {
                self.failure = Some(crate::daemon::Failure::store(&error));
                Err(std::io::Error::other(error))
            }
        }
    }
    pub fn failure(&mut self) -> Option<crate::daemon::Failure> {
        self.failure.take()
    }
}
impl Write for GuardedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        loop {
            match self.activate(|output| output.write(bytes)) {
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::TimedOut
                            | std::io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.activate(|output| output.flush())
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
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<std::io::Result<()>>>,
    socket: Option<PathBuf>,
    quiescing: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl Server {
    pub fn start(
        daemon: Arc<Daemon>,
        operations: Arc<command_surface::OperationTable>,
    ) -> std::io::Result<Self> {
        let listener = bind(&daemon)?;
        let (stop, stopped) = channel();
        let (quiesce, quiescing) = tokio::sync::oneshot::channel();
        let mut server = Self {
            stop: Some(stop),
            thread: None,
            socket: Some(daemon.archive.socket()),
            quiescing: Some(quiescing),
        };
        listener.set_nonblocking(true)?;
        server.thread = Some(
            std::thread::Builder::new()
                .name("polylogue-socket".into())
                .spawn(move || listen(listener, daemon, operations, stopped, quiesce))?,
        );
        Ok(server)
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
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

fn listen(
    listener: UnixListener,
    daemon: Arc<Daemon>,
    operations: Arc<command_surface::OperationTable>,
    stopped: Receiver<()>,
    quiesce: tokio::sync::oneshot::Sender<()>,
) -> std::io::Result<()> {
    let mut clients: Vec<(UnixStream, JoinHandle<()>)> = Vec::new();
    let mut failure = None;
    loop {
        for _ in 0..32 {
            match listener.accept() {
                Ok((stream, _)) => {
                    let to_close = match stream.try_clone() {
                        Ok(to_close) => to_close,
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    };
                    let daemon = Arc::clone(&daemon);
                    let operations = Arc::clone(&operations);
                    match std::thread::Builder::new()
                        .name("polylogue-client".into())
                        .spawn(move || serve_client(stream, &daemon, &operations))
                    {
                        Ok(thread) => clients.push((to_close, thread)),
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        let mut i = 0;
        while i < clients.len() {
            if clients[i].1.is_finished() {
                let (_, thread) = clients.swap_remove(i);
                if thread.join().is_err() {
                    failure.get_or_insert_with(|| std::io::Error::other("socket client panicked"));
                }
            } else {
                i += 1;
            }
        }
        if failure.is_some()
            || !matches!(
                stopped.recv_timeout(Duration::from_millis(20)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            )
        {
            break;
        }
    }
    drop(listener);
    let _ = quiesce.send(());
    // Stop receiving new input while complete queued requests can still receive
    // their refusal and an accepted write can publish its durable receipt.
    for (stream, _) in &clients {
        let _ = stream.shutdown(Shutdown::Read);
    }
    let drain_until = std::time::Instant::now() + Duration::from_secs(5);
    while clients.iter().any(|(_, thread)| !thread.is_finished())
        && std::time::Instant::now() < drain_until
    {
        std::thread::sleep(Duration::from_millis(10));
    }
    // Unresponsive peers cannot hold shutdown past the response drain window.
    for (stream, thread) in &clients {
        if !thread.is_finished() {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
    for (_, thread) in clients {
        if thread.join().is_err() {
            failure.get_or_insert_with(|| std::io::Error::other("socket client panicked"));
        }
    }
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;
