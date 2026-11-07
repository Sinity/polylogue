//! Unix socket framing and listener lifecycle. Application operations live
//! in `service`; ordinary requests receive one response, finite feeds a batch.

use super::Request;
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
        if let Some(incoming) = crate::protocol::wire::incoming(&line) {
            let mut read_scope = None;
            let mut publication = crate::protocol::ReadScopes::default();
            let answer = match incoming {
                Err(error) => command_surface::Answer::error(crate::protocol::failure(
                    "bad_request",
                    error.to_string(),
                )),
                Ok((request, caller, parent_deadline_ms)) => {
                    let (evidence, channel) =
                        crate::protocol::caller::peer(&out, &caller, d.config.requests.machine())
                            .unwrap_or_else(|_| {
                                (
                                    Default::default(),
                                    polylogue_model::authorship::Channel::Cli,
                                )
                            });
                    let mut accepted = crate::protocol::caller::accept(
                        &d.callers,
                        &evidence,
                        channel,
                        cancellation.clone(),
                    );
                    accepted.parent = parent_deadline_ms.map(|deadline| {
                        crate::operation::ParentLifetime::capture(
                            super::execution::limit(Some(deadline), cancellation.clone())
                                .deadline(),
                            cancellation.clone(),
                        )
                    });
                    accepted.invocation = Some(
                        d.invocation(&accepted.authorship)
                            .with_accepted_request(Some(Arc::new(request.clone()))),
                    );
                    accepted.working_directory =
                        crate::protocol::caller::working_directory(&out).ok();
                    let command = crate::commands::command(&request.operation);
                    let caller = crate::commands::Caller {
                        surface: match channel {
                            polylogue_model::authorship::Channel::Mcp => {
                                crate::commands::Surface::Mcp
                            }
                            polylogue_model::authorship::Channel::Hook => {
                                crate::commands::Surface::Hook
                            }
                            _ => crate::commands::Surface::Cli,
                        },
                        cwd: accepted.working_directory.clone(),
                        json: true,
                    };
                    let internal = match command.and_then(|command| command.build_typed) {
                        Some(_) => {
                            match crate::protocol::operations::admit(
                                command.expect("typed command exists"),
                                &request,
                                &caller,
                            ) {
                                Ok(internal) => Some(internal),
                                Err(failure) => {
                                    let answer =
                                        command_surface::Answer::<serde_json::Value>::error(
                                            failure,
                                        );
                                    if writeln!(
                                        out,
                                        "{}",
                                        serde_json::to_string(&answer).expect("answer serializes")
                                    )
                                    .is_err()
                                    {
                                        return;
                                    }
                                    continue;
                                }
                            }
                        }
                        None => None,
                    };
                    accepted.admitted_request = internal.clone();
                    let mut request = request;
                    request.deadline = crate::protocol::operations::admitted_deadline(
                        command,
                        internal.as_ref(),
                        request.deadline,
                    );
                    if crate::protocol::operations::read_scope_required(command, internal.as_ref())
                    {
                        match d.reads.scope_token() {
                            Ok(scope) => read_scope = Some(scope),
                            Err(error) => {
                                let answer = service::store_error(&error);
                                if writeln!(
                                    out,
                                    "{}",
                                    serde_json::to_string(&answer).expect("answer serializes")
                                )
                                .is_err()
                                {
                                    return;
                                }
                                continue;
                            }
                        }
                    }
                    accepted.read_scope = read_scope.clone();
                    if let Some(command) = command
                        && let crate::commands::Action::Stream { .. } = command.action
                        && let Some(internal) = internal.filter(Request::streaming)
                    {
                        let mut limit = request
                            .deadline
                            .duration(crate::protocol::operations::deadlines().get(command.latency))
                            .map_or_else(
                                || {
                                    polylogue_store::read::limit::ReadLimit::without_deadline(
                                        cancellation.clone(),
                                    )
                                },
                                |duration| {
                                    polylogue_store::read::limit::ReadLimit::new(
                                        duration,
                                        cancellation.clone(),
                                    )
                                },
                            );
                        if let Some(parent) = &accepted.parent {
                            let parent_limit = parent.read_limit(cancellation.clone());
                            if parent_limit.deadline().is_some_and(|parent| {
                                limit.deadline().is_none_or(|request| parent < request)
                            }) {
                                limit = parent_limit;
                            }
                        }
                        let _ = out.set_write_timeout(Some(Duration::from_millis(10)));
                        let produced = limit.scope(|| match limit.check() {
                            Ok(()) => {
                                let invocation = accepted
                                    .invocation
                                    .clone()
                                    .expect("canonical request captures admission");
                                d.resolve_operation_context(invocation)
                                    .and_then(|operation| {
                                        crate::protocol::stream::produce_limited(
                                            &internal, d, &operation, &mut out, &limit,
                                        )
                                        .expect("stream command owns its request")
                                    })
                            }
                            Err(error) => Err(crate::daemon::Failure::from(error)),
                        });
                        if let Err(failure) = produced {
                            let _ = crate::protocol::stream::error(&mut out, failure);
                        }
                        let _ = out.set_write_timeout(Some(Duration::from_secs(1)));
                        continue;
                    }
                    let origin = if channel == polylogue_model::authorship::Channel::Mcp {
                        command_surface::Via::Mcp
                    } else {
                        command_surface::Via::Cli
                    };
                    let trusted =
                        command_surface::Caller::authenticated(accepted.scope(), accepted)
                            .with_access(command_surface::Access::Write);
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("operation runtime starts");
                    let dispatched = runtime.block_on(operations.call_as(
                        request,
                        origin,
                        trusted,
                        Default::default(),
                    ));
                    publication = crate::protocol::ReadScopes::single(read_scope.clone());
                    dispatched
                }
            };
            let answer = crate::protocol::scope_answer(answer, &publication);
            if write_scoped_json(&mut out, &answer, &publication).is_err() {
                return;
            }
            continue;
        }
        let incoming = match serde_json::from_str::<super::execution::Incoming>(&line) {
            Ok(incoming) => incoming,
            Err(e) => {
                let _ = writeln!(
                    out,
                    "{}",
                    serde_json::to_string(&error("bad_request", e)).unwrap_or_default()
                );
                continue;
            }
        };
        let (evidence, channel) =
            crate::protocol::caller::peer(&out, &incoming.caller, d.config.requests.machine())
                .unwrap_or_else(|_| {
                    (
                        Default::default(),
                        polylogue_model::authorship::Channel::Cli,
                    )
                });
        let mut accepted =
            crate::protocol::caller::accept(&d.callers, &evidence, channel, cancellation.clone());
        accepted.working_directory = crate::protocol::caller::working_directory(&out).ok();
        let limit = super::execution::limit_for(
            &incoming.request,
            incoming.deadline_ms,
            cancellation.clone(),
        );
        let request = incoming.request;
        let read_scope = if request.read_only() {
            match d.reads.scope_token() {
                Ok(scope) => Some(scope),
                Err(error) => {
                    let _ = writeln!(
                        out,
                        "{}",
                        serde_json::to_string(&service::store_error(&error))
                            .expect("error serializes")
                    );
                    continue;
                }
            }
        } else {
            None
        };
        let _ = out.set_write_timeout(Some(Duration::from_millis(10)));
        let produced = if request.streaming() {
            Some(limit.scope(|| {
                d.operation_context(&accepted.authorship)
                    .and_then(|operation| {
                        crate::protocol::stream::produce_limited(
                            &request, d, &operation, &mut out, &limit,
                        )
                        .expect("stream command owns its request")
                    })
            }))
        } else {
            None
        };
        if let Some(result) = produced {
            if let Err(failure) = result {
                let _ = crate::protocol::stream::error(&mut out, failure);
            }
            let _ = out.set_write_timeout(Some(Duration::from_secs(1)));
            continue;
        }
        let _ = out.set_write_timeout(Some(Duration::from_secs(1)));
        let typed = match &request {
            Request::Status => Some(("status", serde_json::json!({}))),
            Request::Sql { statement } => Some(("sql", serde_json::json!({"statement":statement}))),
            request => crate::protocol::query::Arguments::from_request(request).map(|arguments| {
                (
                    "query",
                    serde_json::to_value(arguments).expect("query arguments serialize"),
                )
            }),
        };
        let mut publication = crate::protocol::ReadScopes::single(read_scope.clone());
        let answer = if let Some((operation, arguments)) = typed {
            let table = operations;
            let cursor = crate::protocol::operations::cursor(&request).map(str::to_owned);
            let request = command_surface::Request {
                operation: operation.into(),
                arguments,
                request_id: uuid::Uuid::now_v7().to_string(),
                deadline: incoming.deadline_ms.map_or(
                    command_surface::Deadline::Default,
                    |deadline| {
                        command_surface::Deadline::Milliseconds(
                            super::execution::remaining(Some(deadline))
                                .as_millis()
                                .try_into()
                                .unwrap_or(u64::MAX),
                        )
                    },
                ),
                budget_tokens: Default::default(),
                cursor,
                fresh: false,
                format: Default::default(),
            };
            let mut accepted = accepted.clone();
            accepted.read_scope = read_scope.clone();
            let scope = accepted.scope();
            let trusted = command_surface::Caller::authenticated(scope, accepted.clone())
                .with_access(command_surface::Access::Write);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("operation runtime starts");
            let dispatched = runtime.block_on(table.call_as(
                request,
                command_surface::Via::Cli,
                trusted,
                Default::default(),
            ));
            publication = crate::protocol::ReadScopes::single(read_scope.clone());
            dispatched
        } else {
            service::handle_with_caller(request, d, &limit, &accepted.authorship)
        };
        let answer = crate::protocol::scope_answer(answer, &publication);
        if write_scoped_json(&mut out, &answer, &publication).is_err() {
            return;
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
    // their refusal and an admitted write can publish its durable receipt.
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
