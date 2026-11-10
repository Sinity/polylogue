//! One socket client for CLI, MCP, browser-host and hook requests.
pub mod stream;
use command_surface::Answer;
use polylogue_store::Archive;
use serde_json::Value;
use std::io;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;

fn runtime() -> io::Result<&'static tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| io::Error::other(error.clone()))
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

/// Sends one request to a running daemon, without waiting for one that is
/// still starting.
pub fn call(
    socket: &Path,
    request: &command_surface::RunnableRequest,
) -> io::Result<Answer<Value>> {
    let budget = connection_budget(request, None);
    let outgoing = crate::protocol::wire::Outgoing {
        request,
        caller: crate::protocol::caller::evidence(),
        parent_deadline_ms: None,
    };
    runtime()?.block_on(async {
        let work = async {
            let stream = tokio::net::UnixStream::connect(socket).await?;
            exchange_message(stream, &outgoing).await
        };
        match budget {
            Some(budget) => match tokio::time::timeout(budget, work).await {
                Ok(result) => result,
                Err(_) => Ok(Answer::error(crate::protocol::failure(
                    "deadline_exceeded",
                    "the daemon did not answer before the deadline",
                ))),
            },
            None => work.await,
        }
    })
}

/// How long a client waits for the answer: the request's read deadline
/// with the protocol's grace for a read-only request, none for one that
/// writes, within the parent's bound.
pub(crate) fn connection_budget(
    request: &command_surface::RunnableRequest,
    parent_deadline_ms: Option<i64>,
) -> Option<Duration> {
    let command = crate::commands::command(&request.operation);
    let class = command.map_or(command_surface::LatencyClass::Local, |command| {
        command.latency
    });
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Cli,
        cwd: None,
    };
    let policy = command.and_then(|command| {
        command.registration().and_then(|registration| {
            registration
                .admit(&request.arguments, command, &caller)
                .ok()
        })
    });
    let deadline = match policy {
        // A request that reads and then writes may wait on the writer after its
        // read finished; the write is the user's, so the client waits for the
        // daemon's answer, which reports a read that overran its deadline itself.
        Some(policy) if !policy.read_only => command_surface::Deadline::None,
        Some(policy)
            if !policy.default_deadline
                && request.deadline == command_surface::Deadline::Default =>
        {
            command_surface::Deadline::None
        }
        Some(policy)
            if policy.default_deadline
                && request.deadline == command_surface::Deadline::Default
                && crate::commands::deadlines().get(class).is_none() =>
        {
            command_surface::Deadline::Milliseconds(
                crate::daemon::execution::DEFAULT_READ_TIMEOUT_MS,
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

async fn exchange_message<T: serde::de::DeserializeOwned>(
    mut stream: tokio::net::UnixStream,
    outgoing: &impl serde::Serialize,
) -> io::Result<T> {
    let mut line = serde_json::to_vec(&outgoing).map_err(io::Error::other)?;
    line.push(b'\n');
    stream.write_all(&line).await?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply).await?;
    if !reply.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "the operation answer ended before its line boundary",
        ));
    }
    serde_json::from_str(&reply).map_err(io::Error::other)
}

/// A hook uses the existing daemon only; its budget covers the entire exchange.
pub(crate) fn call_with_budget(
    socket: &Path,
    request: &command_surface::RunnableRequest,
    budget: Duration,
) -> io::Result<Answer<Value>> {
    let outgoing = crate::protocol::wire::Outgoing {
        request,
        caller: crate::protocol::caller::evidence(),
        parent_deadline_ms: None,
    };
    runtime()?.block_on(async {
        tokio::time::timeout(budget, async {
            let stream = tokio::net::UnixStream::connect(socket).await?;
            exchange_message(stream, &outgoing).await
        })
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "daemon did not answer before the deadline",
            )
        })?
    })
}

/// Executes a declared typed operation and retains its entire shared answer.
pub(crate) async fn call_ready_as(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
    cancelled: &CancellationToken,
    caller: polylogue_model::authorship::CallerEvidence,
) -> io::Result<command_surface::Answer<serde_json::Value>> {
    call_ready_as_bounded(archive, request, cancelled, caller, None).await
}

pub(crate) async fn call_ready_as_bounded(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
    cancelled: &CancellationToken,
    caller: polylogue_model::authorship::CallerEvidence,
    parent_deadline_ms: Option<i64>,
) -> io::Result<Answer<Value>> {
    let budget = connection_budget(request, parent_deadline_ms);
    let ready_deadline = budget.map(crate::daemon::execution::deadline);
    let outgoing = crate::protocol::wire::Outgoing {
        request,
        caller,
        parent_deadline_ms,
    };
    let work = async {
        let stream = crate::cli::when_ready(archive, ready_deadline, || {
            tokio::net::UnixStream::connect(archive.socket())
        })
        .await?;
        exchange_message(stream, &outgoing).await
    };
    let work = async {
        if let Some(budget) = budget {
            match tokio::time::timeout(budget, work).await {
                Ok(result) => result,
                Err(_) => Ok(command_surface::Answer::error(crate::protocol::failure(
                    "deadline_exceeded",
                    "the daemon did not answer before the deadline",
                ))),
            }
        } else {
            work.await
        }
    };
    tokio::select! {
        answer = work => answer,
        _ = cancelled.cancelled() => Ok(command_surface::Answer::error(crate::protocol::failure("cancelled", "the request was cancelled"))),
    }
}

pub(crate) fn call_ready(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
) -> io::Result<command_surface::Answer<serde_json::Value>> {
    runtime()?.block_on(call_ready_as(
        archive,
        request,
        &CancellationToken::new(),
        crate::protocol::caller::evidence(),
    ))
}
