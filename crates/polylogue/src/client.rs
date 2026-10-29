//! One socket client for CLI, MCP, browser-host and hook requests.
pub mod stream;
use crate::daemon::{Request, Response};
use polylogue_store::Archive;
use std::io;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
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

pub fn send(socket: &Path, request: &Request) -> io::Result<Response> {
    let deadline = crate::daemon::execution::request_deadline(request, None);
    runtime()?.block_on(async {
        if let Some(deadline_ms) = deadline {
            match tokio::time::timeout(
                crate::daemon::execution::remaining(Some(deadline_ms)) + Duration::from_secs(1),
                exchange(socket, request, deadline),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Ok(crate::daemon::service::error(
                    "deadline_exceeded",
                    "the daemon did not answer before the deadline",
                )),
            }
        } else {
            exchange(socket, request, deadline).await
        }
    })
}

/// The budget covers connection, writing and reading the complete answer.
pub fn send_with_budget(
    socket: &Path,
    request: &Request,
    budget: Duration,
) -> io::Result<Response> {
    let deadline = Instant::now()
        .checked_add(budget)
        .ok_or_else(|| io::Error::other("socket deadline is out of range"))?;
    let wire_deadline = crate::daemon::execution::request_deadline(
        request,
        Some(crate::daemon::execution::deadline(budget)),
    );
    runtime()?.block_on(async {
        tokio::time::timeout_at(deadline.into(), exchange(socket, request, wire_deadline))
            .await
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    "daemon did not answer before the deadline",
                )
            })?
    })
}

pub fn send_when_ready(
    archive: &Archive,
    request: &Request,
    deadline_ms: Option<i64>,
) -> io::Result<Response> {
    runtime()?.block_on(send_ready(
        archive,
        request,
        deadline_ms,
        &CancellationToken::new(),
    ))
}

/// Cancelling the future drops its socket before the caller finishes.
pub(crate) async fn send_ready(
    archive: &Archive,
    request: &Request,
    deadline_ms: Option<i64>,
    cancelled: &CancellationToken,
) -> io::Result<Response> {
    send_ready_as(
        archive,
        request,
        deadline_ms,
        cancelled,
        crate::protocol::caller::evidence(),
    )
    .await
}

pub(crate) async fn send_ready_as(
    archive: &Archive,
    request: &Request,
    deadline_ms: Option<i64>,
    cancelled: &CancellationToken,
    caller: polylogue_model::authorship::CallerEvidence,
) -> io::Result<Response> {
    let deadline_ms = crate::daemon::execution::request_deadline(request, deadline_ms);
    let exchange = async {
        let stream = crate::cli::when_ready(archive, deadline_ms, || {
            tokio::net::UnixStream::connect(archive.socket())
        })
        .await?;
        exchange_stream(stream, request, deadline_ms, caller).await
    };
    let work = async {
        if let Some(deadline_ms) = deadline_ms {
            match tokio::time::timeout(
                crate::daemon::execution::remaining(Some(deadline_ms)) + Duration::from_secs(1),
                exchange,
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Ok(crate::daemon::service::error(
                    "deadline_exceeded",
                    "the daemon did not answer before the deadline",
                )),
            }
        } else {
            exchange.await
        }
    };
    tokio::select! {
        response = work => response,
        _ = cancelled.cancelled() => Ok(crate::daemon::service::error("cancelled", "the request was cancelled")),
    }
}

async fn exchange(
    socket: &Path,
    request: &Request,
    deadline_ms: Option<i64>,
) -> io::Result<Response> {
    exchange_stream(
        tokio::net::UnixStream::connect(socket).await?,
        request,
        deadline_ms,
        crate::protocol::caller::evidence(),
    )
    .await
}

async fn exchange_stream(
    stream: tokio::net::UnixStream,
    request: &Request,
    deadline_ms: Option<i64>,
    caller: polylogue_model::authorship::CallerEvidence,
) -> io::Result<Response> {
    let outgoing = crate::daemon::execution::Outgoing {
        request,
        deadline_ms,
        caller,
    };
    exchange_message(stream, &outgoing).await
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
    serde_json::from_str(&reply).map_err(io::Error::other)
}

/// Executes a declared typed operation and retains its entire shared answer.
pub(crate) async fn call_ready_as(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
    cancelled: &CancellationToken,
    caller: polylogue_model::authorship::CallerEvidence,
) -> io::Result<command_surface::Answer<serde_json::Value>> {
    let class = crate::commands::command(&request.operation)
        .map(|command| command.latency)
        .unwrap_or(command_surface::LatencyClass::Local);
    let budget = request
        .deadline
        .socket_timeout(crate::protocol::operations::deadlines().get(class));
    let ready_deadline = budget.map(crate::daemon::execution::deadline);
    let outgoing = crate::protocol::wire::Outgoing { request, caller };
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
                    "the daemon did not answer before the deadline".into(),
                    None,
                ))),
            }
        } else {
            work.await
        }
    };
    tokio::select! {
        answer = work => answer,
        _ = cancelled.cancelled() => Ok(command_surface::Answer::error(crate::protocol::failure(
            "cancelled", "the request was cancelled".into(), None))),
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
