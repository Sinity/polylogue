//! One socket client for CLI, MCP, browser-host and hook requests.
use crate::daemon::{Request, Response};
use std::io;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

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
        .map_err(io::Error::other)
}

pub fn send(socket: &Path, request: &Request) -> io::Result<Response> {
    runtime()?.block_on(exchange(socket, request))
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
    runtime()?.block_on(async {
        tokio::time::timeout_at(deadline.into(), exchange(socket, request))
            .await
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    "daemon did not answer before the deadline",
                )
            })?
    })
}

async fn exchange(socket: &Path, request: &Request) -> io::Result<Response> {
    let mut stream = tokio::net::UnixStream::connect(socket).await?;
    let mut line = serde_json::to_vec(request).map_err(io::Error::other)?;
    line.push(b'\n');
    stream.write_all(&line).await?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply).await?;
    serde_json::from_str(&reply).map_err(io::Error::other)
}
