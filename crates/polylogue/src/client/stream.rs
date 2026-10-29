//! One connection and incremental SSE decoder for finite operation streams.
use crate::daemon::Request;
use command_surface::{Answer, SseDecoder, SseEvent};
use polylogue_store::Archive;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("{0:?}")]
    Answer(Answer<serde_json::Value>),
}

pub fn run(
    archive: &Archive,
    request: &Request,
    timeout: Option<Duration>,
    event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> Result<(), Error> {
    let duration = timeout.unwrap_or_else(crate::daemon::execution::timeout);
    let deadline = Some(crate::daemon::execution::deadline(duration));
    let outgoing = crate::daemon::execution::Outgoing::new(request, deadline);
    send(archive, &outgoing, Some(duration), event)
}

/// A typed streaming call uses the same request envelope as a paged call.
pub fn run_call(
    archive: &Archive,
    request: &command_surface::RunnableRequest,
    event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> Result<(), Error> {
    let class = crate::commands::command(&request.operation)
        .map(|command| command.latency)
        .unwrap_or(command_surface::LatencyClass::Local);
    let budget = request
        .deadline
        .socket_timeout(crate::protocol::operations::deadlines().get(class));
    let outgoing = crate::protocol::wire::Outgoing {
        request,
        caller: crate::protocol::caller::evidence(),
    };
    send(archive, &outgoing, budget, event)
}

fn send(
    archive: &Archive,
    outgoing: &impl serde::Serialize,
    budget: Option<Duration>,
    event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> Result<(), Error> {
    let deadline = budget.map(crate::daemon::execution::deadline);
    // Retry connection only. Resending after an event could duplicate output.
    let mut socket =
        crate::cli::wait_for_ready(archive, deadline, || UnixStream::connect(archive.socket()))?;
    socket.set_read_timeout(budget)?;
    socket.set_write_timeout(budget)?;
    serde_json::to_writer(&mut socket, outgoing).map_err(io::Error::other)?;
    socket.write_all(b"\n")?;
    receive(socket, event)
}

pub(crate) fn receive(
    mut input: impl Read,
    mut event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> Result<(), Error> {
    let mut decoder = SseDecoder::new();
    let mut bytes = [0; 8192];
    let mut first = true;
    loop {
        let count = input.read(&mut bytes)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the operation stream stopped before completion",
            )
            .into());
        }
        // Dispatch can refuse before selecting a stream (for example an
        // unknown operation). Its ordinary shared error remains actionable.
        if first && bytes[0] == b'{' {
            let mut reply = bytes[..count].to_vec();
            if !reply.contains(&b'\n') {
                std::io::BufRead::read_until(
                    &mut std::io::BufReader::new(input),
                    b'\n',
                    &mut reply,
                )?;
            }
            let end = reply
                .iter()
                .position(|byte| *byte == b'\n')
                .unwrap_or(reply.len());
            return Err(failure(&reply[..end])?);
        }
        first = false;
        for frame in decoder
            .push(&bytes[..count])
            .map_err(|error| Error::Answer(Answer::error(error)))?
        {
            if frame.event == "error" {
                return Err(failure(frame.data.as_bytes())?);
            }
            if event(frame)? {
                return Ok(());
            }
        }
    }
}

fn failure(bytes: &[u8]) -> io::Result<Error> {
    let answer: Answer<serde_json::Value> =
        serde_json::from_slice(bytes).map_err(io::Error::other)?;
    match answer {
        answer @ Answer::Error { .. } => Ok(Error::Answer(answer)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "an error frame must contain a failure answer",
        )),
    }
}

#[cfg(test)]
mod tests;
