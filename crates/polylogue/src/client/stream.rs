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
    // Retry connection only. Resending after an event could duplicate output.
    let mut socket =
        crate::cli::wait_for_ready(archive, deadline, || UnixStream::connect(archive.socket()))?;
    socket.set_read_timeout(Some(
        crate::daemon::execution::remaining(deadline) + Duration::from_secs(1),
    ))?;
    socket.set_write_timeout(Some(duration))?;
    serde_json::to_writer(
        &mut socket,
        &crate::daemon::execution::Outgoing::new(request, deadline),
    )
    .map_err(io::Error::other)?;
    socket.write_all(b"\n")?;
    receive(socket, event)
}

pub(crate) fn receive(
    mut input: impl Read,
    mut event: impl FnMut(SseEvent) -> io::Result<bool>,
) -> Result<(), Error> {
    let mut decoder = SseDecoder::new();
    let mut bytes = [0; 8192];
    loop {
        let count = input.read(&mut bytes)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the operation stream stopped before completion",
            )
            .into());
        }
        for frame in decoder
            .push(&bytes[..count])
            .map_err(|error| Error::Answer(Answer::error(error)))?
        {
            if frame.event == "error" {
                let answer: Answer<serde_json::Value> =
                    serde_json::from_str(&frame.data).map_err(io::Error::other)?;
                return Err(match answer {
                    answer @ Answer::Error { .. } => Error::Answer(answer),
                    _ => io::Error::new(
                        io::ErrorKind::InvalidData,
                        "an error event must contain a failure answer",
                    )
                    .into(),
                });
            }
            if event(frame)? {
                return Ok(());
            }
        }
    }
}
