//! Printing the change feed. A completion line is forwarded only after every
//! preceding line has been validated and written, then stdout is flushed.
use super::{Archive, Request, Response, fail, wait_for_ready};
use crate::daemon::query::feed::Line;
use polylogue_model::ChangePosition;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::process::ExitCode;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// `changes`: writes one finite batch to stdout, ending with its
/// completion line.
pub fn run(request: &Request, timeout: Option<std::time::Duration>) -> ExitCode {
    let archive = Archive::at(&Archive::default_root());
    match send(&archive, request, &mut io::stdout().lock(), timeout) {
        Ok(code) => ExitCode::from(code),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ) =>
        {
            fail(true, "deadline_exceeded", &e.to_string(), 7)
        }
        Err(e) => fail(true, "io", &e.to_string(), 1),
    }
}

fn send(
    archive: &Archive,
    request: &Request,
    output: &mut impl Write,
    timeout: Option<std::time::Duration>,
) -> io::Result<u8> {
    let Request::Changes { since } = request else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a change-feed request is required",
        ));
    };
    let previous = since
        .as_deref()
        .map(|s| {
            ChangePosition::parse(s)
                .map(|p| p.to_string())
                .ok_or_else(|| invalid("the change position is invalid"))
        })
        .transpose()?;
    // Only the connection is retried. A partial stream is never resent
    // within this output attempt; the caller keeps its previous position.
    let duration = timeout.unwrap_or_else(crate::daemon::execution::timeout);
    let deadline_ms = Some(crate::daemon::execution::deadline(duration));
    let mut socket = wait_for_ready(archive, deadline_ms, || {
        UnixStream::connect(archive.socket())
    })?;
    socket.set_read_timeout(Some(
        crate::daemon::execution::remaining(deadline_ms) + std::time::Duration::from_secs(1),
    ))?;
    socket.set_write_timeout(Some(duration))?;
    let outgoing = crate::daemon::execution::Outgoing {
        request,
        deadline_ms,
    };
    serde_json::to_writer(&mut socket, &outgoing)?;
    socket.write_all(b"\n")?;
    receive(BufReader::new(socket), output, previous.as_deref())
}

fn receive(
    mut input: impl BufRead,
    output: &mut impl Write,
    previous: Option<&str>,
) -> io::Result<u8> {
    let mut begun = None;
    let mut seen = 0_u64;
    let mut text = String::new();
    loop {
        text.clear();
        if input.read_line(&mut text)? == 0 || !text.ends_with('\n') {
            return Err(invalid(
                "the change feed stopped before its completion line",
            ));
        }
        let response: Response = serde_json::from_str(&text).map_err(io::Error::other)?;
        let value = match response {
            Response::Ok { data, .. } => data,
            response @ Response::Error { .. } => {
                serde_json::to_writer(&mut *output, &response)?;
                output.write_all(b"\n")?;
                output.flush()?;
                let Response::Error { kind, .. } = response else {
                    unreachable!()
                };
                return Ok(crate::daemon::execution::exit_code(&kind));
            }
        };
        let line: Line = serde_json::from_value(value.clone()).map_err(io::Error::other)?;
        let complete = match line {
            Line::Begin {
                position,
                generation,
                replacement,
                generation_changed,
            } => {
                if begun.is_some() {
                    return Err(invalid("the change feed repeated its header"));
                }
                let old = previous.and_then(ChangePosition::parse);
                let changed = old.as_ref().is_some_and(|p| p.generation != generation);
                if position.as_deref() != previous
                    || generation.is_empty()
                    || generation_changed != changed
                    || replacement != (old.is_none() || changed)
                {
                    return Err(invalid(
                        "the feed header does not continue the requested position",
                    ));
                }
                begun = Some((position, generation, replacement));
                false
            }
            Line::Change { position, .. } => {
                let Some((previous, _, _)) = &begun else {
                    return Err(invalid("the change feed omitted its header"));
                };
                if &position != previous {
                    return Err(invalid("an event advanced the completed change position"));
                }
                seen = seen
                    .checked_add(1)
                    .ok_or_else(|| invalid("the change count is too large"))?;
                false
            }
            Line::Complete {
                position,
                count,
                replacement,
            } => {
                let Some((previous, generation, expected_replacement)) = &begun else {
                    return Err(invalid("the change feed omitted its header"));
                };
                let target = ChangePosition::parse(&position)
                    .ok_or_else(|| invalid("the completed change position is invalid"))?;
                if &target.generation != generation
                    || count != seen
                    || &replacement != expected_replacement
                {
                    return Err(invalid(
                        "the change feed completion does not describe its delivered batch",
                    ));
                }
                if !replacement
                    && previous
                        .as_ref()
                        .and_then(|p| ChangePosition::parse(p))
                        .is_some_and(|p| {
                            p.generation != target.generation
                                || p.index > target.index
                                || p.archive > target.archive
                        })
                {
                    return Err(invalid(
                        "the change feed completion moved its checkpoint backwards",
                    ));
                }
                true
            }
        };
        output.write_all(polylogue_render::terminal::json(&value, false).as_bytes())?;
        output.write_all(b"\n")?;
        if complete {
            output.flush()?;
            return Ok(0);
        }
    }
}

#[cfg(test)]
mod tests;
