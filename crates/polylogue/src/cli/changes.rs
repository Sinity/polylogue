//! Printing the change feed. A completion line is forwarded only after every
//! preceding line has been validated and written, then stdout is flushed.
use super::{Archive, Request, fail};
use crate::daemon::query::feed::Line;
use polylogue_model::ChangePosition;
#[cfg(test)]
use std::io::Read;
use std::io::{self, Write};
use std::process::ExitCode;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// `changes`: writes one finite batch to stdout, ending with its
/// completion line.
pub fn run(
    _: &crate::commands::Values,
    request: &Request,
    timeout: Option<std::time::Duration>,
) -> ExitCode {
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
    consume(output, previous.as_deref(), |callback| {
        crate::client::stream::run(archive, request, timeout, callback)
    })
}

#[cfg(test)]
fn receive(input: impl Read, output: &mut impl Write, previous: Option<&str>) -> io::Result<u8> {
    consume(output, previous, |callback| {
        crate::client::stream::receive(input, callback)
    })
}

fn consume(
    output: &mut impl Write,
    previous: Option<&str>,
    run: impl FnOnce(
        &mut dyn FnMut(command_surface::SseEvent) -> io::Result<bool>,
    ) -> Result<(), crate::client::stream::Error>,
) -> io::Result<u8> {
    let mut begun = None;
    let mut seen = 0_u64;
    let result = run(&mut |frame| {
        let value: serde_json::Value =
            serde_json::from_str(&frame.data).map_err(io::Error::other)?;
        let line: Line = serde_json::from_value(value.clone()).map_err(io::Error::other)?;
        let expected = match &line {
            Line::Begin { .. } => "begin",
            Line::Change { .. } => "record",
            Line::Complete { .. } => "complete",
        };
        if frame.event != expected {
            return Err(invalid("the stream event does not match its payload"));
        }
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
            return Ok(true);
        }
        Ok(false)
    });
    match result {
        Ok(()) => Ok(0),
        Err(crate::client::stream::Error::Io(error)) => Err(error),
        Err(crate::client::stream::Error::Answer(answer)) => {
            let code = answer.exit_code();
            serde_json::to_writer(&mut *output, &answer)?;
            output.write_all(b"\n")?;
            output.flush()?;
            Ok(code)
        }
    }
}

#[cfg(test)]
mod tests;
