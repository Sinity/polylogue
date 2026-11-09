//! Server-sent-event framing for the daemon's finite streams.
use crate::daemon::Failure;
use serde::Serialize;
use std::io::{self, Write};

pub fn event(output: &mut dyn Write, name: &str, value: &impl Serialize) -> io::Result<()> {
    let event = command_surface::SseEvent {
        id: None,
        event: name.into(),
        data: serde_json::to_string(value)?,
        retry: None,
    };
    let frame = event.encode().map_err(io::Error::other)?;
    output.write_all(frame.as_bytes())?;
    output.flush()
}

/// A failure after the stream began ends it with the error answer.
pub fn error(output: &mut dyn Write, failure: Failure) -> io::Result<()> {
    event(output, "error", &failure.answer::<serde_json::Value>())
}
