//! Shared SSE framing for finite operation streams.
use crate::daemon::{Request, Response, service::Daemon};
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

pub fn error(output: &mut dyn Write, response: &Response) -> io::Result<()> {
    event(output, "error", response)
}

pub fn produce(
    request: &Request,
    daemon: &Daemon,
    output: &mut dyn Write,
) -> Option<Result<(), Response>> {
    crate::commands::COMMANDS.iter().find_map(|command| {
        if let crate::commands::Action::Stream { produce, .. } = command.action {
            produce(request, daemon, output)
        } else {
            None
        }
    })
}
