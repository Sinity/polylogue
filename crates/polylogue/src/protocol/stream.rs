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
    authorship: &polylogue_model::authorship::Authorship,
    output: &mut dyn Write,
) -> Option<Result<(), Response>> {
    crate::commands::COMMANDS.iter().find_map(|command| {
        if let crate::commands::Action::Stream { produce, .. } = command.action {
            produce(request, daemon, authorship, output)
        } else {
            None
        }
    })
}

/// Data writes yield regularly so revoked scopes and cancellations release readers.
pub(crate) struct GuardedWriter<'a> {
    output: &'a mut dyn Write,
    limit: polylogue_store::read::limit::ReadLimit,
    failure: Option<Response>,
}
impl<'a> GuardedWriter<'a> {
    pub fn new(output: &'a mut dyn Write, limit: polylogue_store::read::limit::ReadLimit) -> Self {
        Self {
            output,
            limit,
            failure: None,
        }
    }
    pub fn failure(&mut self) -> Option<Response> {
        self.failure.take()
    }
}
impl Write for GuardedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        loop {
            if let Err(error) = self.limit.check() {
                self.failure = Some(crate::daemon::service::store_error(&error));
                return Err(io::Error::other(error));
            }
            match self.output.write(bytes) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

pub(crate) fn produce_limited(
    request: &Request,
    daemon: &Daemon,
    authorship: &polylogue_model::authorship::Authorship,
    output: &mut dyn Write,
    limit: &polylogue_store::read::limit::ReadLimit,
) -> Option<Result<(), Response>> {
    let mut writer = GuardedWriter::new(output, limit.clone());
    let result = produce(request, daemon, authorship, &mut writer);
    match writer.failure() {
        Some(failure) => Some(Err(failure)),
        None => result,
    }
}
