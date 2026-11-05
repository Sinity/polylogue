//! Shared SSE framing for finite operation streams.
use crate::daemon::{Failure, Request, service::Daemon};
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

pub fn produce(
    request: &Request,
    daemon: &Daemon,
    operation: &std::sync::Arc<crate::operation::Context>,
    output: &mut dyn Write,
) -> Option<Result<(), Failure>> {
    crate::commands::COMMANDS.iter().find_map(|command| {
        if let crate::commands::Action::Stream { produce, .. } = command.action {
            produce(request, daemon, operation, output)
        } else {
            None
        }
    })
}

/// Data writes yield regularly so revoked scopes and cancellations release readers.
pub(crate) struct GuardedWriter<'a> {
    output: &'a mut dyn Write,
    limit: polylogue_store::read::limit::ReadLimit,
    failure: Option<Failure>,
    scope: polylogue_store::ReadScopeToken,
}
impl<'a> GuardedWriter<'a> {
    pub fn new(
        output: &'a mut dyn Write,
        limit: polylogue_store::read::limit::ReadLimit,
        scope: polylogue_store::ReadScopeToken,
    ) -> Self {
        Self {
            output,
            limit,
            failure: None,
            scope,
        }
    }
    /// Actual transport writes are individually bounded; never fence their retry loop.
    fn activate<T>(
        &mut self,
        publish: impl FnOnce(&mut dyn Write) -> io::Result<T>,
    ) -> io::Result<T> {
        if let Err(error) = self.limit.check() {
            self.failure = Some(Failure::store(&error));
            return Err(io::Error::other(error));
        }
        match self.scope.with_current(|| Ok(publish(self.output))) {
            Ok(result) => result,
            Err(error) => {
                self.failure = Some(Failure::store(&error));
                Err(io::Error::other(error))
            }
        }
    }
    pub fn failure(&mut self) -> Option<Failure> {
        self.failure.take()
    }
}
impl Write for GuardedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        loop {
            match self.activate(|output| output.write(bytes)) {
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
        self.activate(|output| output.flush())
    }
}

pub(crate) fn produce_limited(
    request: &Request,
    daemon: &Daemon,
    operation: &std::sync::Arc<crate::operation::Context>,
    output: &mut dyn Write,
    limit: &polylogue_store::read::limit::ReadLimit,
) -> Option<Result<(), Failure>> {
    let scope = match daemon.reads.scope_token() {
        Ok(scope) => scope,
        Err(error) => return Some(Err(Failure::from(error))),
    };
    let mut writer = GuardedWriter::new(output, limit.clone(), scope);
    let result = produce(request, daemon, operation, &mut writer);
    match writer.failure() {
        Some(failure) => Some(Err(failure)),
        None => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn data_activation_is_scope_fenced_even_before_a_reader_binds_the_limit() {
        let daemon = crate::test_daemon::start("native stream activation fence");
        let scope = daemon.daemon.reads.scope_token().unwrap();
        let limit = polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
        assert!(limit.check().is_ok());
        daemon.daemon.archive.advance_generation_for_test().unwrap();
        let mut bytes = Vec::new();
        let mut writer = GuardedWriter::new(&mut bytes, limit, scope);
        assert!(writer.write_all(b"stale complete event").is_err());
        let failure = writer
            .failure()
            .expect("typed scope failure")
            .answer::<serde_json::Value>();
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = failure
        else {
            panic!("scope changed failure");
        };
        assert_eq!(error.code.as_deref(), Some("scope_changed"));
        drop(writer);
        assert!(bytes.is_empty());
    }
}
