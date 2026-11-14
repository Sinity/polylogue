//! Streams for every transport: the producer runs on the executor and
//! writes into a [`stream_buffer`](super::stream_buffer); the transport
//! sends what it takes from the buffer outside any reader.
use super::Failure;
use super::executor::Class;
use super::service::Daemon;
use super::stream_buffer::{StreamInput, StreamOutput};
use crate::operation::{Context, Invocation};
use polylogue_store::read::limit::ReadLimit;
use std::io::Write;
use std::sync::Arc;

/// What produces one stream's events.
pub(crate) type Produce = Box<
    dyn FnOnce(&Daemon, &Invocation, &Arc<Context>, &mut dyn Write) -> Result<(), Failure> + Send,
>;

/// Starts `produce` on the executor under `limit` and returns the buffer
/// the transport sends from.
pub(crate) fn start(
    daemon: &Arc<Daemon>,
    class: Class,
    mut limit: ReadLimit,
    invocation: Invocation,
    produce: Produce,
) -> Result<StreamInput, Failure> {
    // The producing parent's token remains live for any admitted write.
    limit.cancellation = limit.cancellation.child();
    let scope = daemon.reads.scope_token()?;
    let (output, input) = super::stream_buffer::buffer(daemon.archive.staging_dir());
    let worker = Arc::clone(daemon);
    let deadline = limit.deadline();
    let job = move || {
        let _stopping_read = worker.cancel_read_on_stop(&limit.cancellation);
        let mut output = output;
        let result = limit.scope(|| {
            limit.check()?;
            let context = worker.resolve_operation_context(invocation.clone())?;
            let mut writer = GuardedWriter::new(&mut output, limit.clone(), scope);
            let result = produce(&worker, &invocation, &context, &mut writer);
            match writer.failure() {
                Some(failure) => Err(failure),
                None => result,
            }
        });
        output.finish(result);
    };
    if daemon.executor.submit(class, deadline, job) {
        Ok(input)
    } else {
        Err(super::service::shutting_down())
    }
}

/// The producer's writer: each write first checks the request's limit and
/// the access scope it was admitted under, so a revoked scope or a
/// cancellation stops production at the next event.
pub(crate) struct GuardedWriter<'a> {
    output: &'a mut StreamOutput,
    limit: ReadLimit,
    failure: Option<Failure>,
    scope: polylogue_store::ReadScopeToken,
}

impl<'a> GuardedWriter<'a> {
    pub(crate) fn new(
        output: &'a mut StreamOutput,
        limit: ReadLimit,
        scope: polylogue_store::ReadScopeToken,
    ) -> Self {
        Self {
            output,
            limit,
            failure: None,
            scope,
        }
    }

    fn check(&mut self) -> std::io::Result<()> {
        let checked = self.limit.check().and_then(|()| self.scope.check());
        checked.map_err(|error| {
            self.failure = Some(Failure::store(&error));
            std::io::Error::other(error)
        })
    }

    pub(crate) fn failure(&mut self) -> Option<Failure> {
        self.failure.take()
    }
}

impl Write for GuardedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.check()?;
        self.output.write(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.check()
    }
}

#[cfg(test)]
mod tests;
