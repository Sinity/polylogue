//! The buffer between a stream's producer and the transport that sends it.
//!
//! The producer runs on the executor under its read transaction and never
//! waits for the client: what the client has not taken yet is held in memory
//! up to [`MEMORY`] bytes and beyond that in a temporary file in the
//! archive's staging directory. So a slow or stalled client holds only its
//! connection and the buffered bytes, never a reader, an executor place or the
//! write-ahead log. The transport takes chunks in order, outside the reader.
use super::Failure;
use polylogue_model::LockExt as _;
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

/// Bytes kept in memory before the rest goes to the spill file.
const MEMORY: usize = 1 << 20;
/// The largest chunk the transport takes from the spill file at once.
const SPILL_CHUNK: usize = 256 << 10;

#[derive(Default)]
struct State {
    memory: VecDeque<Vec<u8>>,
    in_memory: usize,
    spill: Option<File>,
    /// Bytes written to and taken from the spill file; equal when it holds
    /// nothing unsent.
    spilled: u64,
    taken: u64,
    production: Production,
    input_gone: bool,
}

/// How far the producer has come. It moves forward only: `Unwritten` to
/// `Written` at the first byte, and either of them to `Ended` once.
#[derive(Default)]
enum Production {
    /// Nothing written yet: a failure now is the answer, in place of a
    /// stream.
    #[default]
    Unwritten,
    /// Bytes were written: a failure now is the stream's error event.
    Written,
    /// The producer is done. The failure is one from before the first byte,
    /// held until the transport takes it.
    Ended(Option<Failure>),
}

struct Shared {
    state: Mutex<State>,
    ready: Condvar,
    ready_async: tokio::sync::Notify,
    spill_dir: PathBuf,
}

/// The producer's side; it never blocks on the transport.
pub(crate) struct StreamOutput {
    shared: Arc<Shared>,
}

/// The transport's side.
pub(crate) struct StreamInput {
    shared: Arc<Shared>,
}

/// What the transport takes next.
pub(crate) enum Next {
    Bytes(Vec<u8>),
    End,
    /// The stream failed before its first byte: the transport answers with
    /// the failure instead of a stream.
    Failed(Failure),
}

/// A buffer whose spill file, when needed, is created in `spill_dir`.
pub(crate) fn buffer(spill_dir: PathBuf) -> (StreamOutput, StreamInput) {
    let shared = Arc::new(Shared {
        state: Mutex::default(),
        ready: Condvar::new(),
        ready_async: tokio::sync::Notify::new(),
        spill_dir,
    });
    (
        StreamOutput {
            shared: Arc::clone(&shared),
        },
        StreamInput { shared },
    )
}

impl Shared {
    fn wake(&self) {
        self.ready.notify_all();
        self.ready_async.notify_one();
    }
}

impl StreamOutput {
    /// Ends the stream with the producer's result. A failure after bytes were
    /// written is sent as the stream's error event.
    pub(crate) fn finish(mut self, result: Result<(), Failure>) {
        self.end(result);
    }

    /// The one place production ends; a second call changes nothing.
    fn end(&mut self, result: Result<(), Failure>) {
        let written = match self.shared.state.locked().production {
            Production::Unwritten => false,
            Production::Written => true,
            Production::Ended(_) => return,
        };
        let before_first_byte = match result {
            Ok(()) => None,
            Err(failure) if !written => Some(failure),
            Err(failure) => {
                let _ = crate::protocol::stream::error(self, failure);
                None
            }
        };
        self.shared.state.locked().production = Production::Ended(before_first_byte);
        self.shared.wake();
    }
}

impl Write for StreamOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let mut state = self.shared.state.locked();
        if state.input_gone {
            return Err(std::io::ErrorKind::BrokenPipe.into());
        }
        let spilling = state.spilled > state.taken;
        if !spilling && state.in_memory + bytes.len() <= MEMORY {
            state.in_memory += bytes.len();
            state.memory.push_back(bytes.to_vec());
        } else {
            if state.spill.is_none() {
                state.spill = Some(tempfile::tempfile_in(&self.shared.spill_dir)?);
            }
            let at = state.spilled;
            state
                .spill
                .as_ref()
                .expect("spill file was just created")
                .write_all_at(bytes, at)?;
            state.spilled += bytes.len() as u64;
        }
        if matches!(state.production, Production::Unwritten) {
            state.production = Production::Written;
        }
        drop(state);
        self.shared.wake();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for StreamOutput {
    /// A producer dropped without [`StreamOutput::finish`] never ran to its
    /// end: the executor let its queued job go because the daemon is
    /// stopping. The client hears that, not a complete stream.
    fn drop(&mut self) {
        self.end(Err(super::service::shutting_down()));
    }
}

impl StreamInput {
    /// The next chunk if one is ready; `None` while the producer is still
    /// working.
    fn take(&self) -> std::io::Result<Option<Next>> {
        let mut state = self.shared.state.locked();
        if let Some(bytes) = state.memory.pop_front() {
            state.in_memory -= bytes.len();
            return Ok(Some(Next::Bytes(bytes)));
        }
        if state.spilled > state.taken {
            let length = usize::try_from(state.spilled - state.taken)
                .unwrap_or(usize::MAX)
                .min(SPILL_CHUNK);
            let mut bytes = vec![0; length];
            let at = state.taken;
            let file = state.spill.take().expect("spilled bytes have a file");
            let read = file.read_exact_at(&mut bytes, at);
            let caught_up = state.taken + length as u64 == state.spilled;
            // Caught up: reuse the file from its start.
            let reset = match (&read, caught_up) {
                (Ok(()), true) => file.set_len(0),
                _ => Ok(()),
            };
            state.spill = Some(file);
            read?;
            reset?;
            state.taken += length as u64;
            if caught_up {
                state.taken = 0;
                state.spilled = 0;
            }
            return Ok(Some(Next::Bytes(bytes)));
        }
        Ok(match &mut state.production {
            Production::Unwritten | Production::Written => None,
            Production::Ended(before_first_byte) => Some(match before_first_byte.take() {
                Some(failure) => Next::Failed(failure),
                None => Next::End,
            }),
        })
    }

    /// Waits for the next chunk on this thread.
    pub(crate) fn next_blocking(&self) -> std::io::Result<Next> {
        loop {
            if let Some(next) = self.take()? {
                return Ok(next);
            }
            let state = self.shared.state.locked();
            if state.memory.is_empty()
                && state.spilled == state.taken
                && !matches!(state.production, Production::Ended(_))
            {
                drop(polylogue_model::unpoisoned(self.shared.ready.wait(state)));
            }
        }
    }

    /// Waits for the next chunk without blocking an async runtime thread.
    pub(crate) async fn next(&self) -> std::io::Result<Next> {
        loop {
            let notified = self.shared.ready_async.notified();
            if let Some(next) = self.take()? {
                return Ok(next);
            }
            notified.await;
        }
    }
}

impl Drop for StreamInput {
    fn drop(&mut self) {
        self.shared.state.locked().input_gone = true;
    }
}

#[cfg(test)]
mod tests;
