//! What a request writes back: zero or more events, then exactly one final
//! answer. Each transport has an [`Output`]; the socket and HTTP send
//! events through a [`Spill`], so a request never waits for its client
//! while it holds a reader.
use command_surface::Answer;
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// A transport's side of one reply.
pub(crate) trait Output {
    /// Bytes of server-sent-event frames, in order.
    fn events(&mut self, frames: &[u8]) -> io::Result<()>;
    /// The final answer, and the failure its reply carries inside it (a
    /// line's failed step); nothing follows it.
    fn answer(
        &mut self,
        answer: Answer<Value>,
        inside: Option<command_surface::FailureKind>,
    ) -> io::Result<()>;
}

/// The final answer as the server-sent-event frame that ends a reply.
pub(crate) fn answer_frame(answer: &Answer<Value>) -> io::Result<Vec<u8>> {
    let mut frame = Vec::new();
    crate::protocol::stream::event(&mut frame, "answer", answer)?;
    Ok(frame)
}

/// A reply kept in memory: the events' bytes and the final answer.
#[derive(Default)]
pub(crate) struct Collected {
    pub(crate) events: Vec<u8>,
    pub(crate) answer: Option<Answer<Value>>,
}

impl Output for Collected {
    fn events(&mut self, frames: &[u8]) -> io::Result<()> {
        self.events.extend_from_slice(frames);
        Ok(())
    }
    fn answer(
        &mut self,
        answer: Answer<Value>,
        _: Option<command_surface::FailureKind>,
    ) -> io::Result<()> {
        self.answer = Some(answer);
        Ok(())
    }
}

/// Where a [`Spill`] sends: a client connection.
pub(crate) trait Sink {
    /// Sends what the client takes now of `bytes`, without waiting; 0 when
    /// it takes nothing more now.
    fn try_send(&mut self, bytes: &[u8]) -> io::Result<usize>;
    /// Sends all of `bytes`, at the client's pace.
    fn send(&mut self, bytes: &[u8]) -> io::Result<()>;
}

/// Bytes kept in memory before the rest goes to the spill file.
const MEMORY: usize = 1 << 20;
/// The largest part read back from the spill file at once.
const SPILL_PART: usize = 256 << 10;

/// Sends while a reply is produced without ever waiting for the client:
/// what the client has not taken waits in memory, up to [`MEMORY`] bytes,
/// and beyond that in a file under the archive's staging directory.
/// [`Spill::finish`], called once production has ended and its reader is
/// released, sends the rest at the client's pace.
pub(crate) struct Spill<S: Sink> {
    sink: S,
    memory: VecDeque<u8>,
    file: Option<File>,
    /// Bytes written to and sent from the file; equal when it holds
    /// nothing unsent.
    filed: u64,
    sent: u64,
    part: Vec<u8>,
    dir: PathBuf,
}

impl<S: Sink> Spill<S> {
    pub(crate) fn new(sink: S, dir: PathBuf) -> Self {
        Self {
            sink,
            memory: VecDeque::new(),
            file: None,
            filed: 0,
            sent: 0,
            part: Vec::new(),
            dir,
        }
    }

    fn holding(&self) -> bool {
        !self.memory.is_empty() || !self.part.is_empty() || self.filed > self.sent
    }

    /// Sends `bytes` after whatever is held, without waiting.
    pub(crate) fn write(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        self.send_held(false)?;
        if !self.holding() {
            let sent = self.sink.try_send(bytes)?;
            bytes = &bytes[sent..];
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if self.part.is_empty()
            && self.filed == self.sent
            && self.memory.len() + bytes.len() <= MEMORY
        {
            self.memory.extend(bytes);
            return Ok(());
        }
        let file = match &mut self.file {
            Some(file) => file,
            None => self.file.insert(tempfile::tempfile_in(&self.dir)?),
        };
        file.seek(SeekFrom::Start(self.filed))?;
        file.write_all(bytes)?;
        self.filed += bytes.len() as u64;
        Ok(())
    }

    /// Sends what is held, in order: without waiting, as much as the client
    /// takes now; or, `waiting`, all of it.
    fn send_held(&mut self, waiting: bool) -> io::Result<()> {
        while !self.memory.is_empty() {
            let front = self.memory.as_slices().0;
            let sent = if waiting {
                self.sink.send(front)?;
                front.len()
            } else {
                self.sink.try_send(front)?
            };
            if sent == 0 {
                return Ok(());
            }
            self.memory.drain(..sent);
        }
        loop {
            if self.part.is_empty() {
                if self.filed == self.sent {
                    if let Some(file) = &self.file {
                        file.set_len(0)?;
                    }
                    (self.filed, self.sent) = (0, 0);
                    return Ok(());
                }
                let file = self.file.as_mut().expect("filed bytes have a file");
                let length = usize::try_from(self.filed - self.sent)
                    .unwrap_or(usize::MAX)
                    .min(SPILL_PART);
                self.part.resize(length, 0);
                file.seek(SeekFrom::Start(self.sent))?;
                file.read_exact(&mut self.part)?;
                self.sent += length as u64;
            }
            let sent = if waiting {
                self.sink.send(&self.part)?;
                self.part.len()
            } else {
                self.sink.try_send(&self.part)?
            };
            if sent == 0 {
                return Ok(());
            }
            self.part.drain(..sent);
        }
    }

    /// Sends everything held at the client's pace and returns the sink.
    pub(crate) fn finish(mut self) -> io::Result<S> {
        self.send_held(true)?;
        Ok(self.sink)
    }

    #[cfg(test)]
    pub(crate) fn sink(&mut self) -> &mut S {
        &mut self.sink
    }
}

#[cfg(test)]
mod tests;
