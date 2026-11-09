//! Reads Sinex's confirmed stream off the writer thread. Each page is handed
//! to the writer through its queue, so the writer thread does no network I/O
//! for it; the page is acknowledged to Sinex only after the writer saved it.
use super::*;
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};

/// How long the follower waits between reads of the stream.
const STREAM_POLL: std::time::Duration = std::time::Duration::from_secs(2);
/// The longest wait while reading the stream keeps failing.
const STREAM_POLL_LONGEST: std::time::Duration = std::time::Duration::from_secs(60);
pub(super) struct Follower {
    requests: Sender<Request>,
    handle: Option<JoinHandle<()>>,
}

enum Request {
    Synchronize(SyncSender<Result<i64>>),
    Stop,
}

impl Follower {
    pub(super) fn start(
        sinex: Arc<dyn crate::primary::Sinex>,
        queue: WriterQueue,
        archive: &Archive,
    ) -> Result<Self> {
        let index = crate::archive::open_read(&archive.index_db())?;
        let (requests, receive) = channel();
        let handle = std::thread::Builder::new()
            .name("polylogue-stream".into())
            .spawn(move || run(sinex.as_ref(), &queue, &index, &receive))?;
        Ok(Self {
            requests,
            handle: Some(handle),
        })
    }

    pub(super) fn synchronize(&self) -> Result<i64> {
        let (reply, wait) = sync_channel(1);
        self.requests
            .send(Request::Synchronize(reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    pub(super) fn stop(&mut self) {
        let _ = self.requests.send(Request::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Follower {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(
    sinex: &dyn crate::primary::Sinex,
    queue: &WriterQueue,
    index: &Connection,
    requests: &Receiver<Request>,
) {
    let mut wait = STREAM_POLL;
    loop {
        let reply = match requests.recv_timeout(wait) {
            Ok(Request::Synchronize(reply)) => Some(reply),
            Ok(Request::Stop) | Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => None,
        };
        let outcome = follow_once(sinex, queue, index);
        wait = match &outcome {
            Ok(_) => STREAM_POLL,
            Err(_) => (wait * 2).min(STREAM_POLL_LONGEST),
        };
        let error = outcome.as_ref().err().map(ToString::to_string);
        if queue.send(Msg::StreamFollowed(error)).is_err() {
            return;
        }
        if let Some(reply) = reply {
            let _ = reply.send(outcome);
        }
    }
}

/// One page: read after the applied position, saved by the writer, then
/// acknowledged.
fn follow_once(
    sinex: &dyn crate::primary::Sinex,
    queue: &WriterQueue,
    index: &Connection,
) -> Result<i64> {
    let state = crate::primary::projection::replay_state(index)?;
    let cache = crate::primary::cache_identity(index)?;
    let page = sinex.confirmed(&state, &cache)?;
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::Confirmed(page, reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)??;
    let applied = crate::primary::projection::replay_state(index)?;
    sinex.acknowledge(&applied, &|event| {
        crate::primary::projection::has_fact(index, event)
    })?;
    Ok(applied.applied_position.unwrap_or(0))
}
