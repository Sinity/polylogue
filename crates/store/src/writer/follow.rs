//! Reads Sinex's confirmed stream off the writer thread. Each page is handed
//! to the writer through its queue, so the writer thread does no network I/O
//! for it; the page is acknowledged to Sinex only after the writer saved it.
use super::*;
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};

/// How long the follower waits after reaching the current stream cut.
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
            Ok(followed) if followed.backlog => std::time::Duration::ZERO,
            Ok(_) => STREAM_POLL,
            Err(_) => (wait.max(STREAM_POLL) * 2).min(STREAM_POLL_LONGEST),
        };
        let error = outcome.as_ref().err().map(ToString::to_string);
        if queue.send(Msg::StreamFollowed(error)).is_err() {
            return;
        }
        if let Some(reply) = reply {
            let _ = reply.send(outcome.map(|followed| followed.position));
        }
    }
}

struct Followed {
    position: i64,
    backlog: bool,
}

/// One bounded page per turn, so stop and synchronization requests are checked
/// between applications even while draining a backlog.
fn follow_once(
    sinex: &dyn crate::primary::Sinex,
    queue: &WriterQueue,
    index: &Connection,
) -> Result<Followed> {
    let state = crate::primary::projection::replay_state(index)?;
    let cache = crate::primary::cache_identity(index)?;
    let page = sinex.confirmed(&state, &cache)?;
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::Confirmed(page, reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)??;
    let applied = crate::primary::projection::replay_state(index)?;
    sinex.acknowledge(
        &applied,
        &|event| crate::primary::projection::has_fact(index, event),
        &|source, receipt| crate::primary::projection::processed_source(index, source, receipt),
    )?;
    // A reprovisioned consumer also replays deliveries at applied positions.
    // Drain receipt progress, then wait only once a probe no longer advances.
    let backlog = applied.staged_delivery != state.staged_delivery
        || applied.scope != state.scope
        || applied.projection != state.projection;
    Ok(Followed {
        position: applied.applied_position.unwrap_or(0),
        backlog,
    })
}
