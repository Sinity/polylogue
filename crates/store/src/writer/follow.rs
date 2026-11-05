//! Reads Sinex's confirmed stream off the writer thread. Each page is handed
//! to the writer through its queue, so the writer thread does no network I/O
//! for it; the page is acknowledged to Sinex only after the writer saved it.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// How long the follower waits between reads of the stream.
const STREAM_POLL: std::time::Duration = std::time::Duration::from_secs(2);
/// The longest wait while reading the stream keeps failing.
const STREAM_POLL_LONGEST: std::time::Duration = std::time::Duration::from_secs(60);
/// How often a waiting follower checks that it should stop.
const STOP_CHECK: std::time::Duration = std::time::Duration::from_millis(100);

pub(super) struct Follower {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Follower {
    pub(super) fn start(
        sinex: Arc<dyn crate::primary::Sinex>,
        queue: WriterQueue,
        archive: &Archive,
    ) -> Result<Self> {
        let index = crate::archive::open_read(&archive.index_db())?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("polylogue-stream".into())
            .spawn(move || run(sinex.as_ref(), &queue, &index, &stopping))?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }

    pub(super) fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
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
    stop: &AtomicBool,
) {
    let mut wait = STREAM_POLL;
    loop {
        let mut waited = std::time::Duration::ZERO;
        while waited < wait {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(STOP_CHECK);
            waited += STOP_CHECK;
        }
        let outcome = follow_once(sinex, queue, index);
        wait = match &outcome {
            Ok(()) => STREAM_POLL,
            Err(_) => (wait * 2).min(STREAM_POLL_LONGEST),
        };
        let error = outcome.err().map(|error| error.to_string());
        if queue.send(Msg::StreamFollowed(error)).is_err() {
            return;
        }
    }
}

/// One page: read after the applied position, saved by the writer, then
/// acknowledged.
fn follow_once(
    sinex: &dyn crate::primary::Sinex,
    queue: &WriterQueue,
    index: &Connection,
) -> Result<()> {
    let state = crate::primary::projection::replay_state(index)?;
    let cache = crate::primary::cache_identity(index)?;
    let page = sinex.confirmed(&state, &cache)?;
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::Confirmed(page, reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)??;
    sinex.acknowledge(&crate::primary::projection::replay_state(index)?)
}
