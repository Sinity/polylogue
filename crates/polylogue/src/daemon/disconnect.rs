//! Observe a closed peer even while its application handler is inside SQLite.
use polylogue_store::read::limit::ReadCancellation;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

pub(crate) const INTERVAL: Duration = Duration::from_millis(10);

pub(crate) fn closed(fd: &OwnedFd) -> bool {
    let mut event = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLRDHUP,
        revents: 0,
    };
    // poll borrows one live descriptor and never reads the request bytes.
    let ready = unsafe { libc::poll(&mut event, 1, 0) };
    ready > 0
        && event.revents & (libc::POLLRDHUP | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0
}

pub(crate) struct Watch {
    stop: Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Watch {
    pub(crate) fn start(
        socket: &impl AsFd,
        cancellation: ReadCancellation,
    ) -> std::io::Result<Self> {
        let fd = socket.as_fd().try_clone_to_owned()?;
        let (stop, stopped) = channel();
        let thread = std::thread::Builder::new()
            .name("polylogue-disconnect".into())
            .spawn(move || {
                loop {
                    if closed(&fd) {
                        cancellation.cancel();
                        break;
                    }
                    if !matches!(
                        stopped.recv_timeout(INTERVAL),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                    ) {
                        break;
                    }
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
