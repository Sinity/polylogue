//! Observe a closed peer even while its application handler is inside SQLite.
use polylogue_store::read::limit::ReadCancellation;
use rustix::event::{PollFd, PollFlags, poll};
use std::io::Write;
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::thread::JoinHandle;

pub(crate) struct Watch {
    stop: UnixStream,
    thread: Option<JoinHandle<()>>,
}

impl Watch {
    pub(crate) fn start(
        socket: &impl AsFd,
        cancellation: ReadCancellation,
    ) -> std::io::Result<Self> {
        let fd = socket.as_fd().try_clone_to_owned()?;
        let (stop, stopped) = UnixStream::pair()?;
        let thread = std::thread::Builder::new()
            .name("polylogue-disconnect".into())
            .spawn(move || {
                let mut events = [
                    PollFd::new(&fd, PollFlags::RDHUP),
                    PollFd::new(&stopped, PollFlags::IN),
                ];
                loop {
                    match poll(&mut events, None) {
                        Ok(_) => {
                            if events[0].revents().intersects(
                                PollFlags::RDHUP
                                    | PollFlags::HUP
                                    | PollFlags::ERR
                                    | PollFlags::NVAL,
                            ) {
                                cancellation.cancel();
                            }
                            break;
                        }
                        Err(rustix::io::Errno::INTR) => continue,
                        Err(_) => {
                            cancellation.cancel();
                            break;
                        }
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
        let _ = self.stop.write_all(&[1]);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
