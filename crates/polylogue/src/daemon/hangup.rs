//! One thread watches every connected client for a hang-up, even while the
//! client's request is inside SQLite, and cancels that client's reads.
use polylogue_model::LockExt as _;
use polylogue_store::cancellation::Cancellation;
use rustix::event::epoll;
use std::collections::HashMap;
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// The event data of the watcher's own stop signal.
const STOP: u64 = u64::MAX;

struct Shared {
    epoll: OwnedFd,
    watched: Mutex<HashMap<u64, Cancellation>>,
    next: AtomicU64,
}

pub(crate) struct Watcher {
    shared: Arc<Shared>,
    stop: UnixStream,
    thread: Option<JoinHandle<()>>,
}

/// Watches one connection until dropped; drop it before the connection.
pub(crate) struct Watched {
    shared: Arc<Shared>,
    id: u64,
    fd: OwnedFd,
}

impl Watcher {
    pub(crate) fn start() -> std::io::Result<Self> {
        let epoll = epoll::create(epoll::CreateFlags::CLOEXEC)?;
        let (stop, stopped) = UnixStream::pair()?;
        epoll::add(
            &epoll,
            &stopped,
            epoll::EventData::new_u64(STOP),
            epoll::EventFlags::IN,
        )?;
        let shared = Arc::new(Shared {
            epoll,
            watched: Mutex::default(),
            next: AtomicU64::new(0),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            polylogue_model::spawn_named("polylogue-hangup", move || {
                let _stopped = stopped;
                shared.run();
            })?
        };
        Ok(Self {
            shared,
            stop,
            thread: Some(thread),
        })
    }

    /// Cancels `cancellation` when the peer of `socket` hangs up.
    pub(crate) fn watch(
        &self,
        socket: &impl AsFd,
        cancellation: Cancellation,
    ) -> std::io::Result<Watched> {
        let id = self.shared.next.fetch_add(1, Ordering::Relaxed);
        let fd = socket.as_fd().try_clone_to_owned()?;
        self.shared.watched.locked().insert(id, cancellation);
        // Level-triggered: a peer that hung up before registration is seen.
        if let Err(error) = epoll::add(
            &self.shared.epoll,
            &fd,
            epoll::EventData::new_u64(id),
            epoll::EventFlags::HUP | epoll::EventFlags::ERR | epoll::EventFlags::ONESHOT,
        ) {
            self.shared.watched.locked().remove(&id);
            return Err(error.into());
        }
        Ok(Watched {
            shared: Arc::clone(&self.shared),
            id,
            fd,
        })
    }
}

impl Shared {
    fn run(&self) {
        let mut events = Vec::with_capacity(64);
        loop {
            events.clear();
            match epoll::wait(
                &self.epoll,
                rustix::buffer::spare_capacity(&mut events),
                None,
            ) {
                Ok(_) => {}
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => {
                    polylogue_store::log!(error, "hang-up watcher: {error}");
                    // Without a watcher no read could notice its client
                    // leave; cancel what is watched rather than leave it.
                    for cancellation in self.watched.locked().values() {
                        cancellation.cancel();
                    }
                    return;
                }
            }
            for event in &events {
                let id = event.data.u64();
                if id == STOP {
                    return;
                }
                // A one-shot registration fires once; the entry stays until
                // its connection drops it.
                if let Some(cancellation) = self.watched.locked().get(&id) {
                    cancellation.cancel();
                }
            }
        }
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        let _ = epoll::delete(&self.shared.epoll, &self.fd);
        self.shared.watched.locked().remove(&self.id);
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        let _ = self.stop.write_all(&[1]);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests;
