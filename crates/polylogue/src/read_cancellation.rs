//! One request owns one downstream socket. Cancellation closes its transport;
//! it never undoes a note accepted by the writer or cancels source retention.
use std::{
    io,
    net::Shutdown,
    os::unix::net::UnixStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
/// Clone only to cancel the same request. One token must not be shared by
/// concurrent requests. Cancellation is permanent.
#[derive(Clone, Default)]
pub struct ReadCancellation(Arc<State>);
#[derive(Default)]
struct State {
    cancelled: AtomicBool,
    socket: Mutex<Option<UnixStream>>,
}
impl ReadCancellation {
    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Release);
        if let Some(socket) = self
            .0
            .socket
            .lock()
            .expect("cancellation socket lock")
            .as_ref()
        {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
    pub(crate) fn check(&self) -> io::Result<()> {
        if self.0.cancelled.load(Ordering::Acquire) {
            Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "read request cancelled",
            ))
        } else {
            Ok(())
        }
    }
    pub(crate) fn attach(&self, stream: &UnixStream) -> io::Result<()> {
        self.check()?;
        *self.0.socket.lock().expect("cancellation socket lock") = Some(stream.try_clone()?);
        // Close a socket attached concurrently with cancellation too.
        if let Err(error) = self.check() {
            self.cancel();
            self.detach();
            return Err(error);
        }
        Ok(())
    }
    pub(crate) fn detach(&self) {
        self.0
            .socket
            .lock()
            .expect("cancellation socket lock")
            .take();
    }
}
