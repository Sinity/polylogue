//! Publication runs independently of confirmed-stream reads. A wake is only
//! a hint: durable queue readiness is the work to do.
use crate::{Error, Result, Sinex};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{SyncSender, sync_channel},
};

#[derive(Default)]
pub(crate) struct Stop {
    requested: AtomicBool,
    wake: tokio::sync::Notify,
}
impl Stop {
    pub(crate) async fn until<T>(&self, work: impl Future<Output = Result<T>>) -> Result<T> {
        if self.requested.load(Ordering::Acquire) {
            return Err(Error::Invalid("producer sender stopped".into()));
        }
        tokio::select! {
            result = work => result,
            _ = self.wake.notified() => Err(Error::Invalid("producer sender stopped".into())),
        }
    }
}

fn retryable(error: &Error) -> bool {
    match error {
        Error::Unreachable(_) => true,
        Error::Io(error) | Error::Blobs(hashed_blobs::Error::Io(error)) => {
            polylogue_store::retryable_io(error)
        }
        Error::Store(error) => error.retryable_write(),
        _ => false,
    }
}

pub(crate) struct Sender {
    wake: Option<SyncSender<()>>,
    stop: Arc<Stop>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Sender {
    pub(crate) fn start(worker: Sinex, stop: Arc<Stop>) -> Result<Self> {
        let (wake, requests) = sync_channel(1);
        let thread_stop = stop.clone();
        let thread = std::thread::Builder::new()
            .name("polylogue-send".into())
            .spawn(move || {
                let mut retry = false;
                loop {
                    let awake = if retry {
                        match requests.recv_timeout(std::time::Duration::from_secs(1)) {
                            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => true,
                            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => false,
                        }
                    } else {
                        requests.recv().is_ok()
                    };
                    if !awake || thread_stop.requested.load(Ordering::Acquire) {
                        break;
                    }
                    retry = worker.drain_pending().as_ref().is_err_and(retryable);
                }
            })?;
        Ok(Self {
            wake: Some(wake),
            stop,
            thread: Some(thread),
        })
    }
    pub(crate) fn waker(&self) -> Arc<dyn Fn() + Send + Sync> {
        let wake = self.wake.as_ref().expect("running sender").clone();
        Arc::new(move || {
            let _ = wake.try_send(());
        })
    }
    pub(crate) fn wake(&self) {
        if let Some(wake) = &self.wake {
            let _ = wake.try_send(());
        }
    }
    pub(crate) fn stop(&mut self) {
        self.stop.requested.store(true, Ordering::Release);
        self.stop.wake.notify_one();
        self.wake();
        self.wake.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Drop for Sender {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests;
