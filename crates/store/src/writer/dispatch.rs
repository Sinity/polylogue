//! The writer checks a new operation's producing parent after queue waits.
use super::*;

/// Captured producing-transport lifetime. This is runtime state, never wire data.
/// Unlike a read limit, it carries no SQL stages, counters, or archive scopes.
pub struct ParentLifetime {
    deadline: Option<std::time::Instant>,
    cancellation: crate::read::limit::ReadCancellation,
}
impl ParentLifetime {
    pub fn capture(
        deadline: Option<std::time::Instant>,
        cancellation: crate::read::limit::ReadCancellation,
    ) -> Arc<Self> {
        Arc::new(Self {
            deadline,
            cancellation,
        })
    }
    pub fn read_limit(
        &self,
        cancellation: crate::read::limit::ReadCancellation,
    ) -> crate::read::limit::ReadLimit {
        match self.deadline {
            Some(deadline) => crate::read::limit::ReadLimit::at(deadline, cancellation),
            None => crate::read::limit::ReadLimit::without_deadline(cancellation),
        }
    }
    pub fn check(&self) -> Result<()> {
        self.read_limit(self.cancellation.clone()).check()
    }
}

impl Inner {
    pub(super) fn admit_user_parent(
        &self,
        id: &str,
        parent: Option<&ParentLifetime>,
    ) -> Result<()> {
        #[cfg(any(test, feature = "fixture"))]
        self.dispatch.reach(id, DispatchPhase::BeforeAdmission)?;
        if let Some(parent) = parent {
            parent.check()?;
        }
        // This successful dispatch check transfers ownership to the writer.
        // Expiry after this point must not suppress effects or their ACK cleanup.
        #[cfg(any(test, feature = "fixture"))]
        self.dispatch.reach(id, DispatchPhase::Owned)?;
        let _ = id;
        Ok(())
    }
}

#[cfg(any(test, feature = "fixture"))]
mod fixture {
    use super::*;
    use std::time::Duration;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum DispatchPhase {
        BeforeAdmission,
        Owned,
    }

    struct Pause {
        id: String,
        phase: DispatchPhase,
        reached: SyncSender<()>,
        release: Receiver<()>,
    }

    #[derive(Clone, Default)]
    pub(in crate::writer) struct Barriers(Arc<Mutex<Vec<Pause>>>);

    /// One exact writer dispatch pause. Dropping it releases the native worker.
    pub struct DispatchBarrier {
        reached: Receiver<()>,
        release: Option<SyncSender<()>>,
    }
    impl DispatchBarrier {
        pub fn wait(&self) -> std::result::Result<(), std::sync::mpsc::RecvTimeoutError> {
            self.reached.recv_timeout(Duration::from_secs(5))
        }
        pub fn release(mut self) {
            if let Some(sender) = self.release.take() {
                let _ = sender.send(());
            }
        }
    }
    impl Drop for DispatchBarrier {
        fn drop(&mut self) {
            // A disconnected release channel also lets the writer finish.
            self.release.take();
        }
    }
    impl Barriers {
        pub(in crate::writer) fn reach(&self, id: &str, phase: DispatchPhase) -> Result<()> {
            let pause = {
                let mut pauses = self.0.lock().map_err(|_| StoreError::WriterGone)?;
                pauses
                    .iter()
                    .position(|pause| pause.id == id && pause.phase == phase)
                    .map(|at| pauses.remove(at))
            };
            if let Some(pause) = pause {
                let _ = pause.reached.send(());
                // No registry lock is held while the actual writer is paused.
                let _ = pause.release.recv();
            }
            Ok(())
        }
    }
    impl WriterSender {
        pub fn pause_user_dispatch_for_test(
            &self,
            id: &str,
            phase: DispatchPhase,
        ) -> DispatchBarrier {
            let (reached, wait) = sync_channel(1);
            let (release, resume) = sync_channel(1);
            self.0.dispatch.0.lock().unwrap().push(Pause {
                id: id.into(),
                phase,
                reached,
                release: resume,
            });
            DispatchBarrier {
                reached: wait,
                release: Some(release),
            }
        }
    }
}
#[cfg(any(test, feature = "fixture"))]
pub(super) use fixture::Barriers;
#[cfg(any(test, feature = "fixture"))]
pub use fixture::{DispatchBarrier, DispatchPhase};
