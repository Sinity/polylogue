//! The derived offline completion catalog and cues from committed index changes.
use std::sync::{Arc, Mutex, mpsc};

/// A coalesced cue to read the authoritative index, never a change ledger.
#[derive(Clone, Default)]
pub struct CommitEvents(Arc<Mutex<Vec<mpsc::SyncSender<()>>>>);

pub struct CommitSubscription {
    receiver: mpsc::Receiver<()>,
    wake: mpsc::SyncSender<()>,
}
impl CommitEvents {
    /// Subscribe before taking the startup snapshot so commits cannot fall
    /// between the snapshot and the next incremental read.
    pub fn subscribe(&self) -> CommitSubscription {
        let (wake, receiver) = mpsc::sync_channel(1);
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(wake.clone());
        CommitSubscription { receiver, wake }
    }

    /// Called after a successful logical index commit, including derivations.
    pub(crate) fn committed(&self) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|wake| !matches!(wake.try_send(()), Err(mpsc::TrySendError::Disconnected(()))));
    }
}
impl CommitSubscription {
    pub fn recv(&self) -> Result<(), mpsc::RecvError> {
        self.receiver.recv()
    }
    pub fn try_recv(&self) -> Result<(), mpsc::TryRecvError> {
        self.receiver.try_recv()
    }
    /// The owner sets its stop flag before waking a sleeping publisher.
    pub fn wake_handle(&self) -> mpsc::SyncSender<()> {
        self.wake.clone()
    }
}

mod catalog;
pub use catalog::{
    CatalogReader, CatalogWriter, Kept, Metadata, Parameter, Saved, Session, Setting, ValueCount,
};

mod projection;
pub use projection::{DefinitionChange, NativeKept, NativeSaved, SessionChange};

mod custody;
pub use custody::CatalogPointer;
