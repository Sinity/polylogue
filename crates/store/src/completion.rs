//! The derived offline completion catalog and cues from committed index changes.
use std::sync::{Arc, Mutex, mpsc};

/// What the commits since a subscriber last looked changed, by the kinds of
/// record its readers tell apart. It says which parts to read again, never
/// which rows changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Committed {
    /// Inputs and what is parsed or derived from them: sessions, messages,
    /// events, documents, coverage, vectors.
    pub content: bool,
    /// Authored records: assertions (notes, tags, rules, saved queries,
    /// settings) and the other user writes.
    pub authored: bool,
    /// Queries, their runs and run outcomes, which reads record.
    pub query_records: bool,
}
impl Committed {
    pub const CONTENT: Self = Self {
        content: true,
        authored: false,
        query_records: false,
    };
    pub const AUTHORED: Self = Self {
        content: false,
        authored: true,
        query_records: false,
    };
    pub const QUERY_RECORDS: Self = Self {
        content: false,
        authored: false,
        query_records: true,
    };
    /// A change that may concern any record.
    pub const EVERY: Self = Self {
        content: true,
        authored: true,
        query_records: true,
    };
    pub fn merge(&mut self, other: Self) {
        self.content |= other.content;
        self.authored |= other.authored;
        self.query_records |= other.query_records;
    }
}

struct Subscriber {
    pending: Arc<Mutex<Committed>>,
    wake: mpsc::SyncSender<()>,
}

/// Coalesced cues to read the authoritative index, each saying what kinds
/// of record the commits changed.
#[derive(Clone, Default)]
pub struct CommitEvents(Arc<Mutex<Vec<Subscriber>>>);

pub struct CommitSubscription {
    pending: Arc<Mutex<Committed>>,
    receiver: mpsc::Receiver<()>,
    wake: mpsc::SyncSender<()>,
}
impl CommitEvents {
    /// Subscribe before taking the startup snapshot so commits cannot fall
    /// between the snapshot and the next incremental read.
    pub fn subscribe(&self) -> CommitSubscription {
        let (wake, receiver) = mpsc::sync_channel(1);
        let pending = Arc::default();
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Subscriber {
                pending: Arc::clone(&pending),
                wake: wake.clone(),
            });
        CommitSubscription {
            pending,
            receiver,
            wake,
        }
    }

    /// Called after a successful logical index commit, including derivations,
    /// with what it changed.
    pub(crate) fn committed(&self, changed: Committed) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|subscriber| {
                subscriber
                    .pending
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .merge(changed);
                !matches!(
                    subscriber.wake.try_send(()),
                    Err(mpsc::TrySendError::Disconnected(()))
                )
            });
    }
}
impl CommitSubscription {
    /// Waits for a cue and returns what the commits since the last one
    /// changed; a wake from [`Self::wake_handle`] alone changed nothing.
    pub fn recv(&self) -> Result<Committed, mpsc::RecvError> {
        self.receiver.recv()?;
        Ok(self.take())
    }
    pub fn try_recv(&self) -> Result<Committed, mpsc::TryRecvError> {
        self.receiver.try_recv()?;
        Ok(self.take())
    }
    pub fn recv_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<Committed, mpsc::RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)?;
        Ok(self.take())
    }
    fn take(&self) -> Committed {
        std::mem::take(
            &mut *self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
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
pub use projection::{DefinitionChange, NativeKept, NativeSaved, SessionChange, VocabularyChanges};

mod custody;
pub use custody::CatalogPointer;
