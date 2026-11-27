//! Coalesced cues from committed index changes, and what completion reads
//! from the archive.
use polylogue_model::LockExt;
use std::sync::{Arc, Mutex, mpsc};

/// What the commits since a subscriber last looked changed, by the kinds of
/// record its readers tell apart. It says which parts to read again, never
/// which rows changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Committed {
    /// Inputs and what is parsed or derived from them: sessions, messages,
    /// events, documents, coverage, vectors.
    pub content: bool,
    /// Authored records: annotations (notes, tags, rules, saved queries,
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
    interests: Option<Committed>,
    pending: Arc<Mutex<Committed>>,
    wake: mpsc::SyncSender<()>,
    notify: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// Coalesced cues to read the index, each saying what kinds
/// of record the commits changed, and the count of commits made.
#[derive(Clone, Default)]
pub struct CommitEvents {
    subscribers: Arc<Mutex<Vec<Subscriber>>>,
    made: Arc<std::sync::atomic::AtomicU64>,
}

pub struct CommitSubscription {
    pending: Arc<Mutex<Committed>>,
    receiver: mpsc::Receiver<()>,
    wake: mpsc::SyncSender<()>,
}
impl CommitEvents {
    /// Subscribe before taking the startup snapshot so commits cannot fall
    /// between the snapshot and the next incremental read.
    pub fn subscribe(&self) -> CommitSubscription {
        self.subscribe_to(None, None)
    }

    /// Wake the caller's inbox when selected commits first become pending.
    /// The callback must only queue a wake, never block or call the writer.
    /// Drain the returned subscription to allow another coalesced wake.
    pub fn subscribe_with_wake(
        &self,
        interests: Committed,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> CommitSubscription {
        self.subscribe_to(Some(interests), Some(Arc::new(notify)))
    }

    fn subscribe_to(
        &self,
        interests: Option<Committed>,
        notify: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> CommitSubscription {
        let (wake, receiver) = mpsc::sync_channel(1);
        let pending = Arc::default();
        self.subscribers.locked().push(Subscriber {
            interests,
            pending: Arc::clone(&pending),
            wake: wake.clone(),
            notify,
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
        // Counted before any subscriber wakes, so a reader opened after
        // loading the count sees every commit it counts.
        self.made.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.subscribers.locked().retain(|subscriber| {
            let changed = if let Some(interests) = subscriber.interests {
                let selected = Committed {
                    content: changed.content && interests.content,
                    authored: changed.authored && interests.authored,
                    query_records: changed.query_records && interests.query_records,
                };
                if selected == Committed::default() {
                    return true;
                }
                selected
            } else {
                changed
            };
            subscriber.pending.locked().merge(changed);
            match subscriber.wake.try_send(()) {
                Ok(()) => {
                    if let Some(notify) = &subscriber.notify {
                        notify();
                    }
                    true
                }
                Err(mpsc::TrySendError::Full(())) => true,
                Err(mpsc::TrySendError::Disconnected(())) => false,
            }
        });
    }
}

impl CommitEvents {
    /// The logical index commits made since the writer started. A reader
    /// opened after loading this count sees at least those commits.
    pub fn made(&self) -> u64 {
        self.made.load(std::sync::atomic::Ordering::Acquire)
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
        std::mem::take(&mut *self.pending.locked())
    }
    /// The caller sets its stop flag before waking a sleeping publisher.
    pub fn wake_handle(&self) -> mpsc::SyncSender<()> {
        self.wake.clone()
    }
}

mod values;
pub use values::{KeptName, Session, ValueCount};

#[cfg(test)]
mod tests;
