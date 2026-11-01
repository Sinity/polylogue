//! Briefs handed to sessions whose delivery records the writer has not
//! committed yet. A brief answers before its delivery is recorded, so a
//! resume or compaction brief that follows right away finds its "since"
//! point here until the record is committed.

use crate::briefing::LastBrief;
use polylogue_model::user::UserOutcome;
use polylogue_store::StoreError;
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Mutex, PoisonError};

type Committed = Receiver<Result<UserOutcome, StoreError>>;

/// The latest queued brief per recipient session.
#[derive(Default)]
pub(crate) struct QueuedDeliveries(Mutex<HashMap<String, (LastBrief, Committed)>>);

impl QueuedDeliveries {
    /// Remembers a brief handed to `recipient` until `committed` answers.
    pub fn add(&self, recipient: String, since: LastBrief, committed: Committed) {
        let mut queued = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        prune(&mut queued);
        queued.insert(recipient, (since, committed));
    }

    /// The latest brief handed to `recipient` whose record is still queued.
    pub fn latest(&self, recipient: &str) -> Option<LastBrief> {
        let mut queued = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        prune(&mut queued);
        queued.get(recipient).map(|(since, _)| *since)
    }
}

/// Forgets every brief the writer has answered for: its record is committed
/// (or failed, and the stored deliveries are what remains).
fn prune(queued: &mut HashMap<String, (LastBrief, Committed)>) {
    queued.retain(|_, (_, committed)| matches!(committed.try_recv(), Err(TryRecvError::Empty)));
}
