//! Delivery records the daemon has handed to the writer, or will hand over,
//! without waiting. A brief answers before its delivery is recorded and never
//! waits for room in the writer's queue: when the queue is full, the record
//! waits here, in order with any outcome event for it, and goes to the
//! writer after its next commit. Until a brief's record is committed, it is
//! its recipient's latest brief here, so a resume or compaction brief that
//! follows right away reports only what changed since.

use crate::briefing::LastBrief;
use polylogue_model::user::{UserOutcome, UserRequest};
use polylogue_store::writer::Queued;
use polylogue_store::{ReadScopeToken, StoreError, WriterSender};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Mutex, PoisonError};

type Committed = Receiver<Result<UserOutcome, StoreError>>;

/// A request the writer's queue had no room for.
struct Waiting {
    request: UserRequest,
    scope: Option<ReadScopeToken>,
    /// The recipient whose latest brief this request records.
    recipient: Option<String>,
}

/// A recipient's latest brief and its record: waiting here (`None`) or in
/// the writer's queue.
struct Latest {
    brief: LastBrief,
    request: String,
    committed: Option<Committed>,
}

#[derive(Default)]
struct State {
    latest: HashMap<String, Latest>,
    waiting: VecDeque<Waiting>,
}

#[derive(Default)]
pub(crate) struct QueuedDeliveries(Mutex<State>);

impl QueuedDeliveries {
    /// Hands `request` to the writer without waiting. When `recipient` is
    /// named, it is a brief to that session, remembered until committed.
    pub fn submit(
        &self,
        writes: &WriterSender,
        request: UserRequest,
        scope: Option<ReadScopeToken>,
        recipient: Option<(String, LastBrief)>,
    ) -> Result<(), StoreError> {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        prune(&mut state);
        let recipient = recipient.map(|(recipient, brief)| {
            state.latest.insert(
                recipient.clone(),
                Latest {
                    brief,
                    request: request.id.clone(),
                    committed: None,
                },
            );
            recipient
        });
        state.waiting.push_back(Waiting {
            request,
            scope,
            recipient,
        });
        hand_over(&mut state, writes)
    }

    /// Hands waiting requests to the writer while its queue has room.
    pub fn hand_over(&self, writes: &WriterSender) -> Result<(), StoreError> {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        hand_over(&mut state, writes)
    }

    /// Positive proof only: absence says nothing about a preparation already
    /// committed or an invocation still running. Never acquires an archive reader.
    pub fn owns_preparation(&self, id: &str) -> bool {
        let state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        state.latest.values().any(|latest| {
            latest.request == id
                && match &latest.committed {
                    None => true,
                    Some(committed) => {
                        matches!(committed.try_recv(), Err(TryRecvError::Empty) | Ok(Ok(_)))
                    }
                }
        }) || state.waiting.iter().any(|waiting| {
            waiting.request.id == id
                && matches!(
                    &waiting.request.write,
                    polylogue_model::user::UserWrite::Deliver(_)
                )
        })
    }

    /// The latest brief handed to `recipient` whose record is not committed.
    pub fn latest(&self, recipient: &str) -> Option<LastBrief> {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        prune(&mut state);
        state.latest.get(recipient).map(|latest| latest.brief)
    }
}

/// Waiting requests go in their order; the first that finds the queue full
/// stops the rest, so an outcome event never overtakes its delivery.
fn hand_over(state: &mut State, writes: &WriterSender) -> Result<(), StoreError> {
    while let Some(next) = state.waiting.pop_front() {
        let id = next.request.id.clone();
        match writes.try_enqueue_user(next.request, next.scope)? {
            Queued::Committing(committed) => {
                if let Some(latest) = next
                    .recipient
                    .and_then(|recipient| state.latest.get_mut(&recipient))
                    .filter(|latest| latest.request == id)
                {
                    latest.committed = Some(committed);
                }
            }
            Queued::Full(request, scope) => {
                state.waiting.push_front(Waiting {
                    request: *request,
                    scope,
                    recipient: next.recipient,
                });
                break;
            }
        }
    }
    Ok(())
}

/// Forgets every brief the writer has answered for: its record is committed
/// (or failed, and the stored deliveries are what remains).
fn prune(state: &mut State) {
    state.latest.retain(|_, latest| match &latest.committed {
        None => true,
        Some(committed) => matches!(committed.try_recv(), Err(TryRecvError::Empty)),
    });
}
