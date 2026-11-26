//! Delivery records the daemon has handed to the writer, or will hand over,
//! without waiting. A brief answers before its delivery is recorded and never
//! waits for room in the writer's queue: when the queue is full, the record
//! waits here, in order with any outcome event for it, and goes to the
//! writer after its next commit. Until a delivered brief's record is
//! committed, it is its recipient's latest brief here, so a resume or
//! compaction brief that follows right away reports only what changed since.

use crate::brief::LastBrief;
use polylogue_model::LockExt as _;
use polylogue_model::user::UserRequest;
use polylogue_store::writer::Queued;
use polylogue_store::{ReadScopeToken, StoreError, WriterSender};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, TryRecvError};

type WriteAnswer = Result<
    polylogue_store::archive::writer::Committed<polylogue_store::user_writes::WriteResult>,
    StoreError,
>;

/// The writer's answer for one handed-over request, kept once received so
/// that reading it never consumes it.
struct Committed {
    receiver: Receiver<WriteAnswer>,
    answer: Option<WriteAnswer>,
}

impl Committed {
    fn new(receiver: Receiver<WriteAnswer>) -> Self {
        Self {
            receiver,
            answer: None,
        }
    }

    /// `None` while the writer has not answered; a writer that went away
    /// without answering answers `WriterGone`.
    fn answer(&mut self) -> Option<&WriteAnswer> {
        if self.answer.is_none() {
            self.answer = match self.receiver.try_recv() {
                Ok(answer) => Some(answer),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Err(StoreError::WriterGone)),
            };
        }
        self.answer.as_ref()
    }
}

/// A request the writer's queue had no room for.
struct Waiting {
    request: UserRequest,
    scope: Option<ReadScopeToken>,
    /// The recipient whose latest brief this request records.
    recipient: Option<String>,
}

struct PendingOutcome {
    request: Waiting,
    committed: Committed,
}

/// A recipient's latest brief and its record: waiting here (`None`) or in
/// the writer's queue.
struct Latest {
    brief: LastBrief,
    request: String,
    committed: Option<Committed>,
    /// Its `delivered` outcome has been handed over; only a delivered brief
    /// moves the point a later brief reports changes since.
    delivered: bool,
}

struct PendingAdvice {
    recipient: String,
    refs: Vec<String>,
    committed: Option<Committed>,
}

#[derive(Default)]
struct State {
    latest: HashMap<String, Latest>,
    waiting: VecDeque<Waiting>,
    advice: HashMap<String, PendingAdvice>,
    outcomes: VecDeque<PendingOutcome>,
    last_failure: Option<String>,
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
        let mut state = self.0.locked();
        submit(&mut state, writes, request, scope, recipient)
    }

    /// Optional hints leave immediately if another handoff owns the queue lock.
    pub fn try_submit(
        &self,
        writes: &WriterSender,
        request: UserRequest,
        scope: Option<ReadScopeToken>,
    ) -> Result<bool, StoreError> {
        let Ok(mut state) = self.0.try_lock() else {
            return Ok(false);
        };
        submit(&mut state, writes, request, scope, None)?;
        Ok(true)
    }

    /// Snapshot before opening the reader: a commit is then represented here
    /// or visible to that reader, including an unconfirmed preparation.
    pub fn pending_advice(&self, recipient: &str) -> Option<Vec<(String, Vec<String>)>> {
        let mut state = self.0.try_lock().ok()?;
        prune(&mut state);
        Some(
            state
                .advice
                .iter()
                .filter(|(_, advice)| advice.recipient == recipient)
                .map(|(id, advice)| (id.clone(), advice.refs.clone()))
                .collect(),
        )
    }

    /// Called only for records positively read from current delivery history.
    pub fn advice_applied(&self, ids: &[String]) {
        if let Ok(mut state) = self.0.try_lock() {
            for id in ids {
                state.advice.remove(id);
            }
        }
    }

    /// Hands waiting requests to the writer while its queue has room.
    pub fn hand_over(&self, writes: &WriterSender) -> Result<(), StoreError> {
        let mut state = self.0.locked();
        hand_over(&mut state, writes)
    }

    /// Outcomes stay pending until the writer acknowledges their durable write.
    pub fn pending_outcomes(&self) -> (u64, Option<String>) {
        let mut state = self.0.locked();
        prune(&mut state);
        let waiting = state
            .waiting
            .iter()
            .filter(|next| is_outcome(&next.request))
            .count();
        (
            (waiting + state.outcomes.len()) as u64,
            state.last_failure.clone(),
        )
    }

    /// Producers have stopped. Finish accepted writes while the writer is alive;
    /// a failed final attempt remains pending and is reported to shutdown.
    pub fn drain(&self, writes: &WriterSender) -> Result<(), StoreError> {
        loop {
            {
                let mut state = self.0.locked();
                prune(&mut state);
                if state.waiting.is_empty() && state.outcomes.is_empty() {
                    return Ok(());
                }
            }
            self.hand_over(writes)?;
            // This existing writer barrier serves earlier user requests too.
            writes.flush_rows()?;
            let mut state = self.0.locked();
            prune(&mut state);
            if let Some(error) = &state.last_failure {
                return Err(StoreError::internal(format!(
                    "delivery outcomes remain unwritten after the final attempt: {error}"
                )));
            }
        }
    }

    /// Only a true answer means anything: false says nothing about a preparation already
    /// committed or an invocation still running. Never acquires an archive reader.
    pub fn owns_preparation(&self, id: &str) -> bool {
        let mut state = self.0.locked();
        state.advice.contains_key(id)
            || state.latest.values_mut().any(|latest| {
                latest.request == id
                    && match &mut latest.committed {
                        None => true,
                        Some(committed) => matches!(committed.answer(), None | Some(Ok(_))),
                    }
            })
            || state.waiting.iter().any(|waiting| {
                waiting.request.id == id
                    && matches!(
                        &waiting.request.write,
                        polylogue_model::user::UserWrite::Deliver(_)
                    )
            })
    }

    /// The latest delivered brief to `recipient` whose record is not
    /// committed.
    pub fn latest(&self, recipient: &str) -> Option<LastBrief> {
        let mut state = self.0.locked();
        prune(&mut state);
        state
            .latest
            .get(recipient)
            .filter(|latest| latest.delivered)
            .map(|latest| latest.brief)
    }
}

fn submit(
    state: &mut State,
    writes: &WriterSender,
    request: UserRequest,
    scope: Option<ReadScopeToken>,
    recipient: Option<(String, LastBrief)>,
) -> Result<(), StoreError> {
    prune(state);
    if let polylogue_model::user::UserWrite::Deliver(delivery) = &request.write
        && delivery.event == polylogue_model::DeliveryEvent::Advice
        && let Some(recipient) = &delivery.recipient_id
    {
        state.advice.insert(
            request.id.clone(),
            PendingAdvice {
                recipient: recipient.clone(),
                refs: delivery.refs.clone(),
                committed: None,
            },
        );
    }
    if let polylogue_model::user::UserWrite::DeliveryEvent {
        delivery,
        what: polylogue_model::DeliveryOutcome::Delivered,
        ..
    } = &request.write
    {
        for latest in state.latest.values_mut() {
            if &latest.request == delivery {
                latest.delivered = true;
            }
        }
    }
    // Only session-start, resume and compaction briefs set the point a
    // resume or compaction brief reports changes since.
    let brief = matches!(
        &request.write,
        polylogue_model::user::UserWrite::Deliver(delivery)
            if crate::brief::BRIEF_EVENTS.contains(&delivery.event)
    );
    let recipient = recipient.filter(|_| brief).map(|(recipient, brief)| {
        state.latest.insert(
            recipient.clone(),
            Latest {
                brief,
                request: request.id.clone(),
                committed: None,
                delivered: false,
            },
        );
        recipient
    });
    state.waiting.push_back(Waiting {
        request,
        scope,
        recipient,
    });
    hand_over(state, writes)
}

/// Waiting requests go in their order; the first that finds the queue full
/// stops the rest, so an outcome event never overtakes its delivery.
fn hand_over(state: &mut State, writes: &WriterSender) -> Result<(), StoreError> {
    prune(state);
    // A failed write is attempted at most once in this hand-over turn.
    let waiting = state.waiting.len();
    for _ in 0..waiting {
        let Some(next) = state.waiting.pop_front() else {
            break;
        };
        let id = next.request.id.clone();
        let outcome = is_outcome(&next.request).then(|| Waiting {
            request: next.request.clone(),
            scope: next.scope.clone(),
            recipient: None,
        });
        let queued = writes
            .archive()
            .try_submit(polylogue_store::writer::UserSubmission {
                request: next.request,
                scope: next.scope,
                preparation: None,
                parent: None,
            });
        match queued {
            Err(error) => {
                if let Some(outcome) = outcome {
                    state.waiting.push_front(outcome);
                    state.last_failure = Some(error.to_string());
                } else {
                    // The request is dropped, so nothing will commit it: it
                    // must not stay as pending advice or a latest brief.
                    state.advice.remove(&id);
                    state.latest.retain(|_, latest| latest.request != id);
                }
                return Err(error);
            }
            Ok(Queued::Committing(committed)) => match outcome {
                Some(request) => state.outcomes.push_back(PendingOutcome {
                    request,
                    committed: Committed::new(committed),
                }),
                None => {
                    let committed = Committed::new(committed);
                    if let Some(advice) = state.advice.get_mut(&id) {
                        advice.committed = Some(committed);
                    } else if let Some(latest) = next
                        .recipient
                        .and_then(|recipient| state.latest.get_mut(&recipient))
                        .filter(|latest| latest.request == id)
                    {
                        latest.committed = Some(committed);
                    }
                }
            },
            Ok(Queued::Full(submission)) => {
                state.waiting.push_front(Waiting {
                    request: submission.request,
                    scope: submission.scope,
                    recipient: next.recipient,
                });
                break;
            }
        }
    }
    Ok(())
}

fn is_outcome(request: &UserRequest) -> bool {
    matches!(
        request.write,
        polylogue_model::user::UserWrite::DeliveryEvent { .. }
    )
}

/// Forgets every brief the writer has answered for: its record is committed
/// (or failed, and the stored deliveries are what remains).
fn prune(state: &mut State) {
    let mut failed = VecDeque::new();
    for mut pending in std::mem::take(&mut state.outcomes) {
        let failure = match pending.committed.answer() {
            None => {
                state.outcomes.push_back(pending);
                continue;
            }
            Some(Ok(_)) => continue,
            Some(Err(error)) => error.to_string(),
        };
        state.last_failure = Some(failure);
        failed.push_back(pending.request);
    }
    failed.append(&mut state.waiting);
    state.waiting = failed;
    if state.outcomes.is_empty() && !state.waiting.iter().any(|next| is_outcome(&next.request)) {
        state.last_failure = None;
    }
    state
        .advice
        .retain(|_, advice| match &mut advice.committed {
            None => true,
            // A failed acknowledgment cannot establish stored history. Keep
            // that uncertain pair; it is never permission to prepare it again.
            Some(committed) => {
                !matches!(committed.answer(), Some(Ok(outcome)) if !outcome.outcome.outcome.pending)
            }
        });
    state
        .latest
        .retain(|_, latest| match &mut latest.committed {
            None => true,
            Some(committed) => committed.answer().is_none(),
        });
}

#[cfg(test)]
mod tests;
