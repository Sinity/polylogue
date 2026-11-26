//! Commit-following work and its state in the thread activity.
use crate::{Result, StoreError, completion::CommitSubscription};
use polylogue_model::{Cause, Failure, Timestamp};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// The state of a worker that follows committed changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Worker {
    Running,
    Failing { error: Failure, since: Timestamp },
    Halted { error: Failure },
}

impl Worker {
    /// The problem an operator must see, if work cannot proceed.
    pub fn problem(&self, name: &str) -> Option<String> {
        match self {
            Self::Running => None,
            Self::Failing { error, .. } => Some(format!("{name}: failing: {error}")),
            Self::Halted { error } => Some(format!("{name}: halted: {error}")),
        }
    }
}

/// Runs once, then after each commit. A halted worker stays visible until
/// stopped, without attempting its work again. Returns whether its end was requested
/// or its finite snapshot finished.
pub fn follow(
    name: &'static str,
    subscription: Option<&CommitSubscription>,
    stop: &AtomicBool,
    now: impl Fn() -> Timestamp,
    changed: impl Fn(),
    mut work: impl FnMut() -> Result<()>,
) -> bool {
    let mut state = Worker::Running;
    crate::activity::worker(state.clone());
    loop {
        if stop.load(Ordering::Acquire) {
            return true;
        }
        if !matches!(state, Worker::Halted { .. }) {
            match work() {
                Ok(()) => {
                    if matches!(state, Worker::Failing { .. }) {
                        crate::log!(info, "{name}: recovered");
                        state = Worker::Running;
                        crate::activity::worker(state.clone());
                        changed();
                    }
                }
                Err(error) => {
                    if stop.load(Ordering::Acquire) || matches!(error, StoreError::Stopped) {
                        return true;
                    }
                    if !matches!(error, StoreError::IndexBuilding(_)) {
                        let error = Failure::of(&error);
                        let since = match &state {
                            Worker::Failing { since, .. } => *since,
                            _ => now(),
                        };
                        if !matches!(&state, Worker::Failing { error: previous, .. } if previous == &error)
                        {
                            crate::log!(error, "{name}: {error}");
                        }
                        state = if error.cause == Cause::Transient {
                            Worker::Failing { error, since }
                        } else {
                            Worker::Halted { error }
                        };
                        crate::activity::worker(state.clone());
                        changed();
                    }
                }
            }
        }
        let Some(subscription) = subscription else {
            return matches!(state, Worker::Running);
        };
        if subscription.recv().is_err() {
            return stop.load(Ordering::Acquire);
        }
    }
}

#[cfg(test)]
mod tests;
