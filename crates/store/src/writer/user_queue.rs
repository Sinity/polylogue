//! Bounded user-write admission, with space reserved before a read starts.
use super::{USER_QUEUE, UserMsg};
use crate::read::limit::ReadLimit;
use crate::{Result, StoreError};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

#[derive(Default)]
struct State {
    closed: bool,
    reserved: usize,
    waiting: VecDeque<UserMsg>,
    #[cfg(test)]
    waiters: usize,
}

#[derive(Default)]
pub(super) struct UserQueue {
    state: Mutex<State>,
    available: Condvar,
    #[cfg(test)]
    blocked: Condvar,
}

pub(super) struct Reservation {
    queue: Option<Arc<UserQueue>>,
}

impl UserQueue {
    pub(super) fn reserve(self: &Arc<Self>, limit: Option<&ReadLimit>) -> Result<Reservation> {
        let _cancel = limit.map(|limit| {
            let queue = Arc::clone(self);
            limit.cancellation.on_cancel(move || {
                let _state = queue
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                queue.available.notify_all();
            })
        });
        let mut state = self.state.lock().map_err(|_| StoreError::WriterGone)?;
        loop {
            if let Some(limit) = limit {
                limit.check()?;
            }
            if state.closed {
                return Err(StoreError::WriterGone);
            }
            if state.waiting.len() + state.reserved < USER_QUEUE {
                state.reserved += 1;
                return Ok(Reservation {
                    queue: Some(Arc::clone(self)),
                });
            }
            #[cfg(test)]
            {
                state.waiters += 1;
                self.blocked.notify_all();
            }
            state = match limit.and_then(ReadLimit::remaining) {
                Some(left) => {
                    self.available
                        .wait_timeout(state, left)
                        .map_err(|_| StoreError::WriterGone)?
                        .0
                }
                None => self
                    .available
                    .wait(state)
                    .map_err(|_| StoreError::WriterGone)?,
            };
            #[cfg(test)]
            {
                state.waiters -= 1;
            }
        }
    }

    pub(super) fn try_push(&self, message: UserMsg) -> std::result::Result<(), Option<UserMsg>> {
        let mut state = self.state.lock().map_err(|_| None)?;
        if state.closed {
            return Err(None);
        }
        // Unfinished reads reserve their eventual handoff, not the capacity
        // ordinary user writes need while those reads are still running.
        if state.waiting.len() >= USER_QUEUE {
            return Err(Some(message));
        }
        state.waiting.push_back(message);
        Ok(())
    }

    pub(super) fn push_wait(&self, message: UserMsg) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| StoreError::WriterGone)?;
        loop {
            if state.closed {
                return Err(StoreError::WriterGone);
            }
            if state.waiting.len() < USER_QUEUE {
                state.waiting.push_back(message);
                return Ok(());
            }
            state = self
                .available
                .wait(state)
                .map_err(|_| StoreError::WriterGone)?;
        }
    }

    pub(super) fn waiting(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .waiting
            .len()
    }

    pub(super) fn pop(&self) -> Option<UserMsg> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let next = state.waiting.pop_front();
        if next.is_some() {
            self.available.notify_all();
        }
        next
    }

    /// Closing admission retains every already queued write for the final turn.
    pub(super) fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.closed = true;
        self.available.notify_all();
    }

    /// The receiver has exited; dropping queued replies wakes their callers.
    pub(super) fn disconnect(&self) {
        let pending = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            state.closed = true;
            self.available.notify_all();
            std::mem::take(&mut state.waiting)
        };
        drop(pending);
    }
}

#[cfg(test)]
mod tests;

impl Reservation {
    pub(super) fn push(mut self, message: UserMsg) -> Result<()> {
        let queue = self.queue.take().expect("an unconsumed reservation");
        let mut state = queue.state.lock().map_err(|_| StoreError::WriterGone)?;
        state.reserved -= 1;
        queue.available.notify_all();
        if state.closed {
            return Err(StoreError::WriterGone);
        }
        state.waiting.push_back(message);
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let Some(queue) = self.queue.take() {
            let mut state = queue
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.reserved -= 1;
            queue.available.notify_all();
        }
    }
}
