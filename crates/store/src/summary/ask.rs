//! How a reader asks the summary worker of its process for one session's
//! summary and waits for it: a transcript read of a session that is due
//! shows what the last commit wrote ([`crate::Reader::item_state`]).

use crate::Result;
use crate::StoreError;
use polylogue_model::LockExt;
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// A session a reader waits for, and where the worker answers: `true` once
/// it computed the session's summaries (applied, or refused because its
/// rows changed meanwhile, which leaves it due), `false` when it was not
/// due where the worker reads or the worker cannot compute now.
pub(crate) struct Asked {
    pub session_pk: i64,
    pub reply: SyncSender<Result<bool>>,
}

impl Asked {
    pub(crate) fn answer(self, answer: Result<bool>) {
        let _ = self.reply.send(answer);
    }
}

type Send = dyn Fn(Asked) + std::marker::Send;

/// The summary worker of one archive in this process, shared by every
/// [`crate::archive::Archive`] value for the same root; set while a
/// writer runs.
#[derive(Default)]
pub(crate) struct Asking {
    worker: Mutex<Option<(u64, Box<Send>)>>,
}

impl std::fmt::Debug for Asking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Asking").finish_non_exhaustive()
    }
}

pub(crate) fn state(root: &Path) -> Arc<Asking> {
    static ARCHIVES: OnceLock<Mutex<std::collections::HashMap<std::path::PathBuf, Weak<Asking>>>> =
        OnceLock::new();
    let key = crate::read::scope::root_key(root);
    let mut archives = ARCHIVES.get_or_init(Default::default).locked();
    archives.retain(|_, state| state.strong_count() > 0);
    if let Some(state) = archives.get(&key).and_then(Weak::upgrade) {
        return state;
    }
    let state = Arc::new(Asking::default());
    archives.insert(key, Arc::downgrade(&state));
    state
}

/// A question sent to the worker, waiting for its answer.
pub(crate) struct Answer {
    reply: SyncSender<Result<bool>>,
    answer: Receiver<Result<bool>>,
}

impl Asking {
    /// The worker from now on: `send` delivers a question to it. Returns
    /// the registration that [`Self::end`] takes back.
    pub(crate) fn serve(&self, send: impl Fn(Asked) + std::marker::Send + 'static) -> u64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.worker.locked() = Some((id, Box::new(send)));
        id
    }

    /// The worker registered as `id` stops answering.
    pub(crate) fn end(&self, id: u64) {
        let mut worker = self.worker.locked();
        if worker.as_ref().is_some_and(|(held, _)| *held == id) {
            worker.take();
        }
    }

    /// Asks the worker for `session_pk`; `None` when no worker runs.
    pub(crate) fn ask(&self, session_pk: i64) -> Option<Answer> {
        let worker = self.worker.locked();
        let (_, send) = worker.as_ref()?;
        let (reply, answer) = sync_channel(1);
        send(Asked {
            session_pk,
            reply: reply.clone(),
        });
        Some(Answer { reply, answer })
    }
}

impl Answer {
    /// Waits for the worker's answer, until `limit`'s deadline or
    /// cancellation or the process's stop.
    pub(crate) fn wait(
        self,
        limit: Option<&crate::read::limit::ReadLimit>,
        stop: &crate::cancellation::Cancellation,
    ) -> Result<bool> {
        let wake = |reply: SyncSender<Result<bool>>| {
            move || {
                let _ = reply.try_send(Err(StoreError::Stopped));
            }
        };
        let _stop = stop.on_cancel(wake(self.reply.clone()));
        let _cancel = limit.map(|limit| limit.cancellation.on_cancel(wake(self.reply.clone())));
        drop(self.reply);
        let answer = loop {
            match limit.and_then(crate::read::limit::ReadLimit::remaining) {
                Some(left) => match self.answer.recv_timeout(left) {
                    Ok(answer) => break answer,
                    // Past the deadline the check refuses the read.
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        limit.map_or(Ok(()), crate::read::limit::ReadLimit::check)?;
                    }
                    // The worker ended without answering: the lines as stored.
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(false),
                },
                None => match self.answer.recv() {
                    Ok(answer) => break answer,
                    Err(_) => return Ok(false),
                },
            }
        };
        stop.check()?;
        if let Some(limit) = limit {
            limit.check()?;
        }
        answer
    }
}
