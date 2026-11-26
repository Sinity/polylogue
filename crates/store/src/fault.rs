//! Test-only fault injection. With the `faults` feature a test arms a point
//! to fail, panic, stop the process or hold the thread; without it every
//! check compiles to nothing, and no shipped build enables the feature.

/// Points checked by the store, engine and application.
#[cfg(feature = "faults")]
const DECLARED_POINTS: &[&str] = &[
    "assets.install",
    "bytes.rows",
    "capture.after_commit",
    "capture.between_put_and_commit",
    "completion.before_commit",
    "completion.full_snapshot",
    "database.after_gone",
    "database.after_row",
    "import.after_authored_restore",
    "import.before_authored_restore",
    "import.rows",
    "index.after_creation",
    "index.after_set_aside",
    "index.build.after_finish_sync",
    "index.build.before_rename",
    "index.build.composing",
    "index.build.loading",
    "index.build.prepare",
    "index.deferred",
    "parse",
    "parser.checkpoint",
    "pass.admit",
    "pool.chain_reply",
    "read.annotation_validity",
    "read.chains_behind_index",
    "read.query",
    "read.source_range",
    "read.watched_inputs",
    "reparse.commit",
    "reparse.prepare",
    "repo_lookup.refresh",
    "search.before_queue",
    "search.read_queue",
    "segment.after_append",
    "segment.after_sync",
    "tool_database.comparison_commit",
    "vectors.read",
    "write.archive_commit",
    "write.copy_reading",
    "write.copy_table",
    "write.coverage",
    "write.index_commit",
    "write.input",
    "write.register_sync",
    "write.replay_rows",
    "write.user",
];

/// What an armed point does when reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Return an error, as a failing disk or database would.
    Fail,
    /// Panic, as a bug would, or as the process dying at that step would
    /// look to the transaction it leaves open.
    Panic,
    /// Stop a subprocess at the boundary so its test parent can kill it.
    StopProcess,
    /// Block the thread at the point until the test calls [`release`], so the
    /// test can act while the code is stopped at a known step ([`held`]).
    Hold,
    /// Let the thread pass and record which thread reached the point
    /// ([`recorded`]).
    Record,
}

#[cfg(feature = "faults")]
mod armed {
    use super::Action;
    use polylogue_model::LockExt;
    use std::sync::{Condvar, Mutex};

    /// `(point, subject substring, action, reaches to let pass, hits left)`.
    #[allow(clippy::type_complexity)]
    static POINTS: Mutex<Vec<(String, String, Action, u32, u32)>> = Mutex::new(Vec::new());

    /// Arms `point` for subjects containing `subject`: the first `skip`
    /// reaches pass, the next `times` act.
    pub fn arm(point: &str, subject: &str, action: Action, skip: u32, times: u32) {
        assert!(
            super::DECLARED_POINTS.contains(&point),
            "undeclared fault point: {point}"
        );
        POINTS
            .locked()
            .push((point.to_owned(), subject.to_owned(), action, skip, times));
    }

    /// `(point, subject, thread name)` of each action fired at an armed point.
    static RECORDED: Mutex<Vec<(String, String, String)>> = Mutex::new(Vec::new());

    pub(super) fn record(point: &str, subject: &str) {
        let thread = std::thread::current().name().unwrap_or("").to_owned();
        RECORDED
            .locked()
            .push((point.to_owned(), subject.to_owned(), thread));
    }

    /// The actions fired at `point` so far, in order, as
    /// `(subject, thread name)`.
    pub fn recorded(point: &str) -> Vec<(String, String)> {
        RECORDED
            .locked()
            .iter()
            .filter(|(p, _, _)| p == point)
            .map(|(_, subject, thread)| (subject.clone(), thread.clone()))
            .collect()
    }

    /// Disarms every point and resets the hold. A thread released from a
    /// hold that has not woken yet would wait again, so a test clears only
    /// after the held thread has gone past its point (joined, or its work
    /// observed).
    pub fn clear() {
        POINTS.locked().clear();
        *HOLD.locked() = Hold::default();
        RECORDED.locked().clear();
    }

    #[derive(Default)]
    struct Hold {
        held: bool,
        released: bool,
    }
    static HOLD: Mutex<Hold> = Mutex::new(Hold {
        held: false,
        released: false,
    });
    static HOLD_CHANGED: Condvar = Condvar::new();

    pub(super) fn hold() {
        let mut state = HOLD.locked();
        state.held = true;
        HOLD_CHANGED.notify_all();
        while !state.released {
            state = polylogue_model::unpoisoned(HOLD_CHANGED.wait(state));
        }
    }

    /// Whether a thread has reached a point armed with [`Action::Hold`].
    pub fn held() -> bool {
        HOLD.locked().held
    }

    /// Blocks until a thread has reached a point armed with [`Action::Hold`],
    /// or `timeout` passes. Returns whether a thread is held.
    pub fn wait_held_within(timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut state = HOLD.locked();
        while !state.held {
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return false;
            };
            state = polylogue_model::unpoisoned(HOLD_CHANGED.wait_timeout(state, left)).0;
        }
        true
    }

    /// Blocks until a thread has reached a point armed with [`Action::Hold`];
    /// panics if none does within a minute, so a test never hangs.
    pub fn wait_held() {
        assert!(
            wait_held_within(std::time::Duration::from_secs(60)),
            "no thread reached the held point"
        );
    }

    /// Lets the thread stopped at an [`Action::Hold`] point continue.
    pub fn release() {
        HOLD.locked().released = true;
        HOLD_CHANGED.notify_all();
    }

    pub fn take(point: &str, subject: &str) -> Option<Action> {
        let mut points = POINTS.locked();
        let i = points
            .iter()
            .position(|(p, s, _, _, n)| p == point && subject.contains(s.as_str()) && *n > 0)?;
        let entry = &mut points[i];
        if entry.3 > 0 {
            entry.3 -= 1;
            return None;
        }
        entry.4 -= 1;
        Some(entry.2)
    }
}

#[cfg(feature = "faults")]
pub use armed::{arm, clear, held, recorded, release, wait_held, wait_held_within};

/// Reaching `point` for `subject` (a path, a table): fails, panics, stops the
/// process or holds the thread if a test armed it (a hold or stop returns only
/// once the test releases or continues it), otherwise nothing.
#[inline]
pub fn check(point: &str, subject: &str) -> crate::Result<()> {
    #[cfg(feature = "faults")]
    {
        assert!(
            DECLARED_POINTS.contains(&point),
            "undeclared fault point: {point}"
        );
        let action = armed::take(point, subject);
        if action.is_some() {
            armed::record(point, subject);
        }
        match action {
            Some(Action::Fail) => {
                // An injected failure stands for a step that could not
                // complete (a sync, a commit), never a refusal of content.
                return Err(crate::StoreError::Io(std::io::Error::other(format!(
                    "injected fault at {point} ({subject})"
                ))));
            }
            Some(Action::Panic) => panic!("injected panic at {point} ({subject})"),
            Some(Action::StopProcess) => {
                stop_process();
            }
            Some(Action::Hold) => armed::hold(),
            Some(Action::Record) => {}
            None => {}
        }
    }
    let _ = (point, subject);
    Ok(())
}

#[cfg(feature = "faults")]
#[allow(unsafe_code)]
fn stop_process() {
    // SIGSTOP has no memory preconditions. The test parent owns this process
    // and kills it without running Rust destructors.
    unsafe {
        libc::raise(libc::SIGSTOP);
    }
}
