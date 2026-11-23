//! Test-only fault injection. With the `faults` feature a test arms a point
//! to fail, panic, stop the process or hold the thread; without it every
//! check compiles to nothing, and no shipped build enables the feature.

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
        POINTS
            .locked()
            .push((point.to_owned(), subject.to_owned(), action, skip, times));
    }

    /// Disarms every point and resets the hold. A thread released from a
    /// hold that has not woken yet would wait again, so a test clears only
    /// after the held thread has gone past its point (joined, or its work
    /// observed).
    pub fn clear() {
        POINTS.locked().clear();
        *HOLD.locked() = Hold::default();
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
pub use armed::{arm, clear, held, release};

/// Reaching `point` for `subject` (a path, a table): fails, panics, stops the
/// process or holds the thread if a test armed it (a hold or stop returns only
/// once the test releases or continues it), otherwise nothing.
#[inline]
pub fn check(point: &str, subject: &str) -> crate::Result<()> {
    #[cfg(feature = "faults")]
    match armed::take(point, subject) {
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
        None => {}
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
