//! Test-only fault injection. With the `faults` feature a
//! test arms a point to fail or panic; without it every check compiles to
//! nothing, and no shipped build enables the feature.

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
}

#[cfg(feature = "faults")]
mod armed {
    use super::Action;
    use std::sync::Mutex;

    /// `(point, subject substring, action, reaches to let pass, hits left)`.
    #[allow(clippy::type_complexity)]
    static POINTS: Mutex<Vec<(String, String, Action, u32, u32)>> = Mutex::new(Vec::new());

    /// Arms `point` for subjects containing `subject`: the first `skip`
    /// reaches pass, the next `times` act.
    pub fn arm(point: &str, subject: &str, action: Action, skip: u32, times: u32) {
        POINTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((point.to_owned(), subject.to_owned(), action, skip, times));
    }

    pub fn clear() {
        POINTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    pub fn take(point: &str, subject: &str) -> Option<Action> {
        let mut points = POINTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
pub use armed::{arm, clear};

/// Reaching `point` for `subject` (a path, a table): fails or panics if a
/// test armed it, otherwise nothing.
#[inline]
pub fn check(point: &str, subject: &str) -> crate::Result<()> {
    #[cfg(feature = "faults")]
    match armed::take(point, subject) {
        Some(Action::Fail) => {
            return Err(crate::StoreError::Invalid(format!(
                "injected fault at {point} ({subject})"
            )));
        }
        Some(Action::Panic) => panic!("injected panic at {point} ({subject})"),
        Some(Action::StopProcess) => {
            stop_process();
        }
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
    unsafe { libc::raise(libc::SIGSTOP); }
}
