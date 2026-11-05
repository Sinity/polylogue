//! Checks that take the one answer outcome a test expects and panic, showing
//! the whole answer, on any other. A test names the outcome it means:
//! `ok`, `empty` (exit 1) or `degraded` (exit 5, with its notes).

use command_surface::Answer;
use std::fmt::Debug;

#[track_caller]
pub(crate) fn ok<T: Debug>(answer: Answer<T>) -> T {
    match answer {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(value),
            ..
        } => value,
        other => panic!("expected an ok answer, got {other:?}"),
    }
}

#[track_caller]
pub(crate) fn empty<T: Debug>(answer: Answer<T>) -> T {
    match answer {
        command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            ..
        } => value,
        other => panic!("expected an empty answer, got {other:?}"),
    }
}

/// The value and the notes that say what is missing.
#[track_caller]
pub(crate) fn degraded<T: Debug>(answer: Answer<T>) -> (T, Vec<String>) {
    match answer {
        command_surface::Answer {
            outcome:
                command_surface::Outcome::Degraded {
                    value,
                    missing: notes,
                    ..
                },
            ..
        } => (value, notes),
        other => panic!("expected a degraded answer, got {other:?}"),
    }
}
