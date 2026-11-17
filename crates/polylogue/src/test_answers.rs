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

/// A single successful read step from the declared line operation.
#[track_caller]
pub(crate) fn read(answer: Answer<serde_json::Value>) -> polylogue_model::read::ReadPage {
    use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
    let line: LineAnswer = serde_json::from_value(
        answer
            .value()
            .unwrap_or_else(|| panic!("expected a read value: {answer:?}"))
            .clone(),
    )
    .unwrap();
    let LineAnswer::Steps(mut steps) = line else {
        panic!("expected read steps: {answer:?}")
    };
    assert_eq!(steps.steps.len(), 1, "{answer:?}");
    match steps.steps.remove(0).outcome {
        StepOutcome::Done {
            reply: StepAnswer::Read(page),
            ..
        } => *page,
        other => panic!("expected a completed read: {other:?}"),
    }
}

/// A refusal at acceptance or at the single requested line step.
#[track_caller]
pub(crate) fn refusal(answer: Answer<serde_json::Value>) -> command_surface::Failure {
    use crate::protocol::line::{LineAnswer, StepOutcome};
    if let command_surface::Outcome::Error(error) = &answer.outcome {
        return error.clone();
    }
    let line: LineAnswer = serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    let LineAnswer::Steps(mut steps) = line else {
        panic!("expected refused steps: {answer:?}")
    };
    assert_eq!(steps.steps.len(), 1, "{answer:?}");
    match steps.steps.remove(0).outcome {
        StepOutcome::Failed { error } => error,
        other => panic!("expected a refused read: {other:?}"),
    }
}
