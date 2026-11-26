//! Checks that take the one answer outcome a test expects and panic, showing
//! the whole answer, on any other. A test names the outcome it means:
//! `ok`, `empty` (exit 1) or `degraded` (exit 5, with its notes).

use command_surface::Answer;
use std::fmt::Debug;

/// An ok answer's value as its JSON.
#[track_caller]
pub(crate) fn json<T: Debug + serde::Serialize>(answer: Answer<T>) -> serde_json::Value {
    serde_json::to_value(ok(answer)).expect("a typed answer serializes")
}

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

/// A single successful step through the declared line operation.
#[track_caller]
pub(crate) fn step(answer: Answer<serde_json::Value>) -> crate::protocol::line::StepAnswer {
    use crate::protocol::line::{LineAnswer, StepOutcome};
    let line: LineAnswer = serde_json::from_value(
        answer
            .value()
            .unwrap_or_else(|| panic!("expected a line value: {answer:?}"))
            .clone(),
    )
    .unwrap();
    let LineAnswer::Steps(mut steps) = line else {
        panic!("expected steps: {answer:?}")
    };
    assert_eq!(steps.steps.len(), 1, "{answer:?}");
    match steps.steps.remove(0).outcome {
        StepOutcome::Done { reply, .. } => reply,
        other => panic!("expected a completed step: {other:?}"),
    }
}

/// The listing payload, preserving its empty/degraded outcome and envelope.
#[track_caller]
pub(crate) fn listing(answer: Answer<serde_json::Value>) -> Answer<serde_json::Value> {
    answer.map(|value| {
        let crate::protocol::line::LineAnswer::Listing(reply) =
            serde_json::from_value(value).unwrap()
        else {
            panic!("expected a listing")
        };
        serde_json::to_value(reply).unwrap()
    })
}

/// A single successful read step from the declared line operation.
#[track_caller]
pub(crate) fn read(answer: Answer<serde_json::Value>) -> polylogue_model::read::ReadPage {
    match step(answer) {
        crate::protocol::line::StepAnswer::Read(page) => *page,
        other => panic!("expected a read: {other:?}"),
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
