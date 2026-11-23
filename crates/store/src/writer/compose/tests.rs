use super::*;
use crate::fixture::writer::message as source_message;
use polylogue_model::{Block, Role, Timestamp};

#[test]
fn explicit_content_preference_reads_only_possible_winners_and_keeps_other_groups() {
    let mut capture = source_message(
        "shared",
        "session",
        Role::Assistant,
        0,
        vec![Block::Text {
            text: "capture".into(),
        }],
    );
    capture.at = None;
    capture.model = Some("captured-model".into());
    let live = source_message(
        "shared",
        "session",
        Role::Assistant,
        30,
        vec![Block::Text {
            text: "live".into(),
        }],
    );
    let empty = source_message("shared", "session", Role::Assistant, 20, vec![]);
    let readings = [
        MessageReading {
            input: "capture",
            route: Route::Capture,
            main: true,
            completion: Completion::Finished,
            hash: &[0; 32],
            message: &capture,
        },
        MessageReading {
            input: "live",
            route: Route::Live,
            main: true,
            completion: Completion::Finished,
            hash: &[1; 32],
            message: &live,
        },
        MessageReading {
            input: "sparse",
            route: Route::Live,
            main: false,
            completion: Completion::InProgress,
            hash: &[2; 32],
            message: &empty,
        },
    ];
    // Explicit content outranks even an empty content group. Without any
    // explicit source, populated/main/live wins. Time and model are independent.
    let winners = [1, 0, 1, 1, 2, 0, 1, 1];
    for (mask, expected) in winners.into_iter().enumerate() {
        let mut inspected = Vec::new();
        let result = message(&readings, |n| {
            inspected.push(n);
            Ok(mask & (1 << n) != 0)
        })
        .unwrap()
        .unwrap();
        assert_eq!(result.main, expected);
        assert_eq!(result.message.blocks, readings[expected].message.blocks);
        assert_eq!(result.message.model.as_deref(), Some("captured-model"));
        assert_eq!(result.message.at, Timestamp::from_unix_ms(30));
        let expected_reads = if mask & 2 != 0 {
            vec![1]
        } else if mask & 1 != 0 {
            vec![1, 0]
        } else {
            vec![1, 0, 2]
        };
        assert_eq!(inspected, expected_reads);
    }
}

#[test]
fn a_needed_source_fact_refuses_and_a_losing_source_is_not_read() {
    let body = crate::fixture::writer::message(
        "same",
        "session",
        Role::Assistant,
        1,
        vec![Block::Text {
            text: "same".into(),
        }],
    );
    let readings = [
        MessageReading {
            input: "first",
            route: Route::Live,
            main: true,
            completion: Completion::Finished,
            hash: &[0; 32],
            message: &body,
        },
        MessageReading {
            input: "second",
            route: Route::Live,
            main: true,
            completion: Completion::Finished,
            hash: &[0; 32],
            message: &body,
        },
    ];
    let result = message(&readings, |n| {
        if n == 0 {
            Ok(true)
        } else {
            Err(crate::StoreError::Invalid("unneeded fact".into()))
        }
    })
    .unwrap()
    .unwrap();
    assert_eq!(result.main, 0);
    assert!(matches!(
        message(&readings, |_| Err(crate::StoreError::Invalid(
            "needed fact".into()
        ))),
        Err(crate::StoreError::Invalid(reason)) if reason == "needed fact"
    ));
    assert!(message(&[], |_| unreachable!()).unwrap().is_none());
    assert_eq!(
        message(std::slice::from_ref(&readings[0]), |_| unreachable!())
            .unwrap()
            .unwrap()
            .main,
        0
    );
}
