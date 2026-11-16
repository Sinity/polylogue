use super::*;
use serde_json::{Value, json};

#[test]
fn a_pending_write_is_degraded_with_its_note_and_http_success() {
    let write_result = json!({"id":"record:one", "pending":true});
    let answer = written(write_result.clone(), true);
    assert_eq!(answer.exit_code(), 5);
    assert_eq!(answer.http_status(), Some(200));
    assert_eq!(answer.value(), Some(&write_result));
    assert_eq!(
        crate::protocol::notes(&answer).collect::<Vec<_>>(),
        [AWAITING_CONFIRMATION]
    );
    assert!(matches!(
        written(write_result, false),
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(_),
            ..
        }
    ));
}

#[test]
fn failures_follow_the_shared_table() {
    for (code, exit, http) in [
        (polylogue_model::Code::InvalidQuery, 2, Some(400)),
        (polylogue_model::Code::InvalidCursor, 2, Some(400)),
        (polylogue_model::Code::StaleCursor, 2, Some(400)),
        (polylogue_model::Code::NotBuilt, 2, Some(501)),
        (polylogue_model::Code::Refused, 3, Some(403)),
        (polylogue_model::Code::Conflict, 4, Some(409)),
        (polylogue_model::Code::ScopeChanged, 4, Some(409)),
        (polylogue_model::Code::NotTheClaimant, 4, Some(409)),
        (polylogue_model::Code::JobEnded, 4, Some(409)),
        (polylogue_model::Code::BrowserActionsOff, 4, Some(409)),
        (polylogue_model::Code::UnknownChatAction, 4, Some(404)),
        (polylogue_model::Code::AttachmentUnavailable, 4, Some(404)),
        (polylogue_model::Code::UnsupportedProvider, 2, Some(400)),
        (
            polylogue_model::Code::SubmissionReceiptRequired,
            2,
            Some(400),
        ),
        (
            polylogue_model::Code::SubmissionConversationMismatch,
            2,
            Some(400),
        ),
        (polylogue_model::Code::Io, 4, Some(500)),
        (polylogue_model::Code::StoppedEarly, 5, Some(500)),
        (polylogue_model::Code::DaemonRequired, 6, None),
        (polylogue_model::Code::DeadlineExceeded, 7, Some(504)),
        (polylogue_model::Code::Cancelled, 7, None),
    ] {
        let answer = Answer::<Value>::error(failure(code, "failed"));
        assert_eq!(answer.exit_code(), exit, "{code:?}");
        assert_eq!(answer.http_status(), http, "{code:?}");
        assert!(answer.value().is_none());
    }
}

#[test]
fn a_degraded_answer_keeps_its_outcome_and_notes_on_the_socket_wire() {
    let answer = written(json!({"id":"record:pending", "pending":true}), true);
    let bytes = serde_json::to_vec(&answer).unwrap();
    let read: Answer<Value> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(read, answer);
    assert_eq!(
        crate::protocol::notes(&read).collect::<Vec<_>>(),
        [AWAITING_CONFIRMATION]
    );
    assert_eq!(read.exit_code(), 5);
}

#[test]
fn bundle_failures_distinguish_conflicts_from_invalid_input() {
    use polylogue_model::bundle::BundleError;
    for (issue, kind) in [
        (BundleError::BlockSelectionConflict, FailureKind::Conflict),
        (BundleError::SourceVersionMismatch, FailureKind::Conflict),
        (BundleError::InvalidBlockSelection, FailureKind::BadRequest),
    ] {
        let answer =
            crate::daemon::service::store_error::<()>(&polylogue_store::StoreError::Bundle(issue));
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = answer
        else {
            panic!("block selection failure is not success")
        };
        assert_eq!(error.kind, kind);
    }
}

#[test]
fn an_ambiguous_reference_names_every_candidate() {
    let error = polylogue_store::StoreError::Ambiguous {
        reference: "7f3a".into(),
        candidates: vec!["session:test:7f3a-1".into(), "session:test:7f3a-2".into()],
    };
    let failure = store_failure(&error);
    assert_eq!(failure.kind, FailureKind::Ambiguous);
    assert_eq!(
        failure.candidates(),
        ["session:test:7f3a-1", "session:test:7f3a-2"]
    );
    let selection = polylogue_store::StoreError::Selection {
        code: polylogue_model::Code::StaleCursor,
        message: "the selected row position is unavailable".into(),
    };
    assert_eq!(store_failure(&selection).kind, FailureKind::StaleCursor);
    assert_eq!(
        store_failure(&selection).code.as_deref(),
        Some("stale_cursor")
    );
}

#[test]
fn noncurrent_primary_reads_keep_their_data_and_say_why() {
    use polylogue_model::freshness::IndexFreshness;
    for freshness in [
        IndexFreshness::primary(Some(5), Some(8), Some(0.5), Some(true), None),
        IndexFreshness::primary(Some(5), Some(8), Some(9.0), Some(true), None),
        IndexFreshness::primary(Some(5), None, None, Some(false), Some("unreachable".into())),
    ] {
        let value = json!({"rows":[{"ref":"session:synthetic"}]});
        let freshness = polylogue_model::freshness::ReadFreshness {
            local: polylogue_model::freshness::ReadCompleteness::empty(),
            sinex: freshness.clone(),
        };
        let result = read(value.clone(), &freshness);
        assert_eq!(result.exit_code(), 5);
        assert_eq!(result.http_status(), Some(200));
        assert_eq!(result.value(), Some(&value));
        assert_eq!(
            crate::protocol::notes(&result).collect::<Vec<_>>().len(),
            1,
            "{freshness:?}"
        );
    }
    let current = IndexFreshness::primary(Some(8), Some(8), Some(0.5), Some(true), None);
    assert!(matches!(
        read(
            json!({}),
            &polylogue_model::freshness::ReadFreshness {
                local: polylogue_model::freshness::ReadCompleteness::empty(),
                sinex: current
            }
        ),
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(_),
            ..
        }
    ));
}

#[test]
fn pending_browser_results_keep_their_kind_on_the_wire() {
    use crate::protocol::browser::{ChatAnswer, PendingCapture};
    let capture = written(
        ChatAnswer::Capture(PendingCapture {
            identity: "capture:synthetic".into(),
        }),
        true,
    );
    let bytes = serde_json::to_vec(&capture).unwrap();
    let decoded: command_surface::Answer<ChatAnswer> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded.exit_code(), 5);
    assert!(
        matches!(decoded.value(),Some(ChatAnswer::Capture(write_result)) if write_result.identity=="capture:synthetic")
    );
    let outcome = polylogue_model::user::UserOutcome {
        id: "action:synthetic".into(),
        repeat: true,
        pending: true,
    };
    let authored = written(ChatAnswer::Outcome(outcome.clone()), outcome.pending);
    let bytes = serde_json::to_vec(&authored).unwrap();
    let decoded: command_surface::Answer<ChatAnswer> = serde_json::from_slice(&bytes).unwrap();
    assert!(
        matches!(decoded.value(),Some(ChatAnswer::Outcome(write_result)) if write_result==&outcome)
    );
    assert!(
        serde_json::from_value::<ChatAnswer>(json!({"identity":"capture:synthetic","action":null}))
            .is_err()
    );
}

#[test]
fn typed_failure_codes_keep_their_wire_names_and_classes() {
    use polylogue_model::Code;
    for (code, wire, expected) in [
        (Code::Io, "io", FailureKind::Io),
        (Code::ReadFailed, "read_failed", FailureKind::Io),
        (Code::Empty, "empty", FailureKind::Io),
        (Code::ModelCallsOff, "model_calls_off", FailureKind::Io),
        (
            Code::DaemonUnavailable,
            "daemon_unavailable",
            FailureKind::Io,
        ),
        (Code::Store, "store", FailureKind::Io),
        (Code::Internal, "internal", FailureKind::Io),
        (
            Code::InvalidQuery,
            "invalid_query",
            FailureKind::InvalidQuery,
        ),
        (Code::StaleCursor, "stale_cursor", FailureKind::StaleCursor),
        (
            Code::NotYetAvailable,
            "not_yet_available",
            FailureKind::NotYetAvailable,
        ),
    ] {
        let result = failure(code, "synthetic failure");
        assert_eq!(result.kind, expected);
        assert_eq!(result.code.as_deref(), Some(wire));
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["code"], wire);
        let decoded: command_surface::Failure = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.kind, expected);
        assert_eq!(decoded.code.as_deref(), Some(wire));
    }
}
