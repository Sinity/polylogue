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
        ("invalid_query", 2, Some(400)),
        ("invalid_cursor", 2, Some(400)),
        ("stale_cursor", 2, Some(400)),
        ("not_built", 2, Some(501)),
        ("refused", 3, Some(403)),
        ("conflict", 4, Some(409)),
        ("scope_changed", 4, Some(409)),
        ("not_the_claimant", 4, Some(409)),
        ("job_ended", 4, Some(409)),
        ("browser_actions_off", 4, Some(409)),
        ("unknown_chat_action", 4, Some(404)),
        ("attachment_unavailable", 4, Some(404)),
        ("unsupported_provider", 2, Some(400)),
        ("submission_receipt_required", 2, Some(400)),
        ("submission_conversation_mismatch", 2, Some(400)),
        ("io", 4, Some(500)),
        ("stopped_early", 5, Some(500)),
        ("daemon_required", 6, None),
        ("deadline_exceeded", 7, Some(504)),
        ("cancelled", 7, None),
    ] {
        let answer = Answer::<Value>::error(failure(code, "failed"));
        assert_eq!(answer.exit_code(), exit, "{code}");
        assert_eq!(answer.http_status(), http, "{code}");
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
        code: "stale_cursor",
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
