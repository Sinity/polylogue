use super::*;
use serde_json::json;

#[test]
fn pending_receipt_is_degraded_with_http_success() {
    let receipt = json!({"id":"record:one", "pending":true});
    let answer = answer(receipt.clone());
    assert_eq!(answer.exit_code(), 5);
    assert_eq!(answer.http_status(), Some(200));
    assert_eq!(answer.value(), Some(&receipt));
    assert!(matches!(answer, Answer::Degraded { .. }));
}

#[test]
fn failures_follow_the_shared_table() {
    for (code, exit, http) in [
        ("invalid_query", 2, Some(400)),
        ("stale_cursor", 2, Some(400)),
        ("not_built", 2, Some(501)),
        ("refused", 3, Some(403)),
        ("conflict", 4, Some(409)),
        ("scope_changed", 4, Some(409)),
        ("io", 4, Some(500)),
        ("stopped_early", 5, Some(500)),
        ("daemon_required", 6, None),
        ("deadline_exceeded", 7, Some(504)),
        ("cancelled", 7, None),
    ] {
        let answer = Answer::<Value>::error(failure(code, "failed".into(), None));
        assert_eq!(answer.exit_code(), exit, "{code}");
        assert_eq!(answer.http_status(), http, "{code}");
        assert!(answer.value().is_none());
    }
}

#[test]
fn socket_uses_shared_envelope_without_losing_pending_identity() {
    let response = crate::daemon::service::ok(json!({"id":"record:pending", "pending":true}));
    let bytes = serde_json::to_vec(&response).unwrap();
    let answer: Answer<Value> = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(answer, Answer::Degraded { .. }));
    assert_eq!(answer.value().unwrap()["id"], "record:pending");
    assert_eq!(answer.exit_code(), 5);
    let read: crate::daemon::Response = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(read.answer().exit_code(), 5);
}

#[test]
fn pending_capture_returns_receipt_instead_of_storage_failure() {
    let response =
        crate::daemon::service::store_error(&polylogue_store::StoreError::PrimaryPending {
            identity: "capture:pending".into(),
        });
    let answer = response.answer();
    assert!(matches!(answer, Answer::Degraded { .. }));
    assert_eq!(answer.value().unwrap()["identity"], "capture:pending");
    assert_eq!(answer.http_status(), Some(200));
}

#[test]
fn query_empty_and_incomplete_outcomes_keep_their_shared_exit() {
    assert_eq!(
        answer(json!({"outcome":"empty", "rows":[], "count":0})).exit_code(),
        1
    );
    let incomplete = answer(
        json!({"outcome":"incomplete", "rows":[{"id":"session:one"}], "incomplete":["semantic vectors unavailable"]}),
    );
    assert_eq!(incomplete.exit_code(), 5);
    assert_eq!(incomplete.http_status(), Some(200));
    assert_eq!(
        incomplete.value().unwrap()["rows"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn a_bundle_with_unconfirmed_authored_restorations_is_degraded() {
    let report = json!({"imported":["synthetic.bundle"], "user_state":{"pending":1,"applied":0,"present":0}});
    let result = answer(report.clone());
    assert_eq!(result.exit_code(), 5);
    assert_eq!(result.http_status(), Some(200));
    assert_eq!(result.value(), Some(&report));
}

#[test]
fn projection_failures_distinguish_conflicts_from_invalid_input() {
    use polylogue_model::bundle::BundleError;
    for (issue, kind) in [
        (BundleError::ProjectionConflict, FailureKind::Conflict),
        (BundleError::SourceVersionMismatch, FailureKind::Conflict),
        (BundleError::InvalidBlockSelection, FailureKind::BadRequest),
    ] {
        let response =
            crate::daemon::service::store_error(&polylogue_store::StoreError::Projection(issue));
        let Answer::Error { error, .. } = response.answer() else {
            panic!("projection failure is not success")
        };
        assert_eq!(error.kind, kind);
    }
}

#[test]
fn noncurrent_primary_reads_keep_their_data_and_observed_positions_as_degraded() {
    use polylogue_model::currency::IndexCurrency;
    for currency in [
        IndexCurrency::primary(Some(5), Some(8), Some(0.5), Some(true), None),
        IndexCurrency::primary(Some(5), Some(8), Some(9.0), Some(true), None),
        IndexCurrency::primary(Some(5), None, None, Some(false), Some("unreachable".into())),
    ] {
        let value = json!({"rows":[{"ref":"session:synthetic"}], "read_at":{"currency":currency}});
        let result = answer(value.clone());
        assert_eq!(result.exit_code(), 5);
        assert_eq!(result.http_status(), Some(200));
        assert_eq!(result.value(), Some(&value));
    }
    let value = json!({"read_at":{"currency":IndexCurrency::primary(Some(8),Some(8),Some(0.5),Some(true),None)}});
    assert!(matches!(answer(value), Answer::Ok { .. }));
}
