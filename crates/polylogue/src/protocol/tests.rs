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
