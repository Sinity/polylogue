use super::*;

#[test]
fn a_panicking_operation_answers_internal_and_the_next_request_succeeds() {
    let daemon = crate::test_daemon::start("request panic fixture");
    polylogue_store::fault::arm(
        "read.query",
        "panic-answer-probe",
        polylogue_store::fault::Action::Panic,
        0,
        1,
    );
    let call = request(
        "dev sql",
        json!({"statement":"SELECT 'panic-answer-probe' AS example"}),
        "panic-answer",
    );
    let failed = crate::client::call_ready(&daemon.daemon.archive, &call).unwrap();
    let failure = failed.failure().unwrap();
    assert_eq!(failure.code.as_deref(), Some("internal"));
    assert!(failure.message.contains("panic"), "{failure:?}");
    assert!(failure.message.contains("request ID"), "{failure:?}");
    let next = crate::client::call_ready(&daemon.daemon.archive, &call).unwrap();
    assert_eq!(next.exit_code(), 0, "{next:?}");
    assert_eq!(
        next.value(),
        Some(&json!([{"example":"panic-answer-probe"}]))
    );
}
