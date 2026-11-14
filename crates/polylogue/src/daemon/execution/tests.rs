use super::*;
use crate::daemon::handler::{Policy, within_limit};
use crate::test_daemon;

/// How the daemon treats this call before it runs.
fn policy(call: &test_daemon::Call) -> Policy {
    let row = crate::commands::command(&call.operation).unwrap();
    row.registration()
        .unwrap()
        .admit(
            &call.arguments,
            row,
            &crate::commands::Caller {
                surface: call.surface,
                cwd: None,
            },
        )
        .unwrap()
}

pub(crate) const SLOW: &str = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n";

#[test]
fn background_readers_keep_their_ordinary_busy_wait() {
    let daemon = test_daemon::start("background reader");
    assert!(ReadLimit::current().is_none());
    let reader = daemon.daemon.reads.reader().unwrap();
    let rows = reader
        .query("SELECT timeout FROM pragma_busy_timeout", &[])
        .unwrap();
    assert_eq!(rows[0]["timeout"], 30000);
}

#[test]
fn slow_read_returns_a_typed_deadline_with_unknown_count() {
    let daemon = test_daemon::start("synthetic deadline");
    let limit = ReadLimit::new(Duration::from_millis(50), Default::default());
    let response = crate::test_daemon::Call::new("sql", serde_json::json!({"statement": SLOW}))
        .limit(limit.clone())
        .answer(&daemon.daemon);
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        envelope: command_surface::Envelope { notes, .. },
        ..
    } = &response
    else {
        panic!("expected a deadline refusal");
    };
    assert_eq!(error.code.as_deref(), Some("deadline_exceeded"));
    assert!(error.retryable);
    let [note] = notes.as_slice() else {
        panic!("one note names the stage: {notes:?}")
    };
    let steps: u64 = note
        .strip_prefix("stopped during sql after ")
        .and_then(|rest| rest.split_once(' '))
        .and_then(|(steps, _)| steps.parse().ok())
        .unwrap_or_else(|| panic!("{note}"));
    assert!(steps > 0, "{note}");
    assert!(note.ends_with("rows scanned are unknown"), "{note}");
    assert_eq!(response.exit_code(), 7);
}

#[test]
fn expiry_at_completion_never_returns_a_partial_success() {
    let limit = ReadLimit::new(Duration::from_millis(10), Default::default());
    let response = within_limit(&limit, true, false, "status", None, || {
        std::thread::sleep(Duration::from_millis(20));
        super::super::service::ok(serde_json::json!({"count": 123}))
    });
    assert!(
        matches!(response, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("deadline_exceeded"))
    );
}

#[test]
fn expired_requests_never_start_any_read_operation() {
    let daemon = test_daemon::start("synthetic limits");
    let requests = [
        crate::test_daemon::Call::new("status", serde_json::json!({})),
        crate::test_daemon::Call::new(
            "read",
            serde_json::json!({"target": "session:claude-code:s1", "format": "json"}),
        )
        .cursor(Some("never-open-an-expired-request")),
        crate::test_daemon::Call::new(
            "read",
            serde_json::json!({"target": "session:claude-code:s1", "format": "json"}),
        ),
        crate::test_daemon::Call::new("recall", serde_json::json!({"query": "synthetic"}))
            .id("synthetic-recall"),
        crate::test_daemon::Call::new(
            "query",
            serde_json::json!({"query": "sessions | measure usage"}),
        ),
        crate::test_daemon::Call::new(
            "complete",
            serde_json::json!({"text": "repo:", "position": 5}),
        ),
        crate::test_daemon::Call::new("query", serde_json::json!({"query": "sessions"})),
        crate::test_daemon::Call::new("sql", serde_json::json!({"statement": "SELECT 1"})),
    ];
    for request in requests {
        let limit = ReadLimit::new(Duration::ZERO, Default::default());
        assert!(
            matches!((request).limit(limit.clone()).answer(&daemon.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("deadline_exceeded"))
        );
    }
}

#[test]
fn narrower_suggestions_preserve_quoted_pipes_and_pipeline_stages() {
    for (query, expected) in [
        ("sessions | count", "sessions where since:1d | count"),
        ("\"a|b\"", "(\"a|b\") since:1d"),
        (
            "messages where role:user OR role:assistant | count",
            "messages where (role:user OR role:assistant) since:1d | count",
        ),
    ] {
        assert_eq!(narrower(query).as_deref(), Some(expected));
    }
}

#[test]
fn write_requests_do_not_enter_a_read_scope_even_with_an_expired_cancelled_limit() {
    let limit = ReadLimit::new(Duration::ZERO, Default::default());
    limit.cancellation.cancel();
    for request in [
        crate::test_daemon::Call::new("keep", serde_json::json!({"query": "sessions"}))
            .id("synthetic-keep"),
        crate::test_daemon::Call::new("embed build", serde_json::json!({})),
    ] {
        let policy = policy(&request);
        assert!(!policy.read_phase, "{}", request.operation);
        let response = within_limit(&limit, policy.read_phase, false, "write", None, || {
            assert!(ReadLimit::current().is_none());
            super::super::service::ok(serde_json::json!({"done": true}))
        });
        assert_eq!(
            crate::test_answers::ok(response),
            serde_json::json!({"done": true})
        );
    }
}

#[test]
fn streamed_read_has_no_default_deadline_and_explicit_deadlines_still_apply() {
    let request = crate::test_daemon::Call::new(
        "read",
        serde_json::json!({"target": "sessions", "delivery": "stream", "format": "json"}),
    );
    let policy = policy(&request);
    assert!(policy.read_phase && policy.streaming && !policy.default_deadline);
    let response = within_limit::<()>(
        &ReadLimit::new(Duration::ZERO, Default::default()),
        policy.read_phase,
        false,
        "read",
        None,
        || panic!("expired streamed read started"),
    );
    assert!(
        matches!(&response, command_surface::Answer { outcome: command_surface::Outcome::Error(error), envelope: command_surface::Envelope { notes, .. }, .. } if error.code.as_deref() == Some("deadline_exceeded") && notes[0].starts_with("stopped during read "))
    );
}

#[test]
fn other_read_requests_use_their_operation_name_in_refusals() {
    let daemon = test_daemon::start("synthetic refusal stage");
    let response = crate::test_daemon::Call::new("backfill list", serde_json::json!({}))
        .limit(ReadLimit::new(Duration::ZERO, Default::default()))
        .answer(&daemon.daemon);
    assert!(
        matches!(&response, command_surface::Answer { outcome: command_surface::Outcome::Error(_), envelope: command_surface::Envelope { notes, .. }, .. } if notes[0].starts_with("stopped during backfill list ")),
        "{response:?}"
    );
}

#[test]
fn context_receipts_do_not_disable_the_read_phase_deadline() {
    let daemon = test_daemon::start("synthetic context deadline");
    let request =
        crate::test_daemon::Call::new("takeover", serde_json::json!({"session": "claude-code:s1"}))
            .id("synthetic-context")
            .surface(crate::commands::Surface::Mcp);
    let policy = policy(&request);
    assert!(!policy.read_only && policy.read_phase && policy.default_deadline);
    let response = request
        .limit(ReadLimit::new(Duration::ZERO, Default::default()))
        .answer(&daemon.daemon);
    assert!(
        matches!(response, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("deadline_exceeded"))
    );
}

#[test]
fn an_authored_write_commits_with_an_expired_cancelled_read_budget() {
    let daemon = test_daemon::start("authored write deadline independence");
    let limit = ReadLimit::new(Duration::ZERO, Default::default());
    limit.cancellation.cancel();
    let id = polylogue_store::user_writes::new_request_id();
    let response = crate::test_daemon::Call::new("keep", serde_json::json!({"query": "sessions"}))
        .id(id.clone().as_ref())
        .limit(limit.clone())
        .answer(&daemon.daemon);
    crate::test_answers::ok(response);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .user_request(&id)
            .unwrap()
            .is_some()
    );
}

#[test]
fn continued_read_is_a_page_with_an_ordinary_read_deadline() {
    let request = crate::test_daemon::Call::new(
        "read",
        serde_json::json!({"target": "session:claude-code:s1", "format": "json"}),
    )
    .cursor(Some("opaque-awaiting-owner-validation"));
    let policy = policy(&request);
    assert!(policy.read_only && !policy.streaming && policy.default_deadline);
}

#[test]
fn a_read_first_write_whose_deadline_passes_after_handoff_answers_its_write() {
    let limit = ReadLimit::new(Duration::from_millis(50), Default::default());
    let response = within_limit(&limit, true, false, "assert-query", None, || {
        crate::daemon::handler::finish_read_phase("write").expect("handed over in time");
        assert!(ReadLimit::current().is_none(), "the read phase ended");
        // The writer's commit outlasts the read deadline.
        std::thread::sleep(Duration::from_millis(100));
        crate::daemon::service::ok(serde_json::json!({"written": true}))
    });
    assert_eq!(
        crate::test_answers::ok(response),
        serde_json::json!({"written": true})
    );
}

#[test]
fn a_read_phase_that_has_run_out_refuses_the_handoff() {
    let limit = ReadLimit::new(Duration::ZERO, Default::default());
    limit.scope(|| {
        assert!(crate::daemon::handler::finish_read_phase("write").is_err());
    });
}
