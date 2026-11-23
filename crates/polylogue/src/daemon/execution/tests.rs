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
fn background_readers_keep_their_busy_wait() {
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
    let response = crate::test_daemon::Call::new("dev sql", serde_json::json!({"statement": SLOW}))
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
        .strip_prefix("stopped during dev sql after ")
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
            "rows",
            serde_json::json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "format":"json"}]}),
        )
        .cursor(Some("never-open-an-expired-request")),
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "format":"json"}]}),
        ),
        crate::test_daemon::Call::new("rows", serde_json::json!({"selection":"synthetic", "steps":[{"verb":"brief"}]}))
            .id("synthetic-context"),
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": "sessions | measure usage"}),
        ),
        crate::test_daemon::Call::new(
            "internal complete",
            serde_json::json!({"text": "repo:", "position": 5}),
        ),
        crate::test_daemon::Call::new("rows", serde_json::json!({"selection": "sessions"})),
        crate::test_daemon::Call::new("dev sql", serde_json::json!({"statement": "SELECT 1"})),
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
        crate::test_daemon::Call::new(
            "set",
            serde_json::json!({"key":"browser.actions", "value":"on"}),
        )
        .id("synthetic-keep"),
        crate::test_daemon::Call::new("embed", serde_json::json!({"yes":true})),
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
    let mut request = crate::client::request(
        "rows",
        serde_json::json!({"selection":"sessions", "steps":[{"verb":"read", "delivery":"stream", "format":"json"}]}),
        "stream-deadline".into(),
    );
    let row = crate::commands::command("rows").unwrap();
    let registration = row.registration().unwrap();
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Cli,
        cwd: None,
    };
    let policy = registration
        .admit(&request.arguments, row, &caller)
        .unwrap();
    assert!(policy.read_phase && policy.streaming && !policy.default_deadline);
    request.deadline = policy.deadline(request.deadline);
    let limit_for = |deadline, parent| {
        request_limit(
            deadline,
            row.latency,
            policy.default_deadline,
            parent,
            Default::default(),
        )
    };
    assert_eq!(request.deadline, Deadline::None);
    assert!(limit_for(request.deadline, None).deadline().is_none());

    request.deadline = policy.deadline(Deadline::Milliseconds(5_000));
    assert_eq!(request.deadline, Deadline::Milliseconds(5_000));
    assert!(limit_for(request.deadline, None).deadline().is_some());
    let parent = Instant::now();
    let limit = limit_for(request.deadline, Some(parent));
    assert_eq!(limit.deadline(), Some(parent));
    let response = within_limit::<()>(&limit, policy.read_phase, false, "read", None, || {
        panic!("expired streamed read started")
    });
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
fn context_writes_keep_the_read_phase_deadline() {
    let daemon = test_daemon::start("synthetic context deadline");
    let request = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"session:claude-code:s1", "steps":[{"verb":"brief"}]}),
    )
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
    let response = crate::test_daemon::Call::new(
        "set",
        serde_json::json!({"key":"browser.actions", "value":"on"}),
    )
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
fn continued_read_is_a_page_with_a_read_deadline() {
    let request = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "format":"json"}]}),
    )
    .cursor(Some("opaque-awaiting-daemon-validation"));
    let policy = policy(&request);
    assert!(policy.read_only && !policy.streaming && policy.default_deadline);
}

#[test]
fn a_read_first_write_whose_deadline_passes_after_handoff_answers_its_write() {
    let limit = ReadLimit::new(Duration::from_millis(50), Default::default());
    let response = within_limit(&limit, true, false, "rows", None, || {
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

#[test]
fn deadline_hint_not_repeating_existing_bound() {
    let limit = ReadLimit::new(Duration::ZERO, Default::default());
    for query in [
        "sessions where since:7d",
        "sessions where until:today",
        "sessions where (since:7d repo:morrow)",
        "sessions where last_active >= 2026-10-07",
        "messages where at < 2026-10-07",
        "sessions where last_active between 2026-10-01 and 2026-10-07",
        "messages where at in last week",
    ] {
        polylogue_query::parse(query).unwrap();
        assert_eq!(narrower(query), None, "{query}");
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = stopped_read::<()>(
            &polylogue_store::StoreError::ReadDeadlineExceeded,
            "page",
            Some(query),
            &limit,
        )
        else {
            panic!("expected a deadline refusal");
        };
        assert_eq!(error.suggestion(), None, "{query}");
        assert_eq!(
            error.hint.as_deref(),
            Some("rerun with `--deadline none`, or narrow to one session"),
            "{query}"
        );
    }
}

#[test]
fn deadline_hint_uses_stopped_stage() {
    for stage in ["page", "similarity", "dev sql", "group"] {
        let limit = ReadLimit::new(Duration::ZERO, Default::default());
        limit.scope(|| ReadLimit::set_stage(stage));
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            envelope: command_surface::Envelope { notes, .. },
        } = stopped_read::<()>(
            &polylogue_store::StoreError::ReadDeadlineExceeded,
            "query",
            Some("sessions"),
            &limit,
        )
        else {
            panic!("expected a deadline refusal");
        };
        assert_eq!(
            notes[0],
            format!("stopped during {stage} after 0 SQLite steps; rows scanned are unknown")
        );
        let (suggestion, hint) = if matches!(stage, "page" | "similarity") {
            (
                Some(command_surface::Suggestion::Query(
                    "sessions where since:1d".into(),
                )),
                "restrict the query to a shorter time range or one session",
            )
        } else {
            (
                None,
                "rerun with `--deadline none`, or narrow to one session",
            )
        };
        assert_eq!(error.suggestion(), suggestion.as_ref(), "{stage}");
        assert_eq!(error.hint.as_deref(), Some(hint), "{stage}");
    }
}

#[test]
fn context_deadline_hint_names_the_context_form() {
    for stage in [None, Some("page"), Some("similarity")] {
        let limit = ReadLimit::new(Duration::ZERO, Default::default());
        if let Some(stage) = stage {
            limit.scope(|| ReadLimit::set_stage(stage));
        }
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = stopped_read::<()>(
            &polylogue_store::StoreError::ReadDeadlineExceeded,
            "context",
            Some("sessions where synthetic"),
            &limit,
        )
        else {
            panic!("expected a deadline refusal");
        };
        assert_eq!(error.suggestion(), None, "{stage:?}");
        assert_eq!(
            error.hint.as_deref(),
            Some("rerun with `--deadline none`, or narrow to one session"),
            "{stage:?}"
        );
    }
}
