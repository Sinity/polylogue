use super::*;
use crate::daemon::service::handle_with_limit;
use crate::test_daemon;

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
    let response = handle_with_limit(
        Request::Sql {
            statement: SLOW.into(),
        },
        &daemon.daemon,
        &limit,
    );
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
    let request = Request::Status;
    let limit = ReadLimit::new(Duration::from_millis(10), Default::default());
    let response = run(request, &limit, |_| {
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
        Request::Status,
        Request::Read {
            target: "session:claude-code:s1".into(),
            options: polylogue_model::read::ReadOptions::default(),
            delivery: polylogue_model::read::ReadDelivery::Page,
            format: "json".into(),
            budget: None,
            with_blobs: false,
            cursor: Some("never-open-an-expired-request".into()),
        },
        Request::Read {
            target: "session:claude-code:s1".into(),
            options: polylogue_model::read::ReadOptions::default(),
            delivery: polylogue_model::read::ReadDelivery::Page,
            format: "json".into(),
            budget: None,
            with_blobs: false,
            cursor: None,
        },
        Request::Context {
            budget_tokens: Default::default(),
            request_id: "synthetic-recall".into(),
            ask: crate::briefing::Ask::Recall {
                text: "synthetic".into(),
                cwd: None,
                for_session: None,
                cursor: None,
            },
            sender: None,
        },
        Request::Query {
            text: "sessions | measure usage".into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
        Request::Complete {
            text: "repo:".into(),
            position: 5,
        },
        Request::Query {
            text: "sessions".into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
        Request::Sql {
            statement: "SELECT 1".into(),
        },
    ];
    for request in requests {
        let limit = ReadLimit::new(Duration::ZERO, Default::default());
        assert!(
            matches!(handle_with_limit(request, &daemon.daemon, &limit), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("deadline_exceeded"))
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
        Request::Keep {
            request_id: "synthetic-keep".into(),
            text: "sessions".into(),
            name: None,
        },
        Request::Embed {
            what: "build".into(),
            query: None,
            limit: None,
            yes: false,
        },
    ] {
        assert!(request_deadline(&request, Some(0)).is_none());
        let response = run(request, &limit, |_| {
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
    let request = Request::Read {
        target: "sessions".into(),
        options: polylogue_model::read::ReadOptions::default(),
        delivery: polylogue_model::read::ReadDelivery::Stream,
        format: "json".into(),
        budget: None,
        with_blobs: false,
        cursor: None,
    };
    let limit = limit_for(&request, None, Default::default());
    assert!(limit.check().is_ok());
    assert!(request_deadline(&request, None).is_none());
    assert_eq!(request_deadline(&request, Some(0)), Some(0));
    let response = run(
        request,
        &limit_for(&Request::Status, Some(0), Default::default()),
        |_| panic!("expired streamed read started"),
    );
    assert!(
        matches!(&response, command_surface::Answer { outcome: command_surface::Outcome::Error(error), envelope: command_surface::Envelope { notes, .. }, .. } if error.code.as_deref() == Some("deadline_exceeded") && notes[0].starts_with("stopped during read "))
    );
}

#[test]
fn other_read_requests_use_their_operation_name_in_refusals() {
    let request = Request::BackfillJobs;
    let limit = ReadLimit::new(Duration::ZERO, Default::default());
    let response = run(request, &limit, |_| panic!("expired request started"));
    assert!(
        matches!(&response, command_surface::Answer { outcome: command_surface::Outcome::Error(_), envelope: command_surface::Envelope { notes, .. }, .. } if notes[0].starts_with("stopped during backfill_jobs "))
    );
}

#[test]
fn context_receipts_do_not_disable_the_read_phase_deadline() {
    let request = Request::Context {
        budget_tokens: Default::default(),
        request_id: "synthetic-context".into(),
        ask: crate::briefing::Ask::Takeover {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            for_session: None,
        },
        sender: Some("mcp".into()),
    };
    assert!(!request.read_only());
    assert!(request.has_read_phase());
    assert!(request_deadline(&request, None).is_some());
    assert_eq!(request_deadline(&request, Some(0)), Some(0));
    let response = run(
        request,
        &ReadLimit::new(Duration::ZERO, Default::default()),
        |_| panic!("expired context started"),
    );
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
    let response = handle_with_limit(
        Request::Keep {
            request_id: id.clone(),
            text: "sessions".into(),
            name: None,
        },
        &daemon.daemon,
        &limit,
    );
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
    let request = Request::Read {
        target: "session:claude-code:s1".into(),
        options: polylogue_model::read::ReadOptions::default(),
        delivery: polylogue_model::read::ReadDelivery::Page,
        format: "json".into(),
        budget: None,
        with_blobs: false,
        cursor: Some("opaque-awaiting-owner-validation".into()),
    };
    assert!(request.read_only());
    assert!(!request.streaming());
    assert!(request_deadline(&request, None).is_some());
    assert_eq!(request_deadline(&request, Some(0)), Some(0));
}
