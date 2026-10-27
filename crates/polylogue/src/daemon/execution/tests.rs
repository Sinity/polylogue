use super::*;
use crate::daemon::service::handle_with_limit;
use crate::test_daemon;

pub(crate) const SLOW: &str = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n";

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
    let Response::Error {
        kind,
        detail: Some(detail),
        ..
    } = response
    else {
        panic!("expected a deadline refusal");
    };
    assert_eq!(kind, "deadline_exceeded");
    assert_eq!(detail["stage"], "sql");
    assert!(detail["count"].is_null());
    assert!(detail["rows_scanned"].is_null());
    assert!(detail["sqlite_steps"].as_u64().unwrap() > 0);
    assert_eq!(exit_code(&kind), 7);
    assert_eq!(exit_code("cancelled"), 3);
}

#[test]
fn expiry_at_completion_never_returns_a_partial_success() {
    let request = Request::Status;
    let limit = ReadLimit::new(Duration::from_millis(10), Default::default());
    let response = run(request, &limit, |_| {
        std::thread::sleep(Duration::from_millis(20));
        super::super::service::ok(serde_json::json!({"count": 123}))
    });
    assert!(matches!(response, Response::Error { kind, .. } if kind == "deadline_exceeded"));
}

#[test]
fn expired_requests_never_start_any_read_operation() {
    let daemon = test_daemon::start("synthetic limits");
    let requests = [
        Request::Status,
        Request::Read {
            target: crate::daemon::ReadTarget::Session(
                polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            ),
            view: "full".into(),
            format: "json".into(),
            with_parents: false,
            budget: None,
            cursor: None,
            start: None,
        },
        Request::Context {
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
            word: "repo:".into(),
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
            matches!(handle_with_limit(request, &daemon.daemon, &limit), Response::Error { kind, .. } if kind == "deadline_exceeded")
        );
    }
}
