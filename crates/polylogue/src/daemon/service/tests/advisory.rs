use super::*;
use crate::advisory::tests::{CWD, fixture, operator, selected, statement};
use crate::test_daemon::TestDaemon;
use polylogue_model::{
    Reference, SessionId,
    annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink},
    authorship::{Authorship, Channel},
};
use polylogue_store::read::limit::ReadLimit;
use serde_json::{Value, json};
use std::sync::atomic::Ordering;

fn call_with_limit(w: &TestDaemon, id: &str, session: &str, limit: &ReadLimit) -> Value {
    // This test checks the stored cooldown and history, not warm latency.
    // Enter the unchanged context phase with a finite fixture budget, leaving
    // the typed operation's real 20 ms cap and hook deadline tests separate.
    let operation = w
        .daemon
        .operation_context(&Authorship {
            author: polylogue_model::Author::Agent(None),
            via: Channel::Mcp,
        })
        .unwrap();
    let ask = Ask::Advisory {
        prompt: "Inspect src/parser.rs".into(),
        cwd: CWD.into(),
        session: SessionId::parse(session).unwrap(),
    };
    let answer = limit.scope(|| {
        super::super::context(
            &w.daemon,
            id.into(),
            &ask,
            Some(polylogue_model::DeliverySender::Mcp),
            Default::default(),
            &operation,
        )
    });
    assert!(
        answer.failure().is_none(),
        "advisory request {id} for {session}: {answer:?}"
    );
    serde_json::to_value(crate::test_answers::ok(answer)).unwrap()
}

fn call(w: &TestDaemon, id: &str, session: &str) -> Value {
    let limit = ReadLimit::new(std::time::Duration::from_secs(5), Default::default());
    call_with_limit(w, id, session, &limit)
}

fn positive(w: &TestDaemon, id: &str, session: &str) -> Value {
    let limit = ReadLimit::new(std::time::Duration::from_secs(5), Default::default());
    loop {
        limit
            .check()
            .expect("positive advisory fixture budget expired");
        let answer = call_with_limit(w, id, session, &limit);
        if answer["delivery"] == id {
            return answer;
        }
        assert_eq!(answer["delivery"], Value::Null);
        assert_eq!(answer["prepared"]["empty"], true);
        // A commit can still be changing the scope or handing over delivery
        // records. Wait on those existing locks before another positive attempt.
        w.daemon
            .scope
            .read(limit.deadline(), |_| ())
            .expect("fixture scope is loaded");
        limit
            .check()
            .expect("positive advisory fixture budget expired");
        std::thread::yield_now();
    }
}

#[test]
fn busy_scope_change_returns_empty_without_reserving_advice() {
    let (w, _) = fixture();
    statement(
        &w,
        "locked-file",
        AnnotationKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    w.daemon
        .scope
        .read(None, |_| {
            assert!(w.daemon.scope.try_read(|_| ()).is_none());
            let answer = call(&w, "busy-scope", "codex:advisory-test");
            assert_eq!(answer["delivery"], Value::Null);
            assert_eq!(answer["prepared"]["empty"], true);
        })
        .expect("fixture scope is loaded");
    assert!(
        w.daemon
            .reads
            .reader()
            .unwrap()
            .delivery("busy-scope")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        positive(&w, "after-scope-change", "codex:advisory-test")["delivery"],
        "after-scope-change"
    );
}

#[test]
fn advisory_cooldown_and_once_pair_history_have_distinct_lifetimes() {
    let (mut w, now) = fixture();
    assert!(
        w.daemon
            .reads
            .reader()
            .unwrap()
            .session(&SessionId::parse("codex:advisory-test").unwrap())
            .unwrap()
            .is_none(),
        "first-prompt recipients need not have provider capture yet"
    );
    statement(
        &w,
        "cooldown-file",
        AnnotationKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    // Warm the real selection without preparing a delivery or reserving cooldown.
    assert!(!selected(&w, "Inspect src/parser.rs", CWD).empty);
    let first = positive(&w, "first-advisory", "codex:advisory-test");
    assert_eq!(first["delivery"], "first-advisory");
    assert_eq!(
        first["prepared"]["text"].as_str().unwrap().lines().count(),
        1
    );
    assert!(
        first["prepared"]["text"]
            .as_str()
            .unwrap()
            .contains("first-advisory")
    );
    assert_eq!(
        call(&w, "second-advisory", "codex:advisory-test")["prepared"]["empty"],
        true
    );
    // A synchronous authored write follows the off-writer handoff and proves it committed.
    operator(
        &w,
        "after-advisory",
        polylogue_model::user::UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Note,
                key: None,
                value: Some(json!("synthetic barrier")),
                valid_until: None,
                links: vec![NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::Repo("synthetic".into()),
                }],
            }],
        },
    );
    let row = w
        .daemon
        .reads
        .reader()
        .unwrap()
        .delivery("first-advisory")
        .unwrap()
        .unwrap()
        .0;
    assert!(row.events.is_empty(), "preparation is not emission");
    assert_eq!(
        positive(&w, "different-prefix-recipient", "codex:advisory")["delivery"],
        "different-prefix-recipient",
        "a different identity is not a prefix alias of the original recipient"
    );
    w.restart();
    assert_eq!(
        call(&w, "after-restart", "codex:advisory-test")["prepared"]["empty"],
        true
    );
    now.fetch_add(600_001, Ordering::SeqCst);
    assert!(selected(&w, "Inspect src/parser.rs", CWD).empty);
    statement(
        &w,
        "different-file-decision",
        AnnotationKind::Correction,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    assert!(!selected(&w, "Inspect src/parser.rs", CWD).empty);
    assert_eq!(
        positive(&w, "after-cooldown", "codex:advisory-test")["delivery"],
        "after-cooldown"
    );
}

#[test]
fn an_advisory_store_failure_is_reported() {
    let (w, _) = fixture();
    let reference = statement(
        &w,
        "failing-store-file",
        AnnotationKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    let operation = w
        .daemon
        .operation_context(&Authorship {
            author: polylogue_model::Author::Agent(None),
            via: Channel::Mcp,
        })
        .unwrap();
    let ask = Ask::Advisory {
        prompt: "Inspect src/parser.rs".into(),
        cwd: CWD.into(),
        session: SessionId::parse("codex:advisory-store-failure").unwrap(),
    };
    polylogue_store::fault::arm(
        "read.annotation_validity",
        &reference.to_string(),
        polylogue_store::fault::Action::Fail,
        0,
        1,
    );
    let limit = ReadLimit::new(std::time::Duration::from_secs(5), Default::default());
    // A commit can still be changing the scope, which answers empty before
    // the candidate is read; ask again until the armed read is reached.
    let answer = loop {
        let answer = limit.scope(|| {
            super::super::context(
                &w.daemon,
                "advisory-store-failure".into(),
                &ask,
                Some(polylogue_model::DeliverySender::Mcp),
                Default::default(),
                &operation,
            )
        });
        if answer.failure().is_some() {
            break answer;
        }
        assert_eq!(
            serde_json::to_value(crate::test_answers::ok(answer)).unwrap()["prepared"]["empty"],
            true
        );
        w.daemon
            .scope
            .read(limit.deadline(), |_| ())
            .expect("fixture scope is loaded");
        limit.check().expect("the armed read was never reached");
        std::thread::yield_now();
    };
    let failure = answer.failure().expect("a store failure is not silence");
    assert_eq!(failure.code.as_deref(), Some("io"));
    assert!(
        failure
            .message
            .contains("injected fault at read.annotation_validity"),
        "{failure:?}"
    );
}
