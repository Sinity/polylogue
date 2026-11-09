use super::*;
use crate::advisory::tests::{CWD, fixture, operator, selected, statement};
use crate::test_daemon::TestDaemon;
use polylogue_model::{
    Reference, SessionId,
    assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink},
    authorship::{Authorship, Channel},
};
use serde_json::{Value, json};
use std::sync::atomic::Ordering;

fn call(w: &TestDaemon, id: &str, session: &str) -> Value {
    // This law tests persisted cooldown/history semantics, not warm latency.
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
    let limit = polylogue_store::read::limit::ReadLimit::new(
        std::time::Duration::from_secs(5),
        Default::default(),
    );
    let answer = limit.scope(|| {
        super::super::context(
            &w.daemon,
            id.into(),
            &ask,
            Some("mcp".into()),
            Default::default(),
            &operation,
        )
    });
    assert!(
        answer.failure().is_none(),
        "advisory request {id} for {session}: {answer:?}"
    );
    crate::test_answers::ok(answer)
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
        AssertionKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    // Warm the real selection without preparing a delivery or reserving cooldown.
    assert!(!selected(&w, "Inspect src/parser.rs", CWD).empty);
    let first = call(&w, "first-advisory", "codex:advisory-test");
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
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [NewAssertion {
                kind: AssertionKind::Note,
                key: None,
                value: Some(json!("synthetic barrier")),
                valid_until: None,
                links: vec![NewAssertionLink {
                    role: LinkRole::About,
                    reference: Reference::Repo("synthetic".into()),
                }],
            }]}),
        )
        .id("after-advisory"),
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
        call(&w, "different-prefix-recipient", "codex:advisory")["delivery"],
        "different-prefix-recipient",
        "a different exact identity is not a prefix alias of the original recipient"
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
        AssertionKind::Correction,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    assert!(!selected(&w, "Inspect src/parser.rs", CWD).empty);
    assert_eq!(
        call(&w, "after-cooldown", "codex:advisory-test")["delivery"],
        "after-cooldown"
    );
}
