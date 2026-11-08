use super::*;
use crate::daemon::{Request, service};
use crate::test_daemon::{self, TestDaemon};
use polylogue_model::{
    Author, RevisionReference,
    assertion::{AssertionKind, NewAssertion, NewAssertionLink},
    authorship::{Authorship, Channel},
};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicI64, Ordering};

const CWD: &str = "/synthetic/repo";
fn authored(w: &TestDaemon, request: Request, author: Author) -> Value {
    crate::test_answers::ok(service::handle_with_caller(
        request,
        &w.daemon,
        &crate::daemon::execution::limit(None, Default::default()),
        &Authorship {
            author,
            via: Channel::Cli,
        },
    ))
}
fn operator(w: &TestDaemon, request: Request) -> Value {
    authored(w, request, Author::Operator)
}
fn operation(w: &TestDaemon) -> Arc<crate::operation::Context> {
    w.daemon
        .operation_context(&Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        })
        .unwrap()
}
fn statement(
    w: &TestDaemon,
    id: &str,
    kind: AssertionKind,
    file: &str,
    scope: Option<&str>,
    judged: bool,
) -> RevisionReference {
    let mut links = vec![NewAssertionLink {
        role: LinkRole::About,
        reference: Reference::parse(file).unwrap(),
    }];
    if let Some(scope) = scope {
        links.push(NewAssertionLink {
            role: LinkRole::Scope,
            reference: Reference::parse(scope).unwrap(),
        });
    }
    authored(
        w,
        Request::Assert {
            request_id: id.into(),
            assertions: vec![NewAssertion {
                kind,
                key: Some(id.into()),
                value: Some(json!("ARCHIVED_BODY_NOT_INLINED")),
                valid_until: None,
                links,
            }],
        },
        Author::Agent(None),
    );
    let reader = w.daemon.reads.reader().unwrap();
    let reference = query::assertions::ids(
        &reader,
        &format!(
            "kind:{} key:{}",
            kind.as_str(),
            query::assertions::quoted(id)
        ),
        &operation(w),
    )
    .unwrap()
    .pop()
    .unwrap();
    drop(reader);
    if judged {
        judge(w, &reference, &format!("judge-{id}"));
    }
    reference
}
fn judge(w: &TestDaemon, reference: &RevisionReference, id: &str) {
    operator(
        w,
        Request::Assert {
            request_id: id.into(),
            assertions: vec![NewAssertion {
                kind: AssertionKind::Judgment,
                key: Some("accepted".into()),
                value: None,
                valid_until: None,
                links: vec![NewAssertionLink {
                    role: LinkRole::About,
                    reference: Reference::Assertion(reference.clone()),
                }],
            }],
        },
    );
}
fn selected(w: &TestDaemon, prompt: &str, cwd: &str) -> Prepared {
    let reader = w.daemon.reads.reader().unwrap();
    prepare(
        &reader,
        &w.daemon.scope,
        prompt,
        Path::new(cwd),
        &SessionId::parse("codex:advisory-test").unwrap(),
        "selected-advice",
        &[],
        &operation(w),
    )
    .unwrap()
}
fn fixture() -> (TestDaemon, Arc<AtomicI64>) {
    let now = Arc::new(AtomicI64::new(1_800_000_000_000));
    let clock = Arc::clone(&now);
    let daemon = test_daemon::start_with_clock(
        "synthetic advisory scope",
        polylogue_model::WallClock::new(move || {
            jiff::Timestamp::from_millisecond(clock.load(Ordering::SeqCst)).unwrap()
        }),
    );
    (daemon, now)
}

#[test]
fn advisory_selection_requires_a_current_operator_judgment_and_exact_scoped_file() {
    let (w, _) = fixture();
    let reference = statement(
        &w,
        "file-decision",
        AssertionKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        false,
    );
    assert!(selected(&w, "Inspect src/parser.rs", CWD).empty);
    judge(&w, &reference, "accept-file-decision");
    let prepared = selected(&w, "Inspect `src/parser.rs`", CWD);
    assert!(!prepared.empty);
    assert_eq!(prepared.text.lines().count(), 1);
    assert!(
        prepared
            .refs
            .contains(&Reference::Assertion(reference.clone()).to_string())
    );
    assert!(!prepared.text.contains("ARCHIVED_BODY_NOT_INLINED"));
    assert!(prepared.text.contains("polylogue read"));
    assert!(selected(&w, "Inspect src/parser.rs2", CWD).empty);
    assert!(!selected(&w, "Inspect ../src/parser.rs", "/synthetic/repo/subdir").empty);
    assert!(selected(&w, "Inspect src/parser.rs", "/synthetic/another").empty);
    assert!(selected(&w, "Continue the current work", CWD).empty);
    operator(
        &w,
        Request::Withdraw {
            request_id: "withdraw-file-decision".into(),
            targets: vec![Reference::Assertion(reference)],
            reason: None,
        },
    );
    assert!(selected(&w, "Inspect src/parser.rs", CWD).empty);
}

#[test]
fn unscoped_relative_files_are_not_inferred_global_and_absolute_files_match_exactly() {
    let (w, _) = fixture();
    statement(
        &w,
        "unscoped",
        AssertionKind::Correction,
        "file:src/parser.rs@fixture-commit",
        None,
        true,
    );
    assert!(selected(&w, "Inspect src/parser.rs", CWD).empty);
    statement(
        &w,
        "absolute",
        AssertionKind::Correction,
        "file:/synthetic/repo/src/parser.rs@fixture-commit",
        None,
        true,
    );
    assert!(!selected(&w, "Inspect ./src/../src/parser.rs", CWD).empty);
    assert!(selected(&w, "Inspect src/parser.rs", "/synthetic/another").empty);
}

fn call(w: &TestDaemon, id: &str, session: &str) -> Value {
    let table = crate::protocol::operations::table(Arc::clone(&w.daemon));
    let request = command_surface::Request {
        operation: "advisory".into(),
        arguments: json!({"prompt":"Inspect src/parser.rs", "cwd":CWD,"session":session}),
        request_id: id.into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let answer =
        runtime.block_on(table.call(request, command_surface::Via::Mcp, Default::default()));
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
        Request::Assert {
            request_id: "after-advisory".into(),
            assertions: vec![NewAssertion {
                kind: AssertionKind::Note,
                key: None,
                value: Some(json!("synthetic barrier")),
                valid_until: None,
                links: vec![NewAssertionLink {
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
        call(&w, "different-prefix-recipient", "codex:advisory")["delivery"],
        "different-prefix-recipient",
        "a different exact identity is not a prefix alias of the original recipient"
    );
    w.restart();
    assert_eq!(
        call(&w, "after-restart", "codex:advisory-test")["prepared"]["empty"],
        true
    );
    now.fetch_add(COOLDOWN_MS + 1, Ordering::SeqCst);
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

#[test]
fn advisory_bounds_intersect_parent_and_an_expired_request_prepares_nothing() {
    let (w, _) = fixture();
    let request = Request::Context {
        request_id: "expired-advisory".into(),
        ask: crate::briefing::Ask::Advisory {
            prompt: "Inspect src/parser.rs".into(),
            cwd: CWD.into(),
            session: SessionId::parse("codex:advisory-test").unwrap(),
        },
        sender: Some("hook".into()),
        budget_tokens: Default::default(),
    };
    let limit =
        polylogue_store::read::limit::ReadLimit::at(std::time::Instant::now(), Default::default());
    assert!(
        service::handle_with_limit(request, &w.daemon, &limit)
            .failure()
            .is_some()
    );
    assert!(
        w.daemon
            .reads
            .reader()
            .unwrap()
            .delivery("expired-advisory")
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_pending_exact_pair_is_suppressed_without_assuming_history_application() {
    let (w, _) = fixture();
    let reference = statement(
        &w,
        "pending-file",
        AssertionKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    let reference = Reference::Assertion(reference).to_string();
    let reader = w.daemon.reads.reader().unwrap();
    assert!(reader.delivery("unapplied-advice").unwrap().is_none());
    let pending = vec![vec![reference, "file:src/parser.rs@fixture-commit".into()]];
    let prepared = prepare(
        &reader,
        &w.daemon.scope,
        "Inspect src/parser.rs",
        Path::new(CWD),
        &SessionId::parse("codex:advisory-test").unwrap(),
        "unapplied-advice",
        &pending,
        &operation(&w),
    )
    .unwrap();
    assert!(prepared.empty);
}

#[test]
fn cooldown_is_per_session_and_expires_at_ten_minutes() {
    let cooldowns = Cooldowns::default();
    let first = SessionId::parse("codex:first").unwrap();
    let second = SessionId::parse("codex:second").unwrap();
    cooldowns
        .available(&first, 100)
        .unwrap()
        .insert(first.to_string(), 100);
    assert!(cooldowns.available(&first, 100 + COOLDOWN_MS - 1).is_none());
    assert!(
        cooldowns
            .available(&second, 100 + COOLDOWN_MS - 1)
            .is_some()
    );
    assert!(cooldowns.available(&first, 100 + COOLDOWN_MS).is_some());
}

#[test]
fn a_new_file_assertion_revision_requires_its_own_judgment_and_is_a_distinct_pair() {
    let (w, now) = fixture();
    let old = statement(
        &w,
        "revision-file",
        AssertionKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    let pending = vec![vec![
        Reference::Assertion(old.clone()).to_string(),
        "file:src/parser.rs@fixture-commit".into(),
    ]];
    now.fetch_add(1, Ordering::SeqCst);
    authored(
        &w,
        Request::Revise {
            request_id: "revise-file".into(),
            bases: vec![old],
            patch: polylogue_model::assertion::AssertionPatch {
                value: polylogue_model::assertion::Change::Set(json!(
                    "new exact authored decision"
                )),
                ..Default::default()
            },
        },
        Author::Agent(None),
    );
    assert!(
        selected(&w, "Inspect src/parser.rs", CWD).empty,
        "the predecessor's acceptance does not judge its successor"
    );
    let reader = w.daemon.reads.reader().unwrap();
    let revised = query::assertions::ids(
        &reader,
        "kind:decision key:revision-file current:true",
        &operation(&w),
    )
    .unwrap()
    .pop()
    .unwrap();
    drop(reader);
    now.fetch_add(1, Ordering::SeqCst);
    judge(&w, &revised, "judge-revised-file");
    let reader = w.daemon.reads.reader().unwrap();
    let prepared = prepare(
        &reader,
        &w.daemon.scope,
        "Inspect src/parser.rs",
        Path::new(CWD),
        &SessionId::parse("codex:advisory-test").unwrap(),
        "new-revision-advice",
        &pending,
        &operation(&w),
    )
    .unwrap();
    assert!(!prepared.empty);
    assert!(
        prepared
            .refs
            .contains(&Reference::Assertion(revised).to_string())
    );
    assert_eq!(prepared.text.lines().count(), 1);
}
