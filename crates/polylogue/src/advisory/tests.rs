use super::*;
use crate::test_daemon::{self, TestDaemon};
use polylogue_model::{
    Author, RevisionReference,
    annotation::{AnnotationKind, NewAnnotation, NewAnnotationLink},
    authorship::{Authorship, Channel},
};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicI64, Ordering};

pub(crate) const CWD: &str = "/synthetic/repo";
fn authored(
    w: &TestDaemon,
    id: &str,
    write: polylogue_model::user::UserWrite,
    author: Author,
) -> Value {
    let invocation = w.daemon.invocation(&Authorship {
        author,
        via: Channel::Cli,
    });
    serde_json::to_value(crate::test_answers::ok(crate::daemon::service::write(
        &w.daemon,
        &invocation,
        id.into(),
        write,
    )))
    .unwrap()
}
pub(crate) fn operator(w: &TestDaemon, id: &str, write: polylogue_model::user::UserWrite) -> Value {
    authored(w, id, write, Author::Operator)
}
fn operation(w: &TestDaemon) -> Arc<polylogue_reading::operation::Context> {
    w.daemon
        .operation_context(&Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        })
        .unwrap()
}
pub(crate) fn statement(
    w: &TestDaemon,
    id: &str,
    kind: AnnotationKind,
    file: &str,
    scope: Option<&str>,
    judged: bool,
) -> RevisionReference {
    let mut links = vec![NewAnnotationLink {
        role: LinkRole::About,
        reference: Reference::parse(file).unwrap(),
    }];
    if let Some(scope) = scope {
        links.push(NewAnnotationLink {
            role: LinkRole::Scope,
            reference: Reference::parse(scope).unwrap(),
        });
    }
    authored(
        w,
        id,
        polylogue_model::user::UserWrite::Assert {
            annotations: vec![NewAnnotation {
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
    let reference = query::annotations::ids(
        &reader,
        &format!(
            "kind:{} key:{}",
            kind.as_str(),
            query::annotations::quoted(id)
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
        id,
        polylogue_model::user::UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Judgment,
                key: Some("accepted".into()),
                value: None,
                valid_until: None,
                links: vec![NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::Annotation(reference.clone()),
                }],
            }],
        },
    );
}
/// A scope index at `reader`'s snapshot that only the test holds.
/// Selection reads the scope index without waiting for its lock, and the
/// daemon's scope worker takes that lock on its own thread after every
/// commit, so selecting from the daemon's index could find it held and
/// answer empty.
fn scope_at(reader: &Reader<'_>) -> SharedScopeIndex {
    let scope = SharedScopeIndex::default();
    scope.refresh(reader).unwrap();
    scope
}
pub(crate) fn selected(w: &TestDaemon, prompt: &str, cwd: &str) -> Prepared {
    let reader = w.daemon.reads.reader().unwrap();
    prepare(
        &reader,
        &scope_at(&reader),
        prompt,
        Path::new(cwd),
        &SessionId::parse("codex:advisory-test").unwrap(),
        "selected-advice",
        &[],
        &operation(w),
    )
    .unwrap()
}
pub(crate) fn fixture() -> (TestDaemon, Arc<AtomicI64>) {
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
fn advisory_selection_requires_a_current_operator_judgment_and_scoped_file() {
    let (w, _) = fixture();
    let reference = statement(
        &w,
        "file-decision",
        AnnotationKind::Decision,
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
            .contains(&Reference::Annotation(reference.clone()).to_string())
    );
    assert!(!prepared.text.contains("ARCHIVED_BODY_NOT_INLINED"));
    assert!(
        prepared
            .text
            .contains(&format!("polylogue annotation:{reference} read"))
    );
    assert!(selected(&w, "Inspect src/parser.rs2", CWD).empty);
    assert!(!selected(&w, "Inspect ../src/parser.rs", "/synthetic/repo/subdir").empty);
    assert!(selected(&w, "Inspect src/parser.rs", "/synthetic/another").empty);
    assert!(selected(&w, "Continue the current work", CWD).empty);
    operator(
        &w,
        "withdraw-file-decision",
        polylogue_model::user::UserWrite::WithdrawAnnotations {
            targets: vec![Reference::Annotation(reference)],
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
        AnnotationKind::Correction,
        "file:src/parser.rs@fixture-commit",
        None,
        true,
    );
    assert!(selected(&w, "Inspect src/parser.rs", CWD).empty);
    statement(
        &w,
        "absolute",
        AnnotationKind::Correction,
        "file:/synthetic/repo/src/parser.rs@fixture-commit",
        None,
        true,
    );
    assert!(!selected(&w, "Inspect ./src/../src/parser.rs", CWD).empty);
    assert!(selected(&w, "Inspect src/parser.rs", "/synthetic/another").empty);
    assert!(selected(&w, "Inspect /synthetic/repo/src/parser.rs2", CWD).empty);
    assert!(
        !selected(
            &w,
            "Inspect /synthetic/repo/src/parser.rs",
            "/synthetic/another"
        )
        .empty
    );
}

#[test]
fn an_expired_request_prepares_nothing_where_a_live_one_prepares_the_judged_file() {
    let (w, _) = fixture();
    let judged = statement(
        &w,
        "expiry-file",
        AnnotationKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    let advisory = |id: &str| {
        crate::test_daemon::Call::new("internal advisory", serde_json::json!({"prompt": "Inspect src/parser.rs", "cwd": CWD, "session": SessionId::parse("codex:advisory-test").unwrap()})).id(id).channel(command_surface::Via::Hook)
    };
    let live = polylogue_store::read::limit::ReadLimit::new(
        std::time::Duration::from_secs(60),
        Default::default(),
    );
    let answer = advisory("live-advisory").within(&live).answer(&w.daemon);
    assert!(answer.failure().is_none(), "{answer:?}");
    let value = crate::test_answers::ok(answer);
    assert_eq!(
        value["prepared"]["refs"],
        serde_json::json!([
            Reference::Annotation(judged).to_string(),
            "file:src/parser.rs@fixture-commit"
        ])
    );
    let request = advisory("expired-advisory");
    let limit =
        polylogue_store::read::limit::ReadLimit::at(std::time::Instant::now(), Default::default());
    // docs/agents.md, Advisories: "preparation is capped at 20 ms, intersected
    // with the caller's bound, or the hook says nothing". Past the bound the
    // answer is the empty advisory on every surface, never a timeout.
    let answer = request.within(&limit.clone()).answer(&w.daemon);
    assert!(answer.failure().is_none(), "{answer:?}");
    let value = crate::test_answers::ok(answer);
    assert_eq!(value["delivery"], serde_json::Value::Null);
    assert_eq!(value["prepared"]["text"], "");
    assert_eq!(value["prepared"]["refs"], serde_json::json!([]));
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
fn pending_pair_is_suppressed_without_assuming_history_application() {
    let (w, _) = fixture();
    let reference = statement(
        &w,
        "pending-file",
        AnnotationKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    let reference = Reference::Annotation(reference).to_string();
    let reader = w.daemon.reads.reader().unwrap();
    assert!(reader.delivery("unapplied-advice").unwrap().is_none());
    let pending = vec![vec![reference, "file:src/parser.rs@fixture-commit".into()]];
    let prepared = prepare(
        &reader,
        &scope_at(&reader),
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
    assert!(cooldowns.available(&first, 100 + 600_000 - 1).is_none());
    assert!(cooldowns.available(&second, 100 + 600_000 - 1).is_some());
    assert!(cooldowns.available(&first, 100 + 600_000).is_some());
}

#[test]
fn a_new_file_annotation_revision_requires_its_own_judgment_and_is_a_distinct_pair() {
    let (w, now) = fixture();
    let old = statement(
        &w,
        "revision-file",
        AnnotationKind::Decision,
        "file:src/parser.rs@fixture-commit",
        Some("cwd:/synthetic/repo"),
        true,
    );
    let pending = vec![vec![
        Reference::Annotation(old.clone()).to_string(),
        "file:src/parser.rs@fixture-commit".into(),
    ]];
    now.fetch_add(1, Ordering::SeqCst);
    authored(
        &w,
        "revise-file",
        polylogue_model::user::UserWrite::ReviseAnnotations {
            bases: vec![old.clone()],
            patch: polylogue_model::annotation::AnnotationPatch {
                value: polylogue_model::annotation::Change::Set(json!("new authored decision")),
                ..Default::default()
            },
        },
        Author::Agent(None),
    );
    assert!(
        selected(&w, "Inspect src/parser.rs", CWD).empty,
        "accepting the previous revision does not judge the next one"
    );
    let reader = w.daemon.reads.reader().unwrap();
    let revised = query::annotations::ids(
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
        &scope_at(&reader),
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
            .contains(&Reference::Annotation(revised).to_string())
    );
    assert_eq!(prepared.text.lines().count(), 1);
}
