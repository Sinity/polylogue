use super::*;
use crate::daemon::Request;
use crate::test_daemon;
use polylogue_model::Timestamp;
use polylogue_model::assertion::{
    AssertionKind, AssertionPatch, Change, LinkRole, NewAssertion, NewAssertionLink,
};
use polylogue_model::{Author, SessionId};

fn add(w: &test_daemon::TestDaemon, id: &str, authored: (Author, NewAssertion)) {
    let limit = crate::daemon::execution::limit(None, Default::default());
    let response = crate::daemon::service::handle_with_caller(
        Request::Assert {
            request_id: id.into(),
            assertions: vec![authored.1],
        },
        &w.daemon,
        &limit,
        &polylogue_model::authorship::Authorship {
            author: authored.0,
            via: polylogue_model::authorship::Channel::Cli,
        },
    );
    crate::test_answers::ok(response);
}

fn operator(
    w: &test_daemon::TestDaemon,
    request: Request,
) -> command_surface::Answer<serde_json::Value> {
    crate::daemon::service::handle_with_caller(
        request,
        &w.daemon,
        &crate::daemon::execution::limit(None, Default::default()),
        &polylogue_model::authorship::Authorship {
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
    )
}

fn note(author: Author, target: &str, value: &str) -> (Author, NewAssertion) {
    (
        author,
        NewAssertion {
            kind: AssertionKind::Lesson,
            key: None,
            value: Some(serde_json::json!(value)),
            links: vec![NewAssertionLink {
                role: LinkRole::About,
                reference: Reference::parse(target).unwrap(),
            }],
            valid_until: None,
        },
    )
}

fn values(notes: &[Assertion]) -> Vec<&str> {
    notes.iter().filter_map(|n| n.fact.text()).collect()
}

#[test]
fn notes_are_listed_by_their_current_revision_and_withdrawals_leave_them_out() {
    let w = test_daemon::start("synthetic");
    let now = Timestamp::from_unix_ms(4_000_000_000_000).unwrap();
    let captured = w
        .daemon
        .operation_context(&polylogue_model::authorship::Authorship {
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        })
        .unwrap();
    let context = std::sync::Arc::new(
        captured
            .replay_clock(query_language::time::Clock::new(
                now.into(),
                captured.clock.day.clone(),
            ))
            .unwrap(),
    );
    let first = "018f0000-0000-7000-8000-000000000001";
    add(
        &w,
        first,
        note(Author::Operator, "repo:morrow", "run nextest"),
    );
    let revised = operator(
        &w,
        Request::Revise {
            request_id: "018f0000-0000-7000-8000-000000000002".into(),
            bases: vec![polylogue_model::RevisionReference::exact(
                first.into(),
                first.into(),
            )],
            patch: AssertionPatch {
                value: Change::Set(serde_json::json!("run cargo nextest")),
                ..Default::default()
            },
        },
    );
    crate::test_answers::ok(revised);
    let agent = Author::Agent(Some(SessionId::parse("codex:x").unwrap()));
    add(
        &w,
        "018f0000-0000-7000-8000-000000000003",
        note(agent, "repo:morrow", "an agent's idea"),
    );
    add(
        &w,
        "018f0000-0000-7000-8000-000000000004",
        note(Author::Operator, "repo:morrowland", "another repository"),
    );
    let reader = w.daemon.reads.reader().unwrap();
    let about = about(&[Reference::parse("repo:morrow").unwrap()]);
    assert_eq!(
        values(&list(&reader, &format!("{about} agent:false"), &context).unwrap()),
        ["run cargo nextest"],
        "the current revision only, and a target matches whole"
    );
    assert_eq!(
        count(&reader, &format!("{about} agent:true"), &context).unwrap(),
        1
    );
    drop(reader);

    let withdrawn = operator(
        &w,
        Request::Withdraw {
            request_id: "018f0000-0000-7000-8000-000000000005".into(),
            targets: vec![Reference::Assertion(polylogue_model::RevisionReference {
                root: first.into(),
                revision: None,
            })],
            reason: Some("superseded".into()),
        },
    );
    crate::test_answers::ok(withdrawn);
    let reader = w.daemon.reads.reader().unwrap();
    assert_eq!(
        values(&list(&reader, &about, &context).unwrap()),
        ["an agent's idea"],
        "a withdrawal takes the whole family out, and is not listed itself"
    );
    assert_eq!(
        values(
            &list(
                &reader,
                &format!(
                    "{about} withdrawn:true assertion:{first}@018f0000-0000-7000-8000-000000000002"
                ),
                &context
            )
            .unwrap()
        ),
        ["run cargo nextest"]
    );
}

#[test]
fn a_quoted_value_keeps_quotes_and_backslashes() {
    assert_eq!(quoted(r#"cwd:/a "b"\c"#), r#""cwd:/a \"b\"\\c""#);
}
