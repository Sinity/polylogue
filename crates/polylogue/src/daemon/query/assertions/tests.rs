use super::*;
use crate::test_daemon;
use polylogue_model::Timestamp;
use polylogue_model::assertion::{
    AssertionKind, AssertionPatch, Change, LinkRole, NewAssertion, NewAssertionLink,
};
use polylogue_model::{Author, SessionId};

fn add(w: &test_daemon::TestDaemon, id: &str, authored: (Author, NewAssertion)) {
    let invocation = w
        .daemon
        .invocation(&polylogue_model::authorship::Authorship {
            author: authored.0,
            via: polylogue_model::authorship::Channel::Cli,
        });
    crate::test_answers::ok(crate::daemon::service::write(
        &w.daemon,
        &invocation,
        id.into(),
        polylogue_model::user::UserWrite::Assert {
            assertions: vec![authored.1],
        },
    ));
}

fn operator(w: &test_daemon::TestDaemon, id: &str, write: polylogue_model::user::UserWrite) {
    let invocation = w
        .daemon
        .invocation(&polylogue_model::authorship::Authorship {
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        });
    crate::test_answers::ok(crate::daemon::service::write(
        &w.daemon,
        &invocation,
        id.into(),
        write,
    ));
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
    operator(
        &w,
        "018f0000-0000-7000-8000-000000000002",
        polylogue_model::user::UserWrite::ReviseAssertions {
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

    operator(
        &w,
        "018f0000-0000-7000-8000-000000000005",
        polylogue_model::user::UserWrite::WithdrawAssertions {
            targets: vec![Reference::parse(&format!("assertion:{first}")).unwrap()],
            reason: Some("superseded".into()),
        },
    );
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
