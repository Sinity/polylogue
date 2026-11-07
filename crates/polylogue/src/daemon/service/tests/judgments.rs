use super::*;
use polylogue_model::{Author, Reference};

fn ok(d: &test_daemon::TestDaemon, request: Request) -> serde_json::Value {
    ok_as(d, request, Author::Operator)
}

fn ok_as(d: &test_daemon::TestDaemon, request: Request, author: Author) -> serde_json::Value {
    crate::test_answers::ok(send_as(d, request, author))
}

fn send_as(
    d: &test_daemon::TestDaemon,
    request: Request,
    author: Author,
) -> command_surface::Answer<serde_json::Value> {
    let limit = crate::daemon::execution::limit(None, Default::default());
    handle_with_caller(
        request,
        &d.daemon,
        &limit,
        &polylogue_model::authorship::Authorship {
            author,
            via: polylogue_model::authorship::Channel::Cli,
        },
    )
}

fn assertion(targets: Vec<Reference>, body: &str) -> polylogue_model::assertion::NewAssertion {
    use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
    NewAssertion {
        kind: AssertionKind::Lesson,
        key: None,
        value: Some(serde_json::json!(body)),
        valid_until: None,
        links: targets
            .into_iter()
            .map(|reference| NewAssertionLink {
                role: LinkRole::About,
                reference,
            })
            .collect(),
    }
}

/// A query's whole answer; the caller names the outcome it expects.
fn query(d: &test_daemon::TestDaemon, text: &str) -> command_surface::Answer<serde_json::Value> {
    send_as(
        d,
        Request::Query {
            text: text.into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
        Author::Operator,
    )
}

#[test]
fn the_operator_accepts_or_rejects_an_agents_note() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        Request::Assert {
            request_id: "r1".into(),
            assertions: vec![assertion(vec![session.clone()], "use the fast path")],
        },
        Author::Agent(None),
    );
    let agent_id = format!("assertion:{}", agent["id"].as_str().unwrap());
    let own = ok(
        &d,
        Request::Assert {
            request_id: "r2".into(),
            assertions: vec![assertion(vec![session.clone()], "keep the operator lesson")],
        },
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where agent:true judged:unknown"))["count"],
        1
    );
    // The operator's own notes are revised or withdrawn, not judged.
    let refused = handle(
        Request::Judge {
            request_id: "r3".into(),
            note: format!("assertion:{}", own["id"].as_str().unwrap()),
            accepted: true,
            reason: None,
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("bad_request")),
        "{refused:?}"
    );
    ok(
        &d,
        Request::Judge {
            request_id: "r4".into(),
            note: agent_id.clone(),
            accepted: true,
            reason: Some("checked".into()),
        },
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where judged:accepted"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "assertions where agent:true judged:unknown"))["count"],
        0
    );
    // The brief carries the operator's assertions and the agent assertions
    // the operator accepted.
    let operation = d
        .daemon
        .operation_context(&polylogue_model::authorship::Authorship {
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        })
        .unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let brief = crate::briefing::prepare(
        &reader,
        &d.daemon.scope,
        None,
        &crate::briefing::Ask::SessionStart {
            cwd: "/nonexistent/dir".into(),
            session: Some(polylogue_model::SessionId::parse("claude-code:s1").unwrap()),
            source: crate::briefing::StartSource::Startup,
        },
        "d1",
        &operation,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(brief.text.contains("use the fast path"), "{}", brief.text);
    assert!(
        brief.text.contains("keep the operator lesson"),
        "{}",
        brief.text
    );
    assert!(!brief.text.contains("Assertions held"), "{}", brief.text);
    drop(reader);
    // A later rejection replaces the acceptance.
    ok(
        &d,
        Request::Judge {
            request_id: "r5".into(),
            note: agent_id,
            accepted: false,
            reason: None,
        },
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where judged:rejected"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "assertions where judged:accepted"))["count"],
        0
    );
}

#[test]
fn a_note_applies_to_every_row_of_a_query() {
    let d = test_daemon::start("first prompt");
    let intended = Request::AssertQuery {
        request_id: "r1".into(),
        query: "messages".into(),
        assertion: polylogue_model::assertion::NewAssertion {
            kind: polylogue_model::assertion::AssertionKind::Tag,
            key: Some("reviewed".into()),
            value: None,
            valid_until: None,
            links: vec![],
        },
    };
    let done = ok(&d, intended.clone());
    assert_eq!(done["targets"], 1, "{done}");
    let repeated = ok(&d, intended);
    assert_eq!(repeated["repeat"], true);
    assert_eq!(repeated["targets"], 1);
    let stored = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .user_request("r1")
        .unwrap()
        .unwrap();
    let UserWrite::AssertQuery { resolved, .. } = stored.write else {
        panic!("the actual query and selection must remain in the frozen request");
    };
    assert_eq!(resolved.len(), 1);
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where kind:tag key:reviewed"))["count"],
        1
    );
    let refused = handle(
        Request::AssertQuery {
            request_id: "r2".into(),
            query: "sessions where origin:codex".into(),
            assertion: assertion(vec![], "x"),
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("empty")),
        "{refused:?}"
    );
    let refused = handle(
        Request::AssertQuery {
            request_id: "r3".into(),
            query: "sessions | agg count".into(),
            assertion: assertion(vec![], "x"),
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("bad_request")),
        "{refused:?}"
    );
}

/// Frozen history keeps the original authors of the claim and its judgment.
#[test]
fn frozen_history_preserves_the_note_and_judgment_authors() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        Request::Assert {
            request_id: "r1".into(),
            assertions: vec![assertion(vec![session], "use the fast path")],
        },
        Author::Agent(None),
    );
    ok(
        &d,
        Request::Judge {
            request_id: "r2".into(),
            note: format!("assertion:{}", agent["id"].as_str().unwrap()),
            accepted: true,
            reason: None,
        },
    );
    let reader = d.daemon.reads.reader().unwrap();
    let mut requests = Vec::new();
    reader
        .visit_authored(&mut |prepared| {
            if matches!(prepared.request.id.as_str(), "r1" | "r2") {
                requests.push(prepared.request.clone());
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(
        requests
            .iter()
            .map(|request| request.id.as_str())
            .collect::<Vec<_>>(),
        ["r1", "r2"]
    );
    assert_eq!(requests[0].authorship.author, Author::Agent(None));
    assert_eq!(requests[1].authorship.author, Author::Operator);
}

#[test]
fn a_rejudged_revision_reads_its_latest_operator_judgment_everywhere() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        Request::Assert {
            request_id: "lesson".into(),
            assertions: vec![assertion(vec![session], "prefer the short path")],
        },
        Author::Agent(None),
    );
    let target = format!("assertion:{}", agent["id"].as_str().unwrap());
    for (request_id, accepted) in [("j1", true), ("j2", false), ("j3", true)] {
        ok(
            &d,
            Request::Judge {
                request_id: request_id.into(),
                note: target.clone(),
                accepted,
                reason: None,
            },
        );
    }
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where judged:accepted"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "assertions where judged:rejected"))["count"],
        0
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "assertions where agent:true judged:unknown"))["count"],
        0
    );
    let grouped = crate::test_answers::ok(query(
        &d,
        "assertions where agent:true | group by judged | count",
    ));
    assert_eq!(grouped["rows"][0]["judged"], "accepted", "{grouped}");
    assert_eq!(grouped["rows"][0]["count"], 1, "{grouped}");
}

#[test]
fn changes_select_assertions_by_their_kind() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    ok(
        &d,
        Request::Assert {
            request_id: "lesson".into(),
            assertions: vec![assertion(vec![session], "a lesson")],
        },
    );
    let lessons = crate::test_answers::ok(query(&d, "changes where assertion_kind:lesson"));
    assert_eq!(lessons["count"], 1, "{lessons}");
    assert_eq!(lessons["rows"][0]["kind"], "assertion", "{lessons}");
    assert_eq!(lessons["rows"][0]["assertion_kind"], "lesson", "{lessons}");
    assert_eq!(
        crate::test_answers::empty(query(&d, "changes where assertion_kind:tag"))["count"],
        0
    );
}

#[test]
fn a_structured_value_matches_by_its_text() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    ok(
        &d,
        Request::Assert {
            request_id: "lesson".into(),
            assertions: vec![assertion(vec![session], "use the fast path")],
        },
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where value:\"use the fast path\""))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where value:\"the fast\""))["count"],
        1
    );
}
