use super::*;
use polylogue_model::{Author, Reference};

fn ok(d: &test_daemon::TestDaemon, request: crate::test_daemon::Call) -> serde_json::Value {
    ok_as(d, request, Author::Operator)
}

fn ok_as(
    d: &test_daemon::TestDaemon,
    request: crate::test_daemon::Call,
    author: Author,
) -> serde_json::Value {
    crate::test_answers::ok(send_as(d, request, author))
}

fn send_as(
    d: &test_daemon::TestDaemon,
    request: crate::test_daemon::Call,
    author: Author,
) -> command_surface::Answer<serde_json::Value> {
    let limit = crate::daemon::execution::limit(None, Default::default());
    (request)
        .limit(limit.clone())
        .authorship(
            polylogue_model::authorship::Authorship {
                author,
                via: polylogue_model::authorship::Channel::Cli,
            }
            .clone(),
        )
        .answer(&d.daemon)
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
        crate::test_daemon::Call::new("query", serde_json::json!({"query": text})),
        Author::Operator,
    )
}

#[test]
fn the_operator_accepts_or_rejects_an_agents_note() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        crate::test_daemon::Call::new("assert", serde_json::json!({"assertions": [assertion(vec![session.clone()], "use the fast path")]})).id("r1"),
        Author::Agent(None),
    );
    let agent_id = format!("assertion:{}", agent["id"].as_str().unwrap());
    let own = ok(
        &d,
        crate::test_daemon::Call::new("assert", serde_json::json!({"assertions": [assertion(vec![session.clone()], "keep the operator lesson")]})).id("r2"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where agent:true judged:unknown"))["count"],
        1
    );
    // The operator's own notes are revised or withdrawn, not judged.
    let refused = crate::test_daemon::Call::new("judge", serde_json::json!({"note": format!("assertion:{}", own["id"].as_str().unwrap()), "accepted": true})).id("r3").answer(&d.daemon);
    assert!(
        matches!(refused, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("bad_request")),
        "{refused:?}"
    );
    ok(
        &d,
        crate::test_daemon::Call::new("judge", serde_json::json!({"note": agent_id.clone(), "accepted": true, "reason": Some("checked")})).id("r4"),
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
        &[],
        &crate::briefing::Ask::SessionStart {
            cwd: "/nonexistent/dir".into(),
            session: Some(polylogue_model::SessionId::parse("claude-code:s1").unwrap()),
            source: crate::briefing::StartSource::Startup,
        },
        "d1",
        &operation,
        None,
        None,
        Default::default(),
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
        crate::test_daemon::Call::new(
            "judge",
            serde_json::json!({"note": agent_id, "accepted": false}),
        )
        .id("r5"),
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
    let intended = crate::test_daemon::Call::new("assert-query", serde_json::json!({"query": "messages", "assertion": polylogue_model::assertion::NewAssertion {
            kind: polylogue_model::assertion::AssertionKind::Tag,
            key: Some("reviewed".into()),
            value: None,
            valid_until: None,
            links: vec![],
        }})).id("r1");
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
    let refused = crate::test_daemon::Call::new("assert-query", serde_json::json!({"query": "sessions where origin:codex", "assertion": assertion(vec![], "x")})).id("r2").answer(&d.daemon);
    assert!(
        matches!(refused, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("empty")),
        "{refused:?}"
    );
    let refused = crate::test_daemon::Call::new(
        "assert-query",
        serde_json::json!({"query": "sessions | agg count", "assertion": assertion(vec![], "x")}),
    )
    .id("r3")
    .answer(&d.daemon);
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
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [assertion(vec![session], "use the fast path")]}),
        )
        .id("r1"),
        Author::Agent(None),
    );
    ok(
        &d,
        crate::test_daemon::Call::new("judge", serde_json::json!({"note": format!("assertion:{}", agent["id"].as_str().unwrap()), "accepted": true})).id("r2"),
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
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [assertion(vec![session], "prefer the short path")]}),
        )
        .id("lesson"),
        Author::Agent(None),
    );
    let target = format!("assertion:{}", agent["id"].as_str().unwrap());
    for (request_id, accepted) in [("j1", true), ("j2", false), ("j3", true)] {
        ok(
            &d,
            crate::test_daemon::Call::new(
                "judge",
                serde_json::json!({"note": target.clone(), "accepted": accepted}),
            )
            .id(request_id),
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
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [assertion(vec![session], "a lesson")]}),
        )
        .id("lesson"),
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
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [assertion(vec![session], "use the fast path")]}),
        )
        .id("lesson"),
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

#[test]
fn second_judgment_by_same_author_revises_the_first_regardless_of_clock() {
    use std::sync::atomic::{AtomicI64, Ordering};
    // The writer's clock steps back between the two judgments.
    let now = std::sync::Arc::new(AtomicI64::new(4_000_000_000_000));
    let clock = {
        let now = now.clone();
        polylogue_model::WallClock::new(move || {
            jiff::Timestamp::from_millisecond(now.load(Ordering::SeqCst)).unwrap()
        })
    };
    let d = test_daemon::start_with_clock("first prompt", clock);
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [assertion(vec![session], "prefer the short path")]}),
        )
        .id("lesson"),
        Author::Agent(None),
    );
    let target = format!("assertion:{}", agent["id"].as_str().unwrap());
    ok(
        &d,
        crate::test_daemon::Call::new(
            "judge",
            serde_json::json!({"note": target.clone(), "accepted": true}),
        )
        .id("j1"),
    );
    now.store(3_000_000_000_000, Ordering::SeqCst);
    ok(
        &d,
        crate::test_daemon::Call::new(
            "judge",
            serde_json::json!({"note": target.clone(), "accepted": false}),
        )
        .id("j2"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "assertions where judged:rejected"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "assertions where judged:accepted"))["count"],
        0
    );
    // One current judgment: the second revised the first.
    let current = crate::test_answers::ok(query(&d, "assertions where kind:judgment"));
    assert_eq!(current["count"], 1, "{current}");
    let revisions = crate::test_answers::ok(query(
        &d,
        "assertions where kind:judgment current:* | fields id, root, current | sort by id",
    ));
    assert_eq!(revisions["rows"][0]["root"], "j1", "{revisions}");
    assert_eq!(revisions["rows"][1]["root"], "j1", "{revisions}");
    assert_eq!(revisions["rows"][1]["id"], "j2", "{revisions}");
}

#[test]
fn revising_a_withdrawal_is_refused() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    ok(
        &d,
        crate::test_daemon::Call::new(
            "assert",
            serde_json::json!({"assertions": [assertion(vec![session], "keep the old path")]}),
        )
        .id("note"),
    );
    ok(
        &d,
        crate::test_daemon::Call::new(
            "withdraw",
            serde_json::json!({"targets": ["assertion:note"], "reason": "superseded"}),
        )
        .id("withdrawal"),
    );
    let refused = send_as(
        &d,
        crate::test_daemon::Call::new(
            "revise",
            serde_json::json!({"bases": ["assertion:withdrawal@withdrawal"], "patch": {"value": {"change": "set", "value": "a new reason"}}}),
        )
        .id("revision"),
        Author::Operator,
    );
    assert!(
        matches!(&refused.outcome, command_surface::Outcome::Error(error) if error.message.contains("withdrawal cannot be revised")),
        "{refused:?}"
    );
    // The note stays withdrawn.
    assert_eq!(
        crate::test_answers::empty(query(&d, "assertions where kind:lesson"))["count"],
        0
    );
}
