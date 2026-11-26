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
    match crate::test_answers::step(send_as(d, request, author)) {
        crate::protocol::line::StepAnswer::Wrote(outcome) => serde_json::to_value(outcome).unwrap(),
        crate::protocol::line::StepAnswer::WroteEach(outcome) => {
            serde_json::to_value(outcome).unwrap()
        }
        other => panic!("expected an authored result: {other:?}"),
    }
}

fn send_as(
    d: &test_daemon::TestDaemon,
    request: crate::test_daemon::Call,
    author: Author,
) -> command_surface::Answer<serde_json::Value> {
    let limit = polylogue_store::read::limit::ReadLimit::new(
        crate::daemon::execution::default_read_deadline(),
        Default::default(),
    );
    (request)
        .within(&limit.clone())
        .authorship(
            polylogue_model::authorship::Authorship {
                author,
                via: polylogue_model::authorship::Channel::Cli,
            }
            .clone(),
        )
        .answer(&d.daemon)
}

fn annotation(targets: Vec<Reference>, body: &str) -> serde_json::Value {
    serde_json::json!({"references":targets.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "steps":[{"verb":"note","as":"lesson","text":body}]})
}

/// A query's whole answer; the caller names the outcome it expects.
fn query(d: &test_daemon::TestDaemon, text: &str) -> command_surface::Answer<serde_json::Value> {
    crate::test_answers::listing(send_as(
        d,
        crate::test_daemon::Call::new("rows", serde_json::json!({"selection": text})),
        Author::Operator,
    ))
}

#[test]
fn the_operator_accepts_or_rejects_an_agents_note() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            annotation(vec![session.clone()], "use the fast path"),
        )
        .id("r1"),
        Author::Agent(None),
    );
    let agent_id = format!("annotation:{}", agent["id"].as_str().unwrap());
    let own = ok(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            annotation(vec![session.clone()], "keep the operator lesson"),
        )
        .id("r2"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where agent:true judged:unknown"))["count"],
        1
    );
    // The operator's own notes are revised or withdrawn, not judged.
    let refused = crate::test_daemon::Call::new("rows", serde_json::json!({"selection": format!("annotation:{}", own["id"].as_str().unwrap()), "steps":[{"verb":"judge", "verdict":"accept"}]})).id("r3").answer(&d.daemon);
    assert!(crate::test_answers::refusal(refused).code.as_deref() == Some("bad_request"));
    ok(
        &d,
        crate::test_daemon::Call::new("rows", serde_json::json!({"selection": agent_id.clone(), "steps":[{"verb":"judge", "verdict":"accept", "reason":"checked"}]})).id("r4"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where judged:accepted"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "annotations where agent:true judged:unknown"))["count"],
        0
    );
    // The brief carries the operator's annotations and the agent annotations
    // the operator accepted.
    let operation = d
        .daemon
        .operation_context(&polylogue_model::authorship::Authorship {
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        })
        .unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let brief = crate::brief::prepare(
        crate::brief::BriefContext {
            reader: &reader,
            pending_deliveries: &[],
            delivery_id: "d1",
            context: &operation,
            ranking_identity: None,
            near: None,
        },
        &d.daemon.scope,
        &[],
        &crate::brief::Ask::SessionStart {
            cwd: "/nonexistent/dir".into(),
            session: Some(polylogue_model::SessionId::parse("claude-code:s1").unwrap()),
            source: crate::brief::StartSource::Startup,
        },
        Default::default(),
    )
    .unwrap();
    assert!(brief.text.contains("use the fast path"), "{}", brief.text);
    assert!(
        brief.text.contains("keep the operator lesson"),
        "{}",
        brief.text
    );
    drop(reader);
    // A later rejection replaces the acceptance.
    ok(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": agent_id, "steps":[{"verb":"judge", "verdict":"reject"}]}),
        )
        .id("r5"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where judged:rejected"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "annotations where judged:accepted"))["count"],
        0
    );
}

#[test]
fn a_note_applies_to_every_row_of_a_query() {
    let d = test_daemon::start("first prompt");
    let intended = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"messages", "all":true,
        "steps":[{"verb":"tag","name":"reviewed"}]}),
    )
    .id("r1");
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
        .user_request("r1-1")
        .unwrap()
        .unwrap();
    let UserWrite::AssertQuery { resolved, .. } = stored.write else {
        panic!("the query and selection must stay in the recorded request");
    };
    assert_eq!(resolved.len(), 1);
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where kind:tag key:reviewed"))["count"],
        1
    );
    let refused = crate::test_daemon::Call::new("rows", serde_json::json!({"selection":"sessions where origin:codex", "all":true,"steps":[{"verb":"note","text":"x","as":"lesson"}]})).id("r2").answer(&d.daemon);
    let empty = crate::test_answers::step(refused);
    assert!(
        matches!(&empty, crate::protocol::line::StepAnswer::WroteEach(outcome) if outcome.targets == 0),
        "{empty:?}"
    );
    let refused = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"sessions | agg count", "steps":[{"verb":"note","text":"x","as":"lesson"}]}),
    )
    .id("r3")
    .answer(&d.daemon);
    assert!(crate::test_answers::refusal(refused).code.as_deref() == Some("bad_request"));
}

/// Recorded history keeps the original authors of the note and its judgment.
#[test]
fn recorded_history_keeps_note_and_judgment_authors() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        crate::test_daemon::Call::new("rows", annotation(vec![session], "use the fast path"))
            .id("r1"),
        Author::Agent(None),
    );
    ok(
        &d,
        crate::test_daemon::Call::new("rows", serde_json::json!({"selection": format!("annotation:{}", agent["id"].as_str().unwrap()), "steps":[{"verb":"judge", "verdict":"accept"}]})).id("r2"),
    );
    let reader = d.daemon.reads.reader().unwrap();
    let mut requests = Vec::new();
    reader
        .visit_authored(&mut |prepared| {
            if matches!(prepared.request.id.as_str(), "r1-1" | "r2-1") {
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
        ["r1-1", "r2-1"]
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
        crate::test_daemon::Call::new("rows", annotation(vec![session], "prefer the short path"))
            .id("lesson"),
        Author::Agent(None),
    );
    let target = format!("annotation:{}", agent["id"].as_str().unwrap());
    for (request_id, accepted) in [("j1", true), ("j2", false), ("j3", true)] {
        ok(
            &d,
            crate::test_daemon::Call::new(
                "rows",
                serde_json::json!({"selection": target.clone(), "steps":[{"verb":"judge", "verdict":if accepted {"accept"} else {"reject"}}]}),
            )
            .id(request_id),
        );
    }
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where judged:accepted"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "annotations where judged:rejected"))["count"],
        0
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "annotations where agent:true judged:unknown"))["count"],
        0
    );
    let grouped = crate::test_answers::ok(query(
        &d,
        "annotations where agent:true | group by judged | count",
    ));
    assert_eq!(grouped["rows"][0]["judged"], "accepted", "{grouped}");
    assert_eq!(grouped["rows"][0]["count"], 1, "{grouped}");
}

#[test]
fn changes_select_annotations_by_their_kind() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    ok(
        &d,
        crate::test_daemon::Call::new("rows", annotation(vec![session], "a lesson")).id("lesson"),
    );
    let lessons = crate::test_answers::ok(query(&d, "feed where annotation_kind:lesson"));
    assert_eq!(lessons["count"], 1, "{lessons}");
    assert_eq!(lessons["rows"][0]["kind"], "annotation", "{lessons}");
    assert_eq!(lessons["rows"][0]["annotation_kind"], "lesson", "{lessons}");
    assert_eq!(
        crate::test_answers::empty(query(&d, "feed where annotation_kind:tag"))["count"],
        0
    );
}

#[test]
fn a_structured_value_matches_by_its_text() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    ok(
        &d,
        crate::test_daemon::Call::new("rows", annotation(vec![session], "use the fast path"))
            .id("lesson"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where value:\"use the fast path\""))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where value:\"the fast\""))["count"],
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
        crate::test_daemon::Call::new("rows", annotation(vec![session], "prefer the short path"))
            .id("lesson"),
        Author::Agent(None),
    );
    let target = format!("annotation:{}", agent["id"].as_str().unwrap());
    ok(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": target.clone(), "steps":[{"verb":"judge", "verdict":"accept"}]}),
        )
        .id("j1"),
    );
    now.store(3_000_000_000_000, Ordering::SeqCst);
    ok(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": target.clone(), "steps":[{"verb":"judge", "verdict":"reject"}]}),
        )
        .id("j2"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where judged:rejected"))["count"],
        1
    );
    assert_eq!(
        crate::test_answers::empty(query(&d, "annotations where judged:accepted"))["count"],
        0
    );
    // One current judgment: the second revised the first.
    let current = crate::test_answers::ok(query(&d, "annotations where kind:judgment"));
    assert_eq!(current["count"], 1, "{current}");
    let revisions = crate::test_answers::ok(query(
        &d,
        "annotations where kind:judgment current:* | fields id, root, current | sort by id",
    ));
    assert_eq!(revisions["rows"][0]["root"], "j1-1", "{revisions}");
    assert_eq!(revisions["rows"][1]["root"], "j1-1", "{revisions}");
    assert_eq!(revisions["rows"][1]["id"], "j2-1", "{revisions}");
}

#[test]
fn revising_a_withdrawal_is_refused() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    ok(
        &d,
        crate::test_daemon::Call::new("rows", annotation(vec![session], "keep the old path"))
            .id("note"),
    );
    assert_eq!(
        crate::test_answers::ok(query(&d, "annotations where kind:lesson"))["count"],
        1
    );
    ok(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"annotation:note-1", "steps":[{"verb":"withdraw", "reason":"superseded"}]}),
        )
        .id("withdrawal"),
    );
    let refused = send_as(
        &d,
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"annotation:withdrawal-1@withdrawal-1", "steps":[{"verb":"edit","text":"a new reason"}]}),
        )
        .id("revision"),
        Author::Operator,
    );
    assert!(
        crate::test_answers::refusal(refused)
            .message
            .contains("withdrawal cannot be revised")
    );
    // The note stays withdrawn.
    assert_eq!(
        crate::test_answers::empty(query(&d, "annotations where kind:lesson"))["count"],
        0
    );
}
