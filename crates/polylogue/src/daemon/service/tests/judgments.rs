use super::*;
use polylogue_model::{Author, NewNote, NoteKind, Reference};

fn ok(d: &test_daemon::TestDaemon, request: Request) -> serde_json::Value {
    ok_as(d, request, Author::Operator)
}

fn ok_as(d: &test_daemon::TestDaemon, request: Request, author: Author) -> serde_json::Value {
    let limit = crate::daemon::execution::limit(None, Default::default());
    match handle_with_caller(
        request,
        &d.daemon,
        &limit,
        &polylogue_model::authorship::Authorship {
            author,
            via: polylogue_model::authorship::Channel::Cli,
        },
    ) {
        Response::Ok { data, .. } => data,
        other => panic!("{other:?}"),
    }
}

fn note(targets: Vec<Reference>, body: &str) -> NewNote {
    NewNote {
        kind: NoteKind::Lesson,
        key: None,
        value: Some(body.into()),
        targets,
        relations: vec![],
        revises: None,
        valid_until: None,
    }
}

fn query(d: &test_daemon::TestDaemon, text: &str) -> serde_json::Value {
    ok(
        d,
        Request::Query {
            text: text.into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
    )
}

#[test]
fn the_operator_accepts_or_rejects_an_agents_note() {
    let d = test_daemon::start("first prompt");
    let session = Reference::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok_as(
        &d,
        Request::AddNote {
            request_id: "r1".into(),
            note: note(vec![session.clone()], "use the fast path"),
        },
        Author::Agent(None),
    );
    let agent_id = agent["id"].as_str().unwrap().to_owned();
    let own = ok(
        &d,
        Request::AddNote {
            request_id: "r2".into(),
            note: note(vec![session.clone()], "mine"),
        },
    );
    assert_eq!(
        query(&d, "notes where agent:true judged:unknown")["count"],
        1
    );
    // The operator's own notes are revised or withdrawn, not judged.
    let refused = handle(
        Request::Judge {
            request_id: "r3".into(),
            note: own["id"].as_str().unwrap().into(),
            accepted: true,
            reason: None,
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, Response::Error { ref kind, .. } if kind == "bad_request"),
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
    assert_eq!(query(&d, "notes where judged:accepted")["count"], 1);
    assert_eq!(
        query(&d, "notes where agent:true judged:unknown")["count"],
        0
    );
    // An accepted note reaches the brief as the operator's own.
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
        &crate::briefing::Ask::SessionStart {
            cwd: "/nonexistent/dir".into(),
            session: Some(polylogue_model::SessionId::parse("claude-code:s1").unwrap()),
            source: crate::briefing::StartSource::Startup,
        },
        "d1",
        &operation,
        None,
        None,
        &d.daemon.config.paths.check_history,
    )
    .unwrap();
    assert!(brief.text.contains("use the fast path"), "{}", brief.text);
    assert!(brief.text.contains("(accepted)"), "{}", brief.text);
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
    assert_eq!(query(&d, "notes where judged:rejected")["count"], 1);
    assert_eq!(query(&d, "notes where judged:accepted")["count"], 0);
}

#[test]
fn a_note_applies_to_every_row_of_a_query() {
    let d = test_daemon::start("first prompt");
    let done = ok(
        &d,
        Request::AnnotateQuery {
            request_id: "r1".into(),
            query: "messages".into(),
            note: NewNote {
                kind: NoteKind::Tag,
                key: Some("reviewed".into()),
                ..note(vec![], "")
            },
        },
    );
    assert_eq!(done["targets"], 1, "{done}");
    assert_eq!(query(&d, "notes where kind:tag key:reviewed")["count"], 1);
    let refused = handle(
        Request::AnnotateQuery {
            request_id: "r2".into(),
            query: "sessions where origin:codex".into(),
            note: note(vec![], "x"),
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, Response::Error { ref kind, .. } if kind == "empty"),
        "{refused:?}"
    );
    let refused = handle(
        Request::AnnotateQuery {
            request_id: "r3".into(),
            query: "tool_calls".into(),
            note: note(vec![], "x"),
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, Response::Error { ref kind, .. } if kind == "bad_request"),
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
        Request::AddNote {
            request_id: "r1".into(),
            note: note(vec![session], "use the fast path"),
        },
        Author::Agent(None),
    );
    ok(
        &d,
        Request::Judge {
            request_id: "r2".into(),
            note: agent["id"].as_str().unwrap().into(),
            accepted: true,
            reason: None,
        },
    );
    let reader = d.daemon.reads.reader().unwrap();
    let mut requests = Vec::new();
    reader
        .visit_authored(&mut |prepared, _imports| {
            requests.push(prepared.request.clone());
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
