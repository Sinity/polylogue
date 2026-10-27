use super::*;
use polylogue_model::{Author, NewNote, NoteKind, Target};

fn ok(d: &test_daemon::TestDaemon, request: Request) -> serde_json::Value {
    match handle(request, &d.daemon) {
        Response::Ok { data, .. } => data,
        other => panic!("{other:?}"),
    }
}

fn note(author: Author, targets: Vec<Target>, body: &str) -> NewNote {
    NewNote {
        kind: NoteKind::Lesson,
        key: None,
        value: Some(body.into()),
        author,
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
    let session = Target::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok(
        &d,
        Request::AddNote {
            request_id: "r1".into(),
            note: note(
                Author::Agent(None),
                vec![session.clone()],
                "use the fast path",
            ),
        },
    );
    let agent_id = agent["id"].as_str().unwrap().to_owned();
    let own = ok(
        &d,
        Request::AddNote {
            request_id: "r2".into(),
            note: note(Author::Operator, vec![session.clone()], "mine"),
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
    let reader = d.daemon.reads.reader().unwrap();
    let brief = crate::briefing::prepare(
        &reader,
        &crate::briefing::Ask::SessionStart {
            cwd: "/nonexistent/dir".into(),
            session: Some(polylogue_model::SessionId::parse("claude-code:s1").unwrap()),
            source: crate::briefing::StartSource::Startup,
        },
        "d1",
        polylogue_store::user_writes::now(),
        None,
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
                ..note(Author::Operator, vec![], "")
            },
        },
    );
    assert_eq!(done["targets"], 1, "{done}");
    assert_eq!(query(&d, "notes where kind:tag key:reviewed")["count"], 1);
    let refused = handle(
        Request::AnnotateQuery {
            request_id: "r2".into(),
            query: "sessions where origin:codex".into(),
            note: note(Author::Operator, vec![], "x"),
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
            note: note(Author::Operator, vec![], "x"),
        },
        &d.daemon,
    );
    assert!(
        matches!(refused, Response::Error { ref kind, .. } if kind == "bad_request"),
        "{refused:?}"
    );
}

/// `export user-state` carries the agent's note and the operator's judgment
/// of it, as the requests that wrote them; only as json.
#[test]
fn user_state_exports_notes_and_judgments_as_their_requests() {
    let d = test_daemon::start("first prompt");
    let session = Target::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap());
    let agent = ok(
        &d,
        Request::AddNote {
            request_id: "r1".into(),
            note: note(Author::Agent(None), vec![session], "use the fast path"),
        },
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
    let export = |format: &str| {
        handle(
            Request::Export {
                targets: vec![],
                format: format.into(),
                user_state: true,
            },
            &d.daemon,
        )
    };
    let Response::Ok { data, .. } = export("json") else {
        panic!("export failed");
    };
    let document = polylogue_model::export::Document::from_json(&data.to_string()).unwrap();
    let requests: Vec<&str> = document
        .user_state
        .iter()
        .map(|entry| entry["request"]["id"].as_str().unwrap())
        .collect();
    assert_eq!(requests, ["r1", "r2"]);
    assert!(matches!(export("markdown"), Response::Error { .. }));
}
