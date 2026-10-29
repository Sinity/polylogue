use super::*;
use crate::daemon::service::handle;
use crate::daemon::{Request, Response};
use crate::test_daemon;
use polylogue_model::{Author, NewNote, NoteKind, SessionId};

fn add(w: &test_daemon::TestDaemon, id: &str, note: NewNote) {
    let response = handle(
        Request::AddNote {
            request_id: id.into(),
            note,
        },
        &w.daemon,
    );
    assert!(matches!(response, Response::Ok { .. }), "{response:?}");
}

fn note(author: Author, target: &str, value: &str) -> NewNote {
    NewNote {
        kind: NoteKind::Lesson,
        key: None,
        value: Some(value.into()),
        author,
        targets: vec![Reference::parse(target).unwrap()],
        relations: vec![],
        revises: None,
        valid_until: None,
    }
}

fn values(notes: &[Note]) -> Vec<&str> {
    notes.iter().filter_map(|n| n.value.as_deref()).collect()
}

#[test]
fn notes_are_listed_by_their_current_revision_and_withdrawals_leave_them_out() {
    let w = test_daemon::start("synthetic");
    let now = Timestamp::from_unix_ms(4_000_000_000_000).unwrap();
    let first = "018f0000-0000-7000-8000-000000000001";
    add(
        &w,
        first,
        note(Author::Operator, "repo:morrow", "run nextest"),
    );
    let mut revised = note(Author::Operator, "repo:morrow", "run cargo nextest");
    revised.revises = Some(first.into());
    add(&w, "018f0000-0000-7000-8000-000000000002", revised);
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
        values(&list(&reader, &format!("{about} agent:false"), now).unwrap()),
        ["run cargo nextest"],
        "the current revision only, and a target matches whole"
    );
    assert_eq!(
        count(&reader, &format!("{about} agent:true"), now).unwrap(),
        1
    );
    drop(reader);

    let withdrawn = handle(
        Request::Withdraw {
            request_id: "018f0000-0000-7000-8000-000000000005".into(),
            note: first.into(),
            author: Author::Operator,
            reason: Some("superseded".into()),
        },
        &w.daemon,
    );
    assert!(matches!(withdrawn, Response::Ok { .. }), "{withdrawn:?}");
    let reader = w.daemon.reads.reader().unwrap();
    assert_eq!(
        values(&list(&reader, &about, now).unwrap()),
        ["an agent's idea"],
        "a withdrawal takes the whole family out, and is not listed itself"
    );
    assert_eq!(
        values(
            &list(
                &reader,
                &format!("{about} withdrawn:true current:true"),
                now
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
