use super::*;
use crate::user_writes::{UserRequest, UserWrite};
use crate::{Archive, ReadPool, Writer};
use polylogue_model::{NewNote, SessionId};

fn note(kind: NoteKind, author: Author, target: &str, value: &str) -> NewNote {
    NewNote {
        kind,
        key: None,
        value: Some(value.into()),
        author,
        targets: vec![Target::parse(target).unwrap()],
        relations: vec![],
        revises: None,
        valid_until: None,
    }
}

fn setup() -> (
    tempfile::TempDir,
    Archive,
    crate::ArchiveLock,
    Writer,
    ReadPool,
) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let reads = ReadPool::open(&archive).unwrap();
    (dir, archive, lock, writer, reads)
}

#[test]
fn notes_are_written_once_per_request_and_revised_in_families() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let s = "session:claude-code:abc";
    let first = UserRequest::now(
        "018f0000-0000-7000-8000-000000000001".into(),
        UserWrite::AddNote {
            note: note(NoteKind::Lesson, Author::Operator, s, "run nextest"),
        },
    );
    let out = writer.user(first.clone()).unwrap();
    assert!(!out.repeat);
    // A resend of the same request writes nothing.
    assert!(writer.user(first.clone()).unwrap().repeat);
    // The same ID with other content is a conflict.
    let mut other = first.clone();
    other.write = UserWrite::AddNote {
        note: note(NoteKind::Lesson, Author::Operator, s, "something else"),
    };
    assert!(matches!(
        writer.user(other),
        Err(crate::StoreError::Conflict(_))
    ));

    let mut revised = note(NoteKind::Lesson, Author::Operator, s, "run cargo nextest");
    revised.revises = Some(out.id.clone());
    writer
        .user(UserRequest::now(
            "018f0000-0000-7000-8000-000000000002".into(),
            UserWrite::AddNote { note: revised },
        ))
        .unwrap();

    let r = reads.reader().unwrap();
    let current = r.notes(&NoteFilter::default()).unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].value.as_deref(), Some("run cargo nextest"));
    assert_eq!(current[0].root, out.id);
    let history = r
        .notes(&NoteFilter {
            history: true,
            ..NoteFilter::default()
        })
        .unwrap();
    assert_eq!(history.len(), 2);
}

#[test]
fn withdrawn_families_and_agent_notes_are_filtered() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let s = "repo:polylogue";
    let lesson = writer
        .user(UserRequest::now(
            "018f0000-0000-7000-8000-00000000000a".into(),
            UserWrite::AddNote {
                note: note(NoteKind::Lesson, Author::Operator, s, "keep it plain"),
            },
        ))
        .unwrap();
    let agent = Author::Agent(Some(SessionId::parse("codex:x").unwrap()));
    writer
        .user(UserRequest::now(
            "018f0000-0000-7000-8000-00000000000b".into(),
            UserWrite::AddNote {
                note: note(NoteKind::Lesson, agent, s, "an agent's idea"),
            },
        ))
        .unwrap();

    let r = reads.reader().unwrap();
    let by_operator = r
        .notes(&NoteFilter {
            targets: vec![Target::parse(s).unwrap()],
            agents: Some(false),
            ..NoteFilter::default()
        })
        .unwrap();
    assert_eq!(by_operator.len(), 1);
    assert_eq!(by_operator[0].author, Author::Operator);
    drop(r);

    writer
        .user(UserRequest::now(
            "018f0000-0000-7000-8000-00000000000c".into(),
            UserWrite::Withdraw {
                note: lesson.id.clone(),
                author: Author::Operator,
                reason: Some("superseded".into()),
            },
        ))
        .unwrap();
    let r = reads.reader().unwrap();
    let left = r.notes(&NoteFilter::default()).unwrap();
    assert_eq!(left.len(), 1);
    assert!(left[0].author.is_agent());
    assert!(r.note(&lesson.id).unwrap().is_some());
}

#[test]
fn a_note_needs_a_target_and_a_revision_needs_its_family() {
    let (_dir, _a, _lock, writer, _reads) = setup();
    let mut n = note(NoteKind::Note, Author::Operator, "work:1", "x");
    n.targets.clear();
    assert!(matches!(
        writer.user(UserRequest::now(
            "018f0000-0000-7000-8000-000000000010".into(),
            UserWrite::AddNote { note: n }
        )),
        Err(crate::StoreError::Invalid(_))
    ));
    let mut n = note(NoteKind::Note, Author::Operator, "work:1", "x");
    n.revises = Some("missing".into());
    assert!(matches!(
        writer.user(UserRequest::now(
            "018f0000-0000-7000-8000-000000000011".into(),
            UserWrite::AddNote { note: n }
        )),
        Err(crate::StoreError::NotFound(_))
    ));
}
