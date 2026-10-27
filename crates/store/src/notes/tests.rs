use super::*;
use crate::user_writes::{UserRequest, UserWrite};
use crate::{Archive, ReadPool, Writer};
use polylogue_model::NewNote;

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
    let second = writer
        .user(UserRequest::now(
            "018f0000-0000-7000-8000-000000000002".into(),
            UserWrite::AddNote { note: revised },
        ))
        .unwrap();

    let r = reads.reader().unwrap();
    let current = r.note(&second.id).unwrap().unwrap();
    assert_eq!(current.value.as_deref(), Some("run cargo nextest"));
    assert_eq!(current.first_note, out.id);
    assert_eq!(r.note(&out.id).unwrap().unwrap().first_note, out.id);
}

#[test]
fn a_withdrawal_names_the_family_it_withdraws() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let lesson = writer
        .user(UserRequest::now(
            "018f0000-0000-7000-8000-00000000000a".into(),
            UserWrite::AddNote {
                note: note(
                    NoteKind::Lesson,
                    Author::Operator,
                    "repo:polylogue",
                    "keep it plain",
                ),
            },
        ))
        .unwrap();
    let withdrawal = writer
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
    let named: String = r
        .conn()
        .unwrap()
        .query_row(
            "SELECT other FROM a.note_relation WHERE note_id = ?1 AND kind = 'withdraws'",
            [&withdrawal.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(named, format!("note:{}", lesson.id));
    assert!(r.note(&lesson.id).unwrap().is_some());
    assert!(r.note(&withdrawal.id).unwrap().is_none());
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

#[test]
fn unsupported_note_metadata_returns_an_error_instead_of_an_empty_or_invented_note() {
    let (_dir, archive, _lock, writer, reads) = setup();
    let created = writer
        .user(UserRequest::now(
            "unsupported-note".into(),
            UserWrite::AddNote {
                note: note(
                    NoteKind::Note,
                    Author::Operator,
                    "repo:morrow",
                    "synthetic body",
                ),
            },
        ))
        .unwrap();
    let conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
    for (kind, author, time, until) in [
        ("unrecognized", "operator", 1, None),
        ("note", "unrecognized", 1, None),
        ("note", "operator", i64::MAX, None),
        ("note", "operator", 1, Some(i64::MAX)),
    ] {
        conn.execute(
            "UPDATE note SET kind=?1,author=?2,created_at=?3,valid_until=?4 WHERE id=?5",
            rusqlite::params![kind, author, time, until, created.id],
        )
        .unwrap();
        assert!(matches!(
            reads.reader().unwrap().note(&created.id),
            Err(crate::StoreError::Invalid(_))
        ));
    }
}
