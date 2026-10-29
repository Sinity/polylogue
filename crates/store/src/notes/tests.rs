use super::*;
use crate::{Archive, ReadPool, Writer};
use polylogue_model::NewNote;
use polylogue_model::user::UserWrite;

fn note(kind: NoteKind, author: Author, target: &str, value: &str) -> NewNote {
    NewNote {
        kind,
        key: None,
        value: Some(value.into()),
        author,
        targets: vec![Reference::parse(target).unwrap()],
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
    let first = crate::user_writes::request_now(
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
        .user(crate::user_writes::request_now(
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
        .user(crate::user_writes::request_now(
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
        .user(crate::user_writes::request_now(
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
    let mut n = note(NoteKind::Note, Author::Operator, "repo:synthetic", "x");
    n.targets.clear();
    assert!(matches!(
        writer.user(crate::user_writes::request_now(
            "018f0000-0000-7000-8000-000000000010".into(),
            UserWrite::AddNote { note: n }
        )),
        Err(crate::StoreError::Invalid(_))
    ));
    let mut n = note(NoteKind::Note, Author::Operator, "repo:synthetic", "x");
    n.revises = Some("missing".into());
    assert!(matches!(
        writer.user(crate::user_writes::request_now(
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
        .user(crate::user_writes::request_now(
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

#[test]
fn annotation_retries_and_replay_keep_the_original_targets() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let original = note(NoteKind::Note, Author::Operator, "repo:morrow", "reviewed");
    let request = crate::user_writes::request_now(
        "annotated".into(),
        UserWrite::AnnotateQuery {
            query: "sessions".into(),
            note: original.clone(),
        },
    );
    writer.user(request.clone()).unwrap();
    assert_eq!(
        reads.reader().unwrap().user_request("annotated").unwrap(),
        Some(request.clone())
    );
    let mut changed = original.clone();
    changed.targets = vec![Reference::parse("repo:other").unwrap()];
    assert!(
        writer
            .user(crate::user_writes::request_now(
                "annotated".into(),
                UserWrite::AnnotateQuery {
                    query: "sessions".into(),
                    note: changed
                }
            ))
            .unwrap()
            .repeat
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .note("annotated")
            .unwrap()
            .unwrap()
            .targets,
        original.targets
    );
    let mut changed_query = request.clone();
    if let UserWrite::AnnotateQuery { query, .. } = &mut changed_query.write {
        *query = "messages".into();
    }
    assert!(matches!(
        writer.user(changed_query),
        Err(StoreError::Conflict(_))
    ));
    let (_replay_dir, _replay_archive, _replay_lock, replay_writer, replay_reads) = setup();
    replay_writer.user(request).unwrap();
    assert_eq!(
        replay_reads.reader().unwrap().note("annotated").unwrap(),
        reads.reader().unwrap().note("annotated").unwrap()
    );
}

#[test]
fn kept_result_retries_ignore_resolved_rows_and_time_but_preserve_intent() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let request = crate::user_writes::request_now(
        "kept".into(),
        UserWrite::KeepResult {
            name: Some("before".into()),
            query: "notes".into(),
            rows_json: "[]".into(),
            read_at: "{\"now\":1}".into(),
        },
    );
    writer.user(request.clone()).unwrap();
    let mut changed = request.clone();
    if let UserWrite::KeepResult {
        rows_json, read_at, ..
    } = &mut changed.write
    {
        *rows_json = "[{\"id\":\"new\"}]".into();
        *read_at = "{\"now\":2}".into();
    }
    assert!(writer.user(changed.clone()).unwrap().repeat);
    assert_eq!(
        reads.reader().unwrap().user_request("kept").unwrap(),
        Some(request)
    );
    if let UserWrite::KeepResult { query, .. } = &mut changed.write {
        *query = "sessions".into();
    }
    assert!(matches!(writer.user(changed), Err(StoreError::Conflict(_))));
}

#[test]
fn resolved_saved_revisions_repeat_but_explicit_revisions_conflict() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let mut saved = note(
        NoteKind::SavedQuery,
        Author::Operator,
        "repo:morrow",
        "sessions",
    );
    saved.key = Some("everything".into());
    saved.targets.clear();
    writer
        .user(crate::user_writes::request_now(
            "first".into(),
            UserWrite::SaveQuery {
                note: saved.clone(),
            },
        ))
        .unwrap();
    saved.revises = Some("first".into());
    let resolved = crate::user_writes::request_now(
        "resolved".into(),
        UserWrite::SaveQuery {
            note: saved.clone(),
        },
    );
    writer.user(resolved.clone()).unwrap();
    let explicit = crate::user_writes::request_now(
        "explicit".into(),
        UserWrite::AddNote {
            note: saved.clone(),
        },
    );
    writer.user(explicit.clone()).unwrap();
    saved.revises = Some("resolved".into());
    assert!(
        writer
            .user(crate::user_writes::request_now(
                "resolved".into(),
                UserWrite::SaveQuery {
                    note: saved.clone()
                }
            ))
            .unwrap()
            .repeat
    );
    assert!(matches!(
        writer.user(crate::user_writes::request_now(
            "explicit".into(),
            UserWrite::AddNote { note: saved }
        )),
        Err(StoreError::Conflict(_))
    ));
    assert_eq!(
        reads.reader().unwrap().user_request("resolved").unwrap(),
        Some(resolved.clone())
    );
    assert_eq!(
        reads.reader().unwrap().user_request("explicit").unwrap(),
        Some(explicit)
    );
    let (_dir2, _archive2, _lock2, replay, replay_reads) = setup();
    for change in reads.reader().unwrap().user_state().unwrap() {
        if let crate::primary::RegisterChange::Request { request, .. } = change {
            replay.user(*request).unwrap();
        }
    }
    assert_eq!(
        replay_reads.reader().unwrap().note("resolved").unwrap(),
        reads.reader().unwrap().note("resolved").unwrap()
    );
}

#[test]
fn a_missing_request_is_absent_and_removed_request_content_refuses_retries() {
    let (_dir, archive, _lock, writer, reads) = setup();
    assert_eq!(
        reads.reader().unwrap().user_request("missing").unwrap(),
        None
    );
    writer
        .user(crate::user_writes::request_now(
            "removed".into(),
            UserWrite::AddNote {
                note: note(NoteKind::Note, Author::Operator, "repo:morrow", "body"),
            },
        ))
        .unwrap();
    rusqlite::Connection::open(archive.archive_db())
        .unwrap()
        .execute(
            "UPDATE client_request SET request = NULL WHERE id = 'removed'",
            [],
        )
        .unwrap();
    assert!(matches!(
        reads.reader().unwrap().user_request("removed"),
        Err(StoreError::Conflict(_))
    ));
}
