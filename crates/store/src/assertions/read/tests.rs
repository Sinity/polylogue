use crate::{Archive, ReadPool, StoreError, Writer};
use polylogue_model::{
    Json, Reference, RevisionReference,
    assertion::{AssertionKind, AssertionPatch, Change, LinkRole, NewAssertion, NewAssertionLink},
    user::UserWrite,
};

fn statement(kind: AssertionKind, target: &str, value: &str) -> NewAssertion {
    NewAssertion {
        kind,
        key: None,
        value: Some(Json::String(value.into())),
        links: vec![NewAssertionLink {
            role: LinkRole::About,
            reference: Reference::parse(target).unwrap(),
        }],
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
fn assertions_are_written_once_per_request_and_revised_in_families() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let target = "session:claude-code:abc";
    let first = crate::fixture::authored::operator_request(
        "018f0000-0000-7000-8000-000000000001".into(),
        UserWrite::Assert {
            assertions: vec![statement(AssertionKind::Lesson, target, "run nextest")],
        },
    );
    let out = writer.user(first.clone()).unwrap();
    assert!(!out.repeat);
    assert!(writer.user(first.clone()).unwrap().repeat);
    let mut other = first.clone();
    other.write = UserWrite::Assert {
        assertions: vec![statement(AssertionKind::Lesson, target, "something else")],
    };
    assert!(matches!(writer.user(other), Err(StoreError::Conflict(_))));
    let second = writer
        .user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-000000000002".into(),
            UserWrite::ReviseAssertions {
                bases: vec![RevisionReference::exact(out.id.clone(), out.id.clone())],
                patch: AssertionPatch {
                    value: Change::Set(Json::String("run cargo nextest".into())),
                    ..Default::default()
                },
            },
        ))
        .unwrap();
    let reader = reads.reader().unwrap();
    let current = reader
        .assertion(&RevisionReference {
            root: out.id.clone(),
            revision: None,
        })
        .unwrap()
        .unwrap();
    assert_eq!(current.fact.id, second.id);
    assert_eq!(
        current.fact.value.as_ref().and_then(Json::as_str),
        Some("run cargo nextest")
    );
    assert_eq!(current.fact.root, out.id);
    let original = reader
        .assertion(&RevisionReference::exact(out.id.clone(), out.id.clone()))
        .unwrap()
        .unwrap();
    assert_eq!(original.fact.root, out.id);
    assert_eq!(
        original.fact.value.as_ref().and_then(Json::as_str),
        Some("run nextest")
    );
}

#[test]
fn a_withdrawal_names_its_family_and_keeps_exact_facts_readable() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let lesson = writer
        .user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-00000000000a".into(),
            UserWrite::Assert {
                assertions: vec![statement(
                    AssertionKind::Lesson,
                    "repo:polylogue",
                    "keep it plain",
                )],
            },
        ))
        .unwrap();
    let family = RevisionReference {
        root: lesson.id.clone(),
        revision: None,
    };
    let withdrawal = writer
        .user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-00000000000c".into(),
            UserWrite::WithdrawAssertions {
                targets: vec![Reference::Assertion(family.clone())],
                reason: Some("superseded".into()),
            },
        ))
        .unwrap();
    let reader = reads.reader().unwrap();
    let named: String = reader
        .conn()
        .unwrap()
        .query_row(
            "SELECT ref FROM a.link WHERE record=?1 AND role='withdraws'",
            [&withdrawal.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(named, format!("assertion:{}", lesson.id));
    assert!(
        reader
            .assertion(&RevisionReference::exact(
                lesson.id.clone(),
                lesson.id.clone()
            ))
            .unwrap()
            .is_some()
    );
    assert!(reader.assertion(&family).unwrap().is_none());
    let fact = reader
        .assertion(&RevisionReference::exact(
            withdrawal.id.clone(),
            withdrawal.id.clone(),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(fact.fact.kind, AssertionKind::Withdrawal);
    assert_eq!(
        fact.fact.value.as_ref().and_then(Json::as_str),
        Some("superseded")
    );
    assert!(
        reader
            .assertion_family_currency(&family)
            .unwrap()
            .unwrap()
            .withdrawn()
    );
}

#[test]
fn an_assertion_needs_a_target_and_a_revision_needs_its_family() {
    let (_dir, _a, _lock, writer, _reads) = setup();
    let mut assertion = statement(AssertionKind::Note, "repo:synthetic", "x");
    assertion.links.clear();
    assert!(matches!(
        writer.user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-000000000010".into(),
            UserWrite::Assert {
                assertions: vec![assertion]
            },
        )),
        Err(StoreError::Invalid(_))
    ));
    assert!(matches!(
        writer.user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-000000000011".into(),
            UserWrite::ReviseAssertions {
                bases: vec![RevisionReference::exact("missing".into(), "missing".into())],
                patch: AssertionPatch {
                    value: Change::Set(Json::String("x".into())),
                    ..Default::default()
                },
            },
        )),
        Err(StoreError::NotFound(_))
    ));
}

#[test]
fn unsupported_assertion_metadata_returns_an_error_instead_of_invented_facts() {
    let (_dir, archive, _lock, writer, reads) = setup();
    let created = writer
        .user(crate::fixture::authored::operator_request(
            "unsupported-assertion".into(),
            UserWrite::Assert {
                assertions: vec![statement(
                    AssertionKind::Note,
                    "repo:morrow",
                    "synthetic body",
                )],
            },
        ))
        .unwrap();
    let conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
    // Times are integer milliseconds; i64::MAX is past the last representable year.
    let recorded: i64 = 1;
    for (kind, author, time, until) in [
        ("unrecognized", "operator", recorded, None),
        ("note", "unrecognized", recorded, None),
        ("note", "operator", i64::MAX, None),
        ("note", "operator", recorded, Some(i64::MAX)),
    ] {
        conn.execute(
            "UPDATE assertion SET kind=?1,author=?2,created_at=?3,valid_until=?4 WHERE id=?5",
            rusqlite::params![kind, author, time, until, created.id],
        )
        .unwrap();
        assert!(matches!(
            reads.reader().unwrap().assertion(&RevisionReference::exact(
                created.id.clone(),
                created.id.clone()
            )),
            Err(StoreError::Invalid(_))
        ));
    }
}

#[test]
fn query_assertion_retries_and_replay_keep_the_original_resolved_targets() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let original = Reference::parse("repo:morrow").unwrap();
    let mut assertion = statement(AssertionKind::Note, "repo:morrow", "reviewed");
    assertion.links.clear();
    let request = crate::fixture::authored::operator_request(
        "annotated".into(),
        UserWrite::AssertQuery {
            query: "sessions".into(),
            assertion,
            resolved: vec![original.clone()],
        },
    );
    writer.user(request.clone()).unwrap();
    assert_eq!(
        reads.reader().unwrap().user_request("annotated").unwrap(),
        Some(request.clone())
    );
    let mut changed = request.clone();
    if let UserWrite::AssertQuery { resolved, .. } = &mut changed.write {
        *resolved = vec![Reference::parse("repo:other").unwrap()];
    }
    assert!(writer.user(changed).unwrap().repeat);
    let exact = RevisionReference::exact("annotated".into(), "annotated".into());
    let recorded = reads.reader().unwrap().assertion(&exact).unwrap().unwrap();
    assert_eq!(
        recorded.fact.about().cloned().collect::<Vec<_>>(),
        vec![original]
    );
    let mut changed_query = request.clone();
    if let UserWrite::AssertQuery { query, .. } = &mut changed_query.write {
        *query = "messages".into();
    }
    assert!(matches!(
        writer.user(changed_query),
        Err(StoreError::Conflict(_))
    ));
    let (_replay_dir, _replay_archive, _replay_lock, replay_writer, replay_reads) = setup();
    replay_writer.user(request).unwrap();
    assert_eq!(
        replay_reads
            .reader()
            .unwrap()
            .assertion(&exact)
            .unwrap()
            .unwrap(),
        recorded
    );
}

#[test]
fn kept_result_retries_ignore_resolved_rows_and_time_but_preserve_intent() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let request = crate::fixture::authored::operator_request(
        "kept".into(),
        UserWrite::KeepResult {
            name: Some("before".into()),
            query: "assertions".into(),
            rows_json: "{\"unit\":\"assertions\",\"columns\":[],\"rows\":[]}".into(),
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
    assert!(
        reads
            .reader()
            .unwrap()
            .user_request("kept")
            .unwrap()
            .is_some()
    );
    if let UserWrite::KeepResult { query, .. } = &mut changed.write {
        *query = "sessions".into();
    }
    assert!(matches!(writer.user(changed), Err(StoreError::Conflict(_))));
}

#[test]
fn resolved_saved_revisions_repeat_but_explicit_revisions_conflict() {
    use polylogue_model::{BlobHash, assertion::values::SavedQuery};
    let (_dir, _archive, _lock, writer, reads) = setup();
    let context = || crate::user_writes::PreparationContext::new(vec!["codex".into()]);
    let save = |id: &str, text: &str| {
        crate::fixture::authored::operator_request(
            id.into(),
            UserWrite::SaveQuery {
                name: "everything".into(),
                text: text.into(),
                denominator: None,
            },
        )
    };
    writer
        .user_preparing(save("first", "sessions"), context(), None)
        .unwrap();
    let resolved = save("resolved", "messages");
    writer
        .user_preparing(resolved.clone(), context(), None)
        .unwrap();
    let plan = polylogue_query::plan::prepare(
        "events",
        &polylogue_query::Bindings {
            origins: vec!["codex".into()],
            ..Default::default()
        },
        |_| Ok(()),
    )
    .unwrap();
    let explicit = crate::fixture::authored::operator_request(
        "explicit".into(),
        UserWrite::ReviseAssertions {
            bases: vec![RevisionReference {
                root: "first".into(),
                revision: Some("resolved".into()),
            }],
            patch: AssertionPatch {
                value: Change::Set(
                    serde_json::to_value(SavedQuery {
                        text: "events".into(),
                        hash: BlobHash(plan.question_id().0),
                        denominator: None,
                    })
                    .unwrap(),
                ),
                ..Default::default()
            },
        },
    );
    writer
        .user_preparing(explicit.clone(), context(), None)
        .unwrap();
    // A frozen retry remains available without a new preparation context.
    assert!(writer.user(resolved.clone()).unwrap().repeat);
    let mut changed = explicit.clone();
    if let UserWrite::ReviseAssertions { bases, .. } = &mut changed.write {
        bases[0].revision = Some("first".into());
    }
    assert!(matches!(writer.user(changed), Err(StoreError::Conflict(_))));
    assert_eq!(
        reads.reader().unwrap().user_request("resolved").unwrap(),
        Some(resolved)
    );
    assert_eq!(
        reads.reader().unwrap().user_request("explicit").unwrap(),
        Some(explicit)
    );
    let (_dir2, archive2, _lock2, _replay, replay_reads) = setup();
    let mut destination = crate::archive::open_write(&archive2.archive_db(), 1).unwrap();
    let tx = destination.transaction().unwrap();
    reads
        .reader()
        .unwrap()
        .visit_authored(&mut |prepared| {
            crate::primary::authored::restore(&tx, prepared, 1)?;
            Ok(())
        })
        .unwrap();
    tx.commit().unwrap();
    let exact = RevisionReference {
        root: "first".into(),
        revision: Some("resolved".into()),
    };
    let original = reads.reader().unwrap().assertion(&exact).unwrap().unwrap();
    let restored = replay_reads
        .reader()
        .unwrap()
        .assertion(&exact)
        .unwrap()
        .unwrap();
    assert_eq!(restored.fact, original.fact);
    let mut found = false;
    replay_reads
        .reader()
        .unwrap()
        .visit_authored(&mut |prepared| {
            if prepared.request.id == "resolved" {
                assert_eq!(
                    prepared.effect.assertions.as_slice(),
                    std::slice::from_ref(&restored.fact)
                );
                found = true;
            }
            Ok(())
        })
        .unwrap();
    assert!(found);
}

#[test]
fn a_missing_request_is_absent_and_request_content_cannot_be_removed() {
    let (_dir, archive, _lock, writer, reads) = setup();
    assert_eq!(
        reads.reader().unwrap().user_request("missing").unwrap(),
        None
    );
    let request = crate::fixture::authored::operator_request(
        "kept".into(),
        UserWrite::Assert {
            assertions: vec![statement(AssertionKind::Note, "repo:morrow", "body")],
        },
    );
    writer.user(request).unwrap();
    assert!(
        rusqlite::Connection::open(archive.archive_db())
            .unwrap()
            .execute(
                "UPDATE client_request SET request = NULL WHERE id = 'kept'",
                [],
            )
            .is_err()
    );
    assert!(
        reads
            .reader()
            .unwrap()
            .user_request("kept")
            .unwrap()
            .is_some()
    );
}

#[test]
fn expiry_metadata_does_not_materialize_the_assertion_body() {
    let (_dir, archive, _lock, writer, reads) = setup();
    let mut note = statement(AssertionKind::Note, "repo:morrow", "synthetic body");
    note.valid_until = Some(polylogue_model::Timestamp::from_unix_ms(10).unwrap());
    let created = writer
        .user(crate::fixture::authored::operator_request(
            "expiry-metadata".into(),
            UserWrite::Assert {
                assertions: vec![note],
            },
        ))
        .unwrap();
    // A body that cannot decode discriminates metadata reads from loading the
    // whole assertion. Expiry and identity remain valid synthetic metadata.
    let conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
    conn.execute(
        "UPDATE assertion SET value='malformed-json' WHERE id=?1",
        [&created.id],
    )
    .unwrap();
    let reader = reads.reader().unwrap();
    let reference = RevisionReference::exact(created.id.clone(), created.id);
    assert!(
        reader
            .assertion_is_unexpired(
                &reference,
                polylogue_model::Timestamp::from_unix_ms(9).unwrap()
            )
            .unwrap()
    );
    assert!(
        !reader
            .assertion_is_unexpired(
                &reference,
                polylogue_model::Timestamp::from_unix_ms(10).unwrap()
            )
            .unwrap()
    );
    assert!(reader.assertion(&reference).is_err());
    assert!(
        !reader
            .assertion_is_unexpired(
                &RevisionReference::exact("missing".into(), "missing".into()),
                polylogue_model::Timestamp::from_unix_ms(9).unwrap()
            )
            .unwrap()
    );
}
