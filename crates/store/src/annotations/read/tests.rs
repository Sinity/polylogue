use crate::{Archive, ReadPool, StoreError, Writer};
use polylogue_model::{
    Json, Reference, RevisionReference,
    annotation::{
        AnnotationKind, AnnotationPatch, Change, LinkRole, NewAnnotation, NewAnnotationLink,
    },
    user::UserWrite,
};

fn statement(kind: AnnotationKind, target: &str, value: &str) -> NewAnnotation {
    NewAnnotation {
        kind,
        key: None,
        value: Some(Json::String(value.into())),
        links: vec![NewAnnotationLink {
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
fn annotations_are_written_once_per_request_and_revised_in_families() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let target = "session:claude-code:abc";
    let first = crate::fixture::authored::operator_request(
        "018f0000-0000-7000-8000-000000000001".into(),
        UserWrite::Assert {
            annotations: vec![statement(AnnotationKind::Lesson, target, "run nextest")],
        },
    );
    let out = writer.archive().user(first.clone()).unwrap();
    assert!(!out.repeat);
    assert!(writer.archive().user(first.clone()).unwrap().repeat);
    let mut other = first.clone();
    other.write = UserWrite::Assert {
        annotations: vec![statement(AnnotationKind::Lesson, target, "something else")],
    };
    assert!(matches!(
        writer.archive().user(other),
        Err(StoreError::Conflict(_))
    ));
    let second = writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-000000000002".into(),
            UserWrite::ReviseAnnotations {
                bases: vec![RevisionReference::exact(out.id.clone(), out.id.clone())],
                patch: AnnotationPatch {
                    value: Change::Set(Json::String("run cargo nextest".into())),
                    ..Default::default()
                },
            },
        ))
        .unwrap();
    let reader = reads.reader().unwrap();
    let current = reader
        .annotation(&RevisionReference {
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
        .annotation(&RevisionReference::exact(out.id.clone(), out.id.clone()))
        .unwrap()
        .unwrap();
    assert_eq!(original.fact.root, out.id);
    assert_eq!(
        original.fact.value.as_ref().and_then(Json::as_str),
        Some("run nextest")
    );
}

#[test]
fn withdrawal_names_its_family_and_keeps_facts_readable() {
    let (_dir, _a, _lock, writer, reads) = setup();
    let lesson = writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-00000000000a".into(),
            UserWrite::Assert {
                annotations: vec![statement(
                    AnnotationKind::Lesson,
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
        .archive()
        .user(crate::fixture::authored::operator_request(
            "018f0000-0000-7000-8000-00000000000c".into(),
            UserWrite::WithdrawAnnotations {
                targets: vec![Reference::Annotation(family.clone())],
                reason: Some("superseded".into()),
            },
        ))
        .unwrap();
    let reader = reads.reader().unwrap();
    let named: String = reader
        .conn()
        .unwrap()
        .query_row(
            "SELECT ref FROM a.annotation_link WHERE annotation=?1 AND role='withdraws'",
            [&withdrawal.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(named, format!("annotation:{}", lesson.id));
    assert!(
        reader
            .annotation(&RevisionReference::exact(
                lesson.id.clone(),
                lesson.id.clone()
            ))
            .unwrap()
            .is_some()
    );
    assert!(reader.annotation(&family).unwrap().is_none());
    let fact = reader
        .annotation(&RevisionReference::exact(
            withdrawal.id.clone(),
            withdrawal.id.clone(),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(fact.fact.kind, AnnotationKind::Withdrawal);
    assert_eq!(
        fact.fact.value.as_ref().and_then(Json::as_str),
        Some("superseded")
    );
    assert!(
        reader
            .annotation_family_state(&family)
            .unwrap()
            .unwrap()
            .withdrawn()
    );
}

#[test]
fn an_annotation_needs_a_target_and_a_revision_needs_its_family() {
    let (_dir, _a, _lock, writer, _reads) = setup();
    let mut annotation = statement(AnnotationKind::Note, "repo:synthetic", "x");
    annotation.links.clear();
    assert!(matches!(
        writer
            .archive()
            .user(crate::fixture::authored::operator_request(
                "018f0000-0000-7000-8000-000000000010".into(),
                UserWrite::Assert {
                    annotations: vec![annotation]
                },
            )),
        Err(StoreError::Refused(
            polylogue_model::Reason::BadAnnotationLinks { .. }
        ))
    ));
    assert!(matches!(
        writer
            .archive()
            .user(crate::fixture::authored::operator_request(
                "018f0000-0000-7000-8000-000000000011".into(),
                UserWrite::ReviseAnnotations {
                    bases: vec![RevisionReference::exact("missing".into(), "missing".into())],
                    patch: AnnotationPatch {
                        value: Change::Set(Json::String("x".into())),
                        ..Default::default()
                    },
                },
            )),
        Err(StoreError::NotFound(_))
    ));
}

#[test]
fn unsupported_annotation_metadata_returns_an_error_instead_of_invented_facts() {
    let (_dir, archive, _lock, writer, reads) = setup();
    let created = writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "unsupported-annotation".into(),
            UserWrite::Assert {
                annotations: vec![statement(
                    AnnotationKind::Note,
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
            "UPDATE annotation SET kind=?1,author=?2,created_at=?3,valid_until=?4 WHERE id=?5",
            rusqlite::params![kind, author, time, until, created.id],
        )
        .unwrap();
        assert!(matches!(
            reads
                .reader()
                .unwrap()
                .annotation(&RevisionReference::exact(
                    created.id.clone(),
                    created.id.clone()
                )),
            Err(StoreError::StoredValue { .. })
        ));
    }
}

#[test]
fn query_annotation_retries_and_replay_keep_the_original_resolved_targets() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let original = Reference::parse("repo:morrow").unwrap();
    let mut annotation = statement(AnnotationKind::Note, "repo:morrow", "reviewed");
    annotation.links.clear();
    let request = crate::fixture::authored::operator_request(
        "annotated".into(),
        UserWrite::AssertQuery {
            query: "sessions".into(),
            annotation,
            resolved: vec![original.clone()],
        },
    );
    writer.archive().user(request.clone()).unwrap();
    assert_eq!(
        reads.reader().unwrap().user_request("annotated").unwrap(),
        Some(request.clone())
    );
    let mut changed = request.clone();
    if let UserWrite::AssertQuery { resolved, .. } = &mut changed.write {
        *resolved = vec![Reference::parse("repo:other").unwrap()];
    }
    assert!(writer.archive().user(changed).unwrap().repeat);
    let exact = RevisionReference::exact("annotated".into(), "annotated".into());
    let recorded = reads.reader().unwrap().annotation(&exact).unwrap().unwrap();
    assert_eq!(
        recorded.fact.about().cloned().collect::<Vec<_>>(),
        vec![original]
    );
    let mut changed_query = request.clone();
    if let UserWrite::AssertQuery { query, .. } = &mut changed_query.write {
        *query = "messages".into();
    }
    assert!(matches!(
        writer.archive().user(changed_query),
        Err(StoreError::Conflict(_))
    ));
    let (_replay_dir, _replay_archive, _replay_lock, replay_writer, replay_reads) = setup();
    replay_writer.archive().user(request).unwrap();
    assert_eq!(
        replay_reads
            .reader()
            .unwrap()
            .annotation(&exact)
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
        UserWrite::Save(Box::new(polylogue_model::user::Save {
            name: Some("before".into()),
            query: "annotations".into(),
            rows_json: "{\"unit\":\"annotations\",\"columns\":[],\"rows\":[]}".into(),
            read_at: "{\"now\":1}".into(),
            denominator: None,
            watch: None,
            outcomes: None,
        })),
    );
    writer
        .archive()
        .user(crate::writer::UserSubmission {
            request: request.clone(),
            scope: None,
            preparation: Some(crate::user_writes::PreparationContext::new(vec![
                "codex".into(),
            ])),
            parent: None,
        })
        .unwrap();
    let mut changed = request.clone();
    if let UserWrite::Save(save) = &mut changed.write {
        save.rows_json = "[{\"id\":\"new\"}]".into();
        save.read_at = "{\"now\":2}".into();
    }
    assert!(writer.archive().user(changed.clone()).unwrap().repeat);
    // The retry's rows and read time do not replace the original request.
    let stored = reads
        .reader()
        .unwrap()
        .user_request("kept")
        .unwrap()
        .expect("the kept request is stored");
    let UserWrite::Save(stored) = stored.write else {
        panic!("the stored request is not a save");
    };
    assert_eq!(
        stored.rows_json,
        "{\"unit\":\"annotations\",\"columns\":[],\"rows\":[]}"
    );
    assert_eq!(stored.read_at, "{\"now\":1}");
    if let UserWrite::Save(save) = &mut changed.write {
        save.query = "sessions".into();
    }
    assert!(matches!(
        writer.archive().user(changed),
        Err(StoreError::Conflict(_))
    ));
}

#[test]
fn resolved_saved_revisions_repeat_but_explicit_revisions_conflict() {
    use polylogue_model::{BlobHash, annotation::values::SavedQuery};
    let (_dir, _archive, _lock, writer, reads) = setup();
    let context = || crate::user_writes::PreparationContext::new(vec!["codex".into()]);
    let save = |id: &str, text: &str| {
        crate::fixture::authored::operator_request(
            id.into(),
            UserWrite::Save(Box::new(polylogue_model::user::Save {
                name: Some("everything".into()),
                query: text.into(),
                denominator: None,
                rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
                read_at: "{}".into(),
                watch: None,
                outcomes: None,
            })),
        )
    };
    writer
        .archive()
        .user(crate::writer::UserSubmission {
            request: save("first", "sessions"),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
        .unwrap();
    let resolved = save("resolved", "messages");
    writer
        .archive()
        .user(crate::writer::UserSubmission {
            request: resolved.clone(),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
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
        UserWrite::ReviseAnnotations {
            bases: vec![RevisionReference {
                root: "first".into(),
                revision: Some("resolved".into()),
            }],
            patch: AnnotationPatch {
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
        .archive()
        .user(crate::writer::UserSubmission {
            request: explicit.clone(),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
        .unwrap();
    // A recorded retry remains available without a new preparation context.
    assert!(writer.archive().user(resolved.clone()).unwrap().repeat);
    let mut changed = explicit.clone();
    if let UserWrite::ReviseAnnotations { bases, .. } = &mut changed.write {
        bases[0].revision = Some("first".into());
    }
    assert!(matches!(
        writer.archive().user(changed),
        Err(StoreError::Conflict(_))
    ));
    let mut recorded = resolved;
    let UserWrite::Save(save) = &mut recorded.write else {
        unreachable!()
    };
    save.outcomes = Some(polylogue_model::user::SaveReport {
        request: "resolved".into(),
        saved: Some(polylogue_model::user::SavedPart {
            id: "resolved".into(),
            name: "everything".into(),
            revision: 2,
            new_revision: true,
        }),
        kept: polylogue_model::user::KeptPart {
            result: "result:resolved".into(),
            rows: 0,
            at: recorded.at,
        },
        watch: None,
        repeat: false,
        pending: false,
    });
    assert_eq!(
        reads.reader().unwrap().user_request("resolved").unwrap(),
        Some(recorded)
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
            crate::authored_records::restore(&tx, prepared, 1)?;
            Ok(())
        })
        .unwrap();
    tx.commit().unwrap();
    let exact = RevisionReference {
        root: "first".into(),
        revision: Some("resolved".into()),
    };
    let original = reads.reader().unwrap().annotation(&exact).unwrap().unwrap();
    let restored = replay_reads
        .reader()
        .unwrap()
        .annotation(&exact)
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
                    prepared.effect.annotations.as_slice(),
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
            annotations: vec![statement(AnnotationKind::Note, "repo:morrow", "body")],
        },
    );
    writer.archive().user(request).unwrap();
    let refusal = rusqlite::Connection::open(archive.archive_db())
        .unwrap()
        .execute(
            "UPDATE client_request SET request = NULL WHERE id = 'kept'",
            [],
        )
        .unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("NOT NULL constraint failed: client_request.request"),
        "unexpected refusal: {refusal}"
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
fn expiry_check_does_not_load_the_annotation_body() {
    let (_dir, archive, _lock, writer, reads) = setup();
    let mut note = statement(AnnotationKind::Note, "repo:morrow", "synthetic body");
    note.valid_until = Some(polylogue_model::Timestamp::from_unix_ms(10).unwrap());
    let created = writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "expiry-metadata".into(),
            UserWrite::Assert {
                annotations: vec![note],
            },
        ))
        .unwrap();
    // A body that cannot decode discriminates metadata reads from loading the
    // whole annotation. Expiry and identity remain valid synthetic metadata.
    let conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
    conn.execute(
        "UPDATE annotation SET value='malformed-json' WHERE id=?1",
        [&created.id],
    )
    .unwrap();
    let reader = reads.reader().unwrap();
    let reference = RevisionReference::exact(created.id.clone(), created.id);
    assert!(
        reader
            .annotation_is_unexpired(
                &reference,
                polylogue_model::Timestamp::from_unix_ms(9).unwrap()
            )
            .unwrap()
    );
    assert!(
        !reader
            .annotation_is_unexpired(
                &reference,
                polylogue_model::Timestamp::from_unix_ms(10).unwrap()
            )
            .unwrap()
    );
    let mut linked = Vec::new();
    let _ = reader
        .visit_annotation_references_linked(
            &Reference::parse("repo:morrow").unwrap(),
            &[LinkRole::About],
            |revision| {
                linked.push(revision);
                Ok(std::ops::ControlFlow::Continue(()))
            },
        )
        .unwrap();
    assert_eq!(
        linked.as_slice(),
        std::slice::from_ref(&reference),
        "linked metadata must not decode the malformed body"
    );
    assert!(reader.annotation(&reference).is_err());
    assert!(
        !reader
            .annotation_is_unexpired(
                &RevisionReference::exact("missing".into(), "missing".into()),
                polylogue_model::Timestamp::from_unix_ms(9).unwrap()
            )
            .unwrap()
    );
}

#[test]
fn tagged_reads_only_the_current_revision_of_a_tag() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let first = Reference::parse("session:codex:first").unwrap();
    let mut tag = statement(AnnotationKind::Tag, "session:codex:first", "unused");
    tag.key = Some("status".into());
    tag.value = Some(serde_json::json!({"value":"ready"}));
    let saved = writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "tag-ready".into(),
            UserWrite::Assert {
                annotations: vec![tag],
            },
        ))
        .unwrap();
    writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "tag-done".into(),
            UserWrite::ReviseAnnotations {
                bases: vec![RevisionReference::exact(saved.id.clone(), saved.id.clone())],
                patch: AnnotationPatch {
                    value: Change::Set(serde_json::json!({"value":"done"})),
                    ..Default::default()
                },
            },
        ))
        .unwrap();
    let targets = vec![first.clone()];
    let reader = reads.reader().unwrap();
    let author = polylogue_model::Author::Operator;
    assert!(
        reader
            .tagged(
                &targets,
                "status",
                Some(&serde_json::json!("ready")),
                &author
            )
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        reader
            .tagged(
                &targets,
                "status",
                Some(&serde_json::json!("done")),
                &author
            )
            .unwrap(),
        [first].into_iter().collect()
    );
}

#[test]
fn tagged_reads_current_name_and_value_with_confidence_outside_identity() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let first = Reference::parse("session:codex:first").unwrap();
    let second = Reference::parse("session:codex:second").unwrap();
    let mut tag = statement(AnnotationKind::Tag, "session:codex:first", "unused");
    tag.key = Some("status".into());
    tag.value = Some(serde_json::json!({"value":"ready", "confidence":0.9}));
    let saved = writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "tag-valued".into(),
            UserWrite::Assert {
                annotations: vec![tag],
            },
        ))
        .unwrap();
    let mut bare = statement(AnnotationKind::Tag, "session:codex:second", "unused");
    bare.key = Some("status".into());
    bare.value = None;
    writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "tag-bare".into(),
            UserWrite::Assert {
                annotations: vec![bare],
            },
        ))
        .unwrap();
    let targets = vec![first.clone(), second.clone()];
    {
        let reader = reads.reader().unwrap();
        assert_eq!(
            reader
                .tagged(
                    &targets,
                    "status",
                    Some(&serde_json::json!("ready")),
                    &polylogue_model::Author::Agent(None)
                )
                .unwrap(),
            [first].into_iter().collect()
        );
        assert_eq!(
            reader
                .tagged(&targets, "status", None, &polylogue_model::Author::Operator)
                .unwrap(),
            [second].into_iter().collect()
        );
        assert!(
            reader
                .tagged(
                    &targets,
                    "status",
                    Some(&serde_json::json!("other")),
                    &polylogue_model::Author::Operator
                )
                .unwrap()
                .is_empty()
        );
        assert!(
            reader
                .tagged(
                    &targets,
                    "another",
                    Some(&serde_json::json!("ready")),
                    &polylogue_model::Author::Operator
                )
                .unwrap()
                .is_empty()
        );
    }
    writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "withdraw-valued-tag".into(),
            UserWrite::WithdrawAnnotations {
                targets: vec![Reference::Annotation(RevisionReference::exact(
                    saved.id.clone(),
                    saved.id,
                ))],
                reason: None,
            },
        ))
        .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .tagged(
                &targets,
                "status",
                Some(&serde_json::json!("ready")),
                &polylogue_model::Author::Operator
            )
            .unwrap()
            .is_empty()
    );
}
