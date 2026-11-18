use super::*;

fn r(value: &str) -> Reference {
    Reference::parse(value).unwrap()
}
fn fact(id: &str, name: &str, hash: u8, base: Option<&str>) -> AssertionFact {
    AssertionFact {
        id: id.into(),
        root: "family".into(),
        kind: AssertionKind::SavedQuery,
        key: Some(name.into()),
        value: Some(
            serde_json::to_value(values::SavedQuery {
                text: "sessions".into(),
                hash: BlobHash([hash; 32]),
                denominator: Some(format!("query:{}", "04".repeat(32))),
            })
            .unwrap(),
        ),
        author: Author::Unattributed,
        via: polylogue_model::authorship::Channel::Import,
        machine: "synthetic-host".into(),
        created_at: Some(polylogue_model::Timestamp::from_unix_ms(1).unwrap()),
        valid_until: None,
        links: base
            .into_iter()
            .map(|base| AssertionLink {
                role: LinkRole::Revises,
                reference: r(base),
                seen_hash: None,
                quote: None,
            })
            .collect(),
    }
}

#[test]
fn saved_loader_preserves_declared_hash_denominator_and_cross_name_family_conflicts() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let first = fact("first", "original", 1, None);
    let left = fact("left", "renamed-left", 2, Some("assertion:family@first"));
    let right = fact("right", "renamed-right", 3, Some("assertion:family@first"));
    // Restore facts out of graph order; a local commit order never chooses a head.
    super::super::apply(&conn, &[right, first, left], 7).unwrap();
    conn.execute_batch("BEGIN").unwrap();
    let saved = saved_query_snapshot(&conn, ["original", "renamed-left", "renamed-right"])
        .unwrap()
        .definitions;
    assert!(!saved.contains_key("original"));
    // Only the names asked for are loaded, through whichever revision named them.
    let one = saved_query_snapshot(&conn, ["renamed-left"])
        .unwrap()
        .definitions;
    assert_eq!(one.keys().collect::<Vec<_>>(), ["renamed-left"]);
    for (name, id, hash) in [("renamed-left", "left", 2), ("renamed-right", "right", 3)] {
        let [definition] = saved[name].as_slice() else {
            panic!("one definition under each name")
        };
        assert_eq!(
            definition.reference,
            RevisionReference {
                root: "family".into(),
                revision: Some(id.into())
            }
        );
        assert_eq!(definition.value.text, "sessions");
        assert_eq!(definition.value.hash, BlobHash([hash; 32]));
        assert_eq!(
            definition.value.denominator.as_deref(),
            Some(format!("query:{}", "04".repeat(32)).as_str())
        );
        assert_eq!(
            definition.conflicts,
            [r("assertion:family@left"), r("assertion:family@right")]
        );
    }
}

#[test]
fn saved_revision_numbers_follow_original_links_not_restore_order_or_time() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let mut first = fact("first", "focus", 1, None);
    first.created_at = None;
    let second = fact("second", "focus", 2, Some("assertion:family@first"));
    let third = fact("third", "focus", 3, Some("assertion:family@second"));
    // Original effects may reach this host in the opposite order.
    for (counter, row) in [(1, third), (3, second), (20, first)] {
        super::super::apply(&conn, &[row], counter).unwrap();
    }
    let revisions = named_revisions(&conn, "focus").unwrap();
    assert_eq!(
        revisions
            .iter()
            .map(|row| row.fact.id.as_str())
            .collect::<Vec<_>>(),
        ["first", "second", "third"]
    );
    for (number, id, hash) in [(1, "first", 1), (2, "second", 2), (3, "third", 3)] {
        let definition = named_revision(&conn, "focus", number).unwrap();
        assert_eq!(
            definition.reference,
            RevisionReference::exact("family".into(), id.into())
        );
        assert_eq!(definition.value.hash, BlobHash([hash; 32]));
        assert_eq!(
            saved_revision_number(&conn, &definition.reference).unwrap(),
            number
        );
    }
    assert!(revisions[0].fact.created_at.is_none());
}

#[test]
fn saved_revision_forks_are_ambiguous_and_merged_depth_is_exact() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let first = fact("first", "focus", 1, None);
    let left = fact("left", "focus", 2, Some("assertion:family@first"));
    let right = fact("right", "renamed", 3, Some("assertion:family@first"));
    let mut merged = fact("merged", "focus", 4, Some("assertion:family@left"));
    merged.links.push(AssertionLink {
        role: LinkRole::Revises,
        reference: r("assertion:family@right"),
        seen_hash: None,
        quote: None,
    });
    super::super::apply(&conn, &[merged, right, first, left], 2).unwrap();
    match named_revision(&conn, "focus", 2).unwrap_err() {
        StoreError::Ambiguous {
            reference,
            candidates,
        } => {
            assert_eq!(reference, "query:focus@2");
            assert_eq!(
                candidates,
                ["assertion:family@left", "assertion:family@right"]
            );
        }
        other => panic!("fork must be explicit: {other}"),
    }
    assert_eq!(
        named_revision(&conn, "focus", 3).unwrap().reference,
        RevisionReference::exact("family".into(), "merged".into())
    );
    assert_eq!(
        saved_revision_number(
            &conn,
            &RevisionReference::exact("family".into(), "merged".into())
        )
        .unwrap(),
        3
    );
    assert!(matches!(
        named_revision(&conn, "focus", 4),
        Err(StoreError::NotFound(_))
    ));
}

#[test]
fn saved_revision_history_diagnoses_missing_parents_and_cycles() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let child = fact(
        "child",
        "missing-parent",
        2,
        Some("assertion:family@missing"),
    );
    super::super::apply(&conn, &[child], 1).unwrap();
    match named_revisions(&conn, "missing-parent").unwrap_err() {
        StoreError::ReferenceUnavailable { reference, reason } => {
            assert_eq!(reference, "assertion:family@missing");
            assert!(reason.contains("parent"));
        }
        other => panic!("missing original must be explicit: {other}"),
    }
    let cycle = Connection::open_in_memory().unwrap();
    cycle.execute_batch(crate::schema::ARCHIVE).unwrap();
    super::super::apply(
        &cycle,
        &[
            fact("cycle-a", "cycle", 1, Some("assertion:family@cycle-b")),
            fact("cycle-b", "cycle", 2, Some("assertion:family@cycle-a")),
        ],
        1,
    )
    .unwrap();
    assert!(
        named_revisions(&cycle, "cycle")
            .unwrap_err()
            .to_string()
            .contains("cycle")
    );
}

#[test]
fn saved_query_record_keeps_history_kept_watch_and_dependent_identities() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = crate::fixture::Fixture::new(directory.path()).unwrap();
    let conn = crate::archive::open_archive_writer(&fixture.archive).unwrap();
    let first_plan =
        polylogue_query::plan::prepare("sessions", &polylogue_query::Bindings::default(), |_| {
            Ok(())
        })
        .unwrap();
    let first_object = first_plan.query_object().unwrap();
    crate::query_objects::apply_object(&conn, &first_object, 1).unwrap();
    let mut bindings = polylogue_query::Bindings::default();
    bindings
        .add_saved("focus", &first_plan, "first", "family")
        .unwrap();
    let dependent = polylogue_query::plan::prepare("query:focus | count", &bindings, |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    crate::query_objects::apply_object(&conn, &dependent, 2).unwrap();
    let mut first = fact("first", "focus", 1, None);
    first.created_at = None;
    first.value = Some(
        serde_json::to_value(values::SavedQuery {
            text: "sessions".into(),
            hash: BlobHash(first_object.id.0),
            denominator: None,
        })
        .unwrap(),
    );
    let mut watch = fact("watch", "focus", 1, None);
    watch.root = "watch".into();
    watch.kind = AssertionKind::Signal;
    watch.value = Some(serde_json::json!({"name":"focus","query":"query:focus@1"}));
    super::super::apply(&conn, &[first, watch], 5).unwrap();
    crate::primary::authored::insert_kept(
        &conn,
        &polylogue_model::authored::KeptResult {
            id: "kept-original".into(),
            name: Some("focus".into()),
            query: "sessions".into(),
            ran_at: Timestamp::from_unix_ms(7).unwrap(),
            read_at: "{}".into(),
            rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
        },
    )
    .unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let record = reader.saved_query_record("focus").unwrap().unwrap();
    assert_eq!(record.name, "focus");
    assert_eq!(record.revisions.len(), 1);
    assert_eq!(record.revisions[0].revision, 1);
    assert_eq!(record.revisions[0].reference, "assertion:family@first");
    assert_eq!(
        record.revisions[0].hash,
        format!("query:{}", first_object.id)
    );
    assert!(record.revisions[0].saved_at.is_none());
    assert_eq!(record.revisions[0].author, "unattributed");
    assert_eq!(record.kept.len(), 1);
    assert_eq!(record.kept[0].result, "result:kept-original");
    assert_eq!(record.kept[0].rows, 0);
    assert_eq!(record.kept[0].at, Timestamp::from_unix_ms(7).unwrap());
    assert_eq!(record.watch.as_deref(), Some("assertion:watch@watch"));
    assert_eq!(record.dependents, vec![format!("query:{}", dependent.id)]);
    assert!(reader.saved_query_record("absent").unwrap().is_none());
}

#[test]
fn saved_revision_number_follows_a_merge_of_two_original_families() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let first = fact("first", "focus", 1, None);
    let mut other = fact("other-first", "focus", 2, None);
    other.root = "other-family".into();
    let mut merged = fact("merged", "focus", 3, Some("assertion:family@first"));
    merged.links.push(AssertionLink {
        role: LinkRole::Revises,
        reference: r("assertion:other-family@other-first"),
        seen_hash: None,
        quote: None,
    });
    super::super::apply(&conn, &[merged, other, first], 1).unwrap();
    assert_eq!(
        saved_revision_number(
            &conn,
            &RevisionReference::exact("family".into(), "merged".into())
        )
        .unwrap(),
        2
    );
    assert_eq!(
        named_revision(&conn, "focus", 2).unwrap().reference,
        RevisionReference::exact("family".into(), "merged".into())
    );
    assert!(matches!(
        named_revision(&conn, "focus", 1),
        Err(StoreError::Ambiguous { .. })
    ));
}
