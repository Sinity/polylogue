use super::*;
use polylogue_model::Json;
use polylogue_model::authorship::{Authorship, Channel};
use polylogue_model::user::UserWrite;
use serde_json::json;

fn db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(&crate::schema::ARCHIVE).unwrap();

    conn
}
fn reference(value: &str) -> Reference {
    Reference::parse(value).unwrap()
}
fn request(id: &str) -> UserRequest {
    UserRequest {
        id: id.into(),
        machine: "accepting-machine".into(),
        at: Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z").unwrap(),
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
        write: UserWrite::Assert {
            annotations: vec![],
        },
    }
}
fn statement(
    kind: AnnotationKind,
    key: Option<&str>,
    value: Json,
    target: Option<&str>,
) -> NewAnnotation {
    NewAnnotation {
        kind,
        key: key.map(str::to_owned),
        value: Some(value),
        valid_until: None,
        links: target
            .into_iter()
            .map(|target| NewAnnotationLink {
                role: LinkRole::About,
                reference: reference(target),
            })
            .collect(),
    }
}
fn write_one(conn: &Connection, id: &str, annotation: NewAnnotation, seq: i64) -> UserOutcome {
    write(
        conn,
        &request(id),
        &[annotation],
        seq,
        &|_| Ok(Some(BlobHash([5; 32]))),
        SettingMode::Standalone,
    )
    .unwrap()
}
#[test]
fn cardinality_and_stale_base_use_graph_after_each_overlay_change() {
    let conn = db();
    let title = |value: &str| {
        statement(
            AnnotationKind::Title,
            None,
            json!(value),
            Some("session:codex:subject"),
        )
    };
    write_one(&conn, "first", title("original"), 1);
    write_one(&conn, "second", title("replacement"), 2);
    let second = get(&conn, "second").unwrap().unwrap().fact;
    assert_eq!(second.root, "first");
    assert!(
        second
            .links
            .iter()
            .any(|link| link.role == LinkRole::Revises
                && link.reference == reference("annotation:first@first"))
    );
    let stale = revise(
        &conn,
        &request("third"),
        &[RevisionReference {
            root: "first".into(),
            revision: Some("first".into()),
        }],
        &AnnotationPatch::default(),
        3,
        &|_| Ok(None),
        SettingMode::Standalone,
    );
    assert!(
        matches!(stale,Err(StoreError::Stale {current,..}) if current==vec!["annotation:first@second"])
    );
    let withdrawal = NewAnnotation {
        kind: AnnotationKind::Withdrawal,
        key: None,
        value: None,
        valid_until: None,
        links: vec![NewAnnotationLink {
            role: LinkRole::Withdraws,
            reference: reference("annotation:first@second"),
        }],
    };
    write_one(&conn, "withdraw", withdrawal, 3);
    assert_eq!(
        heads(&conn, &reference("annotation:first")).unwrap(),
        vec![reference("annotation:first@first")]
    );
    revise(
        &conn,
        &request("restored-base"),
        &[RevisionReference {
            root: "first".into(),
            revision: Some("first".into()),
        }],
        &AnnotationPatch {
            value: Change::Set(json!("from exposed older base")),
            ..Default::default()
        },
        4,
        &|_| Ok(None),
        SettingMode::Standalone,
    )
    .unwrap();
    assert_eq!(
        heads(&conn, &reference("annotation:first")).unwrap(),
        vec![reference("annotation:first@restored-base")]
    );
}
#[test]
fn duplicate_tag_uses_mathematical_value_and_keeps_original_hash() {
    let conn = db();
    let first = statement(
        AnnotationKind::Tag,
        Some("priority"),
        json!({"value":1,"confidence":0.5}),
        Some("session:codex:subject"),
    );
    write_one(&conn, "one", first, 1);
    let second = statement(
        AnnotationKind::Tag,
        Some("priority"),
        serde_json::from_str(r#"{"value":1e0,"confidence":1}"#).unwrap(),
        Some("session:codex:subject"),
    );
    let repeated = write(
        &conn,
        &request("two"),
        &[second],
        2,
        &|_| Ok(Some(BlobHash([9; 32]))),
        SettingMode::Standalone,
    )
    .unwrap();
    assert!(repeated.repeat);
    assert_eq!(repeated.id, "one");
    assert!(get(&conn, "two").unwrap().is_none());
    assert_eq!(
        get(&conn, "one").unwrap().unwrap().fact.links[0].seen_hash,
        Some(BlobHash([5; 32]))
    );
}
#[test]
fn recorded_apply_keeps_forks_until_the_next_live_write() {
    let conn = db();
    let title = statement(
        AnnotationKind::Title,
        None,
        json!("title"),
        Some("session:codex:subject"),
    );
    write_one(&conn, "one", title.clone(), 1);
    let mut concurrent = get(&conn, "one").unwrap().unwrap().fact;
    concurrent.id = "other".into();
    concurrent.root = "other".into();
    // Apply keeps the second family in the slot as written elsewhere.
    apply(&conn, &[concurrent], 2).unwrap();
    // A live write into the slot revises every head of both families
    // (docs/annotations.md, several families in one slot).
    write(
        &conn,
        &request("three"),
        &[title],
        3,
        &|_| Ok(None),
        SettingMode::Standalone,
    )
    .unwrap();
    let three = get(&conn, "three").unwrap().unwrap().fact;
    assert_eq!(three.root, "one");
    let mut revised: Vec<String> = three
        .links
        .iter()
        .filter(|link| link.role == LinkRole::Revises)
        .map(|link| link.reference.to_string())
        .collect();
    revised.sort();
    assert_eq!(revised, ["annotation:one@one", "annotation:other@other"]);
}
#[test]
fn sinex_fact_keeps_its_captured_source_and_hash() {
    let conn = db();
    let note = statement(
        AnnotationKind::Note,
        None,
        json!("waiting"),
        Some("message:codex:future"),
    );
    write(
        &conn,
        &request("waiting"),
        &[note],
        1,
        &|_| Ok(None),
        SettingMode::Standalone,
    )
    .unwrap();
    let fact = get(&conn, "waiting").unwrap().unwrap().fact;
    assert_eq!(fact.via, Channel::Cli);
    assert_eq!(fact.machine, "accepting-machine");
    // Stored to the millisecond (`created_at INTEGER`, docs/annotations.md).
    assert_eq!(
        fact.created_at,
        Timestamp::from_unix_ms(request("waiting").at.unix_ms())
    );
    assert_eq!(fact.links[0].seen_hash, None);
    apply(&conn, std::slice::from_ref(&fact), 20).unwrap();
    assert_eq!(get(&conn, "waiting").unwrap().unwrap().fact, fact);
}
#[test]
fn annotation_checks_rubric_and_pass_before_storing() {
    let conn = db();
    let rubric = statement(
        AnnotationKind::Rubric,
        Some("quality"),
        json!({"instructions":"classify","unit_query":"messages","fields":[{"name":"grade","value_type":"integer","minimum":1,"maximum":3}],"abstain":true,"confidence":false,"rationale":false}),
        None,
    );
    write_one(&conn, "rubric", rubric, 1);
    let pass = statement(
        AnnotationKind::Pass,
        None,
        json!({"rubric":"annotation:rubric@rubric","population":null,"counts":{}}),
        None,
    );
    write_one(&conn, "pass", pass, 2);
    let mut annotation = statement(
        AnnotationKind::RubricValue,
        Some("grade"),
        json!({"value":2}),
        Some("message:codex:subject"),
    );
    annotation.links.extend([
        NewAnnotationLink {
            role: LinkRole::DerivedFrom,
            reference: reference("annotation:rubric@rubric"),
        },
        NewAnnotationLink {
            role: LinkRole::Origin,
            reference: reference("annotation:pass@pass"),
        },
    ]);
    write_one(&conn, "allowed", annotation.clone(), 3);
    annotation.value = Some(json!({"value":4}));
    assert!(
        matches!(write(&conn,&request("refused"),&[annotation.clone()],4,&|_| Ok(None),SettingMode::Standalone),Err(StoreError::Refused(Reason::NotAllowedByRubric { key, part })) if key == "grade" && part == "this value")
    );
    annotation.value = Some(json!({"value":2,"confidence":0.5}));
    assert!(matches!(
        write(
            &conn,
            &request("confidence"),
            &[annotation],
            4,
            &|_| Ok(None),
            SettingMode::Standalone
        ),
        Err(StoreError::Refused(Reason::NotAllowedByRubric { key, part })) if key == "grade" && part == "a confidence"
    ));
    assert!(get(&conn, "refused").unwrap().is_none());
}

#[test]
fn bulk_tag_keeps_existing_target_and_writes_each_untagged_target() {
    let conn = db();
    let tag = statement(
        AnnotationKind::Tag,
        Some("research"),
        json!({}),
        Some("session:codex:one"),
    );
    write_one(&conn, "first", tag.clone(), 1);
    let mut bulk = tag;
    bulk.links.push(NewAnnotationLink {
        role: LinkRole::About,
        reference: reference("session:codex:two"),
    });
    let outcome = write(
        &conn,
        &request("bulk"),
        &[bulk],
        2,
        &|_| Ok(None),
        SettingMode::Standalone,
    )
    .unwrap();
    assert!(!outcome.repeat);
    // Writes of one request are numbered from one; the first target was
    // already tagged and wrote nothing.
    assert!(get(&conn, "bulk.1").unwrap().is_none());
    let second = get(&conn, "bulk.2").unwrap().unwrap().fact;
    assert_eq!(
        second.about().cloned().collect::<Vec<_>>(),
        vec![reference("session:codex:two")]
    );
}

#[test]
fn withdrawal_of_conflicted_or_already_withdrawn_root_needs_no_text_winner() {
    let mut conn = db();
    write_one(
        &conn,
        "first",
        statement(
            AnnotationKind::Note,
            None,
            json!("original"),
            Some("session:codex:subject"),
        ),
        1,
    );
    let first = get(&conn, "first").unwrap().unwrap().fact;
    for id in ["left", "right"] {
        let mut fork = first.clone();
        fork.id = id.into();
        fork.links.push(AnnotationLink {
            role: LinkRole::Revises,
            reference: reference("annotation:first@first"),
            seen_hash: Some(BlobHash([4; 32])),
            quote: None,
        });
        apply(&conn, &[fork], 2).unwrap();
    }
    assert_eq!(
        heads(&conn, &reference("annotation:first")).unwrap().len(),
        2
    );
    for (id, seq) in [("withdraw-root", 3), ("already-withdrawn", 4)] {
        let mut r = request(id);
        r.write = UserWrite::WithdrawAnnotations {
            targets: vec![reference("annotation:first")],
            reason: None,
        };
        let tx = conn.transaction().unwrap();
        crate::user_writes::apply(&tx, &r, seq).unwrap();
        let prepared = crate::authored_records::known_request(&tx, &r)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.effect.annotations.len(), 1);
        assert_eq!(
            prepared.effect.annotations[0].links[0].reference,
            reference("annotation:first")
        );
        assert_eq!(prepared.effect.annotations[0].links[0].seen_hash, None);
        tx.commit().unwrap();
        assert!(
            heads(&conn, &reference("annotation:first"))
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn identity_withdrawal_checks_existence_without_calling_the_text_resolver() {
    let conn = db();
    write_one(
        &conn,
        "first",
        statement(
            AnnotationKind::Note,
            None,
            json!("text"),
            Some("session:codex:subject"),
        ),
        1,
    );
    let withdrawal = |target: &str| NewAnnotation {
        kind: AnnotationKind::Withdrawal,
        key: None,
        value: None,
        valid_until: None,
        links: vec![NewAnnotationLink {
            role: LinkRole::Withdraws,
            reference: reference(target),
        }],
    };
    assert!(matches!(
        write(
            &conn,
            &request("missing"),
            &[withdrawal("annotation:missing")],
            2,
            &|_| panic!("identity-only withdrawal requested text"),
            SettingMode::Standalone
        ),
        Err(StoreError::NotFound(_))
    ));
    assert!(get(&conn, "missing").unwrap().is_none());
    write(
        &conn,
        &request("withdrawal"),
        &[withdrawal("annotation:first")],
        2,
        &|_| panic!("identity-only withdrawal requested text"),
        SettingMode::Standalone,
    )
    .unwrap();
    assert!(
        heads(&conn, &reference("annotation:first"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn unavailable_text_link_stores_no_unbound_fact() {
    let conn = db();
    let annotation = statement(
        AnnotationKind::Note,
        None,
        json!("text"),
        Some("message:codex:unavailable"),
    );
    let result = write(
        &conn,
        &request("unavailable"),
        &[annotation],
        1,
        &|target| {
            Err(StoreError::ReferenceUnavailable {
                reference: target.to_string(),
                reason: "source unavailable".into(),
            })
        },
        SettingMode::Standalone,
    );
    assert!(matches!(
        result,
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    assert!(get(&conn, "unavailable").unwrap().is_none());
}

#[test]
fn accepting_withdrawal_refuses_a_cycle_closed_by_an_imported_forward_link() {
    let mut conn = db();
    let imported = AnnotationFact::from_request(
        "imported".into(),
        "imported".into(),
        NewAnnotation {
            kind: AnnotationKind::Withdrawal,
            key: None,
            value: None,
            valid_until: None,
            links: vec![],
        },
        vec![AnnotationLink {
            role: LinkRole::Withdraws,
            reference: reference("annotation:local"),
            seen_hash: None,
            quote: None,
        }],
        &request("imported"),
    );
    apply(&conn, std::slice::from_ref(&imported), 1).unwrap();
    {
        let tx = conn.transaction().unwrap();
        let mut local = request("local");
        local.write = UserWrite::WithdrawAnnotations {
            targets: vec![reference("annotation:imported")],
            reason: None,
        };
        assert!(matches!(
            crate::user_writes::apply(&tx, &local, 2),
            Err(StoreError::Refused(Reason::WithdrawalCycle { .. }))
        ));
        // The accepting writer rolls back the request and every proposed fact.
    }
    assert!(get(&conn, "local").unwrap().is_none());
    assert_eq!(
        get(&conn, "imported").unwrap().unwrap().fact,
        canonical_fact(&imported).unwrap()
    );
    assert!(
        !conn
            .prepare("SELECT 1 FROM client_request WHERE id='local'")
            .unwrap()
            .exists([])
            .unwrap()
    );
}

#[test]
fn normal_revision_of_an_undated_rubric_uses_its_own_request_time_and_slot() {
    let conn = db();
    let req = request("undated-source");
    let statement = statement(
        AnnotationKind::Rubric,
        Some("synthetic-rubric"),
        json!({"instructions":"Classify", "unit_query":"sessions", "fields":[{"name":"category","value_type":"string"}], "abstain":false,"confidence":false,"rationale":false}),
        None,
    );
    let mut original =
        AnnotationFact::from_request("undated".into(), "undated".into(), statement, vec![], &req);
    original.created_at = None;
    apply(&conn, std::slice::from_ref(&original), 1).unwrap();
    let patch = AnnotationPatch {
        value: Change::Set(
            json!({"instructions":"Revised", "unit_query":"sessions", "fields":[{"name":"category","value_type":"string"}], "abstain":false,"confidence":false,"rationale":false}),
        ),
        ..Default::default()
    };
    let request = request("dated-revision");
    let outcome = revise(
        &conn,
        &request,
        &[RevisionReference::exact("undated".into(), "undated".into())],
        &patch,
        2,
        &|_| Ok(None),
        SettingMode::Standalone,
    )
    .unwrap();
    assert!(!outcome.repeat);
    let revised = get(&conn, &outcome.id).unwrap().unwrap().fact;
    assert_eq!(
        revised.created_at,
        Some(Timestamp::from_unix_ms(request.at.unix_ms()).unwrap())
    );
    assert_eq!(revised.root, original.root);
    assert_eq!(get(&conn, "undated").unwrap().unwrap().fact, original);
    assert_eq!(
        crate::annotations::current_revisions::stored_heads(&conn, "undated").unwrap(),
        [Reference::Annotation(RevisionReference::exact(
            "undated".into(),
            outcome.id
        ))]
    );
}

#[test]
fn text_links_written_with_note_keep_other_links_and_source_hashes() {
    let conn = db();
    let text = "See [one](message:codex:one) and [again](message:codex:one).\n\
        ![detail](message:codex:three \"card\")\n\
        Source[^source].\n\n[^source]: [source](message:codex:two)";
    let mut note = statement(
        AnnotationKind::Note,
        None,
        json!(text),
        Some("session:codex:s"),
    );
    note.links.extend([
        NewAnnotationLink {
            role: LinkRole::Evidence,
            reference: reference("message:codex:recorded"),
        },
        NewAnnotationLink {
            role: LinkRole::Mentions,
            reference: reference("message:codex:stale"),
        },
    ]);
    write_one(&conn, "linked-note", note, 1);
    let stored = get(&conn, "linked-note").unwrap().unwrap().fact;
    let links: BTreeSet<_> = stored
        .links
        .iter()
        .map(|link| (link.role, link.reference.to_string()))
        .collect();
    assert_eq!(
        links,
        BTreeSet::from([
            (LinkRole::About, "session:codex:s".into()),
            (LinkRole::Evidence, "message:codex:recorded".into()),
            (LinkRole::Mentions, "message:codex:one".into()),
            (LinkRole::Cites, "message:codex:two".into()),
            (LinkRole::Embeds, "message:codex:three".into()),
        ])
    );
    assert_eq!(stored.links.len(), 5);
    assert!(
        stored
            .links
            .iter()
            .all(|link| link.seen_hash == Some(BlobHash([5; 32])))
    );
    assert_eq!(stored.value, Some(json!(text)));
}

#[test]
fn edit_rewrites_text_links_and_clearing_text_keeps_other_roles() {
    let conn = db();
    let mut note = statement(
        AnnotationKind::Note,
        None,
        json!("[first](message:codex:first)"),
        Some("session:codex:s"),
    );
    note.links.push(NewAnnotationLink {
        role: LinkRole::Evidence,
        reference: reference("message:codex:recorded"),
    });
    write_one(&conn, "text-note", note, 1);
    for (counter, id, base, text, expected) in [
        (
            2,
            "text-edit",
            "text-note",
            "[next](message:codex:next)",
            vec!["message:codex:next"],
        ),
        (
            3,
            "text-clear",
            "text-edit",
            "No links in this revision.",
            vec![],
        ),
    ] {
        revise(
            &conn,
            &request(id),
            &[RevisionReference {
                root: "text-note".into(),
                revision: Some(base.into()),
            }],
            &AnnotationPatch {
                value: Change::Set(json!(text)),
                ..Default::default()
            },
            counter,
            &|_| Ok(Some(BlobHash([5; 32]))),
            SettingMode::Standalone,
        )
        .unwrap();
        let stored = get(&conn, id).unwrap().unwrap().fact;
        let mentions: Vec<_> = stored
            .links
            .iter()
            .filter(|link| link.role == LinkRole::Mentions)
            .map(|link| link.reference.to_string())
            .collect();
        assert_eq!(mentions, expected);
        for (role, target) in [
            (LinkRole::About, "session:codex:s"),
            (LinkRole::Evidence, "message:codex:recorded"),
        ] {
            assert!(
                stored
                    .links
                    .iter()
                    .any(|link| link.role == role && link.reference == reference(target))
            );
        }
        assert_eq!(stored.value, Some(json!(text)));
    }
}

#[test]
fn invalid_text_links_refuse_the_whole_write_with_their_typed_reason() {
    for (id, text, expected) in [
        (
            "bad-reference",
            "[invalid](message:)",
            polylogue_model::Code::BadValue,
        ),
        (
            "session-citation",
            "Source[^source].\n\n[^source]: [session](session:codex:s)",
            polylogue_model::Code::NotBuilt,
        ),
    ] {
        let conn = db();
        let annotations = [
            statement(
                AnnotationKind::Note,
                None,
                json!("A valid first statement."),
                Some("session:codex:s"),
            ),
            statement(
                AnnotationKind::Note,
                None,
                json!(text),
                Some("session:codex:s"),
            ),
        ];
        let result = write(
            &conn,
            &request(id),
            &annotations,
            1,
            &|_| Ok(None),
            SettingMode::Standalone,
        );
        match result {
            Err(StoreError::Selection { code, .. }) => assert_eq!(code, expected),
            other => panic!("the Markdown refusal must retain its code: {other:?}"),
        }
        assert!(
            get(&conn, &format!("{id}.1")).unwrap().is_none(),
            "the earlier preview must be removed"
        );
    }
}
