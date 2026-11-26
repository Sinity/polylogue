use super::*;
use polylogue_model::{Alternate, AlternateEffect, AlternateKind, Json};

fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
fn annotation(
    root: &str,
    id: &str,
    kind: AnnotationKind,
    links: &[(LinkRole, &str)],
) -> AnnotationFact {
    let mut fact = AnnotationFact {
        id: id.into(),
        root: root.into(),
        kind,
        key: None,
        value: Some(Json::String(format!("text {id}"))),
        author: Author::Unattributed,
        via: Channel::Import,
        machine: "synthetic".into(),
        created_at: Some(Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z").unwrap()),
        valid_until: None,
        links: links
            .iter()
            .map(|(role, reference)| AnnotationLink {
                role: *role,
                reference: r(reference),
                seen_hash: role.uses_text_hash().then_some(BlobHash([3; 32])),
                quote: None,
            })
            .collect(),
    };
    if kind.spec().about == polylogue_model::annotation::About::Anything
        && !fact.links.iter().any(|link| link.role == LinkRole::About)
    {
        fact.links.push(AnnotationLink {
            role: LinkRole::About,
            reference: r("session:codex:subject"),
            seen_hash: None,
            quote: None,
        });
    }
    fact
}
fn archive() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(&crate::schema::ARCHIVE).unwrap();

    conn
}

#[test]
fn saved_annotations_replay_facts_and_first_commit() {
    let conn = archive();
    let mut original = annotation(
        "first",
        "first",
        AnnotationKind::Claim,
        &[
            (LinkRole::About, "session:codex:session"),
            (LinkRole::Evidence, "message:codex:evidence"),
            (LinkRole::About, "block:codex:message#0"),
        ],
    );
    original.author = Author::AgentOrigin(polylogue_model::Origin::new("codex").unwrap());
    original.valid_until = original.created_at;
    original.value =
        Some(serde_json::json!({"text":"exact text","check":{"query":"messages","expect":"rows"}}));
    apply(&conn, std::slice::from_ref(&original), 7).unwrap();
    original.links.reverse();
    apply(&conn, std::slice::from_ref(&original), 2).unwrap();
    let actual = get(&conn, "first").unwrap().unwrap();
    original = canonical_fact(&original).unwrap();
    assert_eq!(actual.fact, original);
    assert_eq!(actual.commit_seq, 7);
    let mut changes = Vec::new();
    visit(&conn, Some(6), |record| {
        changes.push(record);
        Ok(())
    })
    .unwrap();
    assert_eq!(changes, vec![actual.clone()]);
    visit(&conn, Some(7), |_| {
        panic!("already committed fact appeared again")
    })
    .unwrap();
    original.value = Some(Json::String("changed".into()));
    assert!(matches!(
        apply(&conn, &[original], 8),
        Err(StoreError::Conflict(_))
    ));
    assert_eq!(get(&conn, "first").unwrap().unwrap(), actual);
    let overlay = archive();
    copy(&conn, &overlay).unwrap();
    copy(&conn, &overlay).unwrap();
    let mut expected = actual;
    expected.commit_seq = 0;
    assert_eq!(get(&overlay, "first").unwrap(), Some(expected));
}
#[test]
fn annotation_graph_shows_forks_in_every_import_order() {
    let facts = [
        annotation("family", "first", AnnotationKind::Note, &[]),
        annotation(
            "family",
            "left",
            AnnotationKind::Note,
            &[(LinkRole::Revises, "annotation:family@first")],
        ),
        annotation(
            "family",
            "right",
            AnnotationKind::Note,
            &[(LinkRole::Revises, "annotation:family@first")],
        ),
        annotation(
            "withdraw",
            "withdraw",
            AnnotationKind::Withdrawal,
            &[
                (LinkRole::Withdraws, "annotation:family@left"),
                (LinkRole::Withdraws, "annotation:family@right"),
            ],
        ),
        annotation(
            "restore",
            "restore",
            AnnotationKind::Withdrawal,
            &[(LinkRole::Withdraws, "annotation:withdraw")],
        ),
    ];
    for order in [[0, 1, 2], [2, 1, 0], [1, 0, 2]] {
        let conn = archive();
        for n in order {
            apply(&conn, std::slice::from_ref(&facts[n]), 9 - n as i64).unwrap();
        }
        assert_eq!(
            heads(&conn, &r("annotation:family")).unwrap(),
            vec![r("annotation:family@left"), r("annotation:family@right")]
        );
        apply(&conn, std::slice::from_ref(&facts[3]), 1).unwrap();
        assert_eq!(
            heads(&conn, &r("annotation:family")).unwrap(),
            vec![r("annotation:family@first")]
        );
        apply(&conn, std::slice::from_ref(&facts[4]), 0).unwrap();
        assert_eq!(
            heads(&conn, &r("annotation:family")).unwrap(),
            vec![r("annotation:family@left"), r("annotation:family@right")]
        );
    }
}
#[test]
fn withdrawals_cover_alternate_roots_and_restore_together() {
    let conn = archive();
    let alternate = Alternate {
        id: "v1".into(),
        root: "v1".into(),
        base: None,
        source: "message:codex:source".into(),
        source_hash: BlobHash([1; 32]),
        kind: AlternateKind::Translation,
        from_lang: Some("en".into()),
        to_lang: Some("pl".into()),
        author: Author::Operator,
        via: "mcp".into(),
        method: None,
        machine: "original".into(),
        created_at: Timestamp::from_unix_ms(1).unwrap(),
        commit_seq: 90,
    };
    crate::alternates::apply(
        &conn,
        &AlternateEffect {
            alternates: vec![alternate],
            alternate_parts: Vec::new(),
        },
        9,
    )
    .unwrap();
    assert_eq!(
        heads(&conn, &r("alternate:v1")).unwrap(),
        vec![r("alternate:v1@v1")]
    );
    let withdrawal = annotation(
        "withdraw",
        "withdraw",
        AnnotationKind::Withdrawal,
        &[(LinkRole::Withdraws, "alternate:v1")],
    );
    apply(&conn, &[withdrawal], 1).unwrap();
    assert!(heads(&conn, &r("alternate:v1")).unwrap().is_empty());
    let restore = annotation(
        "restore",
        "restore",
        AnnotationKind::Withdrawal,
        &[(LinkRole::Withdraws, "annotation:withdraw@withdraw")],
    );
    apply(&conn, &[restore], 0).unwrap();
    assert_eq!(
        heads(&conn, &r("alternate:v1")).unwrap(),
        vec![r("alternate:v1@v1")]
    );
}

fn states(conn: &Connection, root: &str) -> Vec<(String, bool, bool)> {
    conn.prepare("SELECT id,current,withdrawn FROM revision_state WHERE root=?1 ORDER BY id")
        .unwrap()
        .query_map([root], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn stored_heads_keep_forks_and_follow_withdrawals() {
    let conn = archive();
    apply(
        &conn,
        &[
            annotation("family", "first", AnnotationKind::Note, &[]),
            annotation(
                "family",
                "left",
                AnnotationKind::Note,
                &[(LinkRole::Revises, "annotation:family@first")],
            ),
            annotation(
                "family",
                "right",
                AnnotationKind::Note,
                &[(LinkRole::Revises, "annotation:family@first")],
            ),
            annotation("other", "other", AnnotationKind::Note, &[]),
        ],
        1,
    )
    .unwrap();
    assert_eq!(
        states(&conn, "family"),
        vec![
            ("first".into(), false, false),
            ("left".into(), true, false),
            ("right".into(), true, false)
        ]
    );
    assert_eq!(states(&conn, "other"), vec![("other".into(), true, false)]);
    apply(
        &conn,
        &[annotation(
            "withdraw",
            "withdraw",
            AnnotationKind::Withdrawal,
            &[
                (LinkRole::Withdraws, "annotation:family@left"),
                (LinkRole::Withdraws, "annotation:family@right"),
            ],
        )],
        2,
    )
    .unwrap();
    assert_eq!(
        states(&conn, "family"),
        vec![
            ("first".into(), true, false),
            ("left".into(), false, true),
            ("right".into(), false, true)
        ]
    );
    // Withdrawing the withdrawal, by its root, reaches the family it named.
    apply(
        &conn,
        &[annotation(
            "restore",
            "restore",
            AnnotationKind::Withdrawal,
            &[(LinkRole::Withdraws, "annotation:withdraw")],
        )],
        3,
    )
    .unwrap();
    assert_eq!(
        states(&conn, "family"),
        vec![
            ("first".into(), false, false),
            ("left".into(), true, false),
            ("right".into(), true, false)
        ]
    );
    assert_eq!(
        states(&conn, "withdraw"),
        vec![("withdraw".into(), false, true)]
    );
}

#[test]
fn an_annotation_applied_before_its_rubric_occupies_one_slot() {
    let conn = archive();
    let mut annotation = annotation(
        "answer",
        "answer",
        AnnotationKind::RubricValue,
        &[
            (LinkRole::About, "message:codex:row"),
            (LinkRole::DerivedFrom, "annotation:rubric@rubric"),
        ],
    );
    annotation.key = Some("clear".into());
    annotation.value = Some(serde_json::json!({"value": true}));
    apply(&conn, std::slice::from_ref(&annotation), 1).unwrap();
    let slots: i64 = conn
        .query_row(
            "SELECT count(*) FROM annotation_unique_key WHERE id='answer'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(slots, 1);
}

#[test]
fn annotation_links_store_and_read_their_reference_text() {
    let conn = archive();
    let mut fact = annotation("note", "note", AnnotationKind::Note, &[]);
    fact.links = vec![AnnotationLink {
        role: LinkRole::About,
        reference: Reference::parse("annotation:target@revision").unwrap(),
        seen_hash: None,
        quote: None,
    }];
    apply(&conn, &[fact], 1).unwrap();
    let stored: String = conn
        .query_row(
            "SELECT ref FROM annotation_link WHERE annotation='note'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stored, "annotation:target@revision");
    let read = get(&conn, "note").unwrap().unwrap();
    assert_eq!(
        read.fact.links[0].reference.to_string(),
        "annotation:target@revision"
    );
    assert_eq!(
        serde_json::to_value(&read.fact.links[0].reference).unwrap(),
        serde_json::json!(stored)
    );
}
