use super::*;
use polylogue_model::{Json, Variant, VariantEffect, VariantKind};

fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
fn assertion(
    root: &str,
    id: &str,
    kind: AssertionKind,
    links: &[(LinkRole, &str)],
) -> AssertionFact {
    let mut fact = AssertionFact {
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
            .map(|(role, reference)| AssertionLink {
                role: *role,
                reference: r(reference),
                seen_hash: role.uses_text_hash().then_some(BlobHash([3; 32])),
                quote: None,
            })
            .collect(),
    };
    if kind.spec().about == polylogue_model::assertion::About::Anything
        && !fact.links.iter().any(|link| link.role == LinkRole::About)
    {
        fact.links.push(AssertionLink {
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
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();

    conn
}

#[test]
fn frozen_assertions_replay_exact_facts_and_first_admission() {
    let conn = archive();
    let mut original = assertion(
        "first",
        "first",
        AssertionKind::Claim,
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
        panic!("already admitted fact appeared again")
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
fn canonical_assertion_graph_exposes_forks_and_restoration_in_every_import_order() {
    let facts = [
        assertion("family", "first", AssertionKind::Note, &[]),
        assertion(
            "family",
            "left",
            AssertionKind::Note,
            &[(LinkRole::Revises, "assertion:family@first")],
        ),
        assertion(
            "family",
            "right",
            AssertionKind::Note,
            &[(LinkRole::Revises, "assertion:family@first")],
        ),
        assertion(
            "withdraw",
            "withdraw",
            AssertionKind::Withdrawal,
            &[
                (LinkRole::Withdraws, "assertion:family@left"),
                (LinkRole::Withdraws, "assertion:family@right"),
            ],
        ),
        assertion(
            "restore",
            "restore",
            AssertionKind::Withdrawal,
            &[(LinkRole::Withdraws, "assertion:withdraw")],
        ),
    ];
    for order in [[0, 1, 2], [2, 1, 0], [1, 0, 2]] {
        let conn = archive();
        for n in order {
            apply(&conn, std::slice::from_ref(&facts[n]), 9 - n as i64).unwrap();
        }
        assert_eq!(
            heads(&conn, &r("assertion:family")).unwrap(),
            vec![r("assertion:family@left"), r("assertion:family@right")]
        );
        apply(&conn, std::slice::from_ref(&facts[3]), 1).unwrap();
        assert_eq!(
            heads(&conn, &r("assertion:family")).unwrap(),
            vec![r("assertion:family@first")]
        );
        apply(&conn, std::slice::from_ref(&facts[4]), 0).unwrap();
        assert_eq!(
            heads(&conn, &r("assertion:family")).unwrap(),
            vec![r("assertion:family@left"), r("assertion:family@right")]
        );
    }
}
#[test]
fn canonical_withdrawals_cover_variant_roots_and_restore_through_the_same_graph() {
    let conn = archive();
    let variant = Variant {
        id: "v1".into(),
        root: "v1".into(),
        base: None,
        source: "message:codex:source".into(),
        source_hash: BlobHash([1; 32]),
        kind: VariantKind::Translation,
        from_lang: Some("en".into()),
        to_lang: Some("pl".into()),
        author: Author::Operator,
        via: "mcp".into(),
        method: None,
        machine: "original".into(),
        created_at: Timestamp::from_unix_ms(1).unwrap(),
        commit_seq: 90,
    };
    crate::variants::apply(
        &conn,
        &VariantEffect {
            variants: vec![variant],
            variant_parts: Vec::new(),
        },
        9,
    )
    .unwrap();
    assert_eq!(
        heads(&conn, &r("variant:v1")).unwrap(),
        vec![r("variant:v1@v1")]
    );
    let withdrawal = assertion(
        "withdraw",
        "withdraw",
        AssertionKind::Withdrawal,
        &[(LinkRole::Withdraws, "variant:v1")],
    );
    apply(&conn, &[withdrawal], 1).unwrap();
    assert!(heads(&conn, &r("variant:v1")).unwrap().is_empty());
    let restore = assertion(
        "restore",
        "restore",
        AssertionKind::Withdrawal,
        &[(LinkRole::Withdraws, "assertion:withdraw@withdraw")],
    );
    apply(&conn, &[restore], 0).unwrap();
    assert_eq!(
        heads(&conn, &r("variant:v1")).unwrap(),
        vec![r("variant:v1@v1")]
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
fn stored_currency_keeps_every_fork_head_and_follows_withdrawal_and_restoration() {
    let conn = archive();
    apply(
        &conn,
        &[
            assertion("family", "first", AssertionKind::Note, &[]),
            assertion(
                "family",
                "left",
                AssertionKind::Note,
                &[(LinkRole::Revises, "assertion:family@first")],
            ),
            assertion(
                "family",
                "right",
                AssertionKind::Note,
                &[(LinkRole::Revises, "assertion:family@first")],
            ),
            assertion("other", "other", AssertionKind::Note, &[]),
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
        &[assertion(
            "withdraw",
            "withdraw",
            AssertionKind::Withdrawal,
            &[
                (LinkRole::Withdraws, "assertion:family@left"),
                (LinkRole::Withdraws, "assertion:family@right"),
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
        &[assertion(
            "restore",
            "restore",
            AssertionKind::Withdrawal,
            &[(LinkRole::Withdraws, "assertion:withdraw")],
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
fn an_annotation_applied_before_its_rubric_takes_the_rubric_it_was_derived_from() {
    let conn = archive();
    let mut annotation = assertion(
        "answer",
        "answer",
        AssertionKind::Annotation,
        &[
            (LinkRole::About, "message:codex:row"),
            (LinkRole::DerivedFrom, "assertion:rubric@rubric"),
        ],
    );
    annotation.key = Some("clear".into());
    annotation.value = Some(serde_json::json!({"value": true}));
    apply(&conn, std::slice::from_ref(&annotation), 1).unwrap();
    let slots: i64 = conn
        .query_row(
            "SELECT count(*) FROM assertion_slot WHERE id='answer'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(slots, 1);
}
