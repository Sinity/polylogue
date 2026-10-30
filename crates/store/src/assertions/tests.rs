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
        via: Some(Channel::Import),
        machine: Some("synthetic".into()),
        created_at: Some(Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z").unwrap()),
        valid_until: None,
        links: links
            .iter()
            .map(|(role, reference)| AssertionLink {
                role: *role,
                reference: r(reference),
                seen_hash: role.uses_text_hash().then_some(BlobHash([3; 32])),
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
fn historical_assertion_keeps_its_recorded_kind_without_claiming_current_value_schema() {
    use polylogue_model::historical::{
        AssertionEvidence, HistoricalAssertionEvidence, NoteKind, RecordedKind, SourceSchema,
    };
    let conn = archive();
    let fact = HistoricalAssertionEvidence {
        predecessor: None,
        id: "forecast".into(),
        root: "forecast".into(),
        source: SourceSchema::PolylogueArchive { version: 4 },
        kind: RecordedKind::Known(NoteKind::Forecast),
        key: Some("weather".into()),
        value: Some("rain tomorrow".into()),
        recorded_author: Some("operator".into()),
        machine: None,
        created_at: None,
        valid_until: None,
        targets: vec!["repo:morrow".into()],
        relations: vec![],
        links: vec![AssertionLink {
            role: LinkRole::About,
            reference: r("repo:morrow"),
            seen_hash: None,
        }],
    };
    apply_historical(&conn, &fact, 7).unwrap();
    assert!(matches!(
        get(&conn, "forecast"),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    let Some(AssertionEvidence::Historical {
        fact: stored,
        commit_seq,
        imports,
    }) = get_evidence(&conn, "forecast").unwrap()
    else {
        panic!("the original historical fact must remain readable")
    };
    assert_eq!(stored, fact);
    assert_eq!(commit_seq, 7);
    assert!(imports.observed.is_empty());
    assert!(imports.declared.is_empty());
    assert_eq!(
        heads(&conn, &r("assertion:forecast")).unwrap(),
        [r("assertion:forecast@forecast")]
    );
    let mut changed = fact.clone();
    changed.value = Some("sunny tomorrow".into());
    assert!(matches!(
        apply_historical(&conn, &changed, 8),
        Err(StoreError::Conflict(_))
    ));
    let destination = archive();
    copy(&conn, &destination).unwrap();
    let Some(AssertionEvidence::Historical {
        fact: copied,
        commit_seq,
        ..
    }) = get_evidence(&destination, "forecast").unwrap()
    else {
        panic!("scratch must retain the same historical form")
    };
    assert_eq!(copied, fact);
    assert_eq!(commit_seq, 0);
    let mut duplicate = fact.clone();
    duplicate.id = "duplicate-links".into();
    duplicate.root = duplicate.id.clone();
    duplicate.links.push(duplicate.links[0].clone());
    assert!(matches!(
        apply_historical(&conn, &duplicate, 9),
        Err(StoreError::Invalid(_))
    ));
    assert_eq!(get_evidence(&conn, &duplicate.id).unwrap(), None);
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

#[test]
fn pinned_currency_relation_preserves_all_fork_heads_and_exact_status() {
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
    conn.execute_batch("BEGIN").unwrap();
    currency::prepare_root(&conn, "family").unwrap();
    let states: Vec<(String, bool, bool, bool)> = conn
        .prepare("SELECT id,current,withdrawn,conflict FROM temp.assertion_currency ORDER BY id")
        .unwrap()
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        states,
        vec![
            ("first".into(), false, false, true),
            ("left".into(), true, false, true),
            ("right".into(), true, false, true)
        ]
    );
    // Preparing a second indexed candidate cannot change the first family.
    currency::prepare_root(&conn, "other").unwrap();
    currency::prepare_root(&conn, "family").unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.assertion_currency", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        4
    );
    conn.execute_batch("ROLLBACK").unwrap();
    assert_eq!(conn.query_row("SELECT count(*) FROM sqlite_temp_schema WHERE name IN ('assertion_currency','assertion_family_state','assertion_revision_state')",[],|row|row.get::<_,i64>(0)).unwrap(),0);

    // A reused connection computes the new snapshot, including restoration.
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
    conn.execute_batch("BEGIN").unwrap();
    currency::prepare_root(&conn, "family").unwrap();
    let current: String = conn
        .query_row(
            "SELECT id FROM temp.assertion_currency WHERE current=1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(current, "first");
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM temp.assertion_currency WHERE withdrawn=1",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    conn.execute_batch("ROLLBACK").unwrap();
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
    conn.execute_batch("BEGIN").unwrap();
    currency::prepare_root(&conn, "family").unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM temp.assertion_currency WHERE current=1 AND conflict=1",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    conn.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn incomplete_currency_build_is_invisible_and_rebuilt_on_the_same_snapshot() {
    let conn = archive();
    apply(
        &conn,
        &[assertion("root", "first", AssertionKind::Note, &[])],
        1,
    )
    .unwrap();
    assert!(currency::prepare_root(&conn, "root").is_err());
    conn.execute_batch("BEGIN").unwrap();
    currency::prepare_root(&conn, "missing").unwrap();
    // Simulate interruption after inserting a revision but before its family
    // marker. The same published view is used by query and association reads.
    conn.execute(
        "INSERT INTO temp.assertion_revision_state VALUES('partial','root',1,0)",
        [],
    )
    .unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.assertion_currency", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
    currency::prepare_root(&conn, "root").unwrap();
    let id: String = conn
        .query_row("SELECT id FROM temp.assertion_currency", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(id, "first");
    assert!(
        !conn
            .query_row(
                "SELECT present FROM temp.assertion_family_state WHERE root='missing'",
                [],
                |row| row.get::<_, bool>(0)
            )
            .unwrap()
    );
    conn.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn historical_assertion_unknowns_and_provenance_survive_without_a_fabricated_receipt() {
    let conn = archive();
    let mut fact = assertion("historical", "historical", AssertionKind::Note, &[]);
    fact.via = None;
    fact.machine = None;
    fact.created_at = None;
    fact.value = Some(
        serde_json::json!({"text":"original body","predecessor":{"id":"note:old","author":"operator","machine":"claimed-host","created_at":"claimed time","extra":[1,"kept"]}}),
    );
    apply(&conn, std::slice::from_ref(&fact), 17).unwrap();
    let row = get(&conn, "historical").unwrap().unwrap();
    assert_eq!(row.fact, fact);
    assert_eq!(row.commit_seq, 17);
    assert_eq!(row.fact.text(), Some("original body"));
    let raw: (Option<String>, Option<String>, Option<String>) = conn
        .query_row(
            "SELECT via,machine,created_at FROM assertion WHERE id='historical'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(raw, (None, None, None));
    let wire = serde_json::to_value(&fact).unwrap();
    assert!(wire.get("commit_seq").is_none());
    assert_eq!(wire["created_at"], Json::Null);
    let mut changed = fact.clone();
    changed.value.as_mut().unwrap()["predecessor"]["extra"] = serde_json::json!([2, "changed"]);
    assert!(matches!(
        apply(&conn, &[changed], 18),
        Err(StoreError::Conflict(_))
    ));
    apply(&conn, std::slice::from_ref(&fact), 99).unwrap();
    assert_eq!(get(&conn, "historical").unwrap(), Some(row));
}

#[test]
fn independent_restore_preserves_declared_authorship_and_first_observed_ingress() {
    use polylogue_model::authored::ImportAttribution;
    let conn = archive();
    let mut fact = assertion("legacy", "legacy", AssertionKind::Note, &[]);
    fact.author = Author::Operator;
    fact.via = None;
    fact.machine = None;
    fact.created_at = None;
    fact.value = Some(
        serde_json::json!({"text":"historical text","predecessor":{"old_id":"note:legacy","extra":"retained"}}),
    );
    let current = ImportAttribution {
        machine: "receiver".into(),
        input: "incoming".into(),
        chain: 0,
        actor: Author::Agent(None),
        at: Timestamp::from_unix_ms(100).unwrap(),
    };
    let declared = ImportAttribution {
        machine: "sender".into(),
        input: "earlier".into(),
        chain: 1,
        actor: Author::Script("exporter".into()),
        at: Timestamp::from_unix_ms(900).unwrap(),
    };
    restore(&conn, &fact, std::slice::from_ref(&declared), &current, 20).unwrap();
    let mut rollback = current.clone();
    rollback.at = Timestamp::from_unix_ms(1).unwrap();
    restore(&conn, &fact, std::slice::from_ref(&rollback), &rollback, 30).unwrap();
    let actual = get(&conn, "legacy").unwrap().unwrap();
    assert_eq!(actual.fact, fact);
    assert_eq!(actual.commit_seq, 20);
    assert_eq!(actual.imports.observed, vec![current]);
    assert_eq!(actual.imports.declared, vec![declared, rollback]);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM client_request", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let overlay = archive();
    copy(&conn, &overlay).unwrap();
    copy(&conn, &overlay).unwrap();
    let copied = get(&overlay, "legacy").unwrap().unwrap();
    assert_eq!(copied.fact, fact);
    assert_eq!(copied.imports, actual.imports);
    assert_eq!(copied.commit_seq, 0);
    let before = conn.total_changes();
    let receiver = actual.imports.observed[0].clone();
    restore(&conn, &fact, &actual.imports.declared, &receiver, 40).unwrap();
    assert_eq!(conn.total_changes(), before);
}
