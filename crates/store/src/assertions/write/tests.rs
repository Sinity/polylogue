use super::*;
use polylogue_model::Json;
use polylogue_model::authorship::{Authorship, Channel};
use polylogue_model::user::UserWrite;
use serde_json::json;

fn db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();

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
        write: UserWrite::Assert { assertions: vec![] },
    }
}
fn statement(
    kind: AssertionKind,
    key: Option<&str>,
    value: Json,
    target: Option<&str>,
) -> NewAssertion {
    NewAssertion {
        kind,
        key: key.map(str::to_owned),
        value: Some(value),
        valid_until: None,
        links: target
            .into_iter()
            .map(|target| NewAssertionLink {
                role: LinkRole::About,
                reference: reference(target),
            })
            .collect(),
    }
}
fn write_one(conn: &Connection, id: &str, assertion: NewAssertion, seq: i64) -> UserOutcome {
    write(
        conn,
        &request(id),
        &[assertion],
        seq,
        &|_| Ok(Some(BlobHash([5; 32]))),
        Owner::Standalone,
    )
    .unwrap()
}
#[test]
fn cardinality_and_stale_base_use_graph_after_each_overlay_change() {
    let conn = db();
    let title = |value: &str| {
        statement(
            AssertionKind::Title,
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
                && link.reference == reference("assertion:first@first"))
    );
    let stale = revise(
        &conn,
        &request("third"),
        &[RevisionReference {
            root: "first".into(),
            revision: Some("first".into()),
        }],
        &AssertionPatch::default(),
        3,
        &|_| Ok(None),
        Owner::Standalone,
    );
    assert!(
        matches!(stale,Err(StoreError::Stale {current,..}) if current==vec!["assertion:first@second"])
    );
    let withdrawal = NewAssertion {
        kind: AssertionKind::Withdrawal,
        key: None,
        value: None,
        valid_until: None,
        links: vec![NewAssertionLink {
            role: LinkRole::Withdraws,
            reference: reference("assertion:first@second"),
        }],
    };
    write_one(&conn, "withdraw", withdrawal, 3);
    assert_eq!(
        heads(&conn, &reference("assertion:first")).unwrap(),
        vec![reference("assertion:first@first")]
    );
    revise(
        &conn,
        &request("restored-base"),
        &[RevisionReference {
            root: "first".into(),
            revision: Some("first".into()),
        }],
        &AssertionPatch {
            value: Change::Set(json!("from exposed older base")),
            ..Default::default()
        },
        4,
        &|_| Ok(None),
        Owner::Standalone,
    )
    .unwrap();
    assert_eq!(
        heads(&conn, &reference("assertion:first")).unwrap(),
        vec![reference("assertion:first@restored-base")]
    );
}
#[test]
fn duplicate_tag_uses_mathematical_value_and_keeps_original_author_and_hash() {
    let conn = db();
    let first = statement(
        AssertionKind::Tag,
        Some("priority"),
        json!({"value":1,"confidence":0.5}),
        Some("session:codex:subject"),
    );
    write_one(&conn, "one", first, 1);
    let second = statement(
        AssertionKind::Tag,
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
        Owner::Standalone,
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
fn frozen_apply_can_conserve_forks_and_the_next_live_write_resolves_them() {
    let conn = db();
    let title = statement(
        AssertionKind::Title,
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
    // (docs/assertions.md, several families in one slot).
    write(
        &conn,
        &request("three"),
        &[title],
        3,
        &|_| Ok(None),
        Owner::Standalone,
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
    assert_eq!(revised, ["assertion:one@one", "assertion:other@other"]);
}
#[test]
fn native_fact_preserves_captured_provenance_and_original_waiting_hash() {
    let conn = db();
    let note = statement(
        AssertionKind::Note,
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
        Owner::Standalone,
    )
    .unwrap();
    let fact = get(&conn, "waiting").unwrap().unwrap().fact;
    assert_eq!(fact.via, Channel::Cli);
    assert_eq!(fact.machine, "accepting-machine");
    // Stored to the millisecond (`created_at INTEGER`, docs/assertions.md).
    assert_eq!(
        fact.created_at,
        Timestamp::from_unix_ms(request("waiting").at.unix_ms())
    );
    assert_eq!(fact.links[0].seen_hash, None);
    apply(&conn, std::slice::from_ref(&fact), 20).unwrap();
    assert_eq!(get(&conn, "waiting").unwrap().unwrap().fact, fact);
}
#[test]
fn annotation_uses_exact_rubric_and_same_author_pass_before_freezing() {
    let conn = db();
    let rubric = statement(
        AssertionKind::Rubric,
        Some("quality"),
        json!({"instructions":"classify","unit_query":"messages","fields":[{"name":"grade","value_type":"integer","minimum":1,"maximum":3}],"abstain":true,"confidence":false,"rationale":false}),
        None,
    );
    write_one(&conn, "rubric", rubric, 1);
    let pass = statement(
        AssertionKind::Pass,
        None,
        json!({"rubric":"assertion:rubric@rubric","population":null,"counts":{}}),
        None,
    );
    write_one(&conn, "pass", pass, 2);
    let mut annotation = statement(
        AssertionKind::Annotation,
        Some("grade"),
        json!({"value":2}),
        Some("message:codex:subject"),
    );
    annotation.links.extend([
        NewAssertionLink {
            role: LinkRole::DerivedFrom,
            reference: reference("assertion:rubric@rubric"),
        },
        NewAssertionLink {
            role: LinkRole::Origin,
            reference: reference("assertion:pass@pass"),
        },
    ]);
    write_one(&conn, "allowed", annotation.clone(), 3);
    annotation.value = Some(json!({"value":4}));
    assert!(
        matches!(write(&conn,&request("refused"),&[annotation.clone()],4,&|_| Ok(None),Owner::Standalone),Err(StoreError::Invalid(message)) if message.contains("not_allowed_by_rubric"))
    );
    annotation.value = Some(json!({"value":2,"confidence":0.5}));
    assert!(
        write(
            &conn,
            &request("confidence"),
            &[annotation],
            4,
            &|_| Ok(None),
            Owner::Standalone
        )
        .is_err()
    );
    assert!(get(&conn, "refused").unwrap().is_none());
}

#[test]
fn bulk_tag_keeps_existing_target_and_writes_each_untagged_target() {
    let conn = db();
    let tag = statement(
        AssertionKind::Tag,
        Some("research"),
        json!({}),
        Some("session:codex:one"),
    );
    write_one(&conn, "first", tag.clone(), 1);
    let mut bulk = tag;
    bulk.links.push(NewAssertionLink {
        role: LinkRole::About,
        reference: reference("session:codex:two"),
    });
    let outcome = write(
        &conn,
        &request("bulk"),
        &[bulk],
        2,
        &|_| Ok(None),
        Owner::Standalone,
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
            AssertionKind::Note,
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
        fork.links.push(AssertionLink {
            role: LinkRole::Revises,
            reference: reference("assertion:first@first"),
            seen_hash: Some(BlobHash([4; 32])),
            quote: None,
        });
        apply(&conn, &[fork], 2).unwrap();
    }
    assert_eq!(
        heads(&conn, &reference("assertion:first")).unwrap().len(),
        2
    );
    for (id, seq) in [("withdraw-root", 3), ("already-withdrawn", 4)] {
        let mut r = request(id);
        r.write = UserWrite::WithdrawAssertions {
            targets: vec![reference("assertion:first")],
            reason: None,
        };
        let tx = conn.transaction().unwrap();
        crate::user_writes::apply(&tx, &r, seq).unwrap();
        let prepared = crate::primary::authored::known_request(&tx, &r)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.effect.assertions.len(), 1);
        assert_eq!(
            prepared.effect.assertions[0].links[0].reference,
            reference("assertion:first")
        );
        assert_eq!(prepared.effect.assertions[0].links[0].seen_hash, None);
        tx.commit().unwrap();
        assert!(
            heads(&conn, &reference("assertion:first"))
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
            AssertionKind::Note,
            None,
            json!("text"),
            Some("session:codex:subject"),
        ),
        1,
    );
    let withdrawal = |target: &str| NewAssertion {
        kind: AssertionKind::Withdrawal,
        key: None,
        value: None,
        valid_until: None,
        links: vec![NewAssertionLink {
            role: LinkRole::Withdraws,
            reference: reference(target),
        }],
    };
    assert!(matches!(
        write(
            &conn,
            &request("missing"),
            &[withdrawal("assertion:missing")],
            2,
            &|_| panic!("identity-only withdrawal requested text"),
            Owner::Standalone
        ),
        Err(StoreError::NotFound(_))
    ));
    assert!(get(&conn, "missing").unwrap().is_none());
    write(
        &conn,
        &request("withdrawal"),
        &[withdrawal("assertion:first")],
        2,
        &|_| panic!("identity-only withdrawal requested text"),
        Owner::Standalone,
    )
    .unwrap();
    assert!(
        heads(&conn, &reference("assertion:first"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn unavailable_text_link_does_not_freeze_an_unbound_fact() {
    let conn = db();
    let assertion = statement(
        AssertionKind::Note,
        None,
        json!("text"),
        Some("message:codex:unavailable"),
    );
    let result = write(
        &conn,
        &request("unavailable"),
        &[assertion],
        1,
        &|target| {
            Err(StoreError::ReferenceUnavailable {
                reference: target.to_string(),
                reason: "source unavailable".into(),
            })
        },
        Owner::Standalone,
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
    let imported = AssertionFact::from_request(
        "imported".into(),
        "imported".into(),
        NewAssertion {
            kind: AssertionKind::Withdrawal,
            key: None,
            value: None,
            valid_until: None,
            links: vec![],
        },
        vec![AssertionLink {
            role: LinkRole::Withdraws,
            reference: reference("assertion:native"),
            seen_hash: None,
            quote: None,
        }],
        &request("imported"),
    );
    apply(&conn, std::slice::from_ref(&imported), 1).unwrap();
    {
        let tx = conn.transaction().unwrap();
        let mut native = request("native");
        native.write = UserWrite::WithdrawAssertions {
            targets: vec![reference("assertion:imported")],
            reason: None,
        };
        assert!(
            matches!(crate::user_writes::apply(&tx, &native, 2), Err(StoreError::Invalid(message)) if message.contains("form a cycle"))
        );
        // The accepting writer rolls back the request and every proposed fact.
    }
    assert!(get(&conn, "native").unwrap().is_none());
    assert_eq!(
        get(&conn, "imported").unwrap().unwrap().fact,
        canonical_fact(&imported).unwrap()
    );
    assert!(
        !conn
            .prepare("SELECT 1 FROM client_request WHERE id='native'")
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
        AssertionKind::Rubric,
        Some("synthetic-rubric"),
        json!({"instructions":"Classify", "unit_query":"sessions", "fields":[{"name":"category","value_type":"string"}], "abstain":false,"confidence":false,"rationale":false}),
        None,
    );
    let mut original =
        AssertionFact::from_request("undated".into(), "undated".into(), statement, vec![], &req);
    original.created_at = None;
    apply(&conn, std::slice::from_ref(&original), 1).unwrap();
    let patch = AssertionPatch {
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
        Owner::Standalone,
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
        crate::assertions::currency::stored_heads(&conn, "undated").unwrap(),
        [Reference::Assertion(RevisionReference::exact(
            "undated".into(),
            outcome.id
        ))]
    );
}
