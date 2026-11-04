use super::*;
use polylogue_model::{NewVariantPart, user::UserWrite};

fn request(id: &str, base: Option<&str>, text: &str) -> UserRequest {
    UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Agent(None),
            via: polylogue_model::authorship::Channel::Mcp,
        },
        id: id.into(),
        machine: "machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::Variant {
            variant: NewVariant {
                base: base.map(Into::into),
                source: "message:codex:source".into(),
                source_hash: BlobHash([1; 32]),
                kind: VariantKind::Translation,
                from_lang: Some("en".into()),
                to_lang: Some("pl".into()),
                method: Some("translation".into()),
                parts: vec![NewVariantPart {
                    of: "block:codex:source#0".into(),
                    of_hash: BlobHash([2; 32]),
                    relation: VariantRelation::Translates,
                    body: text.into(),
                }],
            },
        },
    }
}
fn write_request(conn: &mut Connection, r: &UserRequest, counter: i64) {
    let tx = conn.transaction().unwrap();
    crate::user_writes::apply(&tx, r, counter).unwrap();
    tx.commit().unwrap();
}
fn archive() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::ensure_archive(&conn).unwrap();
    conn
}
#[test]
fn variant_revisions_use_graph_currency_and_preserve_exact_parts() {
    let mut conn = archive();
    let first = request("first", None, "Pierwszy");
    write_request(&mut conn, &first, 9);
    let correction = request("correction", Some("first"), "Poprawiony");
    write_request(&mut conn, &correction, 2);
    assert_eq!(
        heads(&conn, "first")
            .unwrap()
            .iter()
            .map(|v| v.id.as_str())
            .collect::<Vec<_>>(),
        vec!["correction"]
    );
    let mut body = String::new();
    rows::parts(&conn, "correction", |part| {
        assert_eq!(part.n, 0);
        assert_eq!(part.of_hash, BlobHash([2; 32]));
        body = part.body;
        Ok(())
    })
    .unwrap();
    assert_eq!(body, "Poprawiony");
    let stale = request("stale", Some("first"), "Stary");
    let tx = conn.transaction().unwrap();
    assert!(matches!(
        crate::user_writes::apply(&tx, &stale, 10),
        Err(StoreError::Stale { .. })
    ));
    drop(tx);
    assert!(rows::get(&conn, "stale").unwrap().is_none());
}
#[test]
fn variant_frozen_application_preserves_forks_and_restores_withdrawn_base() {
    let mut from = archive();
    let first = request("first", None, "Pierwszy");
    write_request(&mut from, &first, 1);
    let first_effect = capture(&from, &first, 1).unwrap();
    let correction = request("correction", Some("first"), "Poprawiony");
    write_request(&mut from, &correction, 2);
    let mut effect = capture(&from, &correction, 2).unwrap();
    let mut fork = effect.variants[0].clone();
    fork.id = "fork".into();
    effect.variants.push(fork);
    let mut part = effect.variant_parts[0].clone();
    part.variant = "fork".into();
    effect.variant_parts.push(part);
    let to = archive();
    apply(&to, &first_effect, 10).unwrap();
    apply(&to, &effect, 4).unwrap();
    assert_eq!(heads(&to, "first").unwrap().len(), 2);
    let withdraw = |id: &str, targets: &[&str], counter| {
        use polylogue_model::{
            Author, Reference,
            assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole},
        };
        let fact = AssertionFact {
            id: id.into(),
            root: id.into(),
            kind: AssertionKind::Withdrawal,
            key: None,
            value: None,
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
            machine: "machine".into(),
            created_at: Timestamp::from_unix_ms(1).unwrap(),
            valid_until: None,
            links: targets
                .iter()
                .map(|reference| AssertionLink {
                    role: LinkRole::Withdraws,
                    reference: Reference::parse(reference).unwrap(),
                    seen_hash: None,
                    quote: None,
                })
                .collect(),
        };
        crate::assertions::apply(&to, &[fact], counter).unwrap();
    };
    withdraw(
        "withdraw",
        &["variant:first@correction", "variant:first@fork"],
        20,
    );
    assert_eq!(heads(&to, "first").unwrap()[0].id, "first");
    withdraw("restore", &["assertion:withdraw@withdraw"], 21);
    assert_eq!(heads(&to, "first").unwrap().len(), 2);
    let scratch = archive();
    copy(&to, &scratch).unwrap();
    assert_eq!(rows::get(&scratch, "fork").unwrap().unwrap().commit_seq, 0);
}
#[test]
fn variant_source_binding_cannot_be_changed_by_a_revision() {
    let mut conn = archive();
    write_request(&mut conn, &request("first", None, "Text"), 1);
    let mut changed = request("changed", Some("first"), "Text");
    let UserWrite::Variant { variant } = &mut changed.write else {
        unreachable!()
    };
    variant.source_hash = BlobHash([3; 32]);
    let tx = conn.transaction().unwrap();
    assert!(matches!(
        crate::user_writes::apply(&tx, &changed, 2),
        Err(StoreError::Invalid(_))
    ));
}

#[test]
fn variant_text_is_a_separate_derived_index_and_tracks_canonical_lifecycle() {
    let mut ledger = archive();
    let request = request("first", None, "Tłumaczenie uniquely searchable");
    write_request(&mut ledger, &request, 1);
    let effect = capture(&ledger, &request, 1).unwrap();
    let index = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    index
        .execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    apply(&index, &effect, 7).unwrap();
    project(&index).unwrap();
    project(&index).unwrap();
    let count: i64 = index
        .query_row(
            "SELECT count(*) FROM variant_fts WHERE variant_fts MATCH 'uniquely'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM variant_text", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    index
        .execute("DELETE FROM source_variant_part", [])
        .unwrap();
    index.execute("DELETE FROM source_variant", []).unwrap();
    project(&index).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM variant_fts WHERE variant_fts MATCH 'uniquely'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn variant_restoration_is_exact_idempotent_and_refuses_changed_facts() {
    let mut original = archive();
    let mut request = request("first", None, "Original");
    request.at = Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z").unwrap();
    write_request(&mut original, &request, 1);
    let effect = capture(&original, &request, 1).unwrap();
    assert_eq!(effect.variants[0].created_at, request.at);
    let restored = archive();
    apply(&restored, &effect, 10).unwrap();
    apply(&restored, &effect, 11).unwrap();
    let mut expected = effect.variants[0].clone();
    expected.commit_seq = 10;
    assert_eq!(rows::get(&restored, "first").unwrap(), Some(expected));
    let mut changed = effect.clone();
    changed.variant_parts[0].body = "Changed".into();
    assert!(matches!(
        apply(&restored, &changed, 12),
        Err(StoreError::Conflict(_))
    ));
    let mut body = String::new();
    rows::parts(&restored, "first", |part| {
        body = part.body;
        Ok(())
    })
    .unwrap();
    assert_eq!(body, "Original");
}

#[test]
fn live_variant_reads_use_one_source_snapshot_and_search_the_committed_text() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let before = pool.reader().unwrap();
    assert!(!before.has_variant_source("message:codex:source").unwrap());
    let mut first = request("first", None, "uniqueword");
    let UserWrite::Variant { variant } = &mut first.write else {
        unreachable!()
    };
    variant.parts.extend([
        NewVariantPart {
            of: "block:codex:source#1".into(),
            of_hash: BlobHash([4; 32]),
            relation: VariantRelation::Translates,
            body: "different source".into(),
        },
        NewVariantPart {
            of: "block:codex:source#0".into(),
            of_hash: BlobHash([2; 32]),
            relation: VariantRelation::Translates,
            body: "second rendition part".into(),
        },
    ]);
    archive.writer.user(first).unwrap();
    assert!(!before.has_variant_source("message:codex:source").unwrap());
    drop(before);
    let reader = pool.reader().unwrap();
    assert!(reader.has_variant_source("message:codex:source").unwrap());
    assert!(
        !reader
            .has_variant_source("message:codex:unrelated")
            .unwrap()
    );
    let variant = reader.variant("first").unwrap().unwrap();
    assert_eq!(variant.author, Author::Agent(None));
    assert_eq!(variant.via, "mcp");
    let enabled = vec!["translation:pl".into()];
    assert_eq!(
        reader
            .enabled_variant("message:codex:source", BlobHash([1; 32]), &enabled)
            .unwrap()
            .unwrap()
            .id,
        "first"
    );
    assert!(
        reader
            .enabled_variant("message:codex:source", BlobHash([9; 32]), &enabled)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        reader
            .selected_variant("message:codex:source", BlobHash([9; 32]), &enabled, true)
            .unwrap()
            .unwrap()
            .id,
        "first",
    );
    let mut bodies = Vec::new();
    reader
        .visit_variant_parts("first", |part| {
            bodies.push(part.body);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        bodies,
        ["uniqueword", "different source", "second rendition part"]
    );
    let mut selected = Vec::new();
    reader
        .visit_variant_parts_for("first", "block:codex:source#0", |part| {
            selected.push((part.n, part.body));
            Ok(())
        })
        .unwrap();
    assert_eq!(
        selected,
        vec![
            (0, "uniqueword".into()),
            (2, "second rendition part".into())
        ]
    );
    reader
        .visit_variant_parts_for("first", "block:codex:source#9", |_| {
            panic!("an absent exact source address matched another block")
        })
        .unwrap();
    let mut hits = Vec::new();
    reader
        .visit_variant_search("uniqueword", Some("pl"), |hit| {
            hits.push(hit.part.body);
            Ok(())
        })
        .unwrap();
    assert_eq!(hits, ["uniqueword"]);
    let mut hits = 0;
    reader
        .visit_variant_search("uniqueword", Some("en"), |_| {
            hits += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(hits, 0);
    drop(reader);
    archive
        .writer
        .user(request("second-root", None, "otherword"))
        .unwrap();
    let reader = pool.reader().unwrap();
    assert!(matches!(
        reader.enabled_variant("message:codex:source", BlobHash([1; 32]), &enabled),
        Err(StoreError::Ambiguous { .. })
    ));
    assert!(matches!(
        reader.selected_variant("message:codex:source", BlobHash([9; 32]), &enabled, true),
        Err(StoreError::Ambiguous { .. })
    ));
    drop(reader);
    let mut fresh = request("fresh-root", None, "freshword");
    let UserWrite::Variant { variant } = &mut fresh.write else {
        unreachable!()
    };
    variant.source_hash = BlobHash([9; 32]);
    archive.writer.user(fresh).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .selected_variant("message:codex:source", BlobHash([9; 32]), &enabled, true)
            .unwrap()
            .unwrap()
            .id,
        "fresh-root"
    );
}

#[test]
fn variant_ingress_keeps_original_author_and_first_admitted_time_separate_from_claims() {
    use polylogue_model::{Reference, authored::ImportAttribution};
    let mut origin = archive();
    let request = request("first", None, "Original");
    write_request(&mut origin, &request, 1);
    let effect = capture(&origin, &request, 1).unwrap();
    let target = archive();
    let current = ImportAttribution {
        machine: "receiver".into(),
        input: "local-bundle".into(),
        chain: 1,
        actor: Author::Operator,
        at: Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z").unwrap(),
    };
    restore(&target, &effect, &current, 2).unwrap();
    let mut rollback = current.clone();
    rollback.at = Timestamp::from_unix_ms(0).unwrap();
    restore(&target, &effect, &rollback, 3).unwrap();
    let reference = Reference::parse("variant:first@first").unwrap();
    let part = Reference::parse("variant:first@first#0").unwrap();
    let claim = ImportAttribution {
        machine: "sender".into(),
        input: "declared-bundle".into(),
        chain: 2,
        actor: Author::Script("importer.service".into()),
        at: Timestamp::from_unix_ms(1).unwrap(),
    };
    declare_imports(&target, &reference, std::slice::from_ref(&claim), 4).unwrap();
    let history = imports::history(&target, &reference).unwrap().unwrap();
    assert_eq!(history.observed.as_slice(), std::slice::from_ref(&current));
    assert_eq!(history.declared, [claim]);
    assert_eq!(
        imports::history(&target, &part).unwrap().unwrap().observed,
        [current]
    );
    let original = rows::get(&target, "first").unwrap().unwrap();
    assert_eq!(original.author, Author::Agent(None));
    assert_eq!(original.machine, "machine");
    let copied = archive();
    copy(&target, &copied).unwrap();
    assert_eq!(
        imports::history(&copied, &reference).unwrap().unwrap(),
        history
    );
}

#[test]
fn rendition_early_stop_precedes_later_body_decode() {
    let mut conn = archive();
    let mut request = request("early-stop", None, "first");
    let UserWrite::Variant { variant } = &mut request.write else {
        unreachable!()
    };
    variant.parts.push(NewVariantPart {
        of: "block:codex:source#1".into(),
        of_hash: BlobHash([3; 32]),
        relation: VariantRelation::Translates,
        body: "later".into(),
    });
    write_request(&mut conn, &request, 10);
    conn.execute(
        "UPDATE variant_part SET body=?1 WHERE variant='early-stop' AND n=1",
        [vec![255u8]],
    )
    .unwrap();
    let mut numbers = Vec::new();
    rows::parts_while(&conn, "early-stop", |part| {
        numbers.push(part.n);
        Ok(false)
    })
    .unwrap();
    assert_eq!(numbers, vec![0]);
    assert!(rows::parts(&conn, "early-stop", |_| Ok(())).is_err());
}
