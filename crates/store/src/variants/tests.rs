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
fn variant_revisions_follow_the_graph_and_keep_parts() {
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
fn variant_application_keeps_forks_and_restores_base() {
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
    let sql_heads = |root: &str| {
        let predicate = current_sql("v")
            .replace("source_variant", "variant")
            .replace("source_link", "link")
            .replace("source_revision_state", "revision_state");
        to.prepare(&format!(
            "SELECT v.id FROM variant v WHERE v.root=?1 AND {predicate} ORDER BY v.id"
        ))
        .unwrap()
        .query_map([root], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap()
    };
    let graph_heads = |root: &str| {
        heads(&to, root)
            .unwrap()
            .iter()
            .map(|v| v.id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(sql_heads("first"), graph_heads("first"));
    // VariantLinks scopes incoming base edges by root, including restored rows.
    let mut unrelated = effect.clone();
    unrelated.variants.truncate(1);
    unrelated.variant_parts.truncate(1);
    unrelated.variants[0].id = "unrelated".into();
    unrelated.variants[0].root = "another-root".into();
    unrelated.variant_parts[0].variant = "unrelated".into();
    apply(&to, &unrelated, 5).unwrap();
    assert_eq!(sql_heads("first"), graph_heads("first"));
    assert_eq!(sql_heads("another-root"), graph_heads("another-root"));

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
            created_at: Some(Timestamp::from_unix_ms(1).unwrap()),
            valid_until: None,
            links: targets
                .iter()
                .map(|reference| AssertionLink {
                    role: LinkRole::Withdraws,
                    reference: Reference::parse_stored(reference).unwrap(),
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
    let remaining = heads(&to, "first").unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].id, "first");
    assert_eq!(sql_heads("first"), graph_heads("first"));
    withdraw("restore", &["assertion:withdraw@withdraw"], 21);
    assert_eq!(heads(&to, "first").unwrap().len(), 2);
    assert_eq!(sql_heads("first"), graph_heads("first"));
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
fn variant_text_is_derived_and_tracks_source_lifecycle() {
    let mut standalone = archive();
    let request = request("first", None, "Tłumaczenie uniquely searchable");
    write_request(&mut standalone, &request, 1);
    let effect = capture(&standalone, &request, 1).unwrap();
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
fn variant_restoration_is_idempotent_and_refuses_changed_facts() {
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
fn copied_variant_keeps_authorship() {
    let mut origin = archive();
    let request = request("first", None, "Original");
    write_request(&mut origin, &request, 1);
    let effect = capture(&origin, &request, 1).unwrap();
    let target = archive();
    apply(&target, &effect, 2).unwrap();
    let original = rows::get(&target, "first").unwrap().unwrap();
    assert_eq!(original.author, Author::Agent(None));
    assert_eq!(original.machine, "machine");
    let copied = archive();
    copy(&target, &copied).unwrap();
    let copied = rows::get(&copied, "first").unwrap().unwrap();
    assert_eq!(
        (copied.author, copied.machine, copied.created_at),
        (original.author, original.machine, original.created_at)
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

#[test]
fn live_variant_reads_use_one_source_snapshot_and_search_the_committed_text() {
    let archive = crate::fixture::writer::WriterArchive::new();
    // A reader keeps its snapshot across a write in the installed index's
    // WAL (docs/architecture.md, Durability).
    archive.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let before = pool.reader().unwrap();
    assert!(before.variant("first").unwrap().is_none());
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
    assert!(before.variant("first").unwrap().is_none());
    drop(before);
    let reader = pool.reader().unwrap();
    assert!(reader.variant("first").unwrap().is_some());
    let variant = reader.variant("first").unwrap().unwrap();
    assert_eq!(variant.author, Author::Agent(None));
    assert_eq!(variant.via, "mcp");
    let enabled = vec!["translation:pl".into()];
    assert_eq!(
        reader
            .selected_variant("message:codex:source", BlobHash([1; 32]), &enabled, false)
            .unwrap()
            .unwrap()
            .id,
        "first"
    );
    assert!(
        reader
            .selected_variant("message:codex:source", BlobHash([9; 32]), &enabled, false)
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
        reader.selected_variant("message:codex:source", BlobHash([1; 32]), &enabled, false),
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
