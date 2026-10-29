use super::*;
use polylogue_model::{NewVariantPart, user::UserWrite};

fn request(id: &str, base: Option<&str>, text: &str) -> UserRequest {
    UserRequest {
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
                author: Author::Agent(None),
                via: "mcp".into(),
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
    assert!(
        matches!(crate::user_writes::apply(&tx,&stale,10),Err(StoreError::Invalid(message)) if message.contains("stale"))
    );
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
    to.execute("INSERT INTO note(id,first_note,kind,author,machine,created_at,commit_seq) VALUES('withdraw','withdraw','withdrawal','operator','machine',1,20)",[]).unwrap();
    for id in ["correction", "fork"] {
        to.execute(
            "INSERT INTO note_relation(note_id,kind,other) VALUES('withdraw','withdraws',?1)",
            [format!("variant:first@{id}")],
        )
        .unwrap();
    }
    assert_eq!(heads(&to, "first").unwrap()[0].id, "first");
    to.execute("INSERT INTO note(id,first_note,kind,author,machine,created_at,commit_seq) VALUES('restore','restore','withdrawal','operator','machine',1,21)",[]).unwrap();
    to.execute("INSERT INTO note_relation(note_id,kind,other) VALUES('restore','withdraws','note:withdraw')",[]).unwrap();
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
fn live_variant_reads_use_one_source_snapshot_and_search_the_committed_text() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let first = request("first", None, "uniqueword");
    archive.writer.user(first).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
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
    let mut bodies = Vec::new();
    reader
        .visit_variant_parts("first", |part| {
            bodies.push(part.body);
            Ok(())
        })
        .unwrap();
    assert_eq!(bodies, ["uniqueword"]);
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
}
