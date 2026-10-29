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
