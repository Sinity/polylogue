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
    // Restore facts out of graph order; a local receipt never chooses a head.
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
