use super::*;
fn variant(original: &str, alignment: Vec<TextAlignment>) -> VariantText {
    VariantText {
        mark: VariantMark {
            reference: "variant:synthetic@revision".into(),
            key: "en".into(),
            original: original.into(),
            stale: false,
            show_original: None,
        },
        relations: vec![VariantRelation::Translates],
        text: "replacement".into(),
        alignment,
    }
}
fn source(n: u64) -> Source {
    Source {
        block: Some(n),
        ..Source::record("document:test:document")
    }
}
#[test]
fn document_body_ranges_are_simultaneous_and_inherited_identity_is_unchanged() {
    let key = Reference::Document(crate::DocumentId::parse("test:document").unwrap());
    let mut map = PresentationMap::default();
    map.insert(
        key.clone(),
        variant(
            &key.to_string(),
            vec![
                TextAlignment {
                    source: source(2),
                    start: 1,
                    end: 3,
                },
                TextAlignment {
                    source: source(5),
                    start: 0,
                    end: 1,
                },
            ],
        ),
    )
    .unwrap();
    let mut inherited = source(2);
    inherited.inherited = true;
    assert_eq!(
        map.apply(&inherited, "aλz", &ReadOptions::default())
            .unwrap(),
        "areplacementz"
    );
    assert_eq!(
        map.apply(&source(5), "xy", &ReadOptions::default())
            .unwrap(),
        "y"
    );
    assert_eq!(
        map.apply(&source(1), "uncovered", &ReadOptions::default())
            .unwrap(),
        "uncovered"
    );
    let wire = serde_json::to_string(&map).unwrap();
    assert_eq!(serde_json::from_str::<PresentationMap>(&wire).unwrap(), map);
}
#[test]
fn conflicting_replacements_and_duplicate_json_keys_are_rejected() {
    let first = Reference::parse("block:test:session:message#0").unwrap();
    let mut map = PresentationMap::default();
    let value = variant(&first.to_string(), vec![]);
    map.insert(first.clone(), value.clone()).unwrap();
    assert!(map.insert(first.clone(), value.clone()).is_err());
    let key = serde_json::to_string(&first).unwrap();
    let value = serde_json::to_string(&value).unwrap();
    assert!(
        serde_json::from_str::<PresentationMap>(&format!("{{{key}:{value},{key}:{value}}}"))
            .is_err()
    );
    let other = Reference::parse("message:test:session:message").unwrap();
    assert!(
        map.insert(
            other.clone(),
            variant(
                &other.to_string(),
                vec![TextAlignment {
                    source: Source {
                        block: Some(0),
                        ..Source::record(first.to_string())
                    },
                    start: 1,
                    end: 2
                }]
            )
        )
        .is_err()
    );
}
#[test]
fn utf8_boundaries_are_checked_before_any_replacement() {
    let key = Reference::parse("message:test:session:message").unwrap();
    let mut map = PresentationMap::default();
    map.insert(
        key.clone(),
        variant(
            &key.to_string(),
            vec![TextAlignment {
                source: source(0),
                start: 1,
                end: 2,
            }],
        ),
    )
    .unwrap();
    assert!(
        map.apply(&source(0), "λx", &ReadOptions::default())
            .is_err()
    );
    assert!(map.apply(&source(0), "", &ReadOptions::default()).is_err());
}

#[test]
fn mixed_declared_relations_keep_nonempty_rendition_and_all_omits_remove_it() {
    let key = Reference::parse("block:test:session:message#0").unwrap();
    let mut mixed = variant(&key.to_string(), vec![]);
    mixed.relations = vec![VariantRelation::Omits, VariantRelation::Expands];
    let mut map = PresentationMap::default();
    map.insert(key.clone(), mixed.clone()).unwrap();
    assert_eq!(
        map.apply(
            &Source::record(key.to_string()),
            "original",
            &ReadOptions::default()
        )
        .unwrap(),
        "replacement"
    );
    mixed.relations = vec![VariantRelation::Omits];
    let mut map = PresentationMap::default();
    map.insert(key.clone(), mixed).unwrap();
    assert_eq!(
        map.apply(
            &Source::record(key.to_string()),
            "original",
            &ReadOptions::default()
        )
        .unwrap(),
        ""
    );
}
#[test]
fn many_exact_body_renditions_use_only_their_derived_body_index() {
    let mut map = PresentationMap::default();
    for n in 0..10_000 {
        let key = Reference::parse(&format!("block:test:session:message#{n}")).unwrap();
        map.insert(key.clone(), variant(&key.to_string(), vec![]))
            .unwrap();
    }
    assert_eq!(map.1.len(), 10_000);
    for n in [0, 5_000, 9_999] {
        let source = Source::record(format!("block:test:session:message#{n}"));
        assert_eq!(map.for_source(&source).count(), 1);
        assert_eq!(
            map.apply(&source, "original", &ReadOptions::default())
                .unwrap(),
            "replacement"
        );
    }
}
