use super::*;

fn variant() -> Variant {
    Variant {
        id: "v1".into(),
        root: "v1".into(),
        base: None,
        source: "message:test:one:m1".into(),
        source_hash: BlobHash::of(b"source"),
        kind: VariantKind::Translation,
        from_lang: None,
        to_lang: Some("pl".into()),
        author: Author::Operator,
        via: "cli".into(),
        method: None,
        machine: "synthetic-machine".into(),
        created_at: Timestamp::from_unix_ms(0).unwrap(),
        commit_seq: 1,
    }
}

#[test]
fn absent_optional_fields_are_left_out_and_unknown_keys_are_refused() {
    let json = serde_json::to_value(variant()).unwrap();
    let mut keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "author",
            "commit_seq",
            "created_at",
            "id",
            "kind",
            "machine",
            "root",
            "source",
            "source_hash",
            "to_lang",
            "via",
        ]
    );
    assert_eq!(
        serde_json::from_value::<Variant>(json.clone()).unwrap(),
        variant()
    );
    let mut extra = json;
    extra["extra"] = true.into();
    assert!(serde_json::from_value::<Variant>(extra).is_err());
}
