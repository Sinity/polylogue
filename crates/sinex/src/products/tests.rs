use super::*;
#[test]
fn restored_assertion_keys_retain_each_physical_source_without_receipt_clock_order() {
    let fact = serde_json::from_value(json!({
        "id":"synthetic","root":"synthetic","kind":"note","key":null,
        "value":"Synthetic text", "author":"unattributed","via":"import","machine":"host",
        "created_at":null,"valid_until":null,"links":[]
    }))
    .unwrap();
    let ingress = |input: &str, at: i64| polylogue_model::authored::ImportAttribution {
        input: input.into(),
        chain: 1,
        machine: "receiving-host".into(),
        actor: polylogue_model::Author::Unattributed,
        at: polylogue_model::Timestamp::from_unix_ms(at).unwrap(),
    };
    let first = Product::RestoredAssertion {
        fact,
        import: Some(ingress("one", 1)),
    };
    let mut second = first.clone();
    if let Product::RestoredAssertion { import, .. } = &mut second {
        *import = Some(ingress("two", 2));
    }
    assert_ne!(key(&first, 0).unwrap(), key(&second, 0).unwrap());
    if let Product::RestoredAssertion { import, .. } = &mut second {
        *import = Some(ingress("one", 3));
    }
    assert_eq!(key(&first, 0).unwrap(), key(&second, 0).unwrap());
}
