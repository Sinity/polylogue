use super::*;
#[test]
fn catalog_and_named_host_inputs_are_typed_and_closed() {
    let catalog: Arguments = serde_json::from_value(serde_json::json!({})).unwrap();
    assert!(catalog.key.is_none());
    let host: Arguments = serde_json::from_value(serde_json::json!({
        "key":"signal.quiet_hours", "machine":"named-host"
    }))
    .unwrap();
    assert_eq!(host.machine.as_deref(), Some("named-host"));
    for (input, expected) in [
        (
            serde_json::json!({"machine":true}),
            "invalid type: boolean `true`, expected a string",
        ),
        (
            serde_json::json!({"author":"operator"}),
            "unknown field `author`, expected one of `key`, `value`, `machine`",
        ),
    ] {
        assert_eq!(
            serde_json::from_value::<Arguments>(input)
                .unwrap_err()
                .to_string(),
            expected
        );
    }
}
