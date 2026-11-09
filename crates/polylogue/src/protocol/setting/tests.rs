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
    assert!(serde_json::from_value::<Arguments>(serde_json::json!({"machine":true})).is_err());
    assert!(serde_json::from_value::<Arguments>(serde_json::json!({"author":"operator"})).is_err());
}
