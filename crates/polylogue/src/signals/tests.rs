use super::*;

#[test]
fn rule_duration_uses_shared_parser_without_overflow() {
    assert_eq!(duration("20m"), Some(1_200_000));
    assert_eq!(duration("-1m"), None);
    assert_eq!(duration("999999999999999999999d"), None);
}

#[test]
fn signal_schema_uses_common_closed_rule_and_delivery_contract() {
    let rule: shared_signals::Rule = serde_json::from_value(
        json!({"name":"test","query":"sessions","fires":{"on_change":{"what":"membership"}}}),
    )
    .unwrap();
    assert!(
        rule.validate(&shared_signals::Configuration {
            value_targets: Default::default(),
            action_commands: Default::default()
        })
        .is_ok()
    );
    assert!(
        serde_json::from_value::<shared_signals::Rule>(
            json!({"name":"forged","query":"sessions","operator":true})
        )
        .is_err()
    );
}
