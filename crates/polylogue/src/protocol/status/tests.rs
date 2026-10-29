use super::*;
use serde_json::json;

#[test]
fn held_signal_observations_preserve_reasons_without_inventing_available_rules() {
    let failure =
        crate::daemon::service::error("not_yet_available", "owner context has not been obtained");
    let observed =
        json!({"held":[{"reason":"the operation clock is unavailable","detail":failure}]});
    let typed: Signals = serde_json::from_value(observed.clone()).unwrap();
    let roundtrip = serde_json::to_value(typed).unwrap();
    assert_eq!(roundtrip, observed);
    assert!(roundtrip.get("rules").is_none());
    assert_eq!(
        roundtrip["held"][0]["detail"]["error"]["kind"],
        "not_yet_available"
    );
}

#[test]
fn ordinary_signal_summaries_retain_held_rule_identity() {
    let mut observed = crate::signals::summary(&[], &[]);
    observed["held"] = json!([{"id":"rule-one","reason":"the index is catching up"}]);
    let typed: Signals = serde_json::from_value(observed.clone()).unwrap();
    assert_eq!(serde_json::to_value(typed).unwrap(), observed);
}
