use super::*;
use serde_json::json;

#[test]
fn outlook_labels_estimates_units_coverage_and_missing_configuration() {
    assert!(outlook(&json!({"reason":"plans_not_configured"})).contains("plans_not_configured"));
    assert!(outlook(&json!({"reason":"no_cycle_anchor"})).contains("cycle_anchor_day"));
    let text = outlook(
        &json!({"plan_name":"sample","projection_method":"linear","window":{"start":0,"end":86_400_000},"units":{"api_usd":{"used":1,"burn_per_day":2,"projected":3},"reported_usd":{"used":4,"burn_per_day":5,"projected":6}},"incomplete_usage":true,"quota_reason":"no_quota_configured","plan":{"price_usd":20},"coverage_ratio":0.5}),
    );
    for label in [
        "(estimate)",
        "API-equivalent",
        "tool-reported",
        "incomplete",
        "Quota not configured",
        "coverage: 0.5",
        "1970-01-01 00:00 UTC",
    ] {
        assert!(text.contains(label), "{label}: {text}");
    }
}
