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

#[test]
fn outlook_renders_unobserved_usage_and_missing_prices_as_unknown() {
    let text = outlook(
        &json!({"plan_name":"sample","projection_method":"linear","window":{"start":0,"end":86400000},"units":{},"quota_pressure":{"unit":"credits","quota":10,"used":null,"projected":null,"projected_overage_cost_usd":null,"reason":"quota_unit_unobserved"},"plan":{"price_usd":null},"coverage_ratio":0}),
    );
    assert!(text.contains("used unknown"));
    assert!(text.contains("Configured cycle price USD: unknown"));
    assert!(!text.contains("null"));
}

#[test]
fn outlook_escapes_controls_in_plan_names_units_and_reasons() {
    let data = serde_json::json!({
        "plan_name": "sample\u{1b}[2J",
        "projection_method": "linear",
        "units": {"unit\u{85}": {"used": 1}},
        "quota_pressure": {"unit": "quota\u{202e}", "reason": "reason\u{7}", "used": 1, "projected": 2, "quota": 3, "projected_overage_cost_usd": null}
    });
    let text = outlook(&data);
    assert!(text.contains("sample\\x1b[2J"));
    assert!(text.contains("unit\\x85"));
    assert!(text.contains("quota\\u{202e}"));
    assert!(text.contains("reason\\x07"));
    let missing = outlook(&serde_json::json!({"reason": "missing\u{1b}[2J"}));
    assert!(missing.contains("missing\\x1b[2J"));
    assert!(missing.contains("--option plan_name:method"));
}
