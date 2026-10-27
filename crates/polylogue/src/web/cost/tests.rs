use super::*;
use serde_json::json;

#[test]
fn partial_cost_and_unknown_denominators_match_cli_accounting() {
    let report = json!({
        "messages": null, "sessions": 2,
        "tokens": {"input": {"tokens": 40, "unknown_messages": 1, "response_total_unknown": true}},
        "api_cost": {"known_usd": 1.25, "priced_messages": 1, "unpriced_messages": 1, "response_total_unknown": true},
        "reported_cost": {"known_usd": null, "unpriced_messages": 2},
        "by_model": [], "price_catalog": "synthetic"
    });
    let page = summary("repo:<script>", &report);
    assert!(page.contains("$1.25 known + unknown response usage"));
    assert!(page.contains("unknown responses in 2 sessions"));
    assert!(page.contains("40 known + unknown response usage"));
    assert!(!page.contains("repo:<script>"));
    assert!(page.contains("repo:&lt;script&gt;"));
}

#[test]
fn unavailable_usage_keeps_responses_and_has_no_fabricated_totals() {
    let data = json!({"usage":{"error":"store","message":"synthetic <failure>"},"responses":{"rows":[{"kind":"message","id":"codex:m","label":"assistant","at":null}]},"days":[]});
    let page = render("sessions", &Options::default(), &data);
    assert!(page.contains("Cost unavailable: store: synthetic &lt;failure&gt;"));
    assert!(page.contains("Browse selected sessions"));
    assert!(page.contains("/m/codex%3Am"));
    assert!(!page.contains("0 known response records"));
    assert!(!page.contains("unknown responses in 0 sessions"));
}
