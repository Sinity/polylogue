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
