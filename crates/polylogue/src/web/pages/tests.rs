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
    let page = cost("repo:<script>", &report);
    assert!(page.contains("$1.25 known + unknown response usage"));
    assert!(page.contains("unknown responses in 2 sessions"));
    assert!(page.contains("40 known + unknown response usage"));
    assert!(!page.contains("repo:<script>"));
    assert!(page.contains("repo:&lt;script&gt;"));
}

#[test]
fn incomplete_parents_keep_issues_evidence_and_session_navigation() {
    let status = json!({"complete": false, "issues": [{"kind": "missing_anchor", "parent": "codex:p", "child": "codex:c", "anchor": "<unknown>"}]});
    let links = json!({"columns": [{"name":"parent"}, {"name":"child"}, {"name":"evidence"}, {"name":"pending"}, {"name":"branch_point"}], "rows": [{"parent":"codex:p", "child":"codex:c", "evidence":"native.<field>", "pending":true, "branch_point":null}], "next":"more"});
    let page = parents(&status, &links);
    assert!(page.contains("Some parent sessions are missing"));
    assert!(page.contains("missing_anchor"));
    assert!(page.contains("href=\"/s/codex%3Ap\""), "{page}");
    assert!(page.contains("href=\"/s/codex%3Ac\""), "{page}");
    assert!(page.contains("native.&lt;field&gt;"), "{page}");
    assert!(!page.contains("Only the first 100 links"));
    assert!(!page.contains("<unknown>"));
}

#[test]
fn identical_cell_text_links_only_by_its_column_reference() {
    let id = "same:opaque";
    let data = json!({"columns":[
        {"name":"session","kind":"text","reference":"session"},
        {"name":"document","kind":"text","reference":"document"},
        {"name":"message","kind":"text","reference":"message"},
        {"name":"note","kind":"text","reference":"note"},
        {"name":"id","kind":"text"}],
        "rows":[{"session":id,"document":id,"message":id,"note":id,"id":id}],"count":1});
    let page = results_table(&data);
    assert_eq!(page.matches("href=\"/s/same%3Aopaque\"").count(), 1);
    assert_eq!(page.matches("href=\"/d/same%3Aopaque\"").count(), 1);
    assert_eq!(page.matches("<td>same:opaque</td>").count(), 1);
    assert_eq!(page.matches("href=\"/m/same%3Aopaque\"").count(), 1);
    assert_eq!(page.matches("href=\"/n/same%3Aopaque\"").count(), 1);
    assert!(!page.contains("dedicated readers are unavailable"));
    let old = json!({"unit":"sessions","shape":"rows","columns":[{"name":"id","kind":"text"}],"rows":[{"id":id}],"count":1});
    assert!(
        !results_table(&old).contains("href="),
        "columns without references must not be guessed"
    );
}
