use super::*;
use serde_json::json;

#[test]
fn partial_cost_and_unknown_denominators_match_cli_accounting() {
    let report = json!({
        "messages": null, "sessions": 2,
        "tokens": {"input": {"tokens": 40, "unknown_messages": 1, "response_total_unknown": true}},
        "api_cost": {"known": {"nusd": 1_250_000_000_i64, "usd": 1.25}, "priced_messages": 1, "unpriced_messages": 1, "response_total_unknown": true},
        "reported_cost": {"known": null, "unpriced_messages": 2},
        "by_model": [], "price_catalog": "synthetic"
    });
    let page = cost("repo:<script>", &report);
    assert!(page.contains(
        "unknown responses in 2 sessions · input 40 known + unknown response usage · cached 0"
    ));
    assert!(page.contains(
        "API-equivalent $1.25 known + unknown response usage · tool-reported unknown (2 unpriced)"
    ));
    assert!(!page.contains("repo:<script>"));
    assert!(page.contains("repo:&lt;script&gt;"));
}

#[test]
fn identical_cell_text_links_only_by_its_column_reference() {
    let id = "same:opaque";
    let data = json!({"columns":[
        {"name":"session","kind":"text","reference":"session"},
        {"name":"document","kind":"text","reference":"document"},
        {"name":"message","kind":"text","reference":"message"},
        {"name":"annotation","kind":"text","reference":"annotation"},
        {"name":"id","kind":"text"}],
        "rows":[{"session":id,"document":id,"message":id,"annotation":id,"id":id}],"count":1});
    let page = results_table(&data);
    assert_eq!(page.matches("href=\"/s/same%3Aopaque\"").count(), 1);
    assert_eq!(page.matches("href=\"/d/same%3Aopaque\"").count(), 1);
    assert_eq!(page.matches("<td>same:opaque</td>").count(), 1);
    assert_eq!(page.matches("href=\"/m/same%3Aopaque\"").count(), 1);
    assert_eq!(page.matches("href=\"/a/same%3Aopaque\"").count(), 1);
    assert!(!page.contains("dedicated readers are unavailable"));
    let old = json!({"unit":"sessions","shape":"rows","columns":[{"name":"id","kind":"text"}],"rows":[{"id":id}],"count":1});
    assert!(
        !results_table(&old).contains("href="),
        "columns without references must not be guessed"
    );
}

#[test]
fn layout_uses_the_shared_frame_and_shows_direction_controls_as_text() {
    let output = layout(
        "Title\u{202e}",
        "query",
        "polylogue status",
        "<p>escaped body</p>",
    );
    assert!(output.contains("/assets/shell.js"));
    assert!(output.contains("/assets/tokens.css"));
    assert!(output.contains("/assets/theme.css"));
    assert!(!output.contains('\u{202e}'));
    assert!(output.contains("Title\\u{202e}"));
    assert!(auth().contains("/assets/auth.js"));
}

#[test]
fn the_overview_explains_unavailable_totals() {
    let page = home(&json!({"store": null}), &json!(null));
    assert!(
        page.contains("<div><dt>Archive totals</dt><dd>Archive totals are unavailable.</dd></div>")
    );
}

#[test]
fn the_overview_shows_an_unknown_status_value_as_unknown_and_zero_as_zero() {
    let status = json!({"store": {"search_pending": null, "messages": 0}});
    let page = home(&status, &json!(null));
    assert!(page.contains("<div><dt>search pending</dt><dd>unknown</dd></div>"));
    assert!(page.contains("<div><dt>messages</dt><dd>0</dd></div>"));
}

#[test]
fn the_overview_names_an_index_rebuild_and_unreadable_totals() {
    for (status, expected) in [
        (
            json!({"store":null,"index_build":{"chains":10}}),
            "The index is being rebuilt.",
        ),
        (
            json!({"store":null,"totals_error":"index is unavailable"}),
            "Totals could not be read: index is unavailable",
        ),
        (
            json!({"store":null,"warming":true}),
            "The index is catching up.",
        ),
    ] {
        let page = home(&status, &json!(null));
        assert!(page.contains(&format!(
            "<div><dt>Archive totals</dt><dd>{expected}</dd></div>"
        )));
    }
}
