use super::*;
use serde_json::json;
#[test]
fn archived_metadata_is_inert_and_links_encode_ids_once() {
    let data = json!({"date":"2024-02-29","zone":"Europe/Warsaw","day_start_minutes":240,"across_years":true,"calendar":[{"date":"2024-02-29","records":2}],"activity":{"rows":[{"date":"2024-02-29","kind":"message","id":"codex:a/b%λ","label":"<script>","at":null,"time_source":"provider <time>"}],"next":"opaque+cursor"}});
    let html = render(&data);
    assert!(html.contains("/m/codex%3Aa%2Fb%25%CE%BB"));
    assert!(html.contains("&lt;script&gt;"));
    assert!(!html.contains("<script>"));
    assert!(html.contains("Time unknown"));
    assert!(html.contains("Use February 28"));
    assert!(html.contains("across_years=true&amp;undated=false&amp;cursor=opaque%2Bcursor"));
    assert!(html.contains("<caption>Messages, events and note revisions"));
}
