use super::*;
use serde_json::json;

fn answer() -> Value {
    json!({
        "outcome": "rows",
        "columns": [{"name": "id", "kind": "text"}, {"name": "cost", "kind": "money"}],
        "rows": [
            {"id": "claude-code:a", "cost": 1.5,
             "snippet": {"text": "let's use chrono, not time", "marks": [[10, 16]], "session": "s", "message": "m"}},
            {"id": "codex:b", "cost": null}
        ],
        "count": 3,
        "next": "abc",
        "notes": ["a note"],
        "incomplete": [],
    })
}

fn text(format: Format) -> String {
    let mut out = vec![];
    render(&answer(), format, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

#[test]
fn markdown_marks_matches_and_says_what_is_left() {
    let t = text(Format::Markdown);
    assert!(t.contains("| id | cost | snippet |"), "{t}");
    assert!(t.contains("let's use «chrono», not time"), "{t}");
    assert!(t.contains("| codex:b | – |"), "{t}");
    assert!(t.contains("2 of 3 rows"));
    assert!(t.contains("more: --cursor abc"));
    assert!(t.contains("a note"));
}

#[test]
fn html_escapes_and_marks() {
    let mut a = answer();
    a["rows"][1]["id"] = json!("<b>");
    let mut out = vec![];
    render(&a, Format::Html, &mut out).unwrap();
    let t = String::from_utf8(out).unwrap();
    assert!(t.contains("<mark>chrono</mark>"));
    assert!(t.contains("&lt;b&gt;"));
}

#[test]
fn terminal_bolds_matches_and_explains_empty_results() {
    let t = text(Format::Terminal);
    assert!(t.contains("chrono"));
    assert!(!t.contains("\x1b"));
    let empty = json!({
        "outcome": "empty", "columns": [], "rows": [], "count": 0, "next": null,
        "notes": [], "incomplete": [],
        "why": [{"clause": "repo:other", "matches": 0, "of": 5}]
    });
    let mut out = vec![];
    render(&empty, Format::Markdown, &mut out).unwrap();
    let t = String::from_utf8(out).unwrap();
    assert!(t.contains("no rows"));
    assert!(t.contains("repo:other → 0 of 5"), "{t}");
}

#[test]
fn json_is_the_answer_as_given() {
    let t = text(Format::Json);
    let back: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(back, answer());
}

#[test]
fn linked_html_escapes_targets_and_keeps_links_in_their_own_cells() {
    let data = json!({"columns":[{"name":"id","reference":"document"},{"name":"title"}],"rows":[{"id":"<same>","title":"<same>"}],"count":1});
    let mut out = vec![];
    linked_html(&data, &mut out, &|column, _| {
        (column["reference"] == "document").then(|| "/d/x\"<&".into())
    })
    .unwrap();
    let html = String::from_utf8(out).unwrap();
    assert!(
        html.contains("<a href=\"/d/x&quot;&lt;&amp;\">&lt;same&gt;</a>"),
        "{html}"
    );
    assert!(html.contains("<td>&lt;same&gt;</td>"), "{html}");
}

#[test]
fn row_markup_stays_data_and_link_schemes_are_checked() {
    let data = json!({"columns":[{"name":"text"}],"rows":[{"text":"<script>stored()</script> [x](javascript:foo)"}],"count":1});
    let mut out = vec![];
    render(&data, Format::Markdown, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(!text.contains("<script>"));
    assert!(text.contains("&lt;script&gt;"));
    let mut out = vec![];
    linked_html(&data, &mut out, &|_, _| Some("javascript:stored()".into())).unwrap();
    assert!(!String::from_utf8(out).unwrap().contains("href="));
}
