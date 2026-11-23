use super::*;
use serde_json::json;

fn answer() -> Value {
    json!({
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
    assert_eq!(
        text(Format::Markdown),
        "| id | cost | snippet |\n\
         |---|---|---|\n\
         | claude-code:a | $1.50 | let's use «chrono», not time |\n\
         | codex:b | – |  |\n\
         2 of 3 rows\n\
         more: --cursor abc\n\
         a note\n"
    );
}

#[test]
fn html_escapes_and_marks() {
    let mut a = answer();
    a["rows"][1]["id"] = json!("<b>");
    let mut out = vec![];
    render(&a, Format::Html, &mut out).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "<table>\n\
         <thead><tr><th>id</th><th>cost</th><th>snippet</th></tr></thead><tbody>\n\
         <tr><td>claude-code:a</td><td>$1.50</td><td>let's use <mark>chrono</mark>, not time</td></tr>\n\
         <tr><td>&lt;b&gt;</td><td>–</td><td></td></tr>\n\
         </tbody></table>\n\
         <p>2 of 3 rows</p>\n\
         <p>more: --cursor abc</p>\n\
         <p>a note</p>\n"
    );
}

#[test]
fn terminal_numbers_rows_and_indents_the_snippet() {
    assert_eq!(
        text(Format::Terminal),
        "    id             cost\n\
         @1  claude-code:a  $1.50\n\
         \x20   let's use chrono, not time\n\
         @2  codex:b        –\n\
         2 of 3 rows\n\
         more: --cursor abc\n\
         a note\n"
    );
}

#[test]
fn terminal_with_color_bolds_the_match() {
    let mut out = vec![];
    terminal_with_color(&answer(), true, &mut out).unwrap();
    let t = String::from_utf8(out).unwrap();
    assert!(
        t.contains("\n    let's use \x1b[1mchrono\x1b[0m, not time\n"),
        "{t:?}"
    );
}

#[test]
fn an_empty_answer_says_no_rows_unless_a_part_failed() {
    let empty = json!({
        "columns": [], "rows": [], "count": 0, "next": null,
        "notes": [], "incomplete": [],
        "why": [{"clause": "repo:other", "matches": 0, "of": 5}]
    });
    let mut out = vec![];
    render(&empty, Format::Markdown, &mut out).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "no rows\nrepo:other → 0 of 5\n"
    );
    // Nothing found while a part failed is not an empty answer.
    let mut incomplete = empty.clone();
    incomplete["incomplete"] = json!(["vectors"]);
    let mut out = vec![];
    render(&incomplete, Format::Markdown, &mut out).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "0 rows\nvectors\nrepo:other → 0 of 5\n"
    );
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

#[test]
fn a_deferred_count_does_not_hide_returned_rows() {
    let mut answer = answer();
    answer["count"] = Value::Null;
    let mut out = Vec::new();
    render(&answer, Format::Markdown, &mut out).unwrap();
    let rendered = String::from_utf8(out).unwrap();
    assert!(
        rendered.contains("2 rows shown; count deferred"),
        "{rendered}"
    );
    assert!(!rendered.contains("no rows"), "{rendered}");
}

#[test]
fn money_cells_never_round_a_known_amount_to_zero() {
    let mut a = answer();
    a["rows"][1]["cost"] = serde_json::from_str("0.00001234").unwrap();
    for format in [Format::Markdown, Format::Terminal] {
        let mut out = vec![];
        render(&a, format, &mut out).unwrap();
        let t = String::from_utf8(out).unwrap();
        assert!(t.contains("$0.0000123"), "{t}");
    }
}
