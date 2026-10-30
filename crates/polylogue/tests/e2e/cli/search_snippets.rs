use super::stored_text::{native, with_source};
use super::*;
use serde_json::json;

const PREFIX: &str = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty";

#[test]
fn snippets_shorten_exact_private_key_shaped_source_and_keep_match_offsets() {
    run("wholeprobe", false, false);
}
#[test]
fn snippets_do_not_rewrite_matches_after_an_earlier_private_key_header() {
    run("prefixprobe", true, false);
}
#[test]
fn output_snippets_preserve_exact_tool_result_matches() {
    run("outputprobe", true, true);
}
fn run(term: &str, earlier: bool, output: bool) {
    let text = format!(
        "-----BEGIN PRIVATE KEY----- {PREFIX} {term} -----END PRIVATE KEY----- visible Łódź"
    );
    let mut records = Vec::new();
    if earlier {
        records.push(native(
            "earlier",
            "assistant",
            json!("-----BEGIN PRIVATE KEY----- private header"),
        ));
    }
    records.push(native(
        "matched",
        "user",
        if output {
            json!([{"type":"tool_result","tool_use_id":"missing","content":text}])
        } else {
            json!(text)
        },
    ));
    with_source(&records, |fixture, _| {
        wait_status(fixture, |status| status["store"]["search_pending"] == 0);
        let query = if output {
            format!("output:{term}")
        } else {
            term.to_owned()
        };
        let result = cli(fixture, &["--json", "find", &query]);
        assert!(result.status.success(), "{result:?}");
        let result = json(&result);
        assert_eq!(result["count"], 1, "{result}");
        let snippet = &result["rows"][0]["snippet"];
        assert_eq!(snippet["session"], "claude-code:stored");
        assert_eq!(snippet["message"], "claude-code:matched");
        let shown = snippet["text"].as_str().unwrap();
        assert!(shown.contains(term), "{snippet}");
        // Only the documented edge ellipses may differ from the stored block.
        assert!(
            text.contains(shown.trim_matches('…')),
            "rewritten snippet: {snippet}"
        );
        let marks = snippet["marks"].as_array().unwrap();
        assert!(!marks.is_empty(), "{snippet}");
        for mark in marks {
            let start = mark[0].as_u64().unwrap() as usize;
            let end = mark[1].as_u64().unwrap() as usize;
            assert!(start < end && end <= shown.len());
            assert!(shown.is_char_boundary(start) && shown.is_char_boundary(end));
            assert_eq!(&shown[start..end], term);
        }
        let unicode_query = if output {
            "messages where output:lodz"
        } else {
            "messages where lodz"
        };
        let unicode = cli(fixture, &["--json", "find", unicode_query]);
        assert!(unicode.status.success(), "{unicode:?}");
        let unicode = json(&unicode);
        assert_eq!(unicode["count"], 1, "{unicode}");
        let snippet = &unicode["rows"][0]["snippet"];
        let unicode_text = snippet["text"].as_str().unwrap();
        assert!(
            text.contains(unicode_text.trim_matches('…')),
            "rewritten snippet: {snippet}"
        );
        let marks = snippet["marks"].as_array().unwrap();
        assert!(!marks.is_empty(), "{snippet}");
        for mark in marks {
            let start = mark[0].as_u64().unwrap() as usize;
            let end = mark[1].as_u64().unwrap() as usize;
            assert!(unicode_text.is_char_boundary(start) && unicode_text.is_char_boundary(end));
            assert_eq!(&unicode_text[start..end], "Łódź");
        }
    });
}
