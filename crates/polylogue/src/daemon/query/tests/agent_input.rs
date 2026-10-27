//! What agents paste into a search: every string gets rows or an error
//! that names the problem and gives a corrected query, and a reference
//! that could name several sessions names none of them.

use super::*;

/// Bare words, a pasted session ID with and without its origin, a chat
/// URL and a quoted phrase return rows; unbalanced quotes and
/// parentheses, a misspelt field and a misspelt unit are refused with a
/// corrected query that runs.
#[test]
fn every_agent_string_gets_rows_or_a_corrected_query() {
    let w = morrow();
    for text in [
        "chrono",
        "fix parser",
        "7f3a2c10-0002-4000-8000-000000000002",
        "claude-code:7f3a2c10-0002-4000-8000-000000000002",
        "https://chatgpt.com/c/abc-7",
        "\"fix the parser\"",
        "messages where \"run the tests\"",
    ] {
        let v = run_q(&w, ask(text)).unwrap_or_else(|f| panic!("{text}: {} {}", f.kind, f.message));
        assert!(
            !v["rows"].as_array().unwrap().is_empty(),
            "{text}: no rows: {v}"
        );
    }
    for text in [
        "\"fix the parser",
        "fix the \"parser",
        "(origin:codex OR origin:chatgpt",
        "reop:morrow",
        "messges cargo",
        "sessin where repo:morrow",
    ] {
        let f = run_q(&w, ask(text))
            .map(|v| panic!("{text} was answered: {v}"))
            .unwrap_err();
        assert!(!f.message.is_empty(), "{text}: no message");
        let suggestion = f
            .detail
            .as_ref()
            .and_then(|d| d["suggestion"].as_str())
            .unwrap_or_else(|| panic!("{text}: no corrected query: {} {}", f.kind, f.message))
            .to_owned();
        if let Err(again) = run_q(&w, ask(&suggestion)) {
            panic!(
                "{text}: the corrected query {suggestion:?} fails too: {} {}",
                again.kind, again.message
            );
        }
    }
}

/// A prefix two sessions share is refused listing both, even when a note
/// names one of them; a full ID that is also the prefix of another names
/// only itself.
#[test]
fn a_reference_two_sessions_share_is_refused_listing_both() {
    let mut w = morrow();
    w.fx.note(
        "decision",
        None,
        Some("the one meant"),
        "session:claude-code:7f3a2c10-0002-4000-8000-000000000002",
    )
    .unwrap();
    for text in [
        "7f3a2c10-000",
        "claude-code:7f3a2c10-000",
        "messages where session:claude-code:7f3a2c10-000",
        "id:claude-code:7f3a2c10-000",
        "parent:claude-code:7f3a2c10-000",
        "root:claude-code:7f3a2c10-000",
        "continues:claude-code:7f3a2c10-000",
        "continued_by:claude-code:7f3a2c10-000",
        "connected:claude-code:7f3a2c10-000",
        "tool_calls where child:claude-code:7f3a2c10-000",
        "links where parent:claude-code:7f3a2c10-000",
        "links where child:claude-code:7f3a2c10-000",
        "deliveries where recipient:claude-code:7f3a2c10-000",
    ] {
        let f = run_q(&w, ask(text))
            .map(|v| panic!("{text} was answered: {v}"))
            .unwrap_err();
        assert_eq!(f.kind, "ambiguous", "{text}: {}", f.message);
        let ids: Vec<&str> = f.detail.as_ref().unwrap()["ids"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(ids.len() >= 2, "{text}: {ids:?}");
        for id in &ids {
            assert!(f.message.contains(id), "{text}: {} lacks {id}", f.message);
        }
    }
    let s =
        w.fx.session(SessionFacts {
            id: "claude-code:7f3a2c10-0002-4000-8000-000000000002-copy",
            started: at(9, 9),
            last: at(9, 10),
            ..SessionFacts::default()
        })
        .unwrap();
    w.fx.message(s, msg("copy1", "user", "human", "copy", at(9, 9)))
        .unwrap();
    for text in [
        "claude-code:7f3a2c10-0002-4000-8000-000000000002",
        "7f3a2c10-0002-4000-8000-000000000002",
    ] {
        assert_eq!(ids(&q(&w, text)), ["#2"], "{text}");
    }
    let f = run_q(&w, ask("claude-code:7f3a2c10-9999")).unwrap_err();
    assert_eq!(f.kind, "not_found", "{}", f.message);
}
