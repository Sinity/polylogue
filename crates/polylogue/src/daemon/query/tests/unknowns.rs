//! Unknown never becomes a match or a zero.

use super::*;

/// A session with an unknown model, cost, repository and start, and tool
/// calls whose outcome and exit code are unknown: no `NOT` matches them,
/// and `F:unknown` selects them.
#[test]
fn a_negated_condition_never_matches_an_unknown_value() {
    let mut w = morrow();
    let s = w
        .fx
        .session(SessionFacts {
            id: "claude-code:unknowns",
            ..SessionFacts::default()
        })
        .unwrap();
    let m = w
        .fx
        .message(s, msg("u0", "assistant", "unknown", "", None))
        .unwrap();
    w.fx.tool_call(
        s,
        m,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("cargo test"),
            outcome: "unknown",
            ..CallFacts::default()
        },
    )
    .unwrap();
    let has = |text: &str| ids(&q(&w, text)).contains(&"claude-code:unknowns".to_owned());
    for (negated, unknown) in [
        ("NOT model:claude-opus-5-5", "model:unknown"),
        ("NOT cost > 1", "cost:unknown"),
        ("NOT cost < 1", "cost:unknown"),
        ("NOT repo:morrow", "repo:unknown"),
        ("NOT started < 2026-10-05", "started:unknown"),
    ] {
        assert!(!has(negated), "{negated}");
        assert!(has(unknown), "{unknown}");
    }
    let calls = |text: &str| q(&w, text)["count"].as_u64().unwrap();
    let mine = "tool_calls where session:claude-code:unknowns";
    for (negated, unknown) in [
        ("NOT outcome:ok", "outcome:unknown"),
        ("NOT outcome:error", "outcome:unknown"),
        ("NOT exit_code = 0", "exit_code:unknown"),
        ("exit_code != 0", "exit_code:unknown"),
    ] {
        assert_eq!(calls(&format!("{mine} {negated}")), 0, "{negated}");
        assert_eq!(calls(&format!("{mine} {unknown}")), 1, "{unknown}");
    }
    assert_eq!(
        calls("messages where session:claude-code:unknowns NOT written_by:model"),
        0
    );
    assert_eq!(
        calls("messages where session:claude-code:unknowns written_by:unknown"),
        1
    );
    // `exists` is unknown when no child satisfies the condition and some
    // child is unknown for it, so its `NOT` leaves the session out.
    assert!(has("kind:* exists tool_call(exit_code:unknown)"));
    assert!(!has("kind:* exists tool_call(exit_code = 0)"));
    assert!(!has("kind:* NOT exists tool_call(exit_code = 0)"));
}

/// A tree whose subagent call names a child session that is not in the
/// archive has a tree cost that is a lower bound, reported as partial,
/// never complete.
#[test]
fn a_missing_child_makes_the_tree_cost_partial() {
    let mut w = morrow();
    w.fx.link(
        "claude-code:7f3a2c10-0001-4000-8000-000000000001",
        "claude-code:not-stored",
        "subagent",
        "fresh",
    )
    .unwrap();
    let total = |text: &str| q(&w, text)["rows"][0].clone();
    let one = total(
        "sessions where id:claude-code:7f3a2c10-0001-4000-8000-000000000001 | agg sum:tree_cost",
    );
    assert_eq!(one["partial_tree_cost"], 1, "{one}");
    assert!((one["sum_tree_cost"].as_f64().unwrap() - 1.26).abs() < 1e-9, "{one}");
    // #2's tree has its subagent #3, priced, so it is complete.
    let two = total(
        "sessions where id:claude-code:7f3a2c10-0002-4000-8000-000000000002 | agg sum:tree_cost",
    );
    assert_eq!(two["partial_tree_cost"], 0, "{two}");
}

/// `near:` keeps the matching rows that have no vector, after the ranked
/// rows, and says how many there are.
#[test]
fn near_says_how_many_rows_have_no_vector() {
    let w = morrow();
    let s = super::generated::vectors_for(&w, |session| session.contains("0002-4000"));
    let near = |_: &str| Ok(std::sync::Arc::clone(&s));
    let mut request = ask("kind:* near:\"ambiguous date formats\"");
    request.embeddings = true;
    request.near = Some(&near);
    let v = run_q(&w, request).unwrap();
    assert_eq!(v["count"], 8, "{v}");
    // #2 has vectors; every other session follows it, unranked.
    assert_eq!(v["unknown"]["near"], 7, "{v}");
    assert_eq!(ids(&v)[0], "#2");
    assert!(
        v["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str().is_some_and(|n| n.starts_with("7 of 8 rows have no vector"))),
        "{v}"
    );
}
