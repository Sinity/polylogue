//! Compiler tests: the SQL a query produces, checked by its shape here and by running it in the daemon tests.

use super::*;

fn opts(saved: &[(&str, &str)]) -> Options {
    Options {
        clock: crate::Clock::new(
            1_790_000_000_000,
            polylogue_model::LogicalDay::named("Europe/Warsaw", 240).unwrap(),
        ),
        origins: vec!["claude-code".into(), "codex".into()],
        page_size: None,
        embeddings: false,
        self_session: None,
        saved: saved
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        kept: Default::default(),
    }
}

fn same(a: &str, b: &str, o: &Options) {
    let (pa, pb) = (compile(a, o).unwrap(), compile(b, o).unwrap());
    assert_eq!(pa.unit, pb.unit, "{a} / {b}");
    // The saved condition keeps its own parentheses; the meaning is equal.
    let flat = |s: &str| s.replace(['(', ')'], "");
    assert_eq!(flat(&pa.sql), flat(&pb.sql), "{a} / {b}");
    assert_eq!(pa.params, pb.params, "{a} / {b}");
}

/// Law 127: a saved query selects the same rows used inline or by name.
#[test]
fn a_saved_query_selects_what_it_says_inline_or_by_name() {
    let o = opts(&[
        ("failing", "tool_calls where outcome:error"),
        ("morrow", "tag:project:morrow"),
        ("by-action", "tool_calls | group by action | count"),
    ]);
    // As the source, narrowed by the rest.
    same(
        "query:failing session.repo:morrow",
        "tool_calls where (outcome:error) session.repo:morrow",
        &o,
    );
    // As a condition over the same unit.
    same(
        "tool_calls where query:failing action:shell",
        "tool_calls where (outcome:error) action:shell",
        &o,
    );
    same("query:morrow", "sessions where (tag:project:morrow)", &o);
    // A shaped result is read, then further staged.
    same(
        "query:by-action | limit 2",
        "tool_calls | group by action | count | limit 2",
        &o,
    );
}

#[test]
fn a_saved_query_that_cannot_stand_there_is_refused() {
    let o = opts(&[
        ("failing", "tool_calls where outcome:error"),
        ("by-action", "tool_calls | group by action | count"),
        ("loop", "sessions where query:loop"),
    ]);
    for bad in [
        "tool_calls where query:by-action",
        "messages where query:failing",
        "query:by-action outcome:error",
        "query:nope",
        "query:loop",
    ] {
        assert!(compile(bad, &o).is_err(), "{bad}");
    }
}
