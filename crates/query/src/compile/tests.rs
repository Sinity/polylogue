//! Compiler tests: the SQL a query produces, checked by its shape here and by running it in the daemon tests.

use super::*;

fn opts(saved: &[(&str, &str)]) -> Options {
    Options {
        clock: crate::Clock::new(
            1_790_000_000_000,
            polylogue_model::LogicalDay::named("Europe/Warsaw", 240).unwrap(),
        ),
        origins: vec!["claude-code".into(), "codex".into()],
        command_words: vec!["list".into(), "brief".into()],
        page_size: None,
        embeddings: false,
        self_session: None,
        saved: saved
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        kept: Default::default(),
        fields: vec![],
    }
}

fn c(text: &str) -> Program {
    compile(text, &opts(&[])).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn compile_err(text: &str) -> String {
    compile(text, &opts(&[])).unwrap_err().to_string()
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

#[test]
fn sort_takes_tie_breaking_keys() {
    let p = c("sessions | sort by live_rank, last_event_at desc");
    let order = p.sql.rsplit("ORDER BY").next().unwrap();
    let rank = order.find("WHEN 'waiting'").expect("live_rank first");
    let last = order
        .find("live_last_event_at) DESC")
        .expect("then last_event_at");
    assert!(rank < last, "{order}");
    assert!(compile_err("sessions | group by repo | sort by count, repo").contains("one key"));
}

#[test]
fn measure_ends_a_session_query() {
    let p = c("repo:polylogue | measure usage");
    let m = p.measure.expect("measure");
    assert_eq!(m.what, "usage");
    assert!(!m.everything);
    assert!(
        c("sessions | measure daily:tokens")
            .measure
            .unwrap()
            .everything
    );
    assert!(compile_err("messages | measure usage").contains("session query"));
    assert!(compile_err("sessions | measure usage | sort by cost").contains("last stage"));
}

#[test]
fn fields_choose_the_columns() {
    let p = compile(
        "sessions",
        &Options {
            fields: vec!["id".into(), "cost".into(), "repo".into()],
            ..opts(&[])
        },
    )
    .unwrap();
    let names: Vec<&str> = p.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id", "cost", "repo"]);
}

#[test]
fn changes_continue_from_a_cursor() {
    let p = c("changes where after:12.3");
    assert!(p.sql.contains("> 12") && p.sql.contains("> 3"), "{}", p.sql);
    assert!(compile_err("changes where after:12").contains("INDEX.ARCHIVE"));
}

#[test]
fn deliveries_take_their_outcome_from_the_latest_event() {
    let p = c("deliveries where sender:signal | sort by at desc");
    assert!(p.sql.contains("delivery_event"), "{}", p.sql);
    assert!(!p.sql.contains(".outcome"), "{}", p.sql);
}

#[test]
fn a_query_starting_with_a_command_is_refused() {
    assert!(
        compile_err("list repo:x").contains("is a command"),
        "{}",
        compile_err("list repo:x")
    );
    let near = compile_err("brife now");
    assert!(near.contains("\"brief\""), "{near}");
    assert!(compile(r#""list" repo:x"#, &opts(&[])).is_ok());
}
