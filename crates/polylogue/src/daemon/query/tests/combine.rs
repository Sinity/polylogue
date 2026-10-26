//! Queries inside queries: combining stages, counted conditions, `in`, and
//! saved queries with parameters.

use super::*;

fn count(v: &Value) -> u64 {
    v["count"].as_u64().unwrap()
}

fn error(w: &TestArchive, text: &str) -> Failure {
    match run_q(w, ask(text)) {
        Ok(v) => panic!("{text} answered {v}"),
        Err(f) => f,
    }
}

#[test]
fn set_stages_select_what_their_conditions_would() {
    let w = morrow();
    let union = q(&w, "origin:claude-code | union (origin:codex)");
    assert_eq!(ids(&union), ids(&q(&w, "origin:(claude-code|codex)")));
    assert_eq!(union["ranked"], false, "unranked rows keep their order");
    assert_eq!(
        ids(&q(&w, "origin:claude-code | intersect (cost > 2)")),
        ["#2"]
    );
    assert_eq!(
        ids(&q(&w, "origin:claude-code | except (cost > 2)")),
        ["#4", "#1"]
    );
    // Conditions without a unit are on the rows they combine with.
    assert_eq!(
        count(&q(
            &w,
            "tool_calls where outcome:error | except (command:\"cargo clippy\")"
        )),
        count(&q(
            &w,
            "tool_calls where outcome:error -command:\"cargo clippy\""
        ))
    );
    // Each side keeps its own defaults: the subagent joins only by name.
    assert_eq!(
        ids(&q(
            &w,
            "origin:codex | union (model:claude-opus-5-5 kind:*)"
        )),
        ["#6", "#5", "#3"]
    );
    // Later stages see the combined rows.
    assert_eq!(
        q(&w, "origin:claude-code | union (origin:codex) | count")["rows"][0]["count"],
        5
    );
}

#[test]
fn set_stages_refuse_what_does_not_combine() {
    let w = morrow();
    for bad in [
        "origin:codex | union (messages where chrono)",
        "origin:codex | union (origin:claude-code | count)",
        "origin:codex | count | union (origin:claude-code)",
        "origin:codex | union (origin:claude-code | distinct)",
    ] {
        assert_eq!(error(&w, bad).kind, "invalid_query", "{bad}");
    }
}

#[test]
fn a_union_of_ranked_rows_fuses_their_orders() {
    let w = morrow();
    let mut a = ask("messages where parser | union (messages where chrono)");
    a.page_size = Some(100);
    let v = run_q(&w, a).unwrap();
    assert_eq!(v["ranked"], true);
    assert_eq!(
        count(&v),
        count(&q(&w, "messages where (parser OR chrono)")),
        "{v}"
    );
    let rows = v["rows"].as_array().unwrap();
    let ranks: Vec<f64> = rows.iter().map(|r| r["rank"].as_f64().unwrap()).collect();
    assert!(ranks.windows(2).all(|p| p[0] <= p[1]), "{ranks:?}");
    // `found_in` names the lists that ranked each row.
    for r in rows {
        let text = r["id"].as_str().unwrap();
        let lists: Vec<u64> = r["found_in"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .collect();
        let expected = if text.starts_with("m1p") { 1 } else { 2 };
        assert_eq!(lists, [expected], "{r}");
    }
    // Both first places tie, and each list's later places follow them.
    assert_eq!(ranks[0], ranks[1]);
    // A fuse ranks without adding rows.
    let fused = q(
        &w,
        "messages where parser | fuse (messages where cwd:\"/home/op/morrow\")",
    );
    assert_eq!(count(&fused), count(&q(&w, "messages where parser")));
}

#[test]
fn fused_pages_concatenate_to_the_whole_answer() {
    let w = morrow();
    let text =
        "messages where parser | union (messages where chrono) | union (messages where continue)";
    let mut whole = ask(text);
    whole.page_size = Some(100);
    let whole = run_q(&w, whole).unwrap();
    let mut seen = vec![];
    let mut cursor: Option<String> = None;
    loop {
        let mut a = ask(text);
        a.page_size = Some(2);
        a.cursor = cursor.as_deref();
        let page = run_q(&w, a).unwrap();
        seen.extend(
            page["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["id"].clone()),
        );
        match page["next"].as_str() {
            Some(next) => cursor = Some(next.to_owned()),
            None => break,
        }
    }
    let all: Vec<Value> = whole["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].clone())
        .collect();
    assert_eq!(seen, all);
}

#[test]
fn counted_conditions_count_related_rows() {
    let w = morrow();
    let failing = "sessions where count(tool_call where outcome:error) >= 1";
    assert_eq!(ids(&q(&w, failing)), ["#4", "#2"]);
    assert_eq!(
        ids(&q(&w, "sessions where count(tool_calls) > 2")),
        ids(&q(&w, "sessions where tool_calls > 2"))
    );
    // A count is known for every session: it and its NOT partition them.
    let not = q(
        &w,
        "sessions where NOT count(tool_call where outcome:error) >= 1",
    );
    assert_eq!(
        count(&q(&w, failing)) + count(&not),
        count(&q(&w, "sessions"))
    );
}

#[test]
fn in_compares_with_the_values_a_query_groups_by() {
    let w = morrow();
    assert_eq!(
        ids(&q(
            &w,
            "sessions where id in (messages where chrono | group by session)"
        )),
        ids(&q(&w, "sessions where exists message(chrono)"))
    );
    assert_eq!(
        error(&w, "sessions where id in (messages where chrono)").kind,
        "invalid_query"
    );
}

/// A saved query: a note of kind `saved_query` named by its key.
fn saved(w: &mut TestArchive, name: &str, text: &str) {
    w.fx.note("saved_query", Some(name), Some(text), "repo:synthetic")
        .unwrap();
}

#[test]
fn saved_queries_take_their_parameters_as_values() {
    let mut w = morrow();
    saved(&mut w, "by-outcome", "tool_calls where outcome:$outcome");
    saved(&mut w, "costly", "sessions where cost > $over");
    let v = q(&w, "query:by-outcome(outcome=error)");
    assert_eq!(count(&v), count(&q(&w, "tool_calls where outcome:error")));
    assert_eq!(v["definitions"][0]["name"], "by-outcome");
    assert_eq!(
        ids(&q(
            &w,
            "query:costly(over=2) | union (query:costly(over=1.5))"
        )),
        ids(&q(&w, "sessions where cost > 1.5"))
    );
    // A value is one value: it cannot add a condition.
    assert_eq!(
        error(&w, "query:by-outcome(outcome=\"error OR outcome:ok\")").kind,
        "bad_value"
    );
    for bad in [
        "query:by-outcome",
        "query:by-outcome(outcome=error, extra=1)",
        "tool_calls where query:by-outcome(result=error)",
    ] {
        assert_eq!(error(&w, bad).kind, "invalid_query", "{bad}");
    }
    // `$word` the caller writes is a word.
    assert_eq!(q(&w, "messages where \"$HOME\"")["outcome"], "empty");
}

#[test]
fn a_name_two_saved_queries_share_is_refused() {
    let mut w = morrow();
    saved(&mut w, "twice", "origin:codex");
    saved(&mut w, "twice", "origin:chatgpt");
    let f = error(&w, "query:twice");
    assert_eq!(f.kind, "invalid_query");
    assert!(f.message.contains("2 saved queries"), "{}", f.message);
}
