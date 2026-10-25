use super::*;

fn p(text: &str) -> Query {
    parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
}

fn cond(text: &str) -> CondKind {
    match p(text).source {
        Source::Sessions { cond } => cond.kind,
        other => panic!("{text:?}: {other:?}"),
    }
}

#[test]
fn every_spec_example_parses() {
    // The query examples over the Morrow archive.
    for text in [
        "origin:claude-code",
        "origin:claude-code kind:subagent",
        "kind:* | agg count, sum:cost",
        "sessions | group by origin | count",
        "repo:morrow",
        "tag:project:morrow",
        "sessions where cost > 1",
        "cost:unknown",
        "sessions where human_prompts >= 3",
        "sessions where messages >= 10",
        "sessions where tool_calls > 5",
        "messages where written_by:human \"chrono\"",
        "messages where written_by:human duplicate:* \"chrono\"",
        "tool_calls where outcome:error",
        "tool_calls where outcome:error | group by session | count",
        "tool_calls | group by action | count",
        "tool_calls where tool:Bash | agg count",
        "tool_calls where session.tag:project:morrow | agg count",
        "sessions where exists tool_call(action:shell AND outcome:error)",
        "sessions where seq(command:\"cargo test\" AND outcome:error -> tool:Edit -> command:\"cargo test\" AND outcome:ok)",
        "messages where written_by:human | group by session.origin | count",
        "messages where active:false",
        "parent:claude-code:7f3a2c10-0002-4000-8000-000000000002",
        "lineage:codex:01a20000-0005-7000-8000-000000000005",
        "since:2026-10-05",
        "started between 2026-10-02 and 2026-10-03",
        "has:compaction",
        "model:claude-opus-5-5",
        "origin:(chatgpt|claude-ai)",
        "notes where kind:decision",
        "origin:codex with messages(id, role)[role:assistant, last:1]",
        "near:\"ambiguous date formats\"",
        "repo:other",
        "reop:morrow",
        "links where kind:continuation",
        "link:fork",
        "tree_cost > 2.90",
    ] {
        p(text);
    }
}

#[test]
fn juxtaposition_is_and_and_capitals_are_operators() {
    let CondKind::And { all } = cond("repo:x \"a b\" refactor*") else {
        panic!()
    };
    assert_eq!(all.len(), 3);
    assert!(matches!(&all[1].kind, CondKind::Text { phrase: true, text } if text == "a b"));
    assert!(matches!(&all[2].kind, CondKind::Text { phrase: false, text } if text == "refactor*"));
    // Lower-case `and`/`or` are words to search for.
    let CondKind::And { all } = cond("cats and dogs") else {
        panic!()
    };
    assert_eq!(all.len(), 3);
    assert!(matches!(cond("a OR b"), CondKind::Or { .. }));
    assert!(matches!(cond("NOT tag:x"), CondKind::Not { .. }));
    assert!(matches!(cond("-tag:x"), CondKind::Not { .. }));
}

#[test]
fn values_and_compact_comparisons() {
    assert!(matches!(
        cond("origin:(a|b|c)"),
        CondKind::Match { value: Value::Alternatives { any }, .. } if any.len() == 3
    ));
    assert!(matches!(
        cond("kind:*"),
        CondKind::Match {
            value: Value::Any,
            ..
        }
    ));
    assert!(matches!(
        cond("messages:>=10"),
        CondKind::Compare { op: Op::Ge, value, .. } if value == "10"
    ));
    assert!(matches!(
        cond("tag:project:morrow"),
        CondKind::Match { value: Value::One { text, .. }, .. } if text == "project:morrow"
    ));
    assert!(matches!(
        cond("started > last week"),
        CondKind::Compare { value, .. } if value == "last week"
    ));
}

#[test]
fn references_parse_as_their_own_conditions() {
    assert!(matches!(
        cond("https://chatgpt.com/c/abc"),
        CondKind::Url { .. }
    ));
    assert!(matches!(cond("@3.2"), CondKind::Position { .. }));
}

#[test]
fn a_pasted_call_is_one_search_word() {
    assert!(matches!(
        cond("weekday_offset()"),
        CondKind::Text { text, phrase: false } if text == "weekday_offset()"
    ));
    assert!(matches!(cond("parse(x) fails"), CondKind::And { all } if all.len() == 2));
    // Parentheses with spaces inside still group.
    assert!(matches!(cond("a (b OR c)"), CondKind::And { all } if all.len() == 2));
}

#[test]
fn seq_records_its_gaps() {
    let q = p("tool_calls where seq(tool:Bash -> [next] tool:Edit -> [within:5m] tool:Bash)");
    let Source::Unit {
        cond: Some(Cond {
            kind: CondKind::Seq { steps, gaps },
            ..
        }),
        ..
    } = q.source
    else {
        panic!()
    };
    assert_eq!(steps.len(), 3);
    assert_eq!(gaps, vec![Gap::Next, Gap::Within { ms: 300_000 }]);
}

#[test]
fn a_session_query_can_scope_a_row_query() {
    let q = p("tag:project:morrow | tool_calls where action:shell | group by tool | count");
    assert!(matches!(q.source, Source::Sessions { .. }));
    assert!(matches!(
        q.stages[0].kind,
        StageKind::Scope {
            unit: Unit::ToolCalls,
            ..
        }
    ));
    assert_eq!(q.stages.len(), 3);
}

#[test]
fn a_unit_name_that_is_also_a_field_still_compares() {
    // `messages` is a unit and a session field.
    assert!(matches!(cond("messages >= 10"), CondKind::Compare { .. }));
    assert!(matches!(
        p("messages").source,
        Source::Unit {
            unit: Unit::Messages,
            cond: None
        }
    ));
}

#[test]
fn stages_parse() {
    let q = p(
        "tool_calls | group by day(at), tool | agg count, sum:duration, p90:duration, count_distinct:session | sort by count desc | where count >= 3 | limit 5 | offset 10",
    );
    assert_eq!(q.stages.len(), 6);
    let StageKind::Agg { fns } = &q.stages[1].kind else {
        panic!()
    };
    assert_eq!(fns[2].column(), "p90_duration");
    assert!(matches!(
        p("sessions | sort by random(7)").stages[0].kind,
        StageKind::SortBy {
            key: SortKey::Random { seed: Some(7) },
            ..
        }
    ));
}

#[test]
fn errors_point_at_the_problem_and_offer_a_repair() {
    let e = parse("repo:(a|").unwrap_err();
    let QueryError::Syntax { position, .. } = &e else {
        panic!("{e:?}")
    };
    assert!(*position >= 5);
    let QueryError::Syntax { suggestion, .. } = parse("\"unclosed phrase").unwrap_err() else {
        panic!()
    };
    assert_eq!(suggestion.as_deref(), Some("\"unclosed phrase\""));
    let QueryError::Syntax { suggestion, .. } = parse("messages where").unwrap_err() else {
        panic!()
    };
    assert_eq!(suggestion, None);
}

#[test]
fn nesting_past_the_stack_bound_is_refused_by_name() {
    let deep = format!("{}a{}", "(".repeat(65), ")".repeat(65));
    assert!(matches!(parse(&deep), Err(QueryError::TooDeep { .. })));
    let ok = format!("{}a{}", "(".repeat(64), ")".repeat(64));
    assert!(parse(&ok).is_ok());
}

#[test]
fn a_bare_where_after_session_conditions_is_refused_not_searched() {
    // It used to search the word "where" and answer wrongly (tuu.7).
    let QueryError::Syntax {
        position,
        suggestion,
        ..
    } = parse("kind:* where exists tool_call(outcome:error)").unwrap_err()
    else {
        panic!()
    };
    assert_eq!(position, 7);
    assert_eq!(
        suggestion.as_deref(),
        Some("kind:* exists tool_call(outcome:error)")
    );
    assert!(parse("kind:* NOT where").is_err());
    // The quoted word is searched; `where` inside a row condition, and
    // `sessions where …`, keep their meaning.
    assert!(parse("kind:* \"where\"").is_ok());
    assert!(parse("exists tool_call(where)").is_ok());
    assert!(parse("sessions where exists tool_call(outcome:error)").is_ok());
}

#[test]
fn the_empty_query_is_every_session() {
    assert!(matches!(p("").source, Source::Empty));
    assert!(matches!(p("   ").source, Source::Empty));
}

#[test]
fn unit_names_resolve_through_the_registry() {
    for (text, unit) in [
        ("messages where role:user", Unit::Messages),
        ("message where role:user", Unit::Messages),
        ("TOOL_CALLS", Unit::ToolCalls),
        ("work | count", Unit::Work),
    ] {
        match p(text).source {
            Source::Unit { unit: u, .. } => assert_eq!(u, unit, "{text}"),
            other => panic!("{text:?}: {other:?}"),
        }
    }
    // A lone word that names no unit is searched.
    for text in ["crash", "crash | count", "crash with messages(id)"] {
        assert!(matches!(p(text).source, Source::Sessions { .. }), "{text}");
    }
    // A word before `where`, after `exists` or in a stage must be a unit.
    for (text, fix) in [
        ("mesages where role:user", Some("messages where role:user")),
        ("frobs where x:1", None),
        ("sessions where exists frob(x:1)", None),
        (
            "crash | tool_cals where outcome:error",
            Some("crash | tool_calls where outcome:error"),
        ),
    ] {
        let e = parse(text).unwrap_err();
        assert!(e.to_string().contains("is not a unit"), "{text}: {e}");
        assert_eq!(e.suggestion(), fix, "{text}");
    }
}
