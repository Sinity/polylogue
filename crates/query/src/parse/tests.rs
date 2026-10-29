use super::*;
use crate::ast::*;
fn p(text: &str) -> Query {
    parse(text).unwrap_or_else(|error| panic!("{text}: {error}"))
}
fn cond(text: &str) -> ConditionKind {
    match p(text).source {
        Source::Unit {
            condition: Some(c), ..
        } => c.kind,
        other => panic!("{text}: {other:?}"),
    }
}
#[test]
fn every_spec_example_parses_through_the_product_catalog() {
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
        "connected:codex:01a20000-0005-7000-8000-000000000005",
        "since:2026-10-05",
        "started between 2026-10-02 and 2026-10-03",
        "has:compaction",
        "model:claude-opus-5-5",
        "origin:(chatgpt|claude-ai)",
        "notes where kind:decision",
        "origin:codex with messages(id, role)[role:assistant, last:1]",
        "similar:\"ambiguous date formats\"",
        "repo:other",
        "links where kind:continuation",
        "link:fork",
        "tree_cost > 2.90",
    ] {
        p(text);
    }
}

#[test]
fn shared_tree_retains_boolean_modes_values_and_spans() {
    let ConditionKind::And { conditions } = cond("repo:x \"a b\" refactor*") else {
        panic!()
    };
    assert_eq!(conditions.len(), 3);
    assert!(
        matches!(&conditions[1].kind, ConditionKind::Text { text, mode: TextMode::Phrase } if text == "a b")
    );
    assert!(
        matches!(&conditions[2].kind, ConditionKind::Text { text, mode: TextMode::Prefix } if text == "refactor")
    );
    assert_eq!(conditions[0].span, Span { start: 0, end: 6 });
    assert!(
        matches!(cond("cats and dogs"), ConditionKind::And { conditions } if conditions.len() == 3)
    );
    assert!(matches!(cond("a OR b"), ConditionKind::Or { .. }));
    for text in ["NOT tag:x", "-tag:x"] {
        assert!(matches!(cond(text), ConditionKind::Not { .. }));
    }
    assert!(
        matches!(cond("origin:(a|b|c)"), ConditionKind::Match { value: Value::Set { values }, .. } if values.len() == 3)
    );
    assert!(matches!(
        cond("kind:*"),
        ConditionKind::Match {
            value: Value::Any,
            ..
        }
    ));
    assert!(matches!(
        cond("cost:unknown"),
        ConditionKind::Match {
            value: Value::Unknown,
            ..
        }
    ));
    assert!(
        matches!(cond("tag:project:morrow"), ConditionKind::Match { value: Value::Scalar { text, .. }, .. } if text == "project:morrow")
    );
}

#[test]
fn compact_comparisons_and_unknown_fields_have_executable_corrections() {
    for (text, correction) in [
        ("messages:>=10", "messages >= 10"),
        ("reop:morrow", "repo:morrow"),
    ] {
        let error = parse(text).unwrap_err();
        assert_eq!(error.suggestion(), Some(correction));
        assert!(parse(correction).is_ok());
    }
}

#[test]
fn references_calls_and_sequences_keep_structural_identity() {
    for text in ["https://chatgpt.com/c/abc", "@3.2"] {
        assert!(matches!(cond(text), ConditionKind::Reference { reference } if reference == text));
    }
    assert!(
        matches!(cond("weekday_offset()"), ConditionKind::Text { text, .. } if text == "weekday_offset()")
    );
    assert!(
        matches!(cond("parse(x) fails"), ConditionKind::And { conditions } if conditions.len() == 2)
    );
    let Source::Unit {
        condition:
            Some(Condition {
                kind:
                    ConditionKind::Sequence {
                        unit,
                        steps,
                        gaps,
                        all,
                    },
                ..
            }),
        ..
    } = p("tool_calls where seq(tool:Bash -> [next] tool:Edit -> [within:5m] tool:Bash)").source
    else {
        panic!()
    };
    assert_eq!(unit, "tool_calls");
    assert!(!all);
    assert_eq!(steps.len(), 3);
    assert_eq!(
        gaps,
        vec![
            Gap::Next,
            Gap::Within {
                duration: Value::scalar("5m")
            }
        ]
    );
}

#[test]
fn shared_units_stages_and_nested_queries_use_product_names() {
    for (text, unit) in [
        ("messages where role:user", "messages"),
        ("message where role:user", "messages"),
        ("TOOL_CALLS", "tool_calls"),
        ("work | count", "work"),
        ("", "sessions"),
        ("   ", "sessions"),
    ] {
        assert!(
            matches!(p(text).source, Source::Unit { unit: actual, .. } if actual == unit),
            "{text}"
        );
    }
    assert!(matches!(
        cond("messages >= 10"),
        ConditionKind::Compare { .. }
    ));
    let source = "tool_calls | group by day(at), tool | agg count, sum:duration, p90:duration, count_distinct:session | sort by count desc | where count >= 3 | limit 5 | offset 10";
    let mut catalog = crate::catalog::catalog(&[], &[]);
    catalog
        .units
        .iter_mut()
        .find(|unit| unit.name == "tool_calls")
        .unwrap()
        .fields
        .push(query_language::catalog::Field {
            name: "duration".into(),
            kind: query_language::catalog::FieldKind::Duration,
            ..Default::default()
        });
    let query = with_catalog(source, &catalog).unwrap();
    assert_eq!(
        parse(source).unwrap_err().kind(),
        "unknown_field",
        "elapsed duration has no product producer yet"
    );
    assert_eq!(query.stages.len(), 6);
    let StageKind::Aggregate { aggregates } = &query.stages[1].kind else {
        panic!()
    };
    assert_eq!(aggregates[2].column(), "p90_duration");
    let nested =
        p("origin:codex | union (repo:morrow | where cost > 1) | except (messages where chrono)");
    assert_eq!(nested.stages.len(), 2);
    assert!(
        matches!(cond("count(tool_call where outcome:error) >= 3"), ConditionKind::Count { unit, op: Op::Ge, count: 3, condition: Some(_), .. } if unit == "tool_calls")
    );
    assert!(
        matches!(cond("id in (messages where chrono | group by session)"), ConditionKind::In { query, .. } if query.stages.len() == 1)
    );
}

#[test]
fn repairs_and_product_depth_guard_are_stable() {
    let error = parse("\"unclosed phrase").unwrap_err();
    assert_eq!(error.suggestion(), Some("\"unclosed phrase\""));
    assert!(parse("messages where").is_err());
    let deep = format!("{}a{}", "(".repeat(65), ")".repeat(65));
    assert!(matches!(parse(&deep), Err(QueryError::TooDeep { max: 64 })));
    let allowed = format!("{}a{}", "(".repeat(64), ")".repeat(64));
    assert!(parse(&allowed).is_ok());
    assert!(parse("kind:* where exists tool_call(outcome:error)").is_err());
    assert!(parse("kind:* \"where\"").is_ok());
}

#[test]
fn saved_arguments_are_values_and_literals_do_not_bind_namespaces() {
    let Source::Saved {
        name, arguments, ..
    } = p("query:failing(repo=morrow, phrase=\"two words\")").source
    else {
        panic!()
    };
    assert_eq!(name, "failing");
    assert_eq!(arguments.len(), 2);
    assert_eq!(
        arguments[1].value,
        Value::Scalar {
            text: "two words".into(),
            quoted: true
        }
    );
    assert!(p("repo:\"sx.events\"").namespaces().is_empty());
}

#[test]
fn former_semantic_spelling_refuses_with_the_canonical_correction() {
    let error = parse("near:dates").unwrap_err();
    assert_eq!(error.kind(), "unknown_field");
    assert_eq!(error.suggestion(), Some("similar:dates"));
    assert!(matches!(
        cond("similar:dates"),
        ConditionKind::Similar { .. }
    ));
}

#[test]
fn shorthand_time_bounds_keep_their_shared_typed_meaning() {
    assert!(matches!(
        cond("since:2026-10-05"),
        ConditionKind::Since { .. }
    ));
    assert!(matches!(
        cond("until:2026-10-05"),
        ConditionKind::Until { .. }
    ));
    assert!(matches!(
        cond("messages where at < 2026-10-05"),
        ConditionKind::Compare { op: Op::Lt, .. }
    ));
}

#[test]
fn product_namespace_spelling_binds_to_the_same_declared_unit() {
    let catalog = crate::catalog::catalog(&[], &[]);
    for (qualified, local) in [
        (
            "pl.sessions where repo:morrow",
            "sessions where repo:morrow",
        ),
        ("pl.messages where role:user", "messages where role:user"),
    ] {
        assert_eq!(
            query_language::identity::normal_form(&p(qualified), &catalog).unwrap(),
            query_language::identity::normal_form(&p(local), &catalog).unwrap(),
        );
    }
}
