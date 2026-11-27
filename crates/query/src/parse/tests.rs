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
    let catalog = crate::catalog::catalog(&[]);
    let examples = [
        ("origin:claude-code", "origin:\"claude-code\""),
        (
            "origin:claude-code kind:subagent",
            "(kind:\"subagent\" AND origin:\"claude-code\")",
        ),
        (
            "kind:* | agg count, sum(cost)",
            "kind:* | agg count, sum(cost)",
        ),
        (
            "sessions | group by origin | count",
            "sessions | group by origin | agg count",
        ),
        ("repo:morrow", "repo:\"morrow\""),
        ("tag:project:morrow", "tag:\"project:morrow\""),
        ("sessions where cost > 1", "sessions where cost > \"1\""),
        ("cost:unknown", "cost:unknown"),
        (
            "sessions where human_prompts >= 3",
            "sessions where human_prompts >= \"3\"",
        ),
        (
            "sessions where messages >= 10",
            "sessions where messages >= \"10\"",
        ),
        (
            "sessions where tool_calls > 5",
            "sessions where tool_calls > \"5\"",
        ),
        (
            "messages where written_by:human \"chrono\"",
            "messages where (\"chrono\" AND written_by:\"human\")",
        ),
        (
            "messages where written_by:human duplicate:* \"chrono\"",
            "messages where (\"chrono\" AND duplicate:* AND written_by:\"human\")",
        ),
        (
            "tool_calls where outcome:error",
            "tool_calls where outcome:\"error\"",
        ),
        (
            "tool_calls where outcome:error | group by session | count",
            "tool_calls where outcome:\"error\" | group by session | agg count",
        ),
        (
            "tool_calls | group by action | count",
            "tool_calls | group by action | agg count",
        ),
        (
            "tool_calls where tool:Bash | agg count",
            "tool_calls where tool:\"Bash\" | agg count",
        ),
        (
            "tool_calls where session.tag:project:morrow | agg count",
            "tool_calls where session.tag:\"project:morrow\" | agg count",
        ),
        (
            "sessions where exists tool_calls(action:shell AND outcome:error)",
            "sessions where exists tool_calls((action:\"shell\" AND outcome:\"error\"))",
        ),
        (
            "sessions where seq(command:\"cargo test\" AND outcome:error -> tool:Edit -> command:\"cargo test\" AND outcome:ok)",
            "sessions where seq((command:\"cargo test\" AND outcome:\"error\") -> tool:\"Edit\" -> (command:\"cargo test\" AND outcome:\"ok\"))",
        ),
        (
            "messages where written_by:human | group by session.origin | count",
            "messages where written_by:\"human\" | group by session.origin | agg count",
        ),
        (
            "messages where active:false",
            "messages where active:\"false\"",
        ),
        (
            "parent:claude-code:7f3a2c10-0002-4000-8000-000000000002",
            "parent:\"claude-code:7f3a2c10-0002-4000-8000-000000000002\"",
        ),
        (
            "connected:codex:01a20000-0005-7000-8000-000000000005",
            "connected:\"codex:01a20000-0005-7000-8000-000000000005\"",
        ),
        ("since:2026-10-05", "since:\"2026-10-05\""),
        (
            "started between 2026-10-02 and 2026-10-03",
            "started between \"2026-10-02\" and \"2026-10-03\"",
        ),
        ("has:compaction", "has:\"compaction\""),
        ("model:claude-opus-5-5", "model:\"claude-opus-5-5\""),
        (
            "origin:(chatgpt|claude-ai)",
            "origin:(\"chatgpt\"|\"claude-ai\")",
        ),
        (
            "annotations where kind:decision",
            "annotations where kind:\"decision\"",
        ),
        (
            "origin:codex with messages(id, role)[role:assistant, last:1]",
            "origin:\"codex\" with messages(id, role)[role:\"assistant\", last:1]",
        ),
        (
            "similar:\"ambiguous date formats\"",
            "similar:\"ambiguous date formats\"",
        ),
        ("repo:other", "repo:\"other\""),
        (
            "session_links where kind:continuation",
            "session_links where kind:\"continuation\"",
        ),
        ("link:fork", "link:\"fork\""),
        ("tree_cost > 2.90", "tree_cost > \"2.90\""),
        ("a OR b AND c", "((b AND c) OR a)"),
        ("(a OR b) AND c", "((a OR b) AND c)"),
    ];
    let actual = examples
        .iter()
        .map(|(text, _)| {
            (
                *text,
                query_language::format::format_query(&p(text), &catalog).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let expected = examples
        .iter()
        .map(|(text, expected)| (*text, expected.to_string()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn shared_tree_keeps_boolean_modes_values_and_spans() {
    let ConditionKind::And { conditions } = &cond("repo:x \"a b\" refactor*") else {
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
    assert_eq!(
        parse("cats and dogs").unwrap_err().suggestion(),
        Some("cats AND dogs")
    );
    assert!(
        matches!(&cond("cats \"and\" dogs"), ConditionKind::And { conditions } if conditions.len() == 3)
    );
    assert!(matches!(&cond("a OR b"), ConditionKind::Or { .. }));
    for text in ["NOT tag:x", "-tag:x"] {
        assert!(matches!(&cond(text), ConditionKind::Not { .. }));
    }
    assert!(
        matches!(&cond("origin:(a|b|c)"), ConditionKind::Match { value: Value::Set { values }, .. } if values.len() == 3)
    );
    assert!(matches!(
        &cond("kind:*"),
        ConditionKind::Match {
            value: Value::Any,
            ..
        }
    ));
    assert!(matches!(
        &cond("cost:unknown"),
        ConditionKind::Match {
            value: Value::Unknown,
            ..
        }
    ));
    assert!(
        matches!(&cond("tag:project:morrow"), ConditionKind::Match { value: Value::Scalar { text, .. }, .. } if text == "project:morrow")
    );
}

#[test]
fn compact_comparisons_and_unknown_fields_have_executable_corrections() {
    for (text, correction, code, expected) in [
        (
            "messages:>=10",
            "messages >= 10",
            polylogue_model::Code::InvalidQuery,
            ConditionKind::Compare {
                left: Expression::field("messages"),
                op: Op::Ge,
                right: Expression::value(Value::scalar("10")),
            },
        ),
        (
            "reop:morrow",
            "repo:morrow",
            polylogue_model::Code::UnknownField,
            ConditionKind::Match {
                field: "repo".into(),
                value: Value::scalar("morrow"),
            },
        ),
    ] {
        let error = parse(text).unwrap_err();
        assert_eq!(error.kind(), code);
        assert_eq!(error.suggestion(), Some(correction));
        assert_eq!(cond(correction), expected);
    }
}

#[test]
fn references_calls_and_sequences_keep_structural_identity() {
    for text in ["https://chatgpt.com/c/abc", "@3.2"] {
        assert!(matches!(&cond(text), ConditionKind::Reference { reference } if reference == text));
    }
    assert!(
        matches!(&cond("weekday_offset()"), ConditionKind::Text { text, .. } if text == "weekday_offset()")
    );
    assert!(
        matches!(&cond("parse(x) fails"), ConditionKind::And { conditions } if conditions.len() == 2)
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
    } = &p("tool_calls where seq(tool:Bash -> [next] tool:Edit -> [within 5m] tool:Bash)").source
    else {
        panic!()
    };
    assert_eq!(unit, "tool_calls");
    assert!(!all);
    assert_eq!(steps.len(), 3);
    assert_eq!(
        *gaps,
        vec![
            Gap::Next,
            Gap::Within {
                distance: Distance::Time {
                    duration: Value::scalar("5m")
                }
            }
        ]
    );
}

#[test]
fn shared_units_stages_and_nested_queries_use_product_names() {
    for (text, unit) in [
        ("messages where role:user", "messages"),
        ("TOOL_CALLS", "tool_calls"),
        ("deliveries | count", "deliveries"),
        ("", "sessions"),
        ("   ", "sessions"),
    ] {
        assert!(
            matches!(p(text).source, Source::Unit { unit: actual, .. } if actual == unit),
            "{text}"
        );
    }
    // A unit has one name; its singular is a misspelling with a correction.
    let error = parse("message where role:user").unwrap_err();
    assert_eq!(error.suggestion(), Some("messages where role:user"));
    assert!(matches!(
        &cond("messages >= 10"),
        ConditionKind::Compare { .. }
    ));
    let source = "tool_calls | group by day(at), tool | agg count, sum(duration), p90(duration), count_distinct(session) | sort by count desc | where count >= 3 | limit 5 | offset 10";
    let mut catalog = (*crate::catalog::catalog(&[])).clone();
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
        polylogue_model::Code::UnknownField,
        "elapsed duration has no product producer yet"
    );
    assert_eq!(query.transforms.len(), 6);
    let TransformKind::Aggregate { aggregates } = &query.transforms[1].kind else {
        panic!()
    };
    assert_eq!(aggregates[2].column(), "p90_duration");
    let error = parse("sessions | except (messages where chrono)").unwrap_err();
    let QueryError::Language { error, .. } = error else {
        panic!("expected a query language error, got {error:?}");
    };
    assert_eq!(error.kind, query_language::ErrorKind::InvalidQuery);
    assert_eq!(
        error.message,
        "except needs rows of one unit or the same projected fields and types; project both sides with `| fields …`"
    );
    let accepted = p("sessions | except (sessions where chrono)");
    let TransformKind::Combine { how, query } = &accepted.transforms[0].kind else {
        panic!("expected a combination, got {:?}", accepted.transforms);
    };
    assert_eq!(*how, Combination::Except);
    let Source::Unit {
        unit, condition, ..
    } = &query.source
    else {
        panic!("expected the sessions unit, got {:?}", query.source);
    };
    assert_eq!(unit, "sessions");
    assert_eq!(
        condition.as_ref().unwrap().kind,
        ConditionKind::Text {
            text: "chrono".into(),
            mode: TextMode::Word
        }
    );
    let nested =
        p("origin:codex | union (repo:morrow | where cost > 1) | except (sessions where chrono)");
    assert_eq!(nested.transforms.len(), 2);
    assert!(
        matches!(&cond("count(tool_calls where outcome:error) >= 3"), ConditionKind::Count { unit, op: Op::Ge, count: 3, condition: Some(_), .. } if unit == "tool_calls")
    );
    assert!(
        matches!(&cond("id in (messages where chrono | group by session)"), ConditionKind::In { query, .. } if query.transforms.len() == 1)
    );
}

#[test]
fn repairs_are_stable() {
    let error = parse("\"unclosed phrase").unwrap_err();
    assert_eq!(error.suggestion(), Some("\"unclosed phrase\""));
    for (text, code) in [
        ("messages where", polylogue_model::Code::InvalidQuery),
        (
            "kind:* where exists tool_calls(outcome:error)",
            polylogue_model::Code::UnknownField,
        ),
    ] {
        assert_eq!(parse(text).unwrap_err().kind(), code, "{text}");
    }
    // A quoted `where` is a search word, not the keyword.
    let ConditionKind::And { conditions } = &cond("kind:* \"where\"") else {
        panic!("not a conjunction");
    };
    assert!(matches!(
        &conditions[1].kind,
        ConditionKind::Text { text, .. } if text == "where"
    ));
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
fn former_near_spelling_suggests_similar() {
    let error = parse("near:dates").unwrap_err();
    assert_eq!(error.kind(), polylogue_model::Code::UnknownField);
    assert_eq!(error.suggestion(), Some("similar:dates"));
    assert!(matches!(
        &cond("similar:dates"),
        ConditionKind::Rank {
            list: RankList::Similar,
            ..
        }
    ));
}

#[test]
fn shorthand_time_bounds_keep_their_shared_typed_meaning() {
    assert!(matches!(
        &cond("since:2026-10-05"),
        ConditionKind::Since { .. }
    ));
    assert!(matches!(
        &cond("until:2026-10-05"),
        ConditionKind::Until { .. }
    ));
    assert!(matches!(
        &cond("messages where at < 2026-10-05"),
        ConditionKind::Compare { op: Op::Lt, .. }
    ));
}

#[test]
fn product_namespace_spelling_binds_to_the_same_declared_unit() {
    let catalog = crate::catalog::catalog(&[]);
    for (qualified, local, unit) in [
        (
            "pl.sessions where repo:morrow",
            "sessions where repo:morrow",
            "sessions",
        ),
        (
            "pl.messages where role:user",
            "messages where role:user",
            "messages",
        ),
    ] {
        assert!(
            matches!(&p(qualified).source, Source::Unit { unit: bound, .. } if bound.strip_prefix("pl.").unwrap_or(bound) == unit),
            "{qualified}"
        );
        assert_eq!(
            query_language::identity::normal_form(&p(qualified), &catalog).unwrap(),
            query_language::identity::normal_form(&p(local), &catalog).unwrap(),
        );
    }
}

#[test]
fn time_companion_catalog_keeps_precision_and_unknown() {
    for text in [
        "messages where at.time_quality:operator at.precision:bounded",
        "events where time_quality:\"unknown\" precision:unknown",
        "sessions where started_at.time_quality:captured started_at.precision:minute",
        "sessions where last_active.precision:unknown",
        "tool_calls where result_at.time_quality:unknown",
        "documents where source_time.precision:second",
        "deliveries where at.time_quality:unknown",
    ] {
        p(text);
    }
    // Values outside a companion's closed set are bad values; a companion
    // the unit does not declare is an unknown field.
    for (text, code) in [
        (
            "messages where at.precision:milliseconds",
            polylogue_model::Code::BadValue,
        ),
        (
            "messages where at.time_quality:guessed",
            polylogue_model::Code::BadValue,
        ),
        (
            "sessions where precision:instant",
            polylogue_model::Code::UnknownField,
        ),
        (
            "sessions where started.precision:instant",
            polylogue_model::Code::UnknownField,
        ),
        (
            "events where precision:\"unknown\"",
            polylogue_model::Code::BadValue,
        ),
    ] {
        assert_eq!(parse(text).unwrap_err().kind(), code, "{text}");
    }
    assert!(matches!(
        &cond("events where at.time_quality:unknown"),
        ConditionKind::Match {
            value: Value::Unknown,
            ..
        }
    ));
    assert!(
        matches!(&cond("events where at.time_quality:\"unknown\""), ConditionKind::Match { value: Value::Scalar { text, quoted: true }, .. } if text == "unknown")
    );
    let catalog = crate::catalog::catalog(&[]);
    let text = "events where at.time_quality:unk";
    let completions = query_language::complete::complete(text, text.len(), &catalog);
    let literal = completions
        .iter()
        .find(|candidate| candidate.value == "\"unknown\"")
        .expect("stored unknown quality is a quoted literal completion");
    let completed = format!(
        "{}{}{}",
        &text[..literal.replace.start],
        literal.value,
        &text[literal.replace.end..]
    );
    assert!(
        matches!(&cond(&completed), ConditionKind::Match { value: Value::Scalar { text, quoted: true }, .. } if text == "unknown")
    );
}

#[test]
fn a_lone_exact_annotation_revision_reads_the_annotations_unit() {
    let unit = |text: &str| match p(text).source {
        Source::Unit { unit, .. } => unit,
        other => panic!("{text}: {other:?}"),
    };
    assert_eq!(unit("annotation:root@revision"), "annotations");
    assert_eq!(unit("annotation:root"), "sessions");
    assert_eq!(unit("block:test:message#0"), "sessions");
    assert_eq!(unit("sessions where annotation:root@revision"), "sessions");
}
