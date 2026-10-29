//! Query-language behavior over the archive built through the real loop.
use polylogue_model::LogicalDay;
use polylogue_query::{Options, Param, QueryError, compile};
use polylogue_store::{Reader, SqlValue};
use polylogue_testkit::{Morrow, Set, short};
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn options() -> Options {
    Options {
        clock: polylogue_query::time::clock(
            1_791_453_600_000,
            LogicalDay::named("Europe/Warsaw", 240).unwrap(),
        )
        .unwrap(),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|origin| origin.name.to_owned())
            .collect(),
        command_words: crate::commands::command_words(),
        page_size: None,
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    }
}
fn values(params: &[Param]) -> Vec<SqlValue> {
    params
        .iter()
        .map(|p| match p {
            Param::Int(n) => SqlValue::Int(*n),
            Param::Real(n) => SqlValue::Real(*n),
            Param::Text(s) => SqlValue::Text(s.clone()),
        })
        .collect()
}
fn run(reader: &Reader<'_>, text: &str) -> Result<Value, QueryError> {
    let program = compile(text, &options())?;
    let mut rows = reader
        .query(&program.sql, &values(&program.params))
        .unwrap_or_else(|e| panic!("{text}: {e}"));
    for row in &mut rows {
        row.as_object_mut()
            .unwrap()
            .retain(|name, _| !name.starts_with('_'));
    }
    Ok(json!({"rows": rows, "outcome": if rows.is_empty() {"empty"} else {"rows"}}))
}
fn q(reader: &Reader<'_>, text: &str) -> Value {
    run(reader, text).unwrap_or_else(|e| panic!("{text}: {e}"))
}
fn rows(value: &Value) -> &[Value] {
    value["rows"].as_array().unwrap()
}
fn s(value: &Value) -> String {
    short(value.as_str().unwrap_or(""))
}
fn ids(value: &Value) -> Vec<String> {
    rows(value)
        .iter()
        .map(|r| {
            s(if r["session"].is_string() {
                &r["session"]
            } else {
                &r["id"]
            })
        })
        .collect()
}
fn group_key(row: &Value) -> String {
    row.as_object()
        .unwrap()
        .iter()
        .find(|(key, _)| key.as_str() != "count")
        .map(|(_, v)| s(v))
        .unwrap()
}

/// A grouped result is rows. `limit`, `where` and `sort` after a
/// group act on the groups, never on the ungrouped rows.
#[test]
fn a_grouped_result_is_rows() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let base = "tool_calls | group by action | count";
    let all: Vec<(String, u64)> = rows(&q(&r, base))
        .iter()
        .map(|g| (group_key(g), g["count"].as_u64().unwrap()))
        .collect();
    assert!(all.len() > 2, "{all:?}");
    let limited: Vec<String> = rows(&q(&r, &format!("{base} | limit 1")))
        .iter()
        .map(group_key)
        .collect();
    assert_eq!(limited, [all[0].0.clone()]);
    let mut big: Vec<String> = rows(&q(&r, &format!("{base} | where count >= 6")))
        .iter()
        .map(group_key)
        .collect();
    let mut want: Vec<String> = all
        .iter()
        .filter(|(_, c)| *c >= 6)
        .map(|(k, _)| k.clone())
        .collect();
    big.sort();
    want.sort();
    assert_eq!(big, want);
    let asc: Vec<u64> = rows(&q(&r, &format!("{base} | sort by count asc")))
        .iter()
        .map(|g| g["count"].as_u64().unwrap())
        .collect();
    assert_eq!(asc.len(), all.len());
    assert!(asc.windows(2).all(|w| w[0] <= w[1]), "{asc:?}");
}

/// Three-valued filters. `U where C`, `U where NOT C` and
/// `U where F:unknown` partition U; no row matches both `exists` and
/// `NOT exists`.
#[test]
fn three_valued_filters_partition() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let all: BTreeSet<String> = ids(&q(&r, "kind:*")).into_iter().collect();
    assert_eq!(all.len(), 8);
    for (c, unknown) in [
        ("model:claude-fable-5-1", "model:unknown"),
        ("cost > 1", "cost:unknown"),
        ("repo:morrow", "repo:unknown"),
    ] {
        let yes: BTreeSet<String> = ids(&q(&r, &format!("kind:* {c}"))).into_iter().collect();
        let no: BTreeSet<String> = ids(&q(&r, &format!("kind:* NOT {c}")))
            .into_iter()
            .collect();
        let unk: BTreeSet<String> = ids(&q(&r, &format!("kind:* {unknown}")))
            .into_iter()
            .collect();
        assert!(
            yes.is_disjoint(&no) && yes.is_disjoint(&unk) && no.is_disjoint(&unk),
            "{c}: overlap {yes:?} {no:?} {unk:?}"
        );
        let union: BTreeSet<String> = yes.union(&no).chain(&unk).cloned().collect();
        assert_eq!(
            union, all,
            "{c}: the three results do not cover every session"
        );
    }
    let top: BTreeSet<String> = ids(&q(&r, "sessions")).into_iter().collect();
    let has: BTreeSet<String> = ids(&q(&r, "sessions where exists tool_call(outcome:error)"))
        .into_iter()
        .collect();
    let hasnt: BTreeSet<String> = ids(&q(&r, "sessions where NOT exists tool_call(outcome:error)"))
        .into_iter()
        .collect();
    assert!(has.is_disjoint(&hasnt), "{has:?} {hasnt:?}");
    // The rest have no failing call but a call whose outcome is unknown:
    // `exists` is unknown for them, so neither form returns them.
    let unknown: BTreeSet<String> = ids(&q(&r, "sessions where exists tool_call(outcome:unknown)"))
        .into_iter()
        .filter(|s| !has.contains(s))
        .collect();
    assert_eq!(
        has.union(&hasnt)
            .chain(&unknown)
            .cloned()
            .collect::<BTreeSet<_>>(),
        top,
        "{has:?} {hasnt:?} {unknown:?}"
    );
    assert!(hasnt.is_disjoint(&unknown), "{hasnt:?} {unknown:?}");
}

/// Defaults are bound once. A condition every row satisfies,
/// nested where it is not a top-level condition, leaves the rows unchanged.
#[test]
fn a_nested_condition_does_not_drop_a_default() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let count = |text: &str| rows(&q(&r, text))[0]["count"].as_u64().unwrap();
    assert_eq!(count("tool_calls | agg count"), 37);
    assert_eq!(
        count("tool_calls where (action:shell OR NOT action:shell) | agg count"),
        37
    );
    assert_eq!(
        count("tool_calls where NOT NOT (action:* OR action:shell) | agg count"),
        37
    );
    // The same field at the top level, or `F:*`, drops the default: the
    // code-mode wrappers whose commands are counted come back.
    let every_call = count("tool_calls where action:* | agg count");
    assert!(every_call > 37, "{every_call}");
    assert_eq!(
        count("tool_calls where (action:shell OR NOT action:shell) action:* | agg count"),
        every_call
    );
    // A later `| where` narrows the bound rows; it binds no default again.
    assert_eq!(count("tool_calls | where action:* | agg count"), 37);
    let chrono = |text: &str| rows(&q(&r, text)).len();
    let human = r#"messages where written_by:human "chrono""#;
    assert_eq!(chrono(human), 2);
    for nested in [
        "(duplicate:true OR NOT duplicate:true)",
        "(active:true OR NOT active:true)",
        "NOT (kind:subagent AND kind:main)",
    ] {
        assert_eq!(chrono(&format!("{human} {nested}")), 2, "{nested}");
    }
    assert!(chrono(&format!("{human} duplicate:*")) >= 2);
}

/// A value outside a field's closed set is refused, never an empty
/// answer; every value of the set, `*` and `unknown` (where allowed) pass.
#[test]
fn a_value_outside_a_closed_set_is_refused() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    // A value of another unit's field of the same name, and the unit that has it.
    for (bad, elsewhere) in [
        ("sessions where outcome:error", Some("tool_calls")),
        ("tool_calls where outcome:solved", Some("sessions")),
        ("deliveries where outcome:ok", Some("tool_calls")),
        ("messages where role:asistant", None),
        ("kind:continuation", Some("links")),
        ("kind:mian", None),
    ] {
        let f = run(&r, bad)
            .map(|v| panic!("{bad} was answered ({}) instead of refused", v["outcome"]))
            .unwrap_err();
        assert!(
            f.to_string().contains(" is one of ") && f.to_string().contains(", "),
            "{bad}: the refusal does not name the set: {}",
            f
        );
        if let Some(unit) = elsewhere {
            assert!(
                f.to_string().contains(&format!(" of {unit}")),
                "{bad}: the refusal does not name {unit}: {}",
                f
            );
        }
    }
    // Every value of every closed set, `*`, and `unknown` where the set
    // has it, runs as a query.
    let mut ran = 0;
    for field in polylogue_query::fields::FIELDS {
        let Some(values) = field.values else {
            continue;
        };
        let unit = field.unit.name();
        for value in values.iter().chain(["*"].iter()) {
            let text = format!("{unit} where {}:{value}", field.name);
            if let Err(f) = run(&r, &text) {
                assert!(
                    f.kind() == "not_built",
                    "{text} refused: {} {}",
                    f.kind(),
                    f
                );
            }
            ran += 1;
        }
    }
    assert!(ran > 50, "{ran}");
}

/// The turns unit on Morrow: every session with a prompt has turns, a
/// turn's identity is one of its messages, its span is ordered, and the
/// prompts the turns count are the session's on-path human prompts.
#[test]
fn turns_partition_each_sessions_prompts() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let turns = q(&r, "turns | limit 1000");
    let ts = rows(&turns);
    assert!(!ts.is_empty(), "{turns}");
    for t in ts {
        let id = s(&t["id"]);
        assert!(!id.is_empty(), "{t}");
        let one = q(&r, &format!("messages where id:{id}"));
        assert_eq!(rows(&one).len(), 1, "turn id {id} is a message: {t}");
        if let (Some(a), Some(b)) = (t["start"].as_i64(), t["end"].as_i64()) {
            assert!(a <= b, "{t}");
        }
        assert!(
            matches!(s(&t["path"]).as_str(), "active" | "unknown"),
            "{t}"
        );
    }
    // Every on-path human prompt is counted in exactly one turn.
    let burst: u64 = ts.iter().map(|t| t["burst"].as_u64().unwrap_or(0)).sum();
    let prompts = q(&r, "messages where written_by:human | agg count");
    assert_eq!(
        Some(burst),
        rows(&prompts).first().and_then(|g| g["count"].as_u64()),
        "{prompts}"
    );
    // The session prefix narrows them.
    let scoped = q(&r, "turns where session.kind:main | agg count");
    assert!(!rows(&scoped).is_empty(), "{scoped}");
}

#[test]
fn no_sort_key_holds_bytes() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let opts = options();
    let mut checked = 0;
    for unit in polylogue_query::units::UNITS {
        let questions = std::iter::once(unit.name.to_owned()).chain(
            polylogue_query::fields::FIELDS
                .iter()
                .filter(|f| f.unit == unit.unit && f.sql.is_some())
                .map(|f| format!("{} | sort by {}", unit.name, f.name)),
        );
        for question in questions {
            let Ok(program) = polylogue_query::compile(&question, &opts) else {
                continue;
            };
            if program.order.is_empty() {
                continue;
            }
            let bytes = program
                .order
                .iter()
                .map(|key| format!("typeof(\"{}\") = 'blob'", key.column))
                .collect::<Vec<_>>()
                .join(" OR ");
            let mut params = values(&program.params);
            // The page's row limit.
            params.push(SqlValue::Int(i64::MAX));
            let rows = reader
                .query(
                    &format!(
                        "SELECT count(*) AS n FROM ({}) WHERE {bytes}",
                        program.page_sql(false)
                    ),
                    &params,
                )
                .unwrap_or_else(|e| panic!("{question}: {e}"));
            assert_eq!(rows[0]["n"], 0, "{question}");
            checked += 1;
        }
    }
    assert!(checked > 100, "{checked}");
}
