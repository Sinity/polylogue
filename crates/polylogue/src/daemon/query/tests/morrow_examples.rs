//! The query examples over the Morrow archive built by the real
//! loop and import (`examples/morrow/expected/main/expected.json`).
//!
//! Each example is compared by the shape of its expected value. Examples
//! that need a feature not built yet are listed in [`GAPS`] with the reason;
//! a listed example that starts to match fails the test, so the list is
//! removed from as features land.

use crate::daemon::query::*;
use polylogue_testkit::{Morrow, Set, morrow_dir, short};
use std::collections::{BTreeMap, BTreeSet};

/// 2026-10-08 12:00 Europe/Warsaw (`FACTS.md`, the examples' clock).
const NOW: i64 = 1_791_453_600_000;

/// Examples that do not match yet, by number, with what they wait for.
const GAPS: &[(&str, &str)] = &[
    (
        "3",
        "cost: the example prices from its own table (FACTS.md), not the real price catalog",
    ),
    (
        "6",
        "tags: the example's annotation notes are not written by the fixture",
    ),
    (
        "7",
        "tags: the example's annotation notes are not written by the fixture",
    ),
    (
        "19",
        "tags: the example's annotation notes are not written by the fixture",
    ),
    (
        "31",
        "notes: the example's annotation notes are not written by the fixture",
    ),
    (
        "11",
        "Codex is stored one message per item, so #5 and #6 have more messages than FACTS counts",
    ),
    (
        "14",
        "inherited:* does not yet return inherited copies as rows",
    ),
    ("21", "seq() does not yet take search words over tool calls"),
    (
        "34",
        "an empty repo: result does not yet list the known repositories",
    ),
    ("38", "tree_cost is not built"),
];

fn ask(text: &str) -> Ask<'_> {
    Ask {
        text,
        cursor: None,
        page_size: Some(1000),
        why: false,
        session: None,
        now: NOW,
        day: LogicalDay::named("Europe/Warsaw", 240).unwrap(),
        embeddings: false,
        near: None,
        fields: &[],
    }
}

fn rows(v: &Value) -> &[Value] {
    v["rows"].as_array().map_or(&[], Vec::as_slice)
}

fn s(v: &Value) -> String {
    short(v.as_str().unwrap_or(""))
}

/// The session a row names: a session row's `id`, any other row's `session`.
fn row_session(r: &Value) -> String {
    if r["session"].is_string() {
        s(&r["session"])
    } else {
        s(&r["id"])
    }
}

/// The group key of a grouped row: its one non-count field.
fn group_key(r: &Value) -> String {
    r.as_object()
        .and_then(|o| o.iter().find(|(k, _)| k.as_str() != "count"))
        .map(|(_, v)| s(v))
        .unwrap_or_default()
}

/// Compares one example's result with its expected value; `Err` says how
/// they differ.
fn compare(number: &str, got: &Result<Value, Failure>, want: &Value) -> Result<(), String> {
    let v = match (got, want) {
        (Err(f), _) if want.get("outcome") == Some(&json!("error")) => {
            let _ = f;
            return Ok(());
        }
        (Err(f), _) => return Err(format!("error {} {}", f.kind, f.message)),
        (Ok(v), _) => v,
    };
    let differ = |got: String, want: String| {
        if got == want {
            Ok(())
        } else {
            Err(format!("got {got}, want {want}"))
        }
    };
    match want {
        Value::Number(n) => differ(
            rows(v)
                .first()
                .map_or(json!(null), |r| r["count"].clone())
                .to_string(),
            n.to_string(),
        ),
        Value::Array(items) if items.iter().all(Value::is_string) => {
            let got: Vec<String> = rows(v).iter().map(row_session).collect();
            let want: Vec<String> = items.iter().map(s).collect();
            differ(format!("{got:?}"), format!("{want:?}"))
        }
        Value::Array(items) => {
            // Messages, tool calls and links: compared by the session (or
            // the link's parent and child) and, where given, the command
            // and exit code or how; sorted, since a text search ranks.
            let mut got: Vec<String> = rows(v)
                .iter()
                .map(|r| match number {
                    "15" => format!(
                        "{} {} {}",
                        row_session(r),
                        r["command"].as_str().unwrap_or(""),
                        r["exit_code"]
                    ),
                    "36" => format!(
                        "{} {} {}",
                        s(&r["parent"]),
                        s(&r["child"]),
                        r["how"].as_str().unwrap_or("")
                    ),
                    _ => row_session(r),
                })
                .collect();
            let mut want: Vec<String> = items
                .iter()
                .map(|i| {
                    let i = i.as_array().unwrap();
                    match number {
                        "15" => format!("{} {} {}", s(&i[0]), i[1].as_str().unwrap(), i[2]),
                        "36" => format!("{} {} {}", s(&i[0]), s(&i[1]), i[2].as_str().unwrap()),
                        "31" => s(&i[1]),
                        _ => s(&i[0]),
                    }
                })
                .collect();
            if number != "32" && number != "24" {
                got.sort();
                want.sort();
            }
            differ(format!("{got:?}"), format!("{want:?}"))
        }
        Value::Object(o) if o.contains_key("count") => {
            let r = rows(v).first().cloned().unwrap_or(json!({}));
            let mut diffs = Vec::new();
            for (k, w) in o {
                let g = &r[k.as_str()];
                let same = match (g.as_f64(), w.as_f64()) {
                    (Some(a), Some(b)) => (a - b).abs() < 1e-9,
                    _ => g == w,
                };
                if !same {
                    diffs.push(format!("{k}: got {g}, want {w}"));
                }
            }
            if diffs.is_empty() {
                Ok(())
            } else {
                Err(diffs.join("; "))
            }
        }
        Value::Object(o) if o.contains_key("outcome") => {
            let mut diffs = Vec::new();
            if v["outcome"] != o["outcome"] {
                diffs.push(format!("outcome {} want {}", v["outcome"], o["outcome"]));
            }
            if let Some(Value::Array(ids)) = o.get("rows") {
                let got: Vec<String> = rows(v).iter().map(row_session).collect();
                let want: Vec<String> = ids.iter().map(s).collect();
                if got != want {
                    diffs.push(format!("rows {got:?} want {want:?}"));
                }
            }
            if o.contains_key("known repos") && v.get("known_repos").is_none() {
                diffs.push("no known repositories listed".into());
            }
            if diffs.is_empty() {
                Ok(())
            } else {
                Err(diffs.join("; "))
            }
        }
        Value::Object(o) => {
            // A grouped count: key → count.
            let got: BTreeMap<String, u64> = rows(v)
                .iter()
                .map(|r| (group_key(r), r["count"].as_u64().unwrap_or(0)))
                .collect();
            let Some(want) = o
                .iter()
                .map(|(k, c)| Some((k.clone(), c.as_u64()?)))
                .collect::<Option<BTreeMap<String, u64>>>()
            else {
                return Err(format!("not compared: {want}"));
            };
            differ(format!("{got:?}"), format!("{want:?}"))
        }
        _ => Err(format!("unexpected expected value {want}")),
    }
}

#[test]
fn the_query_examples_match_on_the_real_archive() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(morrow_dir().join("expected/main/expected.json")).unwrap(),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for (key, want) in expected["queries"].as_object().unwrap() {
        let (number, text) = key.split_once(' ').unwrap();
        // 29 lists its variants: the example, with `kind:*`, and `model:unknown`.
        let cases: Vec<(String, Value)> = match (number, want) {
            ("29", Value::Object(o)) => vec![
                (text.to_owned(), o["default"].clone()),
                (format!("{text} kind:*"), o["kind:*"].clone()),
                ("model:unknown".to_owned(), o["model:unknown"].clone()),
            ],
            _ => vec![(text.to_owned(), want.clone())],
        };
        let mut failures = Vec::new();
        for (q, want) in &cases {
            if let Err(why) = compare(number, &run(&r, ask(q)), want) {
                failures.push(format!("{q}: {why}"));
            }
        }
        let gap = GAPS.iter().find(|(n, _)| *n == number);
        match (gap, failures.is_empty()) {
            (None, false) => wrong.push(format!("example {number}: {}", failures.join(" | "))),
            (Some((_, why)), true) => wrong.push(format!(
                "example {number} now matches; remove it from GAPS ({why})"
            )),
            _ => {}
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn counts_match_the_rows_paged() {
    // Every example's `count` equals the rows paged to the end.
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(morrow_dir().join("expected/main/expected.json")).unwrap(),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for key in expected["queries"].as_object().unwrap().keys() {
        let text = key.split_once(' ').unwrap().1;
        let Ok(first) = run(
            &r,
            Ask {
                page_size: Some(1),
                ..ask(text)
            },
        ) else {
            continue;
        };
        if first["shape"] == "summary" {
            continue;
        }
        let count = first["count"].as_u64();
        let mut paged = rows(&first).len() as u64;
        let mut cursor = first["next"].as_str().map(str::to_owned);
        while let Some(c) = cursor {
            let page = run(
                &r,
                Ask {
                    page_size: Some(1),
                    cursor: Some(&c),
                    ..ask(text)
                },
            )
            .unwrap();
            paged += rows(&page).len() as u64;
            cursor = page["next"].as_str().map(str::to_owned);
        }
        if count != Some(paged) {
            wrong.push(format!("{key}: count {count:?}, paged {paged}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn q(r: &Reader<'_>, text: &str) -> Value {
    run(r, ask(text)).unwrap_or_else(|f| panic!("{text}: {} {}", f.kind, f.message))
}

fn ids(v: &Value) -> Vec<String> {
    rows(v).iter().map(row_session).collect()
}

/// A grouped result is rows. `limit`, `where` and `sort` after a
/// group act on the groups, never on the ungrouped rows.
#[test]
fn law_129_a_grouped_result_is_rows() {
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
/// `U where F:unknown` partition U; no owner matches both `exists` and
/// `NOT exists`.
#[test]
fn law_132_three_valued_filters_partition() {
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
    assert_eq!(
        has.union(&hasnt).cloned().collect::<BTreeSet<_>>(),
        top,
        "{has:?} {hasnt:?}"
    );
}

/// Defaults are bound once. A condition every row satisfies,
/// nested where it is not a top-level condition, leaves the rows unchanged.
#[test]
fn law_133_a_nested_condition_does_not_drop_a_default() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let count = |text: &str| rows(&q(&r, text))[0]["count"].as_u64().unwrap();
    assert_eq!(count("tool_calls | agg count"), 37);
    assert_eq!(
        count("tool_calls where (action:shell OR NOT action:shell) | agg count"),
        37
    );
    let chrono = |text: &str| rows(&q(&r, text)).len();
    assert_eq!(chrono(r#"messages where written_by:human "chrono""#), 2);
    assert_eq!(
        chrono(
            r#"messages where written_by:human (inherited:true OR NOT inherited:true) "chrono""#
        ),
        2
    );
}

/// A value outside a field's closed set is refused, never an empty
/// answer; every value of the set, `*` and `unknown` (where allowed) pass.
#[test]
fn law_137_a_value_outside_a_closed_set_is_refused() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    for bad in [
        "tool_calls where outcome:solved",
        "messages where role:asistant",
        "kind:continuation",
        "kind:mian",
    ] {
        match run(&r, ask(bad)) {
            Err(f) => assert!(
                f.message.contains(',') || f.detail.is_some(),
                "{bad}: the refusal does not name the set: {}",
                f.message
            ),
            Ok(v) => panic!("{bad} was answered ({}) instead of refused", v["outcome"]),
        }
    }
    for good in [
        "tool_calls where outcome:ok",
        "tool_calls where outcome:error",
        "tool_calls where outcome:unknown",
        "messages where role:assistant",
        "messages where role:user",
        "kind:subagent",
        "kind:main",
        "kind:*",
    ] {
        if let Err(f) = run(&r, ask(good)) {
            panic!("{good} refused: {} {}", f.kind, f.message);
        }
    }
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
