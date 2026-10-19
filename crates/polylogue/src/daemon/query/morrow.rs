//! The spec's query examples over the Morrow archive built by the real
//! loop and import (`examples/morrow/expected/main/expected.json`).
//!
//! Each example is compared by the shape of its expected value. Examples
//! that need a feature not built yet are listed in [`GAPS`] with the reason;
//! a listed example that starts to match fails the test, so the list is
//! removed from as features land.

use super::*;
use polylogue_testkit::{Morrow, Set, morrow_dir, short};
use std::collections::BTreeMap;

/// 2026-10-08 12:00 Europe/Warsaw (`FACTS.md`, the examples' clock).
const NOW: i64 = 1_791_453_600_000;

/// Examples that do not match yet, by number, with what they wait for.
const GAPS: &[(&str, &str)] = &[
    ("3", "cost: pricing is not built (measurement lane)"),
    ("8", "cost: pricing is not built (measurement lane)"),
    ("9", "cost: pricing is not built (measurement lane)"),
    ("6", "tags: the example's annotation notes are not written by the fixture"),
    ("7", "tags: the example's annotation notes are not written by the fixture"),
    ("19", "tags: the example's annotation notes are not written by the fixture"),
    ("31", "notes: the example's annotation notes are not written by the fixture"),
    (
        "11",
        "Codex is stored one message per item, so #5 and #6 have more messages than FACTS counts",
    ),
    ("14", "inherited:* does not yet return inherited copies as rows"),
    ("21", "seq() does not yet take search words over tool calls"),
    ("34", "an empty repo: result does not yet list the known repositories"),
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
        Value::Number(n) => differ(rows(v).first().map_or(json!(null), |r| r["count"].clone()).to_string(), n.to_string()),
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
                    "15" => format!("{} {} {}", row_session(r), r["command"].as_str().unwrap_or(""), r["exit_code"]),
                    "36" => format!("{} {} {}", s(&r["parent"]), s(&r["child"]), r["how"].as_str().unwrap_or("")),
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
            if diffs.is_empty() { Ok(()) } else { Err(diffs.join("; ")) }
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
            if diffs.is_empty() { Ok(()) } else { Err(diffs.join("; ")) }
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
    // Law 76: every example's `count` equals the rows paged to the end.
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
        let Ok(first) = run(&r, Ask { page_size: Some(1), ..ask(text) }) else {
            continue;
        };
        if first["shape"] == "summary" {
            continue;
        }
        let count = first["count"].as_u64();
        let mut paged = rows(&first).len() as u64;
        let mut cursor = first["next"].as_str().map(str::to_owned);
        while let Some(c) = cursor {
            let page = run(&r, Ask { page_size: Some(1), cursor: Some(&c), ..ask(text) }).unwrap();
            paged += rows(&page).len() as u64;
            cursor = page["next"].as_str().map(str::to_owned);
        }
        if count != Some(paged) {
            wrong.push(format!("{key}: count {count:?}, paged {paged}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
