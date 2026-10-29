use super::fixture_clock;
//! The query examples over the Morrow archive built by the real
//! loop and import (`examples/morrow/expected/main/expected.json`).
//!
//! Each example is compared by the shape of its expected value. Examples
//! that need a feature not built yet are listed in [`GAPS`] with the reason;
//! a listed example that starts to match fails the test, so the list is
//! removed from as features land.

use crate::daemon::query::*;
use polylogue_testkit::{Morrow, Set, morrow_dir, short};
use std::collections::BTreeMap;

/// 2026-10-08 12:00 Europe/Warsaw (`FACTS.md`, the examples' clock).
const NOW: i64 = 1_791_453_600_000;

/// Examples that do not match yet, by number, with what they wait for.
const GAPS: &[(&str, &str)] = &[
    (
        "3",
        "cost: the example prices from its own table (FACTS.md), not the real price catalog",
    ),
    ("21", "seq() does not yet take search words over tool calls"),
    (
        "34",
        "an empty repo: result does not yet list the known repositories",
    ),
    ("38", "tree cost differs from the historical example prices"),
];

fn archive_with_notes() -> Morrow {
    use crate::daemon::service::ArchiveMode;
    use crate::daemon::{Request, Response, service};
    use polylogue_model::{Author, Block, NewNote, NoteKind, Reference};
    use polylogue_testkit::session;

    let archive = Morrow::build(Set::Main);
    let decision = {
        let pool = archive.reads();
        let reader = pool.reader().unwrap();
        reader
            .messages(&session(2), false)
            .unwrap()
            .into_iter()
            .find(|message| {
                message.blocks.iter().any(|block| {
                    matches!(block,
                Block::Text { text } if text.contains("let's use chrono, not time"))
                })
            })
            .unwrap()
            .id
    };
    let mut runtime =
        crate::test_daemon::start_runtime(&archive.root, archive.config(), ArchiveMode::Standalone)
            .unwrap();
    for (id, kind, key, value, targets) in [
        (
            "morrow-project",
            NoteKind::Tag,
            Some("project:morrow"),
            None,
            [1, 2, 4, 5, 6]
                .into_iter()
                .map(|n| Reference::Session(session(n)))
                .collect(),
        ),
        (
            "morrow-research",
            NoteKind::Tag,
            Some("research"),
            None,
            vec![Reference::Session(session(7))],
        ),
        (
            "morrow-decision",
            NoteKind::Decision,
            None,
            Some("decided: chrono, not time"),
            vec![Reference::Message(decision)],
        ),
    ] {
        let response = service::handle(
            Request::AddNote {
                request_id: id.into(),
                note: NewNote {
                    kind,
                    key: key.map(Into::into),
                    value: value.map(Into::into),
                    author: Author::Operator,
                    targets,
                    relations: vec![],
                    revises: None,
                    valid_until: None,
                },
            },
            &runtime.daemon,
        );
        assert!(matches!(response, Response::Ok { .. }), "{response:?}");
    }
    runtime.shutdown().unwrap();
    archive
}

fn ask(text: &str) -> Ask<'_> {
    Ask {
        text,
        cursor: None,
        page_size: Some(1000),
        why: false,
        session: None,
        clock: fixture_clock(NOW, "Europe/Warsaw", 240),
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
fn compare(
    reader: &Reader<'_>,
    number: &str,
    got: &Result<Value, Failure>,
    want: &Value,
) -> Result<(), String> {
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
        Value::Array(items) if number == "31" => {
            use polylogue_model::{Block, Reference};
            let mut got: Vec<_> = rows(v)
                .iter()
                .map(|note| json!([note["value"], note["target"]]).to_string())
                .collect();
            let mut wanted = Vec::new();
            for item in items {
                let number = item[1]
                    .as_str()
                    .unwrap()
                    .strip_prefix('#')
                    .unwrap()
                    .parse::<u8>()
                    .unwrap();
                let prompt = item[2].as_str().unwrap();
                let message = reader
                    .messages(&polylogue_testkit::session(number), false)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .find(|message| {
                        message.blocks.iter().any(|block| {
                            matches!(block,
                            Block::Text { text } if text == prompt)
                        })
                    })
                    .ok_or_else(|| format!("decision target prompt missing in #{number}"))?;
                wanted
                    .push(json!([item[0], Reference::Message(message.id).to_string()]).to_string());
            }
            got.sort();
            wanted.sort();
            differ(format!("{got:?}"), format!("{wanted:?}"))
        }
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
fn codex_message_counts_include_script_commands() {
    let archive = Morrow::build(Set::Main);
    let pool = archive.reads();
    let reader = pool.reader().unwrap();
    let result = run(
        &reader,
        ask("messages where origin:codex | group by session | count"),
    )
    .unwrap();
    let counts: BTreeMap<_, _> = rows(&result)
        .iter()
        .map(|row| (group_key(row), row["count"].as_u64().unwrap()))
        .collect();
    assert_eq!(
        counts,
        BTreeMap::from([("#5".into(), 13), ("#6".into(), 10)])
    );
}

#[test]
fn the_query_examples_match_on_the_real_archive() {
    let w = archive_with_notes();
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
            if let Err(why) = compare(&r, number, &run(&r, ask(q)), want) {
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

/// Every example query, paged by cursor at every page size from 1 to one
/// more than its rows, concatenates to its unpaged rows, and every page
/// carries the unpaged count.
#[test]
fn pages_of_every_size_concatenate_to_the_whole_answer() {
    let w = archive_with_notes();
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(morrow_dir().join("expected/main/expected.json")).unwrap(),
    )
    .unwrap();
    let mut wrong = Vec::new();
    let mut paged = 0;
    for key in expected["queries"].as_object().unwrap().keys() {
        let text = key.split_once(' ').unwrap().1;
        let Ok(whole) = run(&r, ask(text)) else {
            continue;
        };
        let want = rows(&whole);
        for size in 1..=want.len() as u64 + 1 {
            let mut got = Vec::new();
            let mut cursor: Option<String> = None;
            loop {
                let page = run(
                    &r,
                    Ask {
                        page_size: Some(size),
                        cursor: cursor.as_deref(),
                        ..ask(text)
                    },
                )
                .unwrap_or_else(|f| panic!("{key} by {size}: {} {}", f.kind, f.message));
                if page["count"] != whole["count"] {
                    wrong.push(format!(
                        "{key} by {size}: count {} of {}",
                        page["count"], whole["count"]
                    ));
                }
                got.extend(rows(&page).iter().cloned());
                cursor = page["next"].as_str().map(str::to_owned);
                if cursor.is_none() {
                    break;
                }
            }
            if got != want {
                wrong.push(format!("{key} by {size}: the pages differ from the rows"));
            }
        }
        paged += 1;
    }
    assert!(paged > 30, "{paged} examples paged");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
