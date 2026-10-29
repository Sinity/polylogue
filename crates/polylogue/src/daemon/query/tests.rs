//! The query language end to end over a small archive shaped like the
//! Morrow example archive: the same sessions, kinds, costs, links and
//! messages, written row by row so the queries are tested apart from any
//! parser.

use super::*;
use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

const DAY: i64 = 86_400_000;
/// 2026-10-01T12:00Z.
const OCT1: i64 = 1_790_856_000_000;

fn at(day: i64, hour: i64) -> Option<i64> {
    Some(OCT1 + (day - 1) * DAY + (hour - 12) * 3_600_000)
}

struct TestArchive {
    _dir: tempfile::TempDir,
    fx: Fixture,
}

fn msg<'a>(
    id: &'a str,
    role: &'a str,
    by: &'a str,
    text: &'a str,
    at: Option<i64>,
) -> MessageFacts<'a> {
    MessageFacts {
        id,
        role,
        written_by: by,
        prompt_kind: (by == "human").then_some("prose"),
        at,
        text: Some(text),
        ..MessageFacts::default()
    }
}

/// #1–#8 as the example's FACTS table has them, reduced to what the
/// queries below read.
fn morrow() -> TestArchive {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let cc = |n: u8| format!("claude-code:7f3a2c10-000{n}-4000-8000-00000000000{n}");
    let ids: Vec<String> = (1..=4).map(cc).collect();

    // #1 claude-code, Oct 1, cost 1.26, 3 prompts, 6 calls.
    let s1 = fx
        .session(SessionFacts {
            id: &ids[0],
            started: at(1, 9),
            last: at(1, 17),
            cwd: Some("/home/op/morrow"),
            ..SessionFacts::default()
        })
        .unwrap();
    for i in 0..3 {
        let p = fx
            .message(
                s1,
                msg(
                    &format!("m1p{i}"),
                    "user",
                    "human",
                    "please fix the parser",
                    at(1, 9 + i),
                ),
            )
            .unwrap();
        let a = fx
            .message(
                s1,
                MessageFacts {
                    cost_nusd: Some(420_000_000),
                    ..msg(
                        &format!("m1a{i}"),
                        "assistant",
                        "model",
                        "done",
                        at(1, 9 + i),
                    )
                },
            )
            .unwrap();
        let _ = p;
        for j in 0..2 {
            fx.tool_call(
                s1,
                a,
                CallFacts {
                    tool: "Bash",
                    action: "shell",
                    command: Some("cargo build"),
                    outcome: "ok",
                    exit_code: Some(0),
                    at: at(1, 9 + i),
                    output: Some("ok"),
                    ..CallFacts::default()
                },
            )
            .unwrap();
            let _ = j;
        }
    }
    // #2 claude-code, Oct 2, cost 2.85, chrono decision, a failing cargo test.
    let s2 = fx
        .session(SessionFacts {
            id: &ids[1],
            started: at(2, 9),
            last: at(2, 18),
            cwd: Some("/home/op/morrow"),
            ..SessionFacts::default()
        })
        .unwrap();
    fx.message(
        s2,
        msg(
            "m2p0",
            "user",
            "human",
            "let's use chrono, not time",
            at(2, 9),
        ),
    )
    .unwrap();
    fx.message(
        s2,
        msg("m2p1", "user", "human", "now run the tests", at(2, 10)),
    )
    .unwrap();
    fx.message(s2, msg("m2p2", "user", "human", "ship it", at(2, 11)))
        .unwrap();
    let a2 = fx
        .message(
            s2,
            MessageFacts {
                cost_nusd: Some(2_850_000_000),
                ..msg(
                    "m2a0",
                    "assistant",
                    "model",
                    "Switching to chrono.",
                    at(2, 10),
                )
            },
        )
        .unwrap();
    fx.tool_call(
        s2,
        a2,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("cargo test"),
            outcome: "error",
            exit_code: Some(101),
            at: at(2, 10),
            output: Some("test failed: weekday_offset"),
            ..CallFacts::default()
        },
    )
    .unwrap();
    fx.tool_call(
        s2,
        a2,
        CallFacts {
            tool: "Edit",
            action: "file_edit",
            path: Some("src/date.rs"),
            outcome: "ok",
            at: at(2, 11),
            ..CallFacts::default()
        },
    )
    .unwrap();
    fx.tool_call(
        s2,
        a2,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("cargo test"),
            outcome: "ok",
            exit_code: Some(0),
            at: at(2, 12),
            output: Some("ok"),
            ..CallFacts::default()
        },
    )
    .unwrap();
    // #3 subagent of #2, Oct 2, uses claude-opus-5-5, cost 0.16.
    let s3 = fx
        .session(SessionFacts {
            id: &ids[2],
            kind: "subagent",
            started: at(2, 13),
            last: at(2, 14),
            ..SessionFacts::default()
        })
        .unwrap();
    fx.message(
        s3,
        MessageFacts {
            model: Some("claude-opus-5-5"),
            cost_nusd: Some(160_000_000),
            ..msg(
                "m3a0",
                "assistant",
                "model",
                "searched the docs for chrono",
                at(2, 13),
            )
        },
    )
    .unwrap();
    fx.link(&ids[1], &ids[2], "subagent", "fresh").unwrap();
    // #4 claude-code, Oct 3, continues #2, cost 1.59.
    let s4 = fx
        .session(SessionFacts {
            id: &ids[3],
            started: at(3, 9),
            last: at(3, 12),
            cwd: Some("/home/op/morrow"),
            ..SessionFacts::default()
        })
        .unwrap();
    for i in 0..3 {
        fx.message(
            s4,
            msg(
                &format!("m4p{i}"),
                "user",
                "human",
                "continue",
                at(3, 9 + i),
            ),
        )
        .unwrap();
    }
    let a4 = fx
        .message(
            s4,
            MessageFacts {
                cost_nusd: Some(1_590_000_000),
                ..msg("m4a0", "assistant", "model", "continuing", at(3, 10))
            },
        )
        .unwrap();
    fx.tool_call(
        s4,
        a4,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("cargo clippy"),
            outcome: "error",
            exit_code: Some(1),
            at: at(3, 10),
            output: Some("warning"),
            ..CallFacts::default()
        },
    )
    .unwrap();
    fx.link(&ids[1], &ids[3], "continuation", "resume").unwrap();
    fx.event(s4, "compaction", at(3, 11)).unwrap();
    // #5, #6 codex in morrow; #6 forks #5.
    let s5 = fx
        .session(SessionFacts {
            id: "codex:01a20000-0005-7000-8000-000000000005",
            started: at(5, 9),
            last: at(5, 10),
            repo: Some("https://github.com/op/morrow"),
            ..SessionFacts::default()
        })
        .unwrap();
    fx.message(
        s5,
        MessageFacts {
            cost_nusd: None,
            ..msg("m5a0", "assistant", "model", "codex answer five", at(5, 9))
        },
    )
    .unwrap();
    let s6 = fx
        .session(SessionFacts {
            id: "codex:01a20000-0006-7000-8000-000000000006",
            started: at(6, 9),
            last: at(6, 10),
            repo: Some("op/morrow"),
            ..SessionFacts::default()
        })
        .unwrap();
    fx.message(
        s6,
        msg("m6a0", "assistant", "model", "codex answer six", at(6, 9)),
    )
    .unwrap();
    fx.link(
        "codex:01a20000-0005-7000-8000-000000000005",
        "codex:01a20000-0006-7000-8000-000000000006",
        "continuation",
        "fork",
    )
    .unwrap();
    // #7 chatgpt with an abandoned branch; #8 claude-ai, no model.
    let s7 = fx
        .session(SessionFacts {
            id: "chatgpt:abc-7",
            started: at(4, 9),
            last: at(4, 10),
            ..SessionFacts::default()
        })
        .unwrap();
    fx.message(s7, msg("m7p0", "user", "human", "dates", at(4, 9)))
        .unwrap();
    fx.message(
        s7,
        MessageFacts {
            on_active_path: Some(false),
            ..msg("m7p1", "user", "human", "edited dates", at(4, 9))
        },
    )
    .unwrap();
    fx.message(
        s7,
        MessageFacts {
            on_active_path: Some(false),
            ..msg("m7a1", "assistant", "model", "abandoned", at(4, 9))
        },
    )
    .unwrap();
    let s8 = fx
        .session(SessionFacts {
            id: "claude-ai:abc-8",
            started: at(7, 9),
            last: at(7, 10),
            ..SessionFacts::default()
        })
        .unwrap();
    fx.message(
        s8,
        msg(
            "m8p0",
            "user",
            "human",
            "Shorter, and mention chrono.",
            at(7, 9),
        ),
    )
    .unwrap();

    fx.note(
        "tag",
        Some("project"),
        Some("morrow"),
        &format!("session:{}", ids[1]),
    )
    .unwrap();
    fx.note("tag", None, Some("research"), "session:chatgpt:abc-7")
        .unwrap();
    fx.note(
        "decision",
        None,
        Some("decided: chrono, not time"),
        "message:m2p0",
    )
    .unwrap();
    TestArchive { _dir: dir, fx }
}

fn ask(text: &str) -> Ask<'_> {
    Ask {
        text,
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        clock: fixture_clock(OCT1 + 7 * DAY, "UTC", 0),
        embeddings: false,
        near: None,
        fields: &[],
    }
}

fn run_q(w: &TestArchive, a: Ask<'_>) -> Result<Value, Failure> {
    let pool = w.fx.reads().unwrap();
    let r = pool.reader().unwrap();
    run(&r, a)
}

fn q(w: &TestArchive, text: &str) -> Value {
    run_q(w, ask(text)).unwrap_or_else(|f| panic!("{text}: {} {}", f.kind, f.message))
}

/// The `id` column of each row, shortened to the session's number.
fn ids(v: &Value) -> Vec<String> {
    v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let id = r["session"].as_str().or(r["id"].as_str()).unwrap_or("");
            short(id)
        })
        .collect()
}

fn short(id: &str) -> String {
    let n = match id {
        s if s.contains("0001-4000") => "#1",
        s if s.contains("0002-4000") => "#2",
        s if s.contains("0003-4000") => "#3",
        s if s.contains("0004-4000") => "#4",
        s if s.contains("0005-7000") => "#5",
        s if s.contains("0006-7000") => "#6",
        "chatgpt:abc-7" => "#7",
        "claude-ai:abc-8" => "#8",
        other => other,
    };
    n.to_owned()
}

fn groups(v: &Value, key: &str) -> Vec<(String, u64)> {
    v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                short(r[key].as_str().unwrap_or("")),
                r["count"].as_u64().unwrap(),
            )
        })
        .collect()
}

#[test]
fn session_queries_follow_the_morrow_examples() {
    let w = morrow();
    // 1: subagent #3 excluded by default, newest activity first.
    assert_eq!(ids(&q(&w, "origin:claude-code")), ["#4", "#2", "#1"]);
    // 2: naming kind drops the default.
    assert_eq!(ids(&q(&w, "origin:claude-code kind:subagent")), ["#3"]);
    // 5: repo by name matches name, owner/name and URL.
    assert_eq!(ids(&q(&w, "repo:morrow")), ["#6", "#5"]);
    // 8: unknown cost never matches a comparison.
    let v = q(&w, "sessions where cost > 1");
    assert_eq!(ids(&v), ["#4", "#2", "#1"]);
    assert_eq!(v["rows"][1]["cost"], 2.85);
    // 9: cost:unknown.
    assert_eq!(ids(&q(&w, "cost:unknown")), ["#8", "#6", "#5", "#7"]);
    // 11: own messages only.
    let v = q(&w, "sessions where messages >= 6");
    assert_eq!(ids(&v), ["#1"]);
    // 12.
    assert_eq!(ids(&q(&w, "sessions where tool_calls > 2")), ["#2", "#1"]);
    // 26: since a date.
    assert_eq!(ids(&q(&w, "since:2026-10-05")), ["#8", "#6", "#5"]);
    // 27: whole logical days, inclusive.
    assert_eq!(
        ids(&q(&w, "started between 2026-10-02 and 2026-10-03")),
        ["#4", "#2"]
    );
    // 28.
    assert_eq!(ids(&q(&w, "has:compaction")), ["#4"]);
    // 29: only the subagent used the model.
    assert_eq!(q(&w, "model:claude-opus-5-5")["outcome"], "empty");
    assert_eq!(ids(&q(&w, "model:claude-opus-5-5 kind:*")), ["#3"]);
    // 30.
    assert_eq!(ids(&q(&w, "origin:(chatgpt|claude-ai)")), ["#8", "#7"]);
    // 6, 7: tags.
    assert_eq!(ids(&q(&w, "tag:project:morrow")), ["#2"]);
    assert_eq!(ids(&q(&w, "tag:research")), ["#7"]);
}

#[test]
fn aggregates_count_unknowns_apart() {
    let w = morrow();
    // 3: kind:* drops the subagent default; unknown costs are counted, not summed as 0.
    let v = q(&w, "kind:* | agg count, sum:cost");
    let row = &v["rows"][0];
    assert_eq!(v["shape"], "summary");
    assert_eq!(row["count"], 8);
    let sum = row["sum_cost"].as_f64().unwrap();
    assert!((sum - (1.26 + 2.85 + 0.16 + 1.59)).abs() < 1e-9, "{sum}");
    assert_eq!(row["unknown_cost"], 4);
    // 4: group by origin over top-level sessions.
    let v = q(&w, "sessions | group by origin | count");
    assert_eq!(
        groups(&v, "origin"),
        [
            ("claude-code".into(), 3),
            ("codex".into(), 2),
            ("chatgpt".into(), 1),
            ("claude-ai".into(), 1)
        ]
    );
    // After grouping, `| where` filters groups.
    let v = q(&w, "sessions | group by origin | count | where count >= 2");
    assert_eq!(
        groups(&v, "origin"),
        [("claude-code".into(), 3), ("codex".into(), 2)]
    );
}

#[test]
fn row_units_and_their_defaults() {
    let w = morrow();
    // 15: tool calls that failed.
    let v = q(&w, "tool_calls where outcome:error");
    let rows: Vec<(String, String, i64)> = v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                short(r["session"].as_str().unwrap()),
                r["command"].as_str().unwrap().to_owned(),
                r["exit_code"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        [
            ("#2".into(), "cargo test".into(), 101),
            ("#4".into(), "cargo clippy".into(), 1)
        ]
    );
    // 16.
    let v = q(
        &w,
        "tool_calls where outcome:error | group by session | count",
    );
    assert_eq!(groups(&v, "session"), [("#2".into(), 1), ("#4".into(), 1)]);
    // 17.
    let v = q(&w, "tool_calls | group by action | count");
    assert_eq!(
        groups(&v, "action"),
        [("shell".into(), 9), ("file_edit".into(), 1)]
    );
    // 13.
    let v = q(&w, "messages where written_by:human \"chrono\"");
    // Relevance first, then the more recent message.
    assert_eq!(ids(&v), ["#8", "#2"]);
    // 23: the abandoned branch.
    let v = q(&w, "messages where active:false");
    assert_eq!(ids(&v), ["#7", "#7"]);
    // 22: grouping by a session field from a row unit.
    let v = q(
        &w,
        "messages where written_by:human | group by session.origin | count",
    );
    assert_eq!(
        groups(&v, "session_origin"),
        [
            ("claude-code".into(), 9),
            ("chatgpt".into(), 1),
            ("claude-ai".into(), 1)
        ]
    );
    // 31: notes.
    let v = q(&w, "notes where kind:decision");
    assert_eq!(v["rows"][0]["value"], "decided: chrono, not time");
}

#[test]
fn exists_and_seq_match_one_call_and_an_order() {
    let w = morrow();
    // 20: the same call must be both a shell call and an error.
    assert_eq!(
        ids(&q(
            &w,
            "sessions where exists tool_call(action:shell AND outcome:error)"
        )),
        ["#4", "#2"]
    );
    // 21: fail, edit, pass, in that order.
    let v = q(
        &w,
        "sessions where seq(command:\"cargo test\" AND outcome:error -> tool:Edit -> command:\"cargo test\" AND outcome:ok)",
    );
    assert_eq!(ids(&v), ["#2"]);
    let v = q(
        &w,
        "sessions where seq(tool:Edit -> command:\"cargo test\" AND outcome:error)",
    );
    assert_eq!(v["outcome"], "empty");
    let v = q(
        &w,
        "sessions where seq(command:\"cargo test\" -> [next] tool:Edit)",
    );
    assert_eq!(ids(&v), ["#2"]);
}

#[test]
fn matches_return_the_rows_that_matched() {
    let w = morrow();
    // seq: one row per step tuple, with the gap between the steps.
    let v = q(
        &w,
        "sessions where seq(command:\"cargo test\" AND outcome:error -> command:\"cargo test\" AND outcome:ok) | matches",
    );
    assert_eq!(v["unit"], "matches");
    assert_eq!(v["shape"], "matches");
    assert_eq!(ids(&v), ["#2"]);
    let row = &v["rows"][0];
    assert_eq!(row["match"], 1);
    assert_eq!(row["step1_outcome"], "error");
    assert_eq!(row["step2_outcome"], "ok");
    assert_eq!(row["gap_steps"], 1, "the Edit between them");
    assert_eq!(row["gap_time"], 3_600_000 * 2);
    // An aggregate after `| matches` is over the tuples.
    let v = q(
        &w,
        "sessions where seq(command:\"cargo test\" AND outcome:error -> command:\"cargo test\" AND outcome:ok) | matches | agg p50:gap_time, count",
    );
    assert_eq!(v["shape"], "summary");
    assert_eq!(v["rows"][0]["count"], 1);
    // exists: one row per parent row and matching child.
    let v = q(
        &w,
        "sessions where exists tool_call(action:shell AND outcome:error) | matches",
    );
    assert_eq!(v["unit"], "matches");
    let rows = v["rows"].as_array().unwrap();
    assert!(!rows.is_empty());
    assert!(
        rows.iter()
            .all(|r| r["outcome"] == "error" && r["match"].as_u64() >= Some(1))
    );
    // Without one exists or seq, there is nothing to return as matches.
    let err = run_q(&w, ask("sessions where repo:morrow | matches")).unwrap_err();
    assert_eq!(err.kind, "invalid_query");
}

#[test]
fn seq_matches_are_earliest_first_and_never_overlap() {
    let w = worked();
    // Five shell calls: (1, 2), then (3, 4); the fifth starts no match.
    let v = q(
        &w,
        "sessions where seq(action:shell -> action:shell) | matches",
    );
    let rows = v["rows"].as_array().unwrap();
    let pairs: Vec<(String, String)> = rows
        .iter()
        .map(|r| {
            (
                r["step1_command"].as_str().unwrap().to_owned(),
                r["step2_command"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    assert_eq!(
        pairs,
        [
            (
                "bd update demo-7 --claim".to_owned(),
                "cargo test -p store".to_owned()
            ),
            (
                "cargo test -p store".to_owned(),
                "git commit -m fix".to_owned()
            ),
        ]
    );
    assert_eq!(rows[1]["match"], 2);
    // `[next]` with a condition that no later call meets: no match.
    let v = q(
        &w,
        "sessions where seq(command:\"git commit -m fix\" -> [next] outcome:error) | matches",
    );
    assert_eq!(v["outcome"], "empty");
}

#[test]
fn turns_list_their_prompts_and_whether_their_answer_was_corrected() {
    let mut w = morrow();
    let v = q(&w, "turns | limit 100");
    let turns = v["rows"].as_array().unwrap().clone();
    assert!(!turns.is_empty());
    for t in &turns {
        let prompts: Vec<String> =
            serde_json::from_str(t["prompts"].as_str().unwrap_or("[]")).unwrap();
        assert_eq!(
            prompts.len() as u64,
            t["burst"].as_u64().unwrap_or(0),
            "{t}"
        );
    }
    let answered = turns
        .iter()
        .find(|t| t["answer"].as_str().is_some_and(|a| !a.is_empty()))
        .expect("a turn with an answer");
    let answer = answered["answer"].as_str().unwrap().to_owned();
    assert_eq!(q(&w, "turns where corrected:true")["outcome"], "empty");
    w.fx.note(
        "correction",
        None,
        Some("that is wrong"),
        &format!("message:{answer}"),
    )
    .unwrap();
    let v = q(&w, "turns where corrected:true");
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{v}");
    assert_eq!(rows[0]["answer"], answer.as_str());
    let first = serde_json::from_str::<Vec<String>>(answered["prompts"].as_str().unwrap())
        .unwrap()
        .into_iter()
        .next();
    if let Some(p) = first {
        let v = q(&w, &format!("turns where prompts:{p}"));
        assert!(!v["rows"].as_array().unwrap().is_empty(), "{v}");
    }
}

#[test]
fn kept_results_read_back_as_they_were_kept() {
    let mut w = morrow();
    let kept_of = |w: &TestArchive, text: &str| {
        let pool = w.fx.reads().unwrap();
        let r = pool.reader().unwrap();
        keep(&r, ask(text)).unwrap()
    };
    let kept = kept_of(
        &w,
        "sessions where exists tool_call(action:shell AND outcome:error)",
    );
    assert_eq!(ids(&kept), ["#4", "#2"]);
    assert!(kept["match_columns"].is_array(), "match rows kept too");
    w.fx.kept_result(
        "0199-kept",
        "sessions where exists tool_call(action:shell AND outcome:error)",
        &kept.to_string(),
    )
    .unwrap();
    let v = q(&w, "result:0199-kept");
    assert_eq!(v["unit"], "sessions");
    assert_eq!(ids(&v), ["#4", "#2"]);
    let v = q(&w, "result:0199-kept | count");
    assert_eq!(v["rows"][0]["count"], 2);
    let v = q(&w, "result:0199-kept.matches | agg count");
    assert_eq!(
        v["rows"][0]["count"].as_u64(),
        Some(kept["matches"].as_array().unwrap().len() as u64)
    );
    // Conditions do not narrow a kept result; stages do.
    let err = run_q(&w, ask("result:0199-kept repo:morrow")).unwrap_err();
    assert_eq!(err.kind, "invalid_query");
    let err = run_q(&w, ask("result:nope")).unwrap_err();
    assert!(err.message.contains("no kept result"), "{}", err.message);
}

#[test]
fn links_and_parents() {
    let w = morrow();
    // 24: parent drops the subagent default.
    assert_eq!(
        ids(&q(
            &w,
            "parent:claude-code:7f3a2c10-0002-4000-8000-000000000002"
        )),
        ["#4", "#3"]
    );
    // 25.
    assert_eq!(
        ids(&q(
            &w,
            "connected:codex:01a20000-0005-7000-8000-000000000005"
        )),
        ["#6", "#5"]
    );
    // 36.
    let v = q(&w, "links where kind:continuation");
    assert_eq!(v["count"], 2);
    // 37.
    assert_eq!(ids(&q(&w, "link:fork")), ["#6"]);
    // 38: tree cost includes the subagent.
    assert_eq!(ids(&q(&w, "tree_cost > 2.90")), ["#2"]);
    assert_eq!(q(&w, "cost > 2.90")["outcome"], "empty");
}

#[test]
fn text_search_ranks_and_cuts_snippets() {
    let w = morrow();
    let v = q(&w, "chrono");
    assert_eq!(v["ranked"], true);
    // #3's hit makes its root #2 match; subagents are not their own rows.
    let got = ids(&v);
    assert!(
        got.contains(&"#2".to_owned()) && got.contains(&"#8".to_owned()),
        "{got:?}"
    );
    assert!(!got.contains(&"#3".to_owned()));
    let snippet = &v["rows"][0]["snippet"];
    let text = snippet["text"].as_str().unwrap();
    let (s, e) = (
        snippet["marks"][0][0].as_u64().unwrap() as usize,
        snippet["marks"][0][1].as_u64().unwrap() as usize,
    );
    assert_eq!(&text[s..e].to_lowercase(), "chrono");
    // Punctuation inside a word searches its parts.
    assert_eq!(q(&w, "messages where weekday_offset()")["outcome"], "empty");
    let v = q(&w, "tool_calls where output:weekday_offset");
    assert_eq!(v["count"], 1);
    // Trigram substring on commands.
    assert_eq!(q(&w, "tool_calls where command:\"go te\"")["count"], 2);
}

#[test]
fn errors_point_at_the_problem_and_offer_a_fix() {
    let w = morrow();
    // 35.
    let f = run_q(&w, ask("reop:morrow")).unwrap_err();
    assert_eq!(f.kind, "unknown_field");
    assert_eq!(
        f.message,
        "unknown field \"reop\" at position 0; did you mean \"repo\"?"
    );
    assert_eq!(f.detail.unwrap()["suggestion"], "repo:morrow");
    // A closed set names the set and where the value does exist.
    let f = run_q(&w, ask("sessions where outcome:error")).unwrap_err();
    assert!(
        f.message
            .contains("one of solved, partial, abandoned, none"),
        "{}",
        f.message
    );
    assert!(
        f.message.contains("\"error\" is an outcome of tool_calls"),
        "{}",
        f.message
    );
    // Commands and near misses are refused, not searched.
    let f = run_q(&w, ask("brief yesterday")).unwrap_err();
    assert_eq!(f.kind, "not_a_query");
    let f = run_q(&w, ask("messges cargo")).unwrap_err();
    assert!(f.message.contains("\"messages\""), "{}", f.message);
    // Short words and real searches are not near misses.
    assert!(run_q(&w, ask("bug fix")).is_ok());
    assert!(run_q(&w, ask("date parsing")).is_ok());
}

#[test]
fn an_empty_result_explains_itself() {
    let w = morrow();
    // 34.
    let mut a = ask("repo:other");
    a.why = true;
    let v = run_q(&w, a).unwrap();
    assert_eq!(v["outcome"], "empty");
    assert_eq!(v["why"][0]["clause"], "repo:other");
    assert_eq!(v["why"][0]["matches"], 0);
    assert_eq!(v["why"][0]["of"], 7);
    // 33: semantic search is not enabled.
    let v = q(&w, "similar:\"ambiguous date formats\"");
    assert_eq!(v["outcome"], "incomplete");
    assert_eq!(v["count"], 7);
}

/// Similarities for `similar:` computed from stored vectors: the texts of the
/// sessions whose ID is `LIKE like` lie along the query, every other text
/// across it.
fn similarity(w: &TestArchive, like: &str) -> std::sync::Arc<polylogue_store::vectors::Similarity> {
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader
        .query(
            "SELECT hex(t.hash) AS h, EXISTS (SELECT 1 FROM block b JOIN session s ON s.pk = b.session_pk
                 WHERE b.text_id = t.id AND s.id LIKE ?1) AS near FROM text t",
            &[SqlValue::Text(like.into())],
        )
        .unwrap();
    let vectors: Vec<polylogue_store::vectors::Vector> = rows
        .iter()
        .map(|r| {
            let hex = r["h"].as_str().unwrap();
            let text_hash = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            let near = r["near"].as_i64() == Some(1);
            polylogue_store::vectors::Vector {
                text_hash,
                chunk: 0,
                values: if near { vec![1.0, 0.0] } else { vec![0.0, 1.0] },
            }
        })
        .collect();
    let key = "local/test/2/document/unit";
    reader.vectors_writer().unwrap().put(key, &vectors).unwrap();
    let loaded = std::sync::Arc::new(reader.vectors().unwrap().load(key).unwrap());
    std::sync::Arc::new(loaded.similarity(&[1.0, 0.0]).unwrap())
}

#[test]
fn near_ranks_every_row_by_similarity() {
    let w = morrow();
    let s = similarity(&w, "%0002-4000%");
    let near = |q: &str| {
        assert_eq!(q, "ambiguous date formats");
        Ok(std::sync::Arc::clone(&s))
    };
    let mut a = ask("similar:\"ambiguous date formats\"");
    a.embeddings = true;
    a.near = Some(&near);
    let v = run_q(&w, a).unwrap();
    assert_eq!(v["outcome"], "rows", "{v}");
    assert_eq!(v["ranked"], true);
    // Ranking keeps every row; the nearest comes first.
    assert_eq!(v["count"], 7);
    assert_eq!(ids(&v)[0], "#2");
}

#[test]
fn near_applies_its_conditions_before_ranking() {
    let w = morrow();
    // Every text of #2 is nearer than any other, yet a condition that
    // excludes #2 still ranks what it keeps instead of finding nothing.
    let s = similarity(&w, "%0002-4000%");
    let near = |_: &str| Ok(std::sync::Arc::clone(&s));
    let mut a = ask("similar:x -id:claude-code:7f3a2c10-0002");
    a.embeddings = true;
    a.near = Some(&near);
    let v = run_q(&w, a).unwrap();
    assert_eq!(v["count"], 6, "{v}");
    assert!(!ids(&v).contains(&"#2".to_owned()));
    assert!(
        v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["rank"].is_number()),
        "the rows kept have similarities: {v}"
    );
}

#[test]
fn text_and_similarity_rank_together() {
    let w = morrow();
    let s = similarity(&w, "%0002-4000%");
    let near = |_: &str| Ok(std::sync::Arc::clone(&s));
    let mut text_only = ask("messages where parser");
    text_only.page_size = Some(100);
    let lexical = run_q(&w, text_only).unwrap();
    let mut both = ask("messages where parser similar:x");
    both.page_size = Some(100);
    both.embeddings = true;
    both.near = Some(&near);
    let fused = run_q(&w, both).unwrap();
    // The same rows: words filter, similarity only reorders.
    assert_eq!(fused["count"], lexical["count"], "{fused}");
    assert_eq!(fused["ranked"], true);
    let ranks: Vec<f64> = fused["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["rank"].as_f64().unwrap())
        .collect();
    assert!(ranks.windows(2).all(|p| p[0] <= p[1]), "{ranks:?}");
}

#[test]
fn pages_continue_with_a_cursor_and_keep_their_now() {
    let w = morrow();
    let mut a = ask("kind:*");
    a.page_size = Some(3);
    let first = run_q(&w, a).unwrap();
    assert_eq!(first["count"], 8);
    assert_eq!(first["rows"].as_array().unwrap().len(), 3);
    let mut seen = ids(&first);
    let mut cursor = first["next"].as_str().map(str::to_owned);
    while let Some(c) = cursor {
        let mut a = ask("kind:*");
        a.page_size = Some(3);
        a.cursor = Some(&c);
        // A later request still answers the first one's question.
        a.clock.now = a
            .clock
            .now
            .checked_add(jiff::SignedDuration::from_millis(DAY))
            .unwrap();
        let page = run_q(&w, a).unwrap();
        seen.extend(ids(&page));
        cursor = page["next"].as_str().map(str::to_owned);
    }
    assert_eq!(seen.len(), 8);
    let unique: std::collections::HashSet<_> = seen.iter().collect();
    assert_eq!(unique.len(), 8);
    // A cursor continues only its own query.
    let mut a = ask("origin:codex");
    let c = first["next"].as_str().unwrap().to_owned();
    a.cursor = Some(&c);
    assert_eq!(run_q(&w, a).unwrap_err().kind, "bad_request");
}

#[test]
fn attached_rows_follow_each_session() {
    let w = morrow();
    // 32.
    let v = q(
        &w,
        "origin:claude-code with messages(id, role)[role:assistant, last:1]",
    );
    for row in v["rows"].as_array().unwrap() {
        let m = row["messages"].as_array().unwrap();
        assert_eq!(m.len(), 1, "{row}");
        assert_eq!(m[0]["role"], "assistant");
    }
    assert_eq!(v["rows"][1]["messages"][0]["id"], "m2a0");
}

#[test]
fn a_session_query_scopes_a_row_query() {
    let w = morrow();
    let v = q(
        &w,
        "tag:project:morrow | tool_calls where action:shell | agg count",
    );
    assert_eq!(v["rows"][0]["count"], 2);
    // Grouping by logical day.
    let v = q(&w, "tool_calls | group by day(at) | count | sort by day_at");
    assert_eq!(v["rows"][0]["day_at"], "2026-10-01");
}

/// A session that claims a Bead, fails then passes its tests, commits and
/// closes the Bead; derived into the work graph.
fn worked() -> TestArchive {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = fx
        .session(SessionFacts {
            id: "claude-code:w1",
            last: at(2, 13),
            ..SessionFacts::default()
        })
        .unwrap();
    let m = fx
        .message(s, msg("w1a", "assistant", "model", "", at(2, 12)))
        .unwrap();
    let calls: [(&str, &str, &str, i64); 5] = [
        ("bd update demo-7 --claim", "ok", "", 0),
        ("cargo test -p store", "error", "FAILED", 1),
        ("cargo test -p store", "ok", "ok", 2),
        ("git commit -m fix", "ok", "[main 0123abcd] fix", 3),
        ("bd close demo-7", "ok", "", 4),
    ];
    for (command, outcome, output, minute) in calls {
        fx.tool_call(
            s,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some(command),
                outcome,
                at: at(2, 12).map(|t| t + minute * 60_000),
                output: Some(output),
                ..CallFacts::default()
            },
        )
        .unwrap();
    }
    fx.derive().unwrap();
    TestArchive { _dir: dir, fx }
}

#[test]
fn the_work_unit_reads_the_derived_graph() {
    let w = worked();
    let attempts = q(&w, "work where kind:attempt");
    let row = &attempts["rows"][0];
    assert_eq!(attempts["rows"].as_array().unwrap().len(), 1);
    assert_eq!(row["title"], "demo-7");
    assert_eq!(row["state"], "closed");
    assert_eq!(row["goal"], "demo-7");
    let checks = q(&w, "work where kind:verification goal:demo-7");
    let states: Vec<&str> = checks["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["state"].as_str().unwrap())
        .collect();
    assert_eq!(states, ["error", "ok"]);
    assert_eq!(
        q(&w, "work where checks_failed:1")["rows"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    // The short hash the commit command printed answers a full-hash query.
    assert_eq!(
        ids(&q(&w, "commit:0123abcd00112233445566778899aabbccddeeff")),
        ["claude-code:w1"]
    );
}

mod agent_input;
mod combine;
mod day;
mod evidence;
mod generated;
mod live;
mod modes;
mod morrow_examples;
mod references;
mod sequences;
mod unknowns;

#[test]
fn failed_saved_result_storage_is_not_an_empty_catalog() {
    let mut fixture = morrow();
    fixture
        .fx
        .kept_result("broken", "sessions", "not JSON")
        .unwrap();
    let failure = run_q(&fixture, ask("sessions")).unwrap_err();
    assert_eq!(failure.kind, "store");
    assert!(failure.message.contains("malformed JSON"));
    let pool = fixture.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let failure = keep(&reader, ask("sessions")).unwrap_err();
    assert_eq!(failure.kind, "store");
}

#[test]
fn absent_saved_queries_and_results_are_a_valid_empty_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert!(saved_queries(&reader).unwrap().is_empty());
    assert!(kept_results(&reader).unwrap().is_empty());
    assert_eq!(run(&reader, ask("sessions")).unwrap()["count"], 0);
}

#[test]
fn damaged_kept_result_shapes_are_not_defaulted_or_discarded() {
    for (field, body) in [
        ("unit", json!({"columns": []})),
        ("unit", json!({"unit": 7, "columns": []})),
        ("columns", json!({"unit": "sessions"})),
        ("columns", json!({"unit": "sessions", "columns": {}})),
        (
            "columns",
            json!({"unit": "sessions", "columns": [{"name": "id", "kind": "unsupported"}]}),
        ),
        (
            "match_columns",
            json!({"unit": "sessions", "columns": [], "match_columns": null}),
        ),
        (
            "match_columns",
            json!({"unit": "sessions", "columns": [], "match_columns": "wrong"}),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        fixture
            .kept_result("broken", "sessions", &body.to_string())
            .unwrap();
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        for failure in [
            run(&reader, ask("sessions")).unwrap_err(),
            keep(&reader, ask("sessions")).unwrap_err(),
        ] {
            assert_eq!(failure.kind, "store");
            assert!(failure.message.contains("broken"), "{failure:?}");
            assert!(failure.message.contains(field), "{failure:?}");
        }
    }
}

#[test]
fn kept_results_without_match_rows_do_not_need_match_columns() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let body = {
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        keep(&reader, ask("sessions")).unwrap()
    };
    assert!(body.get("match_columns").is_none());
    fixture
        .kept_result("empty", "sessions", &body.to_string())
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let kept = kept_results(&reader).unwrap();
    assert!(kept["empty"].match_columns.is_empty());
    assert_eq!(run(&reader, ask("result:empty")).unwrap()["count"], 0);
}

mod titles;

mod pages;

mod input_events;

#[test]
fn work_queries_distinguish_unknown_check_totals_from_failures() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = fx
        .session(SessionFacts {
            id: "claude-code:unknown-work",
            ..SessionFacts::default()
        })
        .unwrap();
    let m = fx
        .message(
            s,
            msg("unknown-work-call", "assistant", "model", "", at(2, 12)),
        )
        .unwrap();
    for (n, (command, outcome)) in [
        ("bd update demo-7 --claim", "ok"),
        ("cargo test", "unknown"),
        ("cargo check", "error"),
        ("cargo clippy", "ok"),
    ]
    .into_iter()
    .enumerate()
    {
        fx.tool_call(
            s,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some(command),
                outcome,
                at: at(2, 12).map(|t| t + n as i64 * 60_000),
                ..CallFacts::default()
            },
        )
        .unwrap();
    }
    fx.derive().unwrap();
    let w = TestArchive { _dir: dir, fx };
    let fields = ["checks_passed", "checks_failed", "checks_unknown"].map(str::to_owned);
    let totals = run_q(
        &w,
        Ask {
            fields: &fields,
            ..ask("work where kind:attempt")
        },
    )
    .unwrap();
    assert_eq!(
        totals["rows"],
        json!([{"checks_passed":1,"checks_failed":1,"checks_unknown":1}])
    );
    assert_eq!(
        q(
            &w,
            "work where kind:attempt checks_passed:1 checks_failed:1 checks_unknown:1"
        )["rows"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(
        q(&w, "work where kind:attempt checks_failed:2")["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        q(&w, r#"work where kind:verification state:"unknown""#)["rows"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

mod composition;

mod language;
mod measures;

mod shared_laws;

mod reported_timing;

fn fixture_clock(now: i64, zone: &str, start_minutes: u16) -> Clock {
    Clock::new(
        jiff::Timestamp::from_millisecond(now).unwrap(),
        DaySetting::new(
            jiff::tz::TimeZone::get(zone).unwrap(),
            jiff::civil::Time::constant(
                (start_minutes / 60) as i8,
                (start_minutes % 60) as i8,
                0,
                0,
            ),
        )
        .unwrap(),
    )
}
