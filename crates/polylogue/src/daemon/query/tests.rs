//! The query language end to end over a small archive shaped like the
//! spec's Morrow example (spec query.md, "Examples against the Morrow
//! archive"): the same sessions, kinds, costs, links and messages, written
//! row by row so the queries are tested apart from any parser.

use super::*;
use polylogue_store::fixture::{Fixture, M, S, T};

const DAY: i64 = 86_400_000;
/// 2026-10-01T12:00Z.
const OCT1: i64 = 1_790_856_000_000;

fn at(day: i64, hour: i64) -> Option<i64> {
    Some(OCT1 + (day - 1) * DAY + (hour - 12) * 3_600_000)
}

struct World {
    _dir: tempfile::TempDir,
    fx: Fixture,
}

fn msg<'a>(id: &'a str, role: &'a str, by: &'a str, text: &'a str, at: Option<i64>) -> M<'a> {
    M {
        id,
        role,
        written_by: by,
        prompt_kind: (by == "human").then_some("prose"),
        at,
        text: Some(text),
        ..M::default()
    }
}

/// #1–#8 as the example's FACTS table has them, reduced to what the
/// queries below read.
fn morrow() -> World {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let cc = |n: u8| format!("claude-code:7f3a2c10-000{n}-4000-8000-00000000000{n}");
    let ids: Vec<String> = (1..=4).map(cc).collect();

    // #1 claude-code, Oct 1, cost 1.26, 3 prompts, 6 calls.
    let s1 = fx
        .session(S {
            id: &ids[0],
            started: at(1, 9),
            last: at(1, 17),
            cwd: Some("/home/op/morrow"),
            ..S::default()
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
                M {
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
                T {
                    tool: "Bash",
                    action: "shell",
                    command: Some("cargo build"),
                    outcome: "ok",
                    exit_code: Some(0),
                    at: at(1, 9 + i),
                    output: Some("ok"),
                    ..T::default()
                },
            )
            .unwrap();
            let _ = j;
        }
    }
    // #2 claude-code, Oct 2, cost 2.85, chrono decision, a failing cargo test.
    let s2 = fx
        .session(S {
            id: &ids[1],
            started: at(2, 9),
            last: at(2, 18),
            cwd: Some("/home/op/morrow"),
            ..S::default()
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
            M {
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
        T {
            tool: "Bash",
            action: "shell",
            command: Some("cargo test"),
            outcome: "error",
            exit_code: Some(101),
            at: at(2, 10),
            output: Some("test failed: weekday_offset"),
            ..T::default()
        },
    )
    .unwrap();
    fx.tool_call(
        s2,
        a2,
        T {
            tool: "Edit",
            action: "file_edit",
            path: Some("src/date.rs"),
            outcome: "ok",
            at: at(2, 11),
            ..T::default()
        },
    )
    .unwrap();
    fx.tool_call(
        s2,
        a2,
        T {
            tool: "Bash",
            action: "shell",
            command: Some("cargo test"),
            outcome: "ok",
            exit_code: Some(0),
            at: at(2, 12),
            output: Some("ok"),
            ..T::default()
        },
    )
    .unwrap();
    // #3 subagent of #2, Oct 2, uses claude-opus-5-5, cost 0.16.
    let s3 = fx
        .session(S {
            id: &ids[2],
            kind: "subagent",
            started: at(2, 13),
            last: at(2, 14),
            ..S::default()
        })
        .unwrap();
    fx.message(
        s3,
        M {
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
    // #4 claude-code, Oct 3, continues #2, cost 1.59; an inherited copy.
    let s4 = fx
        .session(S {
            id: &ids[3],
            started: at(3, 9),
            last: at(3, 12),
            cwd: Some("/home/op/morrow"),
            ..S::default()
        })
        .unwrap();
    fx.message(
        s4,
        M {
            inherited_from: Some("m2p0"),
            ..msg(
                "m4c0",
                "user",
                "human",
                "let's use chrono, not time",
                at(2, 9),
            )
        },
    )
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
            M {
                cost_nusd: Some(1_590_000_000),
                ..msg("m4a0", "assistant", "model", "continuing", at(3, 10))
            },
        )
        .unwrap();
    fx.tool_call(
        s4,
        a4,
        T {
            tool: "Bash",
            action: "shell",
            command: Some("cargo clippy"),
            outcome: "error",
            exit_code: Some(1),
            at: at(3, 10),
            output: Some("warning"),
            ..T::default()
        },
    )
    .unwrap();
    fx.link(&ids[1], &ids[3], "continuation", "resume").unwrap();
    fx.event(s4, "compaction", at(3, 11)).unwrap();
    // #5, #6 codex in morrow; #6 forks #5.
    let s5 = fx
        .session(S {
            id: "codex:01a20000-0005-7000-8000-000000000005",
            started: at(5, 9),
            last: at(5, 10),
            repo: Some("https://github.com/op/morrow"),
            ..S::default()
        })
        .unwrap();
    fx.message(
        s5,
        M {
            cost_nusd: None,
            ..msg("m5a0", "assistant", "model", "codex answer five", at(5, 9))
        },
    )
    .unwrap();
    let s6 = fx
        .session(S {
            id: "codex:01a20000-0006-7000-8000-000000000006",
            started: at(6, 9),
            last: at(6, 10),
            repo: Some("op/morrow"),
            ..S::default()
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
        .session(S {
            id: "chatgpt:abc-7",
            started: at(4, 9),
            last: at(4, 10),
            ..S::default()
        })
        .unwrap();
    fx.message(s7, msg("m7p0", "user", "human", "dates", at(4, 9)))
        .unwrap();
    fx.message(
        s7,
        M {
            on_active_path: Some(false),
            ..msg("m7p1", "user", "human", "edited dates", at(4, 9))
        },
    )
    .unwrap();
    fx.message(
        s7,
        M {
            on_active_path: Some(false),
            ..msg("m7a1", "assistant", "model", "abandoned", at(4, 9))
        },
    )
    .unwrap();
    let s8 = fx
        .session(S {
            id: "claude-ai:abc-8",
            started: at(7, 9),
            last: at(7, 10),
            ..S::default()
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
    World { _dir: dir, fx }
}

fn ask(text: &str) -> Ask<'_> {
    Ask {
        text,
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        now: OCT1 + 7 * DAY,
        day: LogicalDay::named("UTC", 0).unwrap(),
        embeddings: false,
    }
}

fn run_q(w: &World, a: Ask<'_>) -> Result<Value, Failure> {
    let pool = w.fx.reads().unwrap();
    let r = pool.reader().unwrap();
    run(&r, a)
}

fn q(w: &World, text: &str) -> Value {
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
    // 11: own messages only; #4's inherited copy is not counted.
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
    // 13 and 14: inherited copies drop out unless asked for.
    let v = q(&w, "messages where written_by:human \"chrono\"");
    // Relevance first, then the more recent message.
    assert_eq!(ids(&v), ["#8", "#2"]);
    let v = q(&w, "messages where written_by:human inherited:* \"chrono\"");
    assert_eq!(v["count"], 3);
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
fn links_and_lineage() {
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
        ids(&q(&w, "lineage:codex:01a20000-0005-7000-8000-000000000005")),
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
    let v = q(&w, "near:\"ambiguous date formats\"");
    assert_eq!(v["outcome"], "degraded");
    assert_eq!(v["count"], 7);
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
        a.now += DAY;
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
