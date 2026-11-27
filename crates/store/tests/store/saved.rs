//! A saved query standing where a condition stands keeps its own grouping:
//! `query:NAME` for `a OR b`, narrowed by `c`, is `(a OR b) AND c`
//! (docs/saved-queries.md, saved definitions; docs/query.md, grammar).

use crate::{rows, submit};
use polylogue_model::user::{Save, UserWrite};
use polylogue_store::WriterSender;
use polylogue_testkit::ClaudeHome;
use serde_json::{Value, json};

const SESSION: &str = "5a7ed000-0000-4000-8000-000000000001";

/// One Claude Code record of the session, `n` seconds after 09:00.
fn record(n: u32, uuid: &str, kind: &str, message: Value) -> String {
    format!(
        "{}\n",
        json!({
            "type": kind, "uuid": uuid, "sessionId": SESSION, "cwd": "/home/op/proj",
            "timestamp": format!("2026-10-01T09:00:{n:02}Z"),
            "message": message,
        })
    )
}

/// A session with four tool calls: Bash failing and passing, Read failing
/// on a missing file and passing.
fn four_calls() -> String {
    let call = |id: &str, name: &str, input: Value| json!({"type": "tool_use", "id": id, "name": name, "input": input});
    let result = |id: &str, text: &str, error: bool| json!({"type": "tool_result", "tool_use_id": id, "content": text, "is_error": error});
    [
        record(
            0,
            "u1",
            "user",
            json!({"role": "user", "content": "Build and read the sources."}),
        ),
        record(
            1,
            "a1",
            "assistant",
            json!({"id": "msg_a1", "role": "assistant", "stop_reason": "tool_use", "content": [
                call("toolu_b1", "Bash", json!({"command": "cargo test"})),
                call("toolu_b2", "Bash", json!({"command": "cargo build"})),
                call("toolu_r1", "Read", json!({"file_path": "/home/op/proj/src/missing.rs"})),
                call("toolu_r2", "Read", json!({"file_path": "/home/op/proj/src/lib.rs"})),
            ]}),
        ),
        record(
            2,
            "u2",
            "user",
            json!({"role": "user", "content": [
                result("toolu_b1", "test result: FAILED. exit code: 101", true),
                result("toolu_b2", "Finished `dev` profile", false),
                result("toolu_r1", "File does not exist.", true),
                result("toolu_r2", "pub fn parse() {}", false),
            ]}),
        ),
    ]
    .concat()
}

/// Saves `text` under `name` as `polylogue QUERY save --name NAME` does.
fn save(writer: &WriterSender, name: &str, text: &str) {
    submit(
        writer,
        &format!("save-{name}"),
        UserWrite::Save(Box::new(Save {
            name: Some(name.into()),
            query: text.into(),
            denominator: None,
            rows_json: r#"{"unit":"tool_calls","columns":[],"rows":[]}"#.into(),
            read_position: "{}".into(),
            watch: None,
            outcomes: None,
        })),
    );
}

#[test]
fn a_saved_or_narrowed_by_a_condition_keeps_its_parentheses() {
    let home = ClaudeHome::new();
    home.append(&format!("{SESSION}.jsonl"), &four_calls());
    let (mut engine, _lock) = home.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    save(
        &engine.writer(),
        "failing-or-read",
        "tool_calls where outcome:error OR tool:Read",
    );
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let calls = |text: &str| {
        let mut got: Vec<Value> = rows(&reader, text, &["tool", "outcome"]);
        got.sort_by_key(|row| row.to_string());
        got
    };
    let call = |tool: &str, outcome: &str| json!({"tool": tool, "outcome": outcome});

    assert_eq!(
        calls("tool_calls where query:failing-or-read"),
        [
            call("Bash", "error"),
            call("Read", "error"),
            call("Read", "ok")
        ]
    );
    // (error OR Read) AND Bash; without the parentheses, error OR (Read
    // AND Bash) would add Read's error.
    for narrowed in [
        "tool_calls where query:failing-or-read tool:Bash",
        "tool_calls where tool:Bash AND query:failing-or-read",
    ] {
        assert_eq!(calls(narrowed), [call("Bash", "error")], "{narrowed}");
    }
}
