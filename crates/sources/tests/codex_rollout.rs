//! Codex rollout knowledge from the Python predecessor's
//! `test_parsers_codex.py` and `test_codex_event_stream_contract.py`.
mod support;

use serde_json::{Value, json};
use support::{blocks, jsonl, messages, parse, sessions};

const PATH: &str = ".codex/sessions/2026/03/01/rollout-2026-03-01T10-00-00-019a0000-0000-7000-8000-000000000001.jsonl";
const SID: &str = "019a0000-0000-7000-8000-000000000001";

fn record(kind: &str, payload: Value) -> Value {
    json!({"timestamp": "2026-03-01T10:00:01.000Z", "type": kind, "payload": payload})
}

fn meta(extra: Value) -> Value {
    let mut p = json!({"id": SID, "timestamp": "2026-03-01T10:00:00.000Z", "cwd": "/work",
        "originator": "codex_cli_rs", "cli_version": "0.160.0"});
    p.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    record("session_meta", p)
}

fn call(id: &str, name: &str) -> Value {
    record(
        "response_item",
        json!({"type": "function_call", "name": name,
        "arguments": r#"{"cmd":"ls"}"#, "call_id": id}),
    )
}

fn output(id: &str, text: &str) -> Value {
    record(
        "response_item",
        json!({"type": "function_call_output", "call_id": id, "output": text}),
    )
}

fn user(text: &str) -> Value {
    record(
        "response_item",
        json!({"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}),
    )
}

fn run(records: &[Value]) -> Value {
    parse("codex", PATH, &jsonl(records))
}

/// The outcome of one `exec_command` call whose output is `text`.
fn outcome(text: &str) -> Value {
    let p = run(&[
        meta(json!({})),
        call("c1", "exec_command"),
        output("c1", text),
    ]);
    blocks(&p)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap()["outcome"]
        .clone()
}

#[test]
fn the_exec_envelope_header_gives_the_exit_code() {
    assert_eq!(
        outcome("Chunk ID: a\nWall time: 1.0 seconds\nProcess exited with code 0\nOutput:\nok"),
        json!({"kind": "ok"})
    );
    assert_eq!(
        outcome("Chunk ID: a\nWall time: 1.0 seconds\nProcess exited with code 2\nOutput:\nfail"),
        json!({"kind": "error", "reason": {"kind": "exit_code", "detail": 2}})
    );
    assert_eq!(
        outcome("Exit code: 1\nWall time: 1s\nOutput:\nx"),
        json!({"kind": "error", "reason": {"kind": "exit_code", "detail": 1}})
    );
}

#[test]
fn exit_wording_inside_the_command_output_is_not_the_exit_status() {
    let text = "Chunk ID: abc123\nWall time: 5.0 seconds\nProcess exited with code 0\n\
                Output:\n##[error]Process completed with exit code 1.\nsome CI log tail\n";
    assert_eq!(outcome(text), json!({"kind": "ok"}));
    // No header at all: the marker in the output decides nothing.
    let text = "Chunk ID: a\nWall time: 1.0 seconds\nOutput:\nProcess exited with code 1\n";
    assert_eq!(outcome(text)["kind"], "unknown");
    assert_eq!(
        outcome("The log says: Process exited with code 1")["kind"],
        "unknown"
    );
}

#[test]
fn a_still_running_command_has_no_verdict_yet() {
    assert_eq!(
        outcome(
            "Chunk ID: a\nWall time: 10.0 seconds\nProcess running with session ID 5\nOutput:\nwait"
        ),
        json!({"kind": "unknown", "reason": "running_in_background"})
    );
}

#[test]
fn a_structural_timeout_is_an_error() {
    assert_eq!(
        outcome(r#"{"message": "Wait timed out.", "timed_out": true}"#),
        json!({"kind": "error", "reason": {"kind": "timed_out"}})
    );
}

#[test]
fn a_json_envelope_exit_code_is_structural() {
    assert_eq!(
        outcome(r#"{"output":"x","metadata":{"exit_code":0}}"#),
        json!({"kind": "ok"})
    );
    assert_eq!(
        outcome(r#"{"output":"x","metadata":{"exit_code":3}}"#)["reason"]["detail"],
        3
    );
}

#[test]
fn an_unmarked_command_output_is_unknown_not_success() {
    assert_eq!(
        outcome("file.txt"),
        json!({"kind": "unknown", "reason": "status_field_unreadable"})
    );
}

#[test]
fn a_result_before_its_call_still_pairs_by_call_id() {
    let p = run(&[meta(json!({})), output("c9", "early"), call("c9", "shell")]);
    let ids: Vec<_> = blocks(&p).iter().map(|b| b["call_id"].clone()).collect();
    assert_eq!(ids, [json!("c9"), json!("c9")]);
}

#[test]
fn a_truncated_stream_keeps_the_call_without_a_result() {
    let p = run(&[meta(json!({})), call("c1", "shell")]);
    let b = blocks(&p);
    assert_eq!(b.len(), 1);
    assert_eq!(b[0]["type"], "tool_call");
}

#[test]
fn harness_context_written_as_a_user_message_is_not_typed_by_the_operator() {
    let p = run(&[
        meta(json!({})),
        user("<environment_context>\n<cwd>/w</cwd>\n</environment_context>"),
        user("<user_instructions>do x</user_instructions>"),
        user("real prompt"),
        record(
            "response_item",
            json!({"type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "dev"}]}),
        ),
    ]);
    let kinds: Vec<_> = messages(&p)
        .iter()
        .map(|m| (m["role"].clone(), m["prompt_kind"].clone()))
        .collect();
    assert_eq!(
        kinds,
        [
            (json!("user"), json!("harness")),
            (json!("user"), json!("harness")),
            (json!("user"), json!("prose")),
            (json!("system"), Value::Null),
        ]
    );
}

#[test]
fn reasoning_is_kept_as_summary_text_raw_text_or_a_redacted_mark() {
    let p = run(&[
        meta(json!({})),
        record(
            "response_item",
            json!({"type": "reasoning", "id": "rs1",
            "summary": [{"type": "summary_text", "text": "Sum"}], "encrypted_content": "gAAA"}),
        ),
        record(
            "response_item",
            json!({"type": "reasoning", "id": "rs2", "summary": [], "encrypted_content": "gBBB"}),
        ),
        record(
            "response_item",
            json!({"type": "reasoning", "id": "rs3", "summary": [],
            "content": [{"type": "reasoning_text", "text": "raw"}]}),
        ),
    ]);
    let b = blocks(&p);
    assert_eq!(*b[0], json!({"type": "thinking", "text": "Sum"}));
    assert_eq!(
        *b[1],
        json!({"type": "thinking", "text": null, "redacted": true}),
        "encrypted-only reasoning is still recorded"
    );
    assert_eq!(*b[2], json!({"type": "thinking", "text": "raw"}));
}

#[test]
fn a_fork_names_its_parent_session() {
    let p = run(&[
        meta(json!({"forked_from_id": "019a0000-0000-7000-8000-0000000000ff"})),
        user("hi"),
    ]);
    let links = p["output"]["links"].as_array().unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0]["kind"], "fork");
    assert_eq!(
        links[0]["parent"],
        "codex:019a0000-0000-7000-8000-0000000000ff"
    );
}

#[test]
fn turn_context_directory_changes_join_the_session_directories_in_order() {
    let p = run(&[
        meta(json!({})),
        record("turn_context", json!({"cwd": "/a", "model": "gpt-5"})),
        user("x"),
        record("turn_context", json!({"cwd": "/b", "model": "gpt-5"})),
        user("y"),
        record("turn_context", json!({"cwd": "/b", "model": "gpt-5"})),
        record("turn_context", json!({"cwd": "/a", "model": "gpt-5"})),
    ]);
    // Each change is recorded, a return to an earlier directory included;
    // a repeat of the current one is not.
    assert_eq!(sessions(&p)[0]["cwd"], json!(["/work", "/a", "/b", "/a"]));
}

#[test]
fn a_second_session_meta_does_not_open_another_session() {
    let p = run(&[
        meta(json!({})),
        user("a"),
        meta(json!({"id": "019a0000-0000-7000-8000-000000000002"})),
        user("b"),
    ]);
    assert_eq!(sessions(&p).len(), 1);
    assert!(
        messages(&p)
            .iter()
            .all(|m| m["session"] == format!("codex:{SID}"))
    );
}
