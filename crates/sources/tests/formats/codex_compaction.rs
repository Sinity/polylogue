//! Native compaction and turn-context cases from the Python version's
//! `test_codex_compaction_and_turn_context.py`, through the public parser.

use crate::support::{events, jsonl, messages, parse};
use serde_json::{Value, json};

const PATH: &str = ".codex/sessions/2026/03/01/rollout-2026-03-01T10-00-00-019a0000-0000-7000-8000-000000000001.jsonl";
const SID: &str = "019a0000-0000-7000-8000-000000000001";

fn record(kind: &str, payload: Value) -> Value {
    json!({"timestamp": "2026-03-01T10:00:01Z", "type": kind, "payload": payload})
}

fn meta(extra: Value) -> Value {
    let mut payload = json!({"id": SID});
    payload
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    record("session_meta", payload)
}

fn message(id: &str, role: &str, text: &str) -> Value {
    record(
        "response_item",
        json!({"id": id, "type": "message", "role": role,
        "content": [{"type": if role == "user" { "input_text" } else { "output_text" }, "text": text}]}),
    )
}

fn run(records: &[Value]) -> Value {
    parse("codex", PATH, &jsonl(records))
}

#[test]
fn compaction_keeps_large_native_history_without_creating_messages() {
    let payload = json!({
        "message": "compaction summary",
        "replacement_history": [{"id": "replacement-only", "type": "message", "role": "user",
            "phase": "pre", "model": "history-only-model",
            "content": [{"type": "input_text", "text": "unique context ".repeat(24_000)}]}],
        "replacement_history_items": "native unknown count",
        "after": "native metadata", "range": null,
        "window_id": "window-2", "window_number": 2, "previous_window_id": "window-1",
        "latest_token_usage_record": null, "unknown_context": {"future_field": false}
    });
    let records = [
        meta(json!({})),
        message("before", "user", "original"),
        record("compacted", payload.clone()),
        message("after", "assistant", "continued"),
    ];
    let parsed = run(&records);
    let compacted = events(&parsed, "compaction");
    assert_eq!(compacted.len(), 1);
    let event = compacted[0];
    assert_eq!(event["data"], payload);
    assert_eq!(event["session"], format!("codex:{SID}"));
    assert_eq!(event["after"], "codex:before");
    assert_eq!(event["at"], "2026-03-01T10:00:01Z");
    let start = jsonl(&records[..2]).len();
    assert_eq!(
        event["range"],
        json!([
            start,
            start + serde_json::to_vec(&records[2]).unwrap().len()
        ])
    );
    let authored = messages(&parsed);
    assert_eq!(authored.len(), 2);
    assert_eq!(authored[0]["id"], "codex:before");
    assert_eq!(authored[1]["id"], "codex:after");
    for message in authored {
        assert_eq!(message["session"], format!("codex:{SID}"));
        assert!(message.get("model").is_none());
    }
}

#[test]
fn compaction_preserves_supplied_unknown_history_shapes() {
    let payloads = [
        json!({"message": "older summary"}),
        json!({"replacement_history": null}),
        json!({"replacement_history": false}),
        json!({"replacement_history": "future representation"}),
        json!({"replacement_history": {"items": []}}),
        json!({"replacement_history": []}),
    ];
    for payload in payloads {
        let parsed = run(&[meta(json!({})), record("compacted", payload.clone())]);
        let compacted = events(&parsed, "compaction");
        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0]["data"], payload);
        assert!(compacted[0].get("after").is_none());
        assert!(messages(&parsed).is_empty());
    }
}

#[test]
fn compaction_keeps_parent_history_and_the_existing_fork_anchor() {
    let payload = json!({"replacement_history": [
        {"id": "parent-message", "type": "message", "role": "user", "content": []},
        {"type": "future_context", "unknown_context": null}]});
    let parsed = run(&[
        meta(json!({"forked_from_id": "parent-session"})),
        record("compacted", payload.clone()),
        message("child-message", "user", "child tail"),
    ]);
    assert_eq!(events(&parsed, "compaction")[0]["data"], payload);
    let authored = messages(&parsed);
    assert_eq!(authored.len(), 1);
    assert_eq!(authored[0]["session"], format!("codex:{SID}"));
    assert_eq!(authored[0]["id"], "codex:child-message");
    let links = parsed["output"]["links"].as_array().unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0]["kind"], "fork");
    assert_eq!(links[0]["parent"], "codex:parent-session");
    assert_eq!(links[0]["child"], format!("codex:{SID}"));
    assert_eq!(links[0]["at"], "codex:parent-message");
}

#[test]
fn changed_turn_context_keeps_each_context_and_returns_to_prior_settings() {
    let first = json!({"turn_id": "t1", "user_instructions": "first", "workspace_roots": []});
    let second = json!({"turn_id": "t2", "user_instructions": null, "workspace_roots": []});
    let third = json!({"turn_id": "t3", "root_turn_id": "root", "user_instructions": "first", "workspace_roots": []});
    let fourth = json!({"turn_id": "t4", "user_instructions": "first", "workspace_roots": []});
    let records = [
        meta(json!({})),
        record("turn_context", first.clone()),
        message("prompt", "user", "supplied prompt"),
        record("turn_context", second.clone()),
        record("turn_context", third.clone()),
        record("turn_context", fourth.clone()),
    ];
    let parsed = run(&records);
    let contexts = events(&parsed, "turn_context");
    assert_eq!(contexts.len(), 4);
    for (event, (index, payload)) in
        contexts
            .iter()
            .zip([(1, first), (3, second), (4, third), (5, fourth)])
    {
        assert_eq!(event["data"], payload);
        assert_eq!(event["session"], format!("codex:{SID}"));
        let start = jsonl(&records[..index]).len();
        assert_eq!(
            event["range"],
            json!([
                start,
                start + serde_json::to_vec(&records[index]).unwrap().len()
            ])
        );
    }
    assert!(contexts[0].get("after").is_none());
    assert_eq!(contexts[1]["after"], "codex:prompt");
    assert_eq!(contexts[2]["after"], "codex:prompt");
    assert_eq!(contexts[3]["after"], "codex:prompt");
    assert_eq!(parsed["output"]["events"].as_array().unwrap().len(), 6);
    assert_eq!(messages(&parsed).len(), 1);
}
