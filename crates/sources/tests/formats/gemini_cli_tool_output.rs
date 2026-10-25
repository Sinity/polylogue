//! Inline tool-output knowledge from `test_gemini_cli_tool_output_losses.py`.
//! Stored-file sidecar joins belong to import, outside this parser.

use crate::support::{blocks, message, parse};
use serde_json::{Value, json};

const PATH: &str = ".gemini/tmp/synthetic/chats/session-s1.json";

fn call(response: Option<Value>, display: Option<Value>, status: Option<Value>) -> Value {
    let mut call =
        json!({"id": "c1", "name": "run_shell_command", "args": {"command": "printf synthetic"}});
    if let Some(response) = response {
        call["result"] = json!([{"functionResponse": {"id": "c1", "name": "run_shell_command", "response": response}}]);
    }
    if let Some(display) = display {
        call["resultDisplay"] = display;
    }
    if let Some(status) = status {
        call["status"] = status;
    }
    call
}

fn parsed(call: &Value) -> Value {
    let messages = json!([{"id": "a1", "type": "gemini", "timestamp": "2026-03-14T21:41:02Z", "toolCalls": [call]}]);
    let source = format!(r#"{{"sessionId":"s1","messages":{messages}}}"#);
    let origin = polylogue_sources::origin::by_name("gemini-cli").unwrap();
    assert!((origin.recognize)(
        std::path::Path::new(PATH),
        source.as_bytes()
    ));
    let parsed = parse("gemini-cli", PATH, source.as_bytes());
    assert_eq!(message(&parsed, ":a1")["role"], "assistant");
    assert_eq!(message(&parsed, ":a1")["session"], "gemini-cli:s1");
    parsed
}

fn result(parsed: &Value) -> &Value {
    let results: Vec<_> = blocks(parsed)
        .into_iter()
        .filter(|b| b["type"] == "tool_result")
        .collect();
    assert_eq!(
        results.len(),
        1,
        "one carrier keeps the tool's renderings"
    );
    results[0]
}

#[test]
fn gemini_cli_output_and_error_survive_with_identical_or_divergent_display() {
    for display in ["PARTIAL_STDOUT", "TERMINAL_RENDERING"] {
        let wire = call(
            Some(json!({"output": "PARTIAL_STDOUT", "error": "DISTINCT_STDERR"})),
            Some(json!(display)),
            Some(json!("error")),
        );
        let parsed = parsed(&wire);
        let result = result(&parsed);
        assert_eq!(result["output"], "PARTIAL_STDOUT\nDISTINCT_STDERR");
        assert_eq!(result["call_id"], "c1");
        assert_eq!(result["outcome"], json!({"kind": "error"}));
        assert_eq!(result["data"]["result"], wire["result"]);
        assert_eq!(result["data"]["resultDisplay"], display);
    }
}

#[test]
fn gemini_cli_marker_text_and_longer_ui_do_not_replace_native_streams() {
    for output in [
        "<tool_output_masked>head ... tail</tool_output_masked>",
        "Output too large. Showing first 8,000 and last 32,000 characters\nhead ... tail",
        "the tool literally printed '<tool_output_masked>' in its documentation",
        "example: Output too large. Showing first 8,000 and last 32,000 characters",
        "Output too large. Showing first several and last some characters",
        "ordinary output",
    ] {
        let display = "unrelated terminal status and decorative rendering\n".repeat(400);
        let wire = call(
            Some(json!({"output": output, "error": "DISTINCT_STDERR"})),
            Some(json!(display)),
            Some(json!("error")),
        );
        let parsed = parsed(&wire);
        let result = result(&parsed);
        assert_eq!(result["output"], format!("{output}\nDISTINCT_STDERR"));
        assert_eq!(result["data"]["result"], wire["result"]);
        assert_eq!(result["data"]["resultDisplay"], display);
        assert!(result["data"].get("output_was_truncated").is_none());
    }
}

#[test]
fn gemini_cli_display_length_and_presence_do_not_establish_model_visible_output() {
    let output = "<tool_output_masked>kept head and tail excerpts</tool_output_masked>";
    for display in ["two words".into(), "雪".repeat(400)] {
        let wire = call(
            Some(json!({"output": output})),
            Some(json!(display)),
            Some(json!("success")),
        );
        let parsed = parsed(&wire);
        assert_eq!(result(&parsed)["output"], output);
        assert_eq!(result(&parsed)["data"]["resultDisplay"], display);
    }
    let display = "The terminal showed this; the model-facing result is absent.";
    let parsed = parsed(&call(None, Some(json!(display)), Some(json!("success"))));
    assert_eq!(result(&parsed)["output"], Value::Null);
    assert_eq!(result(&parsed)["data"]["resultDisplay"], display);
    assert_eq!(result(&parsed)["outcome"], json!({"kind": "ok"}));
}

#[test]
fn gemini_cli_display_cells_and_structured_media_remain_tool_data_not_authored_blocks() {
    let cells = json!([[{"text": "FILE", "fg": "cyan", "bold": true}], [{"text": "alpha.txt", "fg": "white", "bold": false}]]);
    let media = json!({"inlineData": {"mimeType": "image/png", "data": "c3ludGhldGlj"}, "empty": [], "flag": false, "zero": 0});
    for response in [json!({"output": "alpha.txt\n"}), json!({"output": media})] {
        let wire = call(
            Some(response.clone()),
            Some(cells.clone()),
            Some(json!("success")),
        );
        let parsed = parsed(&wire);
        let kinds: Vec<_> = blocks(&parsed)
            .into_iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["tool_call", "tool_result"]);
        assert_eq!(result(&parsed)["data"]["resultDisplay"], cells);
        assert_eq!(result(&parsed)["data"]["result"], wire["result"]);
        if response["output"].is_string() {
            assert_eq!(result(&parsed)["output"], "alpha.txt\n");
        } else {
            assert_eq!(result(&parsed)["output"], Value::Null);
        }
        assert!(result(&parsed).get("attachments").is_none());
    }
}

#[test]
fn gemini_cli_empty_null_and_absent_streams_and_display_remain_distinct() {
    for (response, expected) in [
        (Value::Null, Value::Null),
        (json!([]), Value::Null),
        (json!({}), Value::Null),
        (json!({"output": null, "error": null}), Value::Null),
        (json!({"output": "", "error": null}), json!("")),
        (json!({"error": ""}), json!("")),
        (json!({"output": false, "error": 0}), Value::Null),
        (json!({"output": "", "error": "stderr"}), json!("stderr")),
        (json!({"output": "same", "error": "same"}), json!("same")),
    ] {
        let wire = call(Some(response), Some(Value::Null), Some(json!("success")));
        let parsed = parsed(&wire);
        assert_eq!(result(&parsed)["output"], expected);
        assert_eq!(result(&parsed)["data"]["result"], wire["result"]);
        assert_eq!(
            result(&parsed)["data"].get("resultDisplay"),
            Some(&Value::Null)
        );
    }
    for display in [
        None,
        Some(Value::Null),
        Some(json!("")),
        Some(json!([])),
        Some(json!({})),
    ] {
        let wire = call(None, display.clone(), None);
        let parsed = parsed(&wire);
        assert!(result(&parsed)["data"].get("result").is_none());
        assert_eq!(
            result(&parsed)["data"].get("resultDisplay"),
            display.as_ref()
        );
        assert_eq!(result(&parsed)["output"], Value::Null);
    }
    for supplied in [Value::Null, json!([])] {
        let mut wire = call(None, None, None);
        wire["result"] = supplied.clone();
        let parsed = parsed(&wire);
        assert_eq!(result(&parsed)["data"].get("result"), Some(&supplied));
        assert_eq!(result(&parsed)["output"], Value::Null);
        assert_eq!(result(&parsed)["outcome"], json!({"kind": "no_result"}));
    }
}

#[test]
fn gemini_cli_results_keep_structural_outcomes_without_guessing_from_text() {
    for (status, expected) in [
        (Some(json!("success")), json!({"kind": "ok"})),
        (Some(json!("error")), json!({"kind": "error"})),
        (
            Some(json!("cancelled")),
            json!({"kind": "error", "reason": {"kind": "interrupted"}}),
        ),
        (
            Some(json!("provider-new-status")),
            json!({"kind": "unknown", "reason": "status_field_unreadable"}),
        ),
        (
            Some(json!(false)),
            json!({"kind": "unknown", "reason": "status_field_unreadable"}),
        ),
        (
            Some(Value::Null),
            json!({"kind": "unknown", "reason": "status_field_unreadable"}),
        ),
        (
            None,
            json!({"kind": "unknown", "reason": "status_field_unreadable"}),
        ),
    ] {
        let wire = call(
            Some(json!({"output": "prose mentions error and Exit code 2"})),
            None,
            status,
        );
        let parsed = parsed(&wire);
        assert_eq!(result(&parsed)["outcome"], expected);
        assert_eq!(result(&parsed)["data"].get("status"), wire.get("status"));
    }
    let absent = parsed(&call(None, None, None));
    assert_eq!(result(&absent)["outcome"], json!({"kind": "no_result"}));
    let pending = parsed(&call(None, None, Some(json!("executing"))));
    assert_eq!(result(&pending)["outcome"], json!({"kind": "no_result"}));
}
