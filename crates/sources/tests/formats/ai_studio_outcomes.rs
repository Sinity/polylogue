//! AI Studio execution outcomes use Google's CodeExecutionResult enum,
//! not the result prose or an absent-success default.

use crate::support::{blocks, messages, parse};
use serde_json::{Value, json};

fn parsed(result: &Value, in_parts: bool) -> Value {
    let code = json!({"language": "PYTHON", "code": "print('synthetic')"});
    let mut chunk = json!({"id": "answer", "role": "model", "text": "Authored explanation"});
    let media = [
        json!({"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}),
        json!({"fileData": {"mimeType": "text/plain", "fileUri": "drive://synthetic-output"}}),
    ];
    let mut parts = vec![json!({"text": "Authored explanation"})];
    if in_parts {
        parts.extend([
            json!({"executableCode": code}),
            json!({"codeExecutionResult": result}),
        ]);
    } else {
        chunk["executableCode"] = code;
        chunk["codeExecutionResult"] = result.clone();
    }
    parts.extend(media);
    chunk["parts"] = json!(parts);
    let prompt = json!({"chunkedPrompt": {"chunks": [chunk]}});
    let parsed = parse(
        "ai-studio",
        "polylogue/drive/ai-studio/tool-outcomes.json",
        &serde_json::to_vec(&prompt).unwrap(),
    );
    let authored = messages(&parsed);
    assert_eq!(authored.len(), 1);
    assert_eq!(authored[0]["session"], "ai-studio:tool-outcomes");
    assert_eq!(authored[0]["role"], "assistant");
    assert_eq!(authored[0]["blocks"][0]["text"], "Authored explanation");
    let all = blocks(&parsed);
    let calls: Vec<_> = all.iter().filter(|b| b["type"] == "tool_call").collect();
    let results: Vec<_> = all.iter().filter(|b| b["type"] == "tool_result").collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(results.len(), 1);
    assert_eq!(calls[0]["call_id"], "13:tool-outcomes#id:6:answer:code1");
    assert_eq!(results[0]["call_id"], calls[0]["call_id"]);
    assert_eq!(results[0]["data"], *result);
    assert_eq!(results[0]["output"], result["output"]);
    let attachments: Vec<_> = all
        .iter()
        .filter(|b| b["type"] == "image" || b["type"] == "file")
        .collect();
    assert_eq!(attachments.len(), 2);
    for attachment in attachments {
        assert_eq!(attachment["direction"], "output");
    }
    assert!(all.iter().any(|b| b["url"] == "drive://synthetic-output"));
    parsed
}

fn outcome(parsed: &Value) -> &Value {
    &blocks(parsed)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap()["outcome"]
}

#[test]
fn unreported_or_unreadable_execution_outcomes_stay_unknown() {
    for status in [
        None,
        Some(Value::Null),
        Some(json!(false)),
        Some(json!(true)),
        Some(json!(0)),
        Some(json!(1)),
        Some(json!({})),
        Some(json!([])),
        Some(json!("OUTCOME_UNSPECIFIED")),
        Some(json!("OUTCOME_FUTURE")),
        Some(json!("outcome_ok")),
        Some(json!("OUTCOME_OK_EXTRA")),
    ] {
        let mut result =
            json!({"output": "Error: synthetic command failed", "future_field": {"kept": false}});
        if let Some(value) = status {
            result["outcome"] = value;
        }
        for in_parts in [false, true] {
            let parsed = parsed(&result, in_parts);
            assert_eq!(
                outcome(&parsed),
                &json!({"kind": "unknown", "reason": "status_field_unreadable"}),
                "{result}"
            );
        }
    }
}

#[test]
fn documented_execution_outcomes_override_prose_and_keep_timeout_reason() {
    for (status, output, expected) in [
        (
            "OUTCOME_OK",
            "Error: synthetic text quotes a failure",
            json!({"kind": "ok"}),
        ),
        (
            "OUTCOME_FAILED",
            "Success: synthetic text quotes a success",
            json!({"kind": "error"}),
        ),
        (
            "OUTCOME_DEADLINE_EXCEEDED",
            "partial synthetic output",
            json!({"kind": "error", "reason": {"kind": "timed_out"}}),
        ),
    ] {
        let result = json!({"outcome": status, "output": output});
        for in_parts in [false, true] {
            let parsed = parsed(&result, in_parts);
            assert_eq!(outcome(&parsed), &expected);
        }
    }
}
