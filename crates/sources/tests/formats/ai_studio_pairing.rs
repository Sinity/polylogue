//! Execution results without IDs need a preceding unanswered call.

use crate::support::{blocks, parse};
use serde_json::{Value, json};

fn code(number: u64) -> Value {
    json!({"executableCode": {"language": "PYTHON", "code": format!("print({number})")}})
}

fn result(label: &str) -> Value {
    json!({"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": label,
        "future_field": {"kept": false}}})
}

fn run(mut chunk: Value) -> Value {
    chunk["id"] = json!("turn");
    chunk["role"] = json!("model");
    let parsed = run_chunks(&[chunk]);
    assert_eq!(parsed["output"]["messages"].as_array().unwrap().len(), 1);
    parsed
}

fn run_chunks(chunks: &[Value]) -> Value {
    let bytes = serde_json::to_vec(&json!({"chunkedPrompt": {"chunks": chunks}})).unwrap();
    let parsed = parse(
        "ai-studio",
        "polylogue/drive/ai-studio/pairing.json",
        &bytes,
    );
    for row in parsed["output"]["messages"].as_array().unwrap() {
        let start = row["offset"].as_u64().unwrap() as usize;
        let end = row["end"].as_u64().unwrap() as usize;
        let chunk = serde_json::from_slice::<Value>(&bytes[start..end]).unwrap();
        assert!(chunks.contains(&chunk));
        assert_eq!(row["message"]["session"], "ai-studio:pairing");
        assert_eq!(
            row["message"]["id"],
            format!("ai-studio:pairing#{}", chunk["id"].as_str().unwrap())
        );
    }
    parsed
}

fn check_results(parsed: &Value, expected: &[(&Value, Option<&str>)]) {
    let results: Vec<_> = blocks(parsed)
        .into_iter()
        .filter(|b| b["type"] == "tool_result")
        .collect();
    assert_eq!(results.len(), expected.len());
    for (actual, (native, owner)) in results.iter().zip(expected) {
        assert_eq!(actual["data"], native["codeExecutionResult"]);
        assert_eq!(actual["output"], native["codeExecutionResult"]["output"]);
        assert_eq!(actual["outcome"], json!({"kind": "ok"}));
        assert_eq!(actual["call_id"], json!(owner));
    }
}

#[test]
fn a_result_only_chunk_or_part_keeps_an_unresolved_call_identity() {
    let orphan = result("orphan");
    for chunk in [orphan.clone(), json!({"parts": [orphan.clone()]})] {
        let parsed = run(chunk);
        check_results(&parsed, &[(&orphan, None)]);
        assert!(!blocks(&parsed).iter().any(|b| b["type"] == "tool_call"));
    }
}

#[test]
fn a_result_before_code_cannot_borrow_that_future_call_identity() {
    let before = result("before");
    let owned = result("owned");
    let parsed = run(json!({"parts": [before, code(1), owned,
        {"fileData": {"mimeType": "text/plain", "fileUri": "drive://synthetic-output"}}]}));
    check_results(
        &parsed,
        &[(&before, None), (&owned, Some("pairing#turn:code1"))],
    );
    let calls: Vec<_> = blocks(&parsed)
        .into_iter()
        .filter(|b| b["type"] == "tool_call")
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["call_id"], "pairing#turn:code1");
    let file = blocks(&parsed)
        .into_iter()
        .find(|b| b["type"] == "file")
        .unwrap();
    assert_eq!(file["url"], "drive://synthetic-output");
    assert_eq!(file["direction"], "output");
}

#[test]
fn an_answered_execution_cannot_lend_its_identity_to_an_extra_result() {
    let first = result("first");
    let stray = result("stray");
    let second = result("second");
    let parsed = run(json!({"parts": [code(1), first, stray, code(2), second]}));
    check_results(
        &parsed,
        &[
            (&first, Some("pairing#turn:code1")),
            (&stray, None),
            (&second, Some("pairing#turn:code2")),
        ],
    );
    let calls: Vec<_> = blocks(&parsed)
        .into_iter()
        .filter(|b| b["type"] == "tool_call")
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["call_id"], "pairing#turn:code1");
    assert_eq!(calls[1]["call_id"], "pairing#turn:code2");
}

#[test]
fn multiple_unanswered_executions_do_not_establish_a_unique_result_owner() {
    let ambiguous = result("ambiguous");
    let stray = result("still unowned");
    let fresh = result("fresh pair");
    let parsed = run(json!({"parts": [code(1), code(2), ambiguous, stray, code(3), fresh]}));
    check_results(
        &parsed,
        &[
            (&ambiguous, None),
            (&stray, None),
            (&fresh, Some("pairing#turn:code3")),
        ],
    );
    let calls: Vec<_> = blocks(&parsed)
        .into_iter()
        .filter(|b| b["type"] == "tool_call")
        .collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0]["call_id"], "pairing#turn:code1");
    assert_eq!(calls[1]["call_id"], "pairing#turn:code2");
    assert_eq!(calls[2]["call_id"], "pairing#turn:code3");
}

#[test]
fn consecutive_model_chunks_keep_the_actual_preceding_call_identity() {
    let native = result("owned across chunks");
    let parsed = run_chunks(&[
        json!({"id": "call", "role": "model", "executableCode": code(1)["executableCode"]}),
        json!({"id": "explanation", "role": "model", "text": "Continuing explanation"}),
        json!({"id": "answer", "role": "model", "codeExecutionResult": native["codeExecutionResult"]}),
    ]);
    check_results(&parsed, &[(&native, Some("pairing#call:code1"))]);
    let rows = parsed["output"]["messages"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1]["message"]["parent"], rows[0]["message"]["id"]);
    assert_eq!(rows[2]["message"]["parent"], rows[1]["message"]["id"]);
}

#[test]
fn user_or_unreadable_chunks_end_the_preceding_unanswered_call() {
    let native = result("unowned after boundary");
    for boundary in [
        json!({"id": "boundary", "role": "user", "text": "new prompt"}),
        json!({"id": "boundary", "role": "user", "executableCode": code(2)["executableCode"]}),
        json!({"id": "boundary", "role": "future-role", "text": "unknown role"}),
        json!({"id": "boundary", "role": null}),
        json!({"id": "boundary", "role": "model", "parts": []}),
    ] {
        let expected_rows = if boundary["role"] == "user" { 3 } else { 2 };
        let parsed = run_chunks(&[
            json!({"id": "call", "role": "model", "executableCode": code(1)["executableCode"]}),
            boundary,
            json!({"id": "answer", "role": "model", "codeExecutionResult": native["codeExecutionResult"]}),
        ]);
        check_results(&parsed, &[(&native, None)]);
        assert_eq!(
            parsed["output"]["messages"].as_array().unwrap().len(),
            expected_rows
        );
    }
}

#[test]
fn several_unanswered_calls_across_chunks_do_not_establish_a_unique_owner() {
    let native = result("ambiguous across chunks");
    let parsed = run_chunks(&[
        json!({"id": "first", "role": "model", "executableCode": code(1)["executableCode"]}),
        json!({"id": "second", "role": "model", "executableCode": code(2)["executableCode"]}),
        json!({"id": "third", "role": "model", "executableCode": code(3)["executableCode"]}),
        json!({"id": "answer", "role": "model", "codeExecutionResult": native["codeExecutionResult"]}),
    ]);
    check_results(&parsed, &[(&native, None)]);
    assert_eq!(
        blocks(&parsed)
            .into_iter()
            .filter(|b| b["type"] == "tool_call")
            .count(),
        3
    );
}
