//! Expectations below come from provider fields, independently of parser output.
use super::support::*;
use serde_json::json;

#[test]
fn unused_leaf_counted_and_outcomes_sum_with_malformed_line() {
    let mut input = jsonl(&[
        json!({"type":"session_meta","payload":{"id":"morrow"}}),
        json!({"type":"response_item","payload":{"type":"message","id":"answer","role":"assistant","content":[{"type":"output_text","text":"done","future":{"nested":[1,2]}}]}}),
    ]);
    input.extend_from_slice(b"{broken}\n");
    let parsed = parse("codex", "sessions/rollout-morrow.jsonl", &input);
    assert_eq!(parsed["counts"]["records"], 3);
    assert_eq!(parsed["counts"]["used"]["session_meta"], 1);
    assert_eq!(parsed["counts"]["used"]["response_item/message"], 1);
    assert_eq!(
        parsed["counts"]["unread_fields"]["response_item/message.content[].future.nested[]"],
        2
    );
    let unreadable = parsed["counts"]["unreadable"]
        .as_object()
        .unwrap()
        .values()
        .map(|value| value.as_u64().unwrap())
        .sum::<u64>();
    assert_eq!(unreadable, 1);
}

#[test]
fn claude_usage_signature_and_multi_result_metadata_are_read() {
    let input = jsonl(&[
        json!({"type":"assistant","uuid":"a","sessionId":"morrow","message":{"id":"native","role":"assistant","content":[{"type":"thinking","signature":"opaque"}],"usage":{"thinking_tokens":9,"server_tool_use":{"web_search_requests":2,"web_fetch_requests":0},"iterations":3}}}),
        json!({"type":"user","uuid":"u","sessionId":"morrow","toolUseResult":[{"tool_use_id":"read","file":{"filePath":"/home/op/morrow/a.rs","numLines":2}},{"tool_use_id":"edit","exitCode":7,"durationMs":125,"structuredPatch":[{"oldStart":1,"newStart":1,"lines":["-a","+b"]}]}],"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"edit","content":[{"type":"tool_reference","tool_name":"Read"},{"type":"text","text":"failed"}]},{"type":"tool_result","tool_use_id":"read","content":"two lines"}]}}),
    ]);
    let parsed = parse("claude-code", "projects/morrow/morrow.jsonl", &input);
    let assistant = message(&parsed, ":a");
    assert_eq!(assistant["usage"]["reasoning"], 9);
    assert_eq!(
        assistant["production"],
        json!({"server_tool_use":{"web_search_requests":2,"web_fetch_requests":0},"iterations":3})
    );
    assert_eq!(
        assistant["blocks"],
        json!([{"type":"thinking","text":null,"signature":"opaque"}])
    );
    let results = &message(&parsed, ":u")["blocks"];
    assert_eq!(results[0]["reported_duration_ms"], 125);
    assert_eq!(
        results[0]["outcome"],
        json!({"kind":"error","reason":{"kind":"exit_code","detail":7}})
    );
    assert_eq!(
        results[0]["data"]["content"][0],
        json!({"type":"tool_reference","tool_name":"Read"})
    );
    assert_eq!(
        results[0]["data"]["toolUseResult"]["structuredPatch"],
        json!([{"oldStart":1,"newStart":1,"lines":["-a","+b"]}])
    );
    assert_eq!(
        results[1]["data"]["file"],
        json!({"filePath":"/home/op/morrow/a.rs","numLines":2})
    );
}

#[test]
fn codex_session_metadata_phase_encrypted_reasoning_and_reported_duration_survive() {
    let input = jsonl(&[
        json!({"type":"session_meta","payload":{"id":"morrow","base_instructions":{"text":"Be clear."},"model_provider":"openai","git":{"commit_hash":"abc123"},"originator":"cli","workspace_roots":["/home/op/morrow"]}}),
        json!({"type":"turn_context","payload":{"cwd":"/home/op/morrow","model":"gpt-5","context_window":200000,"user_instructions":"Use Rust."}}),
        json!({"type":"response_item","payload":{"type":"message","id":"answer","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"Done."}]}}),
        json!({"type":"response_item","payload":{"type":"reasoning","id":"reason","encrypted_content":"sealed","summary":[]}}),
        json!({"type":"response_item","payload":{"type":"function_call","call_id":"call","name":"exec_command","arguments":"{\"cmd\":\"true\"}"}}),
        json!({"type":"response_item","timestamp":"2026-10-01T12:00:00Z","payload":{"type":"function_call_output","call_id":"call","output":"Wall time: 0.125 seconds\nProcess exited with code 0\nOutput:\nok"}}),
    ]);
    let parsed = parse("codex", "sessions/rollout-morrow.jsonl", &input);
    let session = sessions(&parsed)[0];
    assert_eq!(session["instructions"], "Be clear.");
    assert_eq!(session["model_provider"], "openai");
    assert_eq!(session["git_commit"], "abc123");
    assert_eq!(
        message(&parsed, ":answer")["production"]["phase"],
        "final_answer"
    );
    assert_eq!(
        message(&parsed, ":reason")["blocks"],
        json!([{"type":"thinking","text":null,"signature":"sealed","redacted":true}])
    );
    let result = blocks(&parsed)
        .into_iter()
        .find(|block| block["type"] == "tool_result")
        .unwrap();
    assert_eq!(result["reported_duration_ms"], 125);
    assert_eq!(result["outcome"], json!({"kind":"ok"}));
    assert_eq!(
        events(&parsed, "session_meta")[0]["data"]["workspace_roots"],
        json!(["/home/op/morrow"])
    );
    assert_eq!(
        events(&parsed, "turn_context")[0]["data"]["context_window"],
        200000
    );
}

#[test]
fn import_single_conversations_preserves_provider_metadata_and_token_budget() {
    let chatgpt = parse("chatgpt", "one.json", br#"{"conversation_id":"morrow","mapping":{"u":{"id":"u","message":{"id":"u","author":{"role":"user"},"content":{"content_type":"text","parts":["Hi"]},"weight":0,"metadata":{"citations":[{"url":"https://example.com"}],"canvas":{"id":"c"}}}}}}"#);
    assert_eq!(
        message(&chatgpt, ":u")["attrs"],
        json!({"weight":0,"metadata":{"citations":[{"url":"https://example.com"}],"canvas":{"id":"c"}}})
    );
    let claude = parse("claude-ai", "one.json", br#"{"uuid":"morrow","chat_messages":[{"uuid":"a","sender":"assistant","content":[{"type":"token_budget","remaining":99},{"type":"thinking","thinking":"plan","summaries":[{"text":"summary"}],"start_timestamp":"2026-10-01T12:00:00Z"}]}]}"#);
    assert_eq!(
        message(&claude, ":a")["blocks"][0],
        json!({"type":"other","type_name":"token_budget","data":{"type":"token_budget","remaining":99}})
    );
    assert_eq!(
        message(&claude, ":a")["attrs"]["content"][1]["fields"]["summaries"],
        json!([{"text":"summary"}])
    );
}

#[test]
fn agent_progress_embedded_transcript_keeps_native_message_and_spawn_identity() {
    let input = jsonl(&[
        json!({"type":"progress","sessionId":"morrow","parentToolUseID":"spawn","timestamp":"2026-10-01T12:00:00Z","data":{"type":"agent_progress","agentId":"child","message":{"type":"assistant","uuid":"child-answer","timestamp":"2026-10-01T12:00:01Z","message":{"id":"response","role":"assistant","content":[{"type":"text","text":"Checked."}]}}}}),
    ]);
    let parsed = parse("claude-code", "projects/morrow/morrow.jsonl", &input);
    let answer = message(&parsed, ":child-answer");
    assert_eq!(answer["session"], "claude-code:agent-child");
    assert_eq!(answer["blocks"], json!([{"type":"text","text":"Checked."}]));
    assert_eq!(answer["at"], "2026-10-01T12:00:01Z");
    assert_eq!(parsed["output"]["links"][0]["parent"], "claude-code:morrow");
    assert_eq!(
        parsed["output"]["links"][0]["child"],
        "claude-code:agent-child"
    );
    assert_eq!(parsed["output"]["links"][0]["call_id"], "spawn");
    assert_eq!(parsed["counts"]["records"], 1);
    assert_eq!(parsed["counts"]["used"]["progress/agent_progress"], 1);
}

#[test]
fn nested_export_record_accounts_unknown_leaf_once_at_its_record_path() {
    let parsed = parse("chatgpt", "one.json", br#"{"conversation_id":"morrow","mapping":{"u":{"id":"u","message":{"id":"u","author":{"role":"user"},"content":{"content_type":"text","parts":["Hi"],"future":{"leaf":1}}}}}}"#);
    assert_eq!(
        parsed["counts"]["unread_fields"]["text.content.future.leaf"],
        1
    );
    assert!(
        !parsed["counts"]["unread_fields"]
            .as_object()
            .unwrap()
            .keys()
            .any(|key| key.starts_with("node.message"))
    );
}
