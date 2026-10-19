use super::*;
use crate::parse::run_document;
use serde_json::json;

fn attr(k: &str, v: Json) -> Json {
    let value = match v {
        Json::String(s) => json!({"stringValue": s}),
        Json::Number(n) => json!({"intValue": n.to_string()}),
        other => other,
    };
    json!({"key": k, "value": value})
}

fn chat_span(id: &str, start: u64, inputs: Json, outputs: Json, tokens: (u64, u64)) -> Json {
    json!({"traceId": "t1", "spanId": id, "name": "chat gpt-4o",
    "startTimeUnixNano": start.to_string(),
    "endTimeUnixNano": (start + 2_000_000_000).to_string(),
    "attributes": [
        attr("gen_ai.operation.name", json!("chat")),
        attr("gen_ai.conversation.id", json!("conv-1")),
        attr("gen_ai.request.model", json!("gpt-4o")),
        attr("gen_ai.input.messages", json!(inputs.to_string())),
        attr("gen_ai.output.messages", json!(outputs.to_string())),
        attr("gen_ai.usage.input_tokens", json!(tokens.0)),
        attr("gen_ai.usage.output_tokens", json!(tokens.1)),
    ]})
}

#[test]
fn repeated_history_is_written_once_in_order() {
    let user = json!({"role": "user", "parts": [{"type": "text", "content": "hi"}]});
    let reply = json!({"role": "assistant", "parts": [{"type": "text", "content": "hello"}],
                       "finish_reason": "stop"});
    let user2 = json!({"role": "user", "parts": [{"type": "text", "content": "more"}]});
    let reply2 = json!({"role": "assistant", "parts": [
        {"type": "reasoning", "content": "think"},
        {"type": "tool_call", "id": "c1", "name": "search", "arguments": {"q": "x"}}]});
    let doc = json!({"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{
    "scope": {"name": "s"},
    "spans": [
        chat_span("b", 2_000_000_000_000, json!([user, reply, user2]), json!([reply2]), (30, 9)),
        chat_span("a", 1_000_000_000_000, json!([user]), json!([reply]), (10, 4)),
        {"traceId": "t1", "spanId": "c", "name": "http GET", "attributes": []}
    ]}]}]});
    let step = run_document(&OtelGenAi, &serde_json::to_vec(&doc).unwrap()).unwrap();
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    let texts: Vec<String> = msgs.iter().map(|m| m.text()).collect();
    assert_eq!(texts, ["hi", "hello", "more", "think"]);
    assert_eq!(msgs[1].usage.output, Some(4));
    assert_eq!(msgs[1].model.as_deref(), Some("gpt-4o"));
    assert_eq!(msgs[1].duration_ms, Some(2000));
    assert_eq!(msgs[3].usage.input, Some(30));
    assert!(matches!(&msgs[3].blocks[1], Block::ToolCall { tool: Some(t), .. } if t == "search"));
    assert_eq!(msgs[3].id.native().as_str(), "conv-1#3");
    assert_eq!(step.output.sessions.len(), 1);
    assert_eq!(step.accounting.framed, 3);
}

#[test]
fn line_per_document_exports_are_read() {
    let one = json!({"resourceSpans": [{"scopeSpans": [{"spans": [
        chat_span("a", 1_000_000_000_000,
                  json!([{"role": "user", "parts": [{"type": "text", "content": "q"}]}]),
                  json!([{"role": "assistant", "parts": [{"type": "text", "content": "a"}]}]),
                  (1, 1))]}]}]});
    let bytes = format!("{one}\n{one}\n");
    let step = run_document(&OtelGenAi, bytes.as_bytes()).unwrap();
    assert_eq!(step.output.messages.len(), 2);
    assert!(recognize(
        Path::new("/x/traces.jsonl"),
        br#"{"resourceSpans":"#
    ));
}
