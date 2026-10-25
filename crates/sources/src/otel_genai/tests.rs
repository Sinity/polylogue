use super::*;
fn parse(bytes: &[u8]) -> Step {
    crate::export::fixture::parse(&mut open(Path::new("traces.json")).unwrap(), bytes).unwrap()
}
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
    let step = parse(&serde_json::to_vec(&doc).unwrap());
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
    assert_eq!(step.counts.records, 3);
}

#[test]
fn line_per_document_exports_are_read() {
    let one = json!({"resourceSpans": [{"scopeSpans": [{"spans": [
        chat_span("a", 1_000_000_000_000,
                  json!([{"role": "user", "parts": [{"type": "text", "content": "q"}]}]),
                  json!([{"role": "assistant", "parts": [{"type": "text", "content": "a"}]}]),
                  (1, 1))]}]}]});
    let bytes = format!("{one}\n{one}\n");
    let step = parse(bytes.as_bytes());
    assert_eq!(step.output.messages.len(), 2);
    assert!(recognize(
        Path::new("/x/traces.jsonl"),
        br#"{"resourceSpans":"#
    ));
}

fn wrapped(spans: Vec<Json>) -> Json {
    json!({"resourceSpans":[{"scopeSpans":[{"spans":spans}]}]})
}
fn named(id: &str, time: Option<u64>, conversation: &str) -> Json {
    let mut span = json!({"traceId":"trace","spanId":id,"attributes":[
        attr("gen_ai.operation.name",json!("chat")),
        attr("gen_ai.conversation.id",json!(conversation)),
        attr("gen_ai.input.messages",json!(json!([{"role":"user","parts":[{"type":"text","content":conversation}]}]).to_string())),
    ]});
    if let Some(time) = time {
        span["startTimeUnixNano"] = json!(time.to_string());
    }
    span
}

#[test]
fn malformed_middle_lines_discard_their_prefix_and_preserve_later_spans() {
    let first = wrapped(vec![named("b", Some(2_000_000_000_000), "kept")]);
    let bad = wrapped(vec![named("a", Some(1_000_000_000_000), "refused")]);
    let last = wrapped(vec![named("c", Some(3_000_000_000_000), "later")]);
    let bytes = format!("{first}\n{bad}!\n{last}\n");
    let step = parse(bytes.as_bytes());
    assert_eq!(
        step.output
            .messages
            .iter()
            .map(|r| r.message.text())
            .collect::<Vec<_>>(),
        ["kept", "later"]
    );
    assert_eq!(step.counts.records, 3);
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 1);
    for event in &step.output.events {
        let (a, b) = event.range.unwrap();
        let raw: Json = serde_json::from_slice(&bytes.as_bytes()[a as usize..b as usize]).unwrap();
        assert_eq!(raw, event.data);
        assert_ne!(raw["spanId"], "a");
    }
    for row in &step.output.messages {
        let raw: Json =
            serde_json::from_slice(&bytes.as_bytes()[row.offset as usize..row.end as usize])
                .unwrap();
        assert!(raw.get("spanId").is_some());
    }
}

#[test]
fn nanoseconds_are_exact_unknowns_precede_known_and_equal_keys_keep_physical_order() {
    let spans = vec![
        named("last", Some(u64::MAX), "max"),
        named("tie", Some(42), "tie first"),
        named("b", None, "unknown b"),
        named("big", Some(9_007_199_254_740_993), "big next"),
        named("big", Some(9_007_199_254_740_992), "big first"),
        named("tie", Some(42), "tie second"),
        named("a", None, "unknown a"),
        named("zero", Some(0), "zero"),
    ];
    let step = parse(wrapped(spans).to_string().as_bytes());
    assert_eq!(
        step.output
            .messages
            .iter()
            .map(|r| r.message.text())
            .collect::<Vec<_>>(),
        [
            "unknown a",
            "unknown b",
            "zero",
            "tie first",
            "tie second",
            "big first",
            "big next",
            "max"
        ]
    );
}

#[test]
fn duplicate_selected_containers_and_bad_suffixes_emit_no_prefix_spans() {
    let span = named("a", Some(1), "prefix");
    let duplicate = format!(
        r#"{{"resourceSpans":[{{"scopeSpans":[{{"spans":[{span}]}}]}}],"resourceSpans":[]}}"#
    );
    for bytes in [duplicate, format!("{} trailing", wrapped(vec![span]))] {
        let step = parse(bytes.as_bytes());
        assert!(step.output.messages.is_empty());
        assert!(step.output.events.is_empty());
        assert_eq!(step.counts.unreadable.values().sum::<u64>(), 1);
    }
}

#[test]
fn an_out_of_range_numeric_timestamp_stays_unknown_without_dropping_its_span() {
    let bad = wrapped(vec![named("a", Some(1), "large timestamp")])
        .to_string()
        .replace("\"startTimeUnixNano\":\"1\"", "\"startTimeUnixNano\":1e999");
    assert!(bad.contains("1e999"));
    let good = wrapped(vec![named("b", Some(2), "good")]);
    let step = parse(format!("{bad}\n{good}\n").as_bytes());
    assert_eq!(
        step.output
            .messages
            .iter()
            .map(|r| r.message.text())
            .collect::<Vec<_>>(),
        ["large timestamp", "good"]
    );
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 0);
    assert!(step.output.events[0].at.is_none());
}
