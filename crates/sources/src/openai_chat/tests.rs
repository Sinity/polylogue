use super::*;
use serde_json::json;

fn parse(file: &str, doc: Json) -> crate::parse::Step {
    let parser = OpenAiChat {
        file: file.into(),
        state: ChatState::default(),
    };
    crate::export::fixture::parse(
        &mut Parse::Export(Box::new(parser)),
        &serde_json::to_vec(&doc).unwrap(),
    )
    .unwrap()
}

#[test]
fn an_array_with_a_tool_round_trip() {
    let step = parse(
        "chat-1",
        json!([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": [{"type": "text", "text": "weather?"},
                                          {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
            {"role": "assistant", "content": null, "reasoning_content": "need a tool",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "weather", "arguments": "{\"city\":\"Oslo\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "rain"},
            {"role": "assistant", "content": "Rain."}
        ]),
    );
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs.len(), 5);
    assert_eq!(msgs[0].role, Role::System);
    assert!(
        matches!(&msgs[1].blocks[1], Block::Image(a) if a.url.as_deref() == Some("https://x/y.png"))
    );
    assert!(matches!(&msgs[2].blocks[0], Block::Thinking { .. }));
    match &msgs[2].blocks[1] {
        Block::ToolCall { tool, input, .. } => {
            assert_eq!(tool.as_deref(), Some("weather"));
            assert_eq!(input["city"], "Oslo");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(msgs[3].prompt_kind, Some(PromptKind::Harness));
    assert!(matches!(&msgs[3].blocks[0], Block::ToolResult { output: Some(o), .. } if o == "rain"));
    assert_eq!(msgs[4].id.native().as_str(), "chat-1#4");
    assert_eq!(
        msgs[4].parent.as_ref().unwrap().native().as_str(),
        "chat-1#3"
    );
    assert!(
        step.counts.unread_fields.is_empty(),
        "{:?}",
        step.counts.unread_fields
    );
}

#[test]
fn an_object_names_its_own_session() {
    let step = parse(
        "file",
        json!({"messages": [{"role": "user", "content": "hi"}], "id": "conv-9",
               "model": "gpt-4o", "title": "Greeting"}),
    );
    assert_eq!(step.output.sessions[0].id.native().as_str(), "conv-9");
    assert_eq!(
        step.output.sessions[0].title.as_ref().unwrap().text,
        "Greeting"
    );
}

#[test]
fn recognizes_chat_arrays_and_objects() {
    assert!(recognize(Path::new("/x/a.json"), br#"[{"role": "user""#));
    assert!(recognize(Path::new("/x/a.json"), br#"{"messages": ["#));
    assert!(!recognize(Path::new("/x/a.json"), br#"[{"uuid": "x""#));
    assert!(!recognize(Path::new("/x/a.txt"), br#"[{"role": "user""#));
}

#[test]
fn refused_and_empty_messages_keep_raw_ordinals_without_breaking_parent_links() {
    let step = parse(
        "p",
        json!([
            {"role":"user","content":"question"},
            {"role":"future_role","content":"unread"},
            {"role":"assistant","content":""},
            {"role":"assistant","content":"answer"}
        ]),
    );
    assert_eq!(step.output.messages.len(), 2);
    assert_eq!(step.output.messages[1].message.id.native().as_str(), "p#3");
    assert_eq!(
        step.output.messages[1].message.parent.as_ref(),
        Some(&step.output.messages[0].message.id)
    );
    assert_eq!(step.counts.records, 5);
    assert_eq!(
        step.counts
            .unreadable
            .get("future_role: unknown role future_role"),
        Some(&1)
    );
    assert_eq!(
        step.counts
            .skipped
            .get("assistant: a message with no content"),
        Some(&1)
    );
}

#[test]
fn invalid_wrappers_refuse_the_input_without_emitting_child_messages() {
    for (file, doc, reason) in [
        ("p", json!({"messages":false}), "no messages array"),
        (
            "",
            json!({"id":"","messages":[{"role":"user","content":"not emitted"}]}),
            "unusable session name",
        ),
        ("p", json!(true), "not an array or object"),
    ] {
        let step = parse(file, doc);
        assert!(step.output.sessions.is_empty());
        assert!(step.output.messages.is_empty());
        assert_eq!(step.counts.records, 1);
        assert_eq!(
            step.counts.unreadable.get(&format!("chat: {reason}")),
            Some(&1)
        );
    }
}

#[test]
fn an_empty_wrapper_id_still_uses_a_nonempty_file_name() {
    let step = parse(
        "file",
        json!({"id":"","messages":[{"role":"user","content":"kept"}]}),
    );
    assert_eq!(
        step.output.messages[0].message.id.native().as_str(),
        "file#0"
    );
    assert!(step.counts.unreadable.is_empty());
}
