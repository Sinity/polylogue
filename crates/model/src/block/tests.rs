use super::*;

#[test]
fn blocks_serialize_with_a_type_tag() {
    let b = Block::ToolResult {
        call_id: Some("t1".into()),
        outcome: ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(2)),
        },
        output: Some("boom".into()),
        data: None,
        attachments: vec![],
    };
    let v = serde_json::to_value(&b).unwrap();
    assert_eq!(v["type"], "tool_result");
    assert_eq!(v["outcome"]["kind"], "error");
    assert_eq!(v["outcome"]["reason"]["kind"], "exit_code");
    let back: Block = serde_json::from_value(v).unwrap();
    assert_eq!(back, b);
}

#[test]
fn text_covers_prose_and_thinking_only() {
    assert_eq!(Block::Text { text: "a".into() }.text(), Some("a"));
    let call = Block::ToolCall {
        call_id: None,
        tool: None,
        input: Json::Null,
        command: None,
        path: None,
    };
    assert_eq!(call.text(), None);
}
