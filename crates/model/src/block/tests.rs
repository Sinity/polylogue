use super::*;

#[test]
fn blocks_serialize_with_a_type_tag() {
    let b = Block::ToolResult {
        reported_duration_ms: None,
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
fn a_tool_call_stored_without_an_action_reads_as_other() {
    // A block stored before calls carried an action reads as `other`.
    let old: Block = serde_json::from_str(
        r#"{"type": "tool_call", "call_id": null, "tool": "x", "input": null}"#,
    )
    .unwrap();
    assert!(matches!(
        old,
        Block::ToolCall {
            action: ToolAction::Other,
            ..
        }
    ));
}

#[test]
fn text_covers_prose_and_thinking_only() {
    assert_eq!(Block::Text { text: "a".into() }.text(), Some("a"));
    let call = Block::ToolCall {
        written_text: None,
        run_by_script: false,
        call_id: None,
        tool: None,
        action: ToolAction::Other,
        input: Json::Null,
        command: None,
        path: None,
    };
    assert_eq!(call.text(), None);
}

#[test]
fn explicit_unknown_attachment_direction_keeps_known_wire_values() {
    for (direction, wire) in [
        (AttachmentDirection::Input, "input"),
        (AttachmentDirection::Output, "output"),
        (AttachmentDirection::Unknown, "unknown"),
    ] {
        let value = Attachment {
            source_id: Some("images/neutral.png".into()),
            direction,
            ..Attachment::default()
        };
        let encoded = serde_json::to_value(&value).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({"source_id":"images/neutral.png","direction":wire})
        );
        assert_eq!(
            serde_json::from_value::<Attachment>(encoded).unwrap(),
            value
        );
    }
    assert_eq!(Attachment::default().direction, AttachmentDirection::Input);
}

#[test]
fn attachments_keep_direct_and_tool_result_source_order() {
    let first = Attachment {
        name: Some("first.png".into()),
        media_type: Some("image/png".into()),
        direction: AttachmentDirection::Input,
        ..Attachment::default()
    };
    let second = Attachment {
        name: Some("second.txt".into()),
        direction: AttachmentDirection::Output,
        ..Attachment::default()
    };
    assert_eq!(
        Block::Image(first.clone()).attachments(),
        std::slice::from_ref(&first)
    );
    assert_eq!(
        Block::File(second.clone()).attachments(),
        std::slice::from_ref(&second)
    );
    let result = Block::ToolResult {
        reported_duration_ms: None,
        call_id: None,
        outcome: ToolOutcome::Ok,
        output: None,
        data: None,
        attachments: vec![first.clone(), second.clone()],
    };
    assert_eq!(result.attachments(), &[first, second]);
    let without_attachments = [
        Block::Text {
            text: "text".into(),
        },
        Block::Thinking {
            text: Some("thought".into()),
            signature: None,
            redacted: false,
        },
        Block::ToolCall {
            call_id: None,
            tool: None,
            action: ToolAction::Other,
            input: serde_json::json!({"attachments": [{"name": "input.json"}]}),
            written_text: None,
            command: None,
            path: None,
            run_by_script: false,
        },
        Block::Other {
            type_name: "outside".into(),
            data: serde_json::json!({}),
        },
    ];
    for block in without_attachments {
        assert_eq!(block.attachments(), &[]);
    }
}
