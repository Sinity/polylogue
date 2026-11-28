use crate::{Block, BlockKind, DeliverySender, LINK_KIND_NAMES, LINK_REASON_NAMES, ToolAction};

#[test]
fn block_kind_names_are_the_serde_tags() {
    let blocks = [
        Block::Text { text: "t".into() },
        Block::Thinking {
            text: None,
            signature: None,
            redacted: false,
        },
        Block::ToolCall {
            written_text: None,
            run_by_script: false,
            call_id: None,
            tool: None,
            action: ToolAction::Other,
            input: serde_json::Value::Null,
            command: None,
            path: None,
        },
        Block::ToolResult {
            reported_duration_ms: None,
            call_id: None,
            outcome: crate::ToolOutcome::Ok,
            output: None,
            data: None,
            attachments: vec![],
        },
        Block::Image(Default::default()),
        Block::File(Default::default()),
        Block::Other {
            type_name: "x".into(),
            data: serde_json::Value::Null,
        },
    ];
    let kinds: Vec<BlockKind> = blocks.iter().map(Block::kind).collect();
    assert_eq!(kinds, BlockKind::ALL);
    for b in &blocks {
        let tag = serde_json::to_value(b).unwrap()["type"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(tag, b.kind().as_str());
    }
}

#[test]
fn link_names_cover_every_kind_once() {
    assert_eq!(
        LINK_KIND_NAMES,
        ["spawn", "fork", "continuation", "resume", "reference"]
    );
    assert_eq!(LINK_REASON_NAMES, ["briefing", "send", "mention"]);
}

/// The declared name is the serialized form, and a name outside the set is
/// refused when read.
#[test]
fn declared_names_are_the_serde_form() {
    assert_eq!(
        serde_json::to_value(DeliverySender::Send).unwrap(),
        serde_json::json!("send")
    );
    assert_eq!(
        serde_json::from_value::<DeliverySender>(serde_json::json!("act")).unwrap(),
        DeliverySender::Act
    );
    assert!(serde_json::from_value::<DeliverySender>(serde_json::json!("Send")).is_err());
}

#[test]
fn record_and_bundle_names_keep_their_wire_bytes() {
    use crate::annotation::values::{FieldType, Layout, Outcome, SlotMode};
    use crate::block::{AttachmentDirection, UnknownReason};
    use crate::bundle::{Format, MemberKind, fragments::seek::TextField};

    fn matches_wire<T>(variants: &[T], wire: &str)
    where
        T: std::fmt::Debug + PartialEq + serde::Serialize + serde::de::DeserializeOwned,
    {
        assert_eq!(serde_json::to_string(variants).unwrap(), wire);
        assert_eq!(serde_json::from_str::<Vec<T>>(wire).unwrap(), variants);
    }

    matches_wire(
        FieldType::ALL,
        r#"["string","integer","number","boolean","enum"]"#,
    );
    matches_wire(Layout::ALL, r#"["linear","board","wiki","table"]"#);
    matches_wire(SlotMode::ALL, r#"["follow","fixed"]"#);
    matches_wire(Outcome::ALL, r#"["solved","partial","abandoned"]"#);
    matches_wire(AttachmentDirection::ALL, r#"["unknown","input","output"]"#);
    matches_wire(
        UnknownReason::ALL,
        r#"["status_field_unreadable","running_in_background"]"#,
    );
    matches_wire(Format::ALL, r#"["json","jsonl","markdown","html"]"#);
    matches_wire(MemberKind::ALL, r#"["records","rendered","blob"]"#);
    matches_wire(TextField::ALL, r#"["text","thinking","tool_output"]"#);
}
