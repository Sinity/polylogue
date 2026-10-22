use super::*;
use serde_json::json;

#[test]
fn event_identity_distinguishes_message_anchors_and_ignores_physical_ranges() {
    let mut event = Event {
        session: polylogue_model::SessionId::parse("chatgpt:example").unwrap(),
        kind: "active_leaf".into(),
        at: None,
        after: Some(polylogue_model::MessageId::parse("chatgpt:a").unwrap()),
        range: Some((10, 20)),
        data: json!(null),
    };
    let first = pack_event(&event).unwrap().0;
    event.range = Some((100, 200));
    assert_eq!(pack_event(&event).unwrap().0, first);
    event.after = Some(polylogue_model::MessageId::parse("chatgpt:b").unwrap());
    let second = pack_event(&event).unwrap().0;
    assert_ne!(second, first);
    event.after = None;
    let unanchored = pack_event(&event).unwrap().0;
    assert_ne!(unanchored, first);
    assert_ne!(unanchored, second);
}

#[test]
fn text_rows_read_older_compression_levels_and_reuse_a_packer() {
    let mut packer = RowPacker::new().unwrap();
    for text in ["", "Łódź café 🦀", "a repeated body", "a repeated body"] {
        let old = zstd::bulk::compress(text.as_bytes(), 3).unwrap();
        assert_eq!(unpack_text(&old).unwrap(), text);
        assert_eq!(unpack_text(&pack_text(text).unwrap()).unwrap(), text);
        assert_eq!(unpack_text(&packer.pack_text(text).unwrap()).unwrap(), text);
    }
}

#[test]
fn batch_encoder_keeps_independent_frames_across_row_types_and_sizes() {
    let mut packer = RowPacker::new().unwrap();
    for size in [0, 64 * 1024, 3, 1024 * 1024, 1] {
        let text = "λ".repeat(size);
        let value = json!({"text":text,"nested":[null,false,42],"unknown":{"empty":[]}});
        assert_eq!(packer.pack_text(&text).unwrap(), pack_text(&text).unwrap());
        assert_eq!(
            packer.pack_json(&value).unwrap(),
            pack_json(&value).unwrap()
        );
        let event = Event {
            session: polylogue_model::SessionId::parse("claude-code:session").unwrap(),
            kind: "synthetic".into(),
            at: None,
            after: None,
            range: None,
            data: value.clone(),
        };
        assert_eq!(
            packer.pack_event(&event).unwrap(),
            pack_event(&event).unwrap()
        );
        let message: Message = serde_json::from_value(json!({
            "id":"claude-code:message", "session":"claude-code:session", "role":"assistant",
            "blocks":[{"type":"tool_call","input":value}], "usage":{}
        }))
        .unwrap();
        let prepared = packer.pack_message(&message).unwrap();
        assert_eq!(prepared, PreparedMessage::new(&message).unwrap());
        assert_eq!(prepared.blocks[0], pack_block(&message.blocks[0]).unwrap());
        assert_eq!(
            prepared.hash,
            *blake3::hash(&serde_json::to_vec(&message).unwrap()).as_bytes()
        );
        let mut metadata = message;
        metadata.blocks.clear();
        assert_eq!(prepared.attrs, pack_json(&metadata).unwrap());
    }
}

#[test]
fn block_rows_separate_only_text_and_restore_the_complete_value() {
    for value in [
        json!({"type": "text", "text": "prose"}),
        json!({"type": "thinking", "text": "reasoning", "signature": "sig", "redacted": false}),
        json!({"type": "tool_result", "call_id": "call:1", "output": "result", "data": {"kept": [1, null]}, "outcome": {"kind":"ok"}}),
        json!({"type": "tool_call", "call_id": "call:1", "tool": "shell", "input": {"command": "true"}}),
    ] {
        let block: Block = serde_json::from_value(value).unwrap();
        let text = block_text(&block).map(str::to_owned);
        let mut restored = unpack_block(&pack_block(&block).unwrap()).unwrap();
        if let Some(text) = text {
            assert!(block_text(&restored).is_none());
            restore_text(&mut restored, text);
        }
        assert_eq!(restored, block);
    }
}

#[test]
fn damaged_compression_and_json_are_distinct_failures() {
    assert!(matches!(unpack_text(b"not zstd"), Err(StoreError::Io(_))));
    assert!(matches!(
        unpack_text(&pack_bytes(&[0xff]).unwrap()),
        Err(StoreError::Io(_))
    ));
    assert!(matches!(
        unpack_json::<serde_json::Value>(&pack_text("not JSON").unwrap()),
        Err(StoreError::Json(_))
    ));
    let value = json!({"nested": [null, "Łódź", 42]});
    assert_eq!(
        unpack_json::<serde_json::Value>(&pack_json(&value).unwrap()).unwrap(),
        value
    );
}

#[test]
fn prepared_rows_match_previous_codec_for_every_block_kind() {
    let blocks = vec![
        json!({"type":"text","text":"Łódź prose"}),
        json!({"type":"text","text":""}),
        json!({"type":"thinking","text":null,"signature":"sig","redacted":true}),
        json!({"type":"thinking","text":"reasoning","signature":"sig"}),
        json!({"type":"tool_call","call_id":"call:1","tool":"shell","action":"shell","input":{"nested":[null,42,"payload"]},"command":"true","path":"/synthetic"}),
        json!({"type":"tool_result","call_id":"call:1","outcome":{"kind":"ok"},"output":"large output","data":{"nested":[false,"kept"]},"attachments":[{"name":"file","direction":"output"}]}),
        json!({"type":"tool_result","call_id":null,"outcome":{"kind":"no_result"},"output":null}),
        json!({"type":"image","name":"image","direction":"input"}),
        json!({"type":"file","name":"file","direction":"output"}),
        json!({"type":"other","type_name":"future","data":{"kept":[1,null,"payload"]}}),
    ];
    let message: Message = serde_json::from_value(json!({
        "id":"claude-code:message", "session":"claude-code:session",
        "parent":"claude-code:parent", "role":"assistant", "model":"synthetic-model",
        "prompt_kind":"model", "blocks":blocks, "usage":{}, "stop_reason":"end_turn",
        "reported_cost_usd":0.25, "duration_ms":42,
        "production":{"effort":"high","permission":{"decision":"allowed"},"request_id":"request"}
    }))
    .unwrap();
    let prepared = PreparedMessage::new(&message).unwrap();
    let mut old_metadata = message.clone();
    old_metadata.blocks.clear();
    assert_eq!(prepared.attrs, pack_json(&old_metadata).unwrap());
    assert_eq!(
        prepared.hash,
        *blake3::hash(&serde_json::to_vec(&message).unwrap()).as_bytes()
    );
    for (block, packed) in message.blocks.iter().zip(&prepared.blocks) {
        let mut old_body = block.clone();
        match &mut old_body {
            Block::Text { text }
            | Block::Thinking {
                text: Some(text), ..
            }
            | Block::ToolResult {
                output: Some(text), ..
            } => text.clear(),
            _ => {}
        }
        assert_eq!(*packed, pack_json(&old_body).unwrap());
    }
}
