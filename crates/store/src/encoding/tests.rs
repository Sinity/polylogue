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
    let mut packer = TextPacker::new().unwrap();
    for text in ["", "Łódź café 🦀", "a repeated body", "a repeated body"] {
        let old = zstd::bulk::compress(text.as_bytes(), 3).unwrap();
        assert_eq!(unpack_text(&old).unwrap(), text);
        assert_eq!(unpack_text(&pack_text(text).unwrap()).unwrap(), text);
        assert_eq!(unpack_text(&packer.pack(text).unwrap()).unwrap(), text);
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
