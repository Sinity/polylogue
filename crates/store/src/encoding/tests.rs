use super::*;
use serde_json::json;

#[test]
fn event_identity_distinguishes_message_anchors_and_ignores_physical_ranges() {
    let mut event = Event {
        session: Some(polylogue_model::SessionId::parse("chatgpt:example").unwrap()),
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
        let packed = packer.pack_json(&value).unwrap();
        assert!(packed.len() <= pack_json(&value).unwrap().len());
        assert_eq!(unpack_json::<serde_json::Value>(&packed).unwrap(), value);
        let event = Event {
            session: Some(polylogue_model::SessionId::parse("claude-code:session").unwrap()),
            kind: "synthetic".into(),
            at: None,
            after: None,
            range: None,
            data: value.clone(),
        };
        let (id, packed) = packer.pack_event(&event).unwrap();
        let (old_id, old_packed) = pack_event(&event).unwrap();
        assert_eq!(id, old_id);
        assert!(packed.len() <= old_packed.len());
        assert_eq!(
            unpack_json::<serde_json::Value>(&packed).unwrap(),
            event.data
        );
        let message: Message = serde_json::from_value(json!({
            "id":"claude-code:message", "session":"claude-code:session", "role":"assistant",
            "blocks":[{"type":"tool_call","input":value}], "usage":{}
        }))
        .unwrap();
        let prepared = packer.pack_message(&message).unwrap();
        assert_eq!(prepared, PreparedMessage::new(&message).unwrap());
        assert_eq!(
            unpack_block(&prepared.blocks[0]).unwrap(),
            unpack_block(&pack_block(&message.blocks[0]).unwrap()).unwrap()
        );
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
fn prepared_rows_preserve_values_for_every_block_kind() {
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
        assert_eq!(unpack_block(packed).unwrap(), old_body);
    }
}

#[test]
fn dictionary_rows_and_plain_sorted_frames_remain_distinct() {
    let message: Message = serde_json::from_value(json!({
        "id":"codex:sample-42", "session":"codex:sample-session",
        "parent":"codex:sample-parent", "role":"assistant", "model":"synthetic-model",
        "prompt_kind":"model", "blocks":[], "usage":{}, "stop_reason":"end_turn"
    }))
    .unwrap();
    let dictionary = zstd::zstd_safe::get_dict_id_from_dict(ROW_DICTIONARY).unwrap();
    let row = RowPacker::new().unwrap().pack_json(&message).unwrap();
    assert_eq!(
        zstd::zstd_safe::get_dict_id_from_frame(&row),
        Some(dictionary)
    );
    assert_eq!(unpack_json::<Message>(&row).unwrap(), message);
    for plain in [
        pack_json(&message).unwrap(),
        pack_text("archive_conn text").unwrap(),
    ] {
        assert!(zstd::zstd_safe::get_dict_id_from_frame(&plain).is_none());
        assert_eq!(
            unpack_bytes(&plain).unwrap(),
            zstd::stream::decode_all(&plain[..]).unwrap()
        );
    }
    let bytes = serde_json::to_vec(&message).unwrap();
    for level in [1, 3] {
        let other_level = zstd::stream::encode_all(&bytes[..], level).unwrap();
        assert_eq!(unpack_json::<Message>(&other_level).unwrap(), message);
    }
}

#[test]
fn unknown_dictionary_and_truncated_dictionary_frames_refuse() {
    // The formatted dictionary begins with its magic and little-endian ID.
    let mut other = ROW_DICTIONARY.to_vec();
    let id = u32::from_le_bytes(other[4..8].try_into().unwrap());
    other[4..8].copy_from_slice(&(id + 1).to_le_bytes());
    let bytes = br#"{"type":"text","text":""}"#;
    let packed = zstd::bulk::Compressor::with_dictionary(LEVEL, &other)
        .unwrap()
        .compress(bytes)
        .unwrap();
    assert_ne!(
        zstd::zstd_safe::get_dict_id_from_frame(&packed),
        zstd::zstd_safe::get_dict_id_from_dict(ROW_DICTIONARY)
    );
    assert!(matches!(
        unpack_json::<serde_json::Value>(&packed),
        Err(StoreError::Io(_))
    ));
    let packed = RowPacker::new().unwrap().pack_row(bytes).unwrap();
    assert!(zstd::zstd_safe::get_dict_id_from_frame(&packed).is_some());
    assert!(unpack_bytes(&packed[..packed.len() - 1]).is_err());
}

#[test]
fn unrelated_row_values_never_grow_relative_to_plain_encoding() {
    let mut packer = RowPacker::new().unwrap();
    for size in [0_usize, 1, 1024, 64 * 1024] {
        let values = (0..size)
            .map(|n| blake3::hash(&n.to_le_bytes()).to_hex().to_string())
            .collect::<Vec<_>>();
        let value = json!({"unfamiliar":[null, false, values],"integer":u64::MAX});
        let packed = packer.pack_json(&value).unwrap();
        assert!(packed.len() <= pack_json(&value).unwrap().len());
        assert_eq!(unpack_json::<serde_json::Value>(&packed).unwrap(), value);
    }
}

#[test]
fn boundary_suppliedness_and_unassociated_usage_survive_row_codecs() {
    for boundary in [None, Some(true), Some(false)] {
        let mut message: Message =
            serde_json::from_value(json!({"id":"codex:item", "session":"codex:thread",
            "role":"assistant", "blocks":[]}))
            .unwrap();
        message.production.response_boundary_unknown = boundary;
        let row = RowPacker::new().unwrap().pack_json(&message).unwrap();
        assert_eq!(unpack_json::<Message>(&row).unwrap(), message);
    }
    let reading = polylogue_model::UsageObservation {
        observation: "synthetic-reading".into(),
        usage: polylogue_model::Usage {
            input: Some(0),
            output: None,
            ..Default::default()
        },
        model: None,
        response: None,
        completion_observed: None,
        observed_at: None,
        input_including_cache: Some(0),
        output_including_reasoning: None,
    };
    let event = Event {
        session: Some(polylogue_model::SessionId::parse("codex:thread").unwrap()),
        kind: "usage_observation".into(),
        at: None,
        after: None,
        range: Some((1, 2)),
        data: serde_json::to_value(&reading).unwrap(),
    };
    let (_, packed) = RowPacker::new().unwrap().pack_event(&event).unwrap();
    assert_eq!(
        unpack_json::<polylogue_model::UsageObservation>(&packed).unwrap(),
        reading
    );
}
