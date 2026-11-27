use super::*;
use serde_json::json;

#[test]
fn repeated_text_blocks_reuse_the_same_frame_and_keep_distinct_text_ids() {
    let blocks = [
        Block::Text {
            text: "first".into(),
        },
        Block::Text {
            text: "second".into(),
        },
    ];
    let expected = RowPacker::new()
        .unwrap()
        .pack_row(&block_bytes(&blocks[0]).unwrap())
        .unwrap();
    let mut packer = RowPacker::new().unwrap();
    let packed = packer.pack_blocks(&blocks, [0; 32]).unwrap();
    assert_eq!(packed.blocks, [expected.clone(), expected.clone()]);
    assert_eq!(
        packer.empty_text_block.as_deref(),
        Some(expected.as_slice())
    );
    assert_ne!(packed.texts[0], packed.texts[1]);
    let mut block = unpack_block(&packed.blocks[1]).unwrap();
    restore_text(&mut block, "second".into());
    assert_eq!(block, blocks[1]);
}

#[test]
fn event_identity_depends_on_its_message_not_its_range() {
    let mut event = Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(polylogue_model::SessionId::parse("chatgpt:example").unwrap()),
        kind: "active_leaf".into(),
        at: None,
        after: Some(polylogue_model::MessageId::parse("chatgpt:a").unwrap()),
        range: Some((10, 20)),
        data: json!(null),
    };
    let first = event_content_id(&event).unwrap();
    event.range = Some((100, 200));
    assert_eq!(event_content_id(&event).unwrap(), first);
    event.after = Some(polylogue_model::MessageId::parse("chatgpt:b").unwrap());
    let second = event_content_id(&event).unwrap();
    assert_ne!(second, first);
    event.after = None;
    let unplaced = event_content_id(&event).unwrap();
    assert_ne!(unplaced, first);
    assert_ne!(unplaced, second);
}

#[test]
fn event_id_independent_of_map_insertion_order() {
    let event = |data: serde_json::Value| Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(polylogue_model::SessionId::parse("chatgpt:example").unwrap()),
        kind: "settings".into(),
        at: None,
        after: None,
        range: None,
        data,
    };
    // The workspace builds serde_json with preserve_order, so these two maps
    // keep their members in the order written here.
    let forward: serde_json::Value =
        serde_json::from_str(r#"{"a": 1, "b": {"x": 1.50, "y": [true]}}"#).unwrap();
    let reverse: serde_json::Value =
        serde_json::from_str(r#"{"b": {"y": [true], "x": 1.50}, "a": 1}"#).unwrap();
    assert_ne!(forward.to_string(), reverse.to_string());
    let id = event_content_id(&event(forward)).unwrap();
    assert_eq!(id, event_content_id(&event(reverse)).unwrap());
    let respelled: serde_json::Value =
        serde_json::from_str(r#"{"a": 1, "b": {"x": 1.5, "y": [true]}}"#).unwrap();
    assert_ne!(id, event_content_id(&event(respelled)).unwrap());
}

#[test]
fn text_bodies_read_any_compression_level() {
    for text in ["", "Łódź café 🦀", "a repeated body", "a repeated body"] {
        let old = zstd::bulk::compress(text.as_bytes(), 3).unwrap();
        assert_eq!(unpack_text(&old).unwrap(), text);
        assert_eq!(unpack_text(&pack_text(text).unwrap()).unwrap(), text);
    }
}

#[test]
fn row_frames_and_message_content_read_back_across_sizes() {
    let mut packer = RowPacker::new().unwrap();
    for size in [0, 64 * 1024, 3, 1024 * 1024, 1] {
        let text = "λ".repeat(size);
        let value = json!({"text":text,"nested":[null,false,42],"unknown":{"empty":[]}});
        let packed = packer.pack_json(&value).unwrap();
        assert!(packed.len() <= pack_json(&value).unwrap().len());
        assert_eq!(unpack_json::<serde_json::Value>(&packed).unwrap(), value);
        let mut message: Message = serde_json::from_value(json!({
            "id":"claude-code:message", "session":"claude-code:session", "role":"assistant",
            "blocks":[{"type":"tool_call","input":value}], "usage":{}
        }))
        .unwrap();
        let before = message.clone();
        let content = message_content(&mut message, &[]).unwrap();
        assert_eq!(message, before);
        assert_eq!(read_message_content(&content, true).unwrap().0, message);
    }
}

#[test]
fn block_rows_separate_text_and_data_and_restore_the_complete_value() {
    for value in [
        json!({"type": "text", "text": "prose"}),
        json!({"type": "thinking", "text": "reasoning", "signature": "sig", "redacted": false}),
        json!({"type": "tool_result", "call_id": "call:1", "output": "result", "data": {"kept": [1, null]}, "outcome": {"kind":"ok"}}),
        json!({"type": "tool_result", "call_id": "call:2", "output": null, "data": [true], "outcome": {"kind":"ok"}}),
        json!({"type": "tool_call", "call_id": "call:1", "tool": "shell", "input": {"command": "true"}}),
        json!({"type": "tool_call", "call_id": "call:3", "tool": "write", "input": {"path": "a"}, "written_text": "new file body"}),
    ] {
        let block: Block = serde_json::from_value(value).unwrap();
        let text = block_text(&block).map(str::to_owned);
        let data = block_data(&block).cloned();
        let mut restored = unpack_block(&pack_block(&block).unwrap()).unwrap();
        assert!(block_data(&restored).is_none());
        if let Some(text) = text {
            assert!(block_text(&restored).is_none());
            restore_text(&mut restored, text);
        }
        if let Some(data) = data {
            set_data(&mut restored, data);
        }
        assert_eq!(restored, block);
    }
}

#[test]
fn damaged_compression_and_json_are_distinct_failures() {
    assert!(matches!(unpack_text(b"not zstd"), Err(StoreError::Io(_))));
    // Text that is not UTF-8 is stored data this build cannot read back.
    assert!(matches!(
        unpack_text(&pack_bytes(&[0xff]).unwrap()),
        Err(StoreError::StoredValue { column: None, .. })
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
    let mut stored = message.clone();
    let content = message_content(&mut stored, &[]).unwrap();
    let (read, pieces) = read_message_content(&content, false).unwrap();
    assert_eq!(pieces.len(), message.blocks.len());
    for ((block, piece), read) in message.blocks.iter().zip(&pieces).zip(&read.blocks) {
        let mut body = block.clone();
        match &mut body {
            Block::Text { text }
            | Block::Thinking {
                text: Some(text), ..
            } => text.clear(),
            Block::ToolResult { output, data, .. } => {
                if let Some(text) = output {
                    text.clear();
                }
                *data = None;
            }
            _ => {}
        }
        assert_eq!(*read, body);
        assert_eq!(
            read_json::<Block>(&content[piece.start..piece.start + piece.length]).unwrap(),
            body
        );
        let data = piece.data_length.map(|length| {
            let at = piece.start + piece.length;
            read_json::<serde_json::Value>(&content[at..at + length]).unwrap()
        });
        assert_eq!(data.as_ref(), block_data(block));
    }
    let (mut whole, _) = read_message_content(&content, true).unwrap();
    for (block, original) in whole.blocks.iter_mut().zip(&message.blocks) {
        if let Some(text) = block_text(original) {
            restore_text(block, text.to_owned());
        }
    }
    assert_eq!(whole, message);
}

#[test]
fn dictionary_rows_and_plain_frames_remain_distinct() {
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
        pack_text("archive text").unwrap(),
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
fn concatenated_frames_and_alternating_frame_kinds_decode_whole() {
    let first = pack_bytes(b"first frame, ").unwrap();
    let joined = [first.clone(), pack_bytes(b"second frame").unwrap()].concat();
    // The first frame's stated size covers only that frame.
    assert_eq!(
        unpack_bytes(&joined).unwrap(),
        b"first frame, second frame".to_vec()
    );
    let bytes = br#"{"type":"text","text":"a dictionary row"}"#;
    let row = RowPacker::new().unwrap().pack_row(bytes).unwrap();
    assert!(zstd::zstd_safe::get_dict_id_from_frame(&row).is_some());
    for _ in 0..2 {
        assert_eq!(unpack_bytes(&row).unwrap(), bytes.to_vec());
        assert_eq!(unpack_bytes(&first).unwrap(), b"first frame, ".to_vec());
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
    assert!(matches!(
        unpack_json::<serde_json::Value>(&packed[..packed.len() - 1]),
        Err(StoreError::Io(_))
    ));
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
    let reading = polylogue_model::UsageReport {
        report_id: "synthetic-reading".into(),
        usage: polylogue_model::Usage {
            input: Some(0),
            output: None,
            ..Default::default()
        },
        model: None,
        response: None,
        measured_completion: None,
        measured_at: None,
        input_including_cache: Some(0),
        output_including_reasoning: None,
    };
    let event = Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(polylogue_model::SessionId::parse("codex:thread").unwrap()),
        kind: "usage_report".into(),
        at: None,
        after: None,
        range: Some((1, 2)),
        data: serde_json::to_value(&reading).unwrap(),
    };
    let mut packer = crate::content::Packer::default();
    let mut stated = event.clone();
    let prepared = PreparedEvent::new(&mut stated, &[], &mut packer).unwrap();
    assert_eq!(stated, event);
    let piece = packer.finish().unwrap().read(prepared.content).unwrap();
    let (fields, data) = piece.split_at(prepared.fields_length as usize);
    assert_eq!(
        read_json::<polylogue_model::UsageReport>(data).unwrap(),
        reading
    );
    let mut fields: Event = read_json(fields).unwrap();
    fields.data = event.data.clone();
    assert_eq!(fields, event);
}

mod columns;
mod dictionary;

proptest::proptest! {
    /// Comparing two session orders compares (order time, the input's
    /// digest, sequence, record position), with an unknown time first.
    #[test]
    fn session_order_bytes_sort_as_their_fields(
        a in (proptest::option::of(proptest::num::i64::ANY), "[a-c]{0,3}", proptest::num::i64::ANY, -2i64..3),
        b in (proptest::option::of(proptest::num::i64::ANY), "[a-c]{0,3}", proptest::num::i64::ANY, -2i64..3),
    ) {
        let bytes = |(time, input, seq, position): &(Option<i64>, String, i64, i64)| {
            session_order(*time, input, *seq, *position)
        };
        let digest = |input: &str| blake3::hash(input.as_bytes()).as_bytes()[..16].to_vec();
        let fields = |(time, input, seq, position): &(Option<i64>, String, i64, i64)| {
            (*time, digest(input), *seq, *position)
        };
        proptest::prop_assert_eq!(bytes(&a).cmp(&bytes(&b)), fields(&a).cmp(&fields(&b)));
        proptest::prop_assert_eq!(bytes(&a).len(), ORDER_LENGTH);
    }
}

#[test]
fn text_hash_is_blake3_of_the_composed_form() {
    let composed = *blake3::hash("caf\u{e9}".as_bytes()).as_bytes();
    assert_eq!(normalized_text_hash("caf\u{e9}"), composed);
    assert_eq!(normalized_text_hash("cafe\u{301}"), composed);
}
