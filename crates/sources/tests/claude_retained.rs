//! Native Claude source controls: no database or public-renderer qualification.
use polylogue_model::{BlockContent, NativeId, RawJson, Role};
use polylogue_sources::claude_code::{FileCtx, KeyedRecords, keyed::MemoryReadView};
use polylogue_sources::frame::{FailurePhase, ParseError, drive_jsonl};

#[test]
fn native_ids_opaque_tokens_and_text_repairs_remain_distinct() {
    let bytes = br#"{"type":"user","uuid":"\ud800","sessionId":"s","promptSource":"typed","message":{"content":"text \ud800"},"uninterpreted":"\udfff"}
{"type":"user","uuid":"~u:eda080","sessionId":"s","parentUuid":"\ud800","promptSource":"typed","message":{"content":"literal"}}
{"type":"user","uuid":"\ufffd","sessionId":"s","promptSource":"typed","message":{"content":"scalar"}}
{"type":"assistant","uuid":"a","sessionId":"s","message":{"id":"r","content":[{"type":"tool_use","id":"\ud800","name":"Read","input":{"n":1e+02,"fraction":1.00,"integer":9007199254740993,"\ud800":"opaque\udfff"}}]}}
"#;
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        bytes,
        0,
    );
    assert!(framed.counts.balanced());
    assert!(framed.counts.undecodable.is_empty());
    assert_eq!(
        framed.counts.text_repairs.count(),
        1,
        "only semantic text is repaired"
    );
    let output = framed.output.unwrap();
    let messages: Vec<_> = output
        .delta
        .responses
        .iter()
        .map(|(_, _, c)| &c.message)
        .collect();
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages
            .iter()
            .map(|m| &m.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        4
    );
    let lone_key = RawJson::parse(br#""\ud800""#).unwrap();
    let encoded_key = serde_json::to_string(lone_key.native().unwrap()).unwrap();
    let lone = &output
        .delta
        .responses
        .iter()
        .find(|(_, key, _)| key.role == Role::User && key.response == encoded_key)
        .unwrap()
        .2
        .message;
    assert!(messages.iter().any(|m| m.parent.as_ref() == Some(&lone.id)));
    let tool = output
        .delta
        .contributions
        .iter()
        .flat_map(|(_, _, c)| &c.blocks)
        .find_map(|b| {
            if let BlockContent::ToolCall(c) = b {
                Some(c)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(tool.call_id.as_ref().unwrap().encoding(), Some("wtf8"));
    let input = tool.input.as_ref().unwrap();
    assert_eq!(input.get("n").unwrap().canonical(), b"1e+02");
    assert_eq!(input.get("fraction").unwrap().canonical(), b"1.00");
    assert_eq!(
        input.get("integer").unwrap().canonical(),
        b"9007199254740993"
    );
    let lone_key = RawJson::parse(br#""\ud800""#).unwrap();
    let native = lone_key.native().unwrap();
    assert!(input.as_object().unwrap().contains_key(native));
    assert_ne!(native, &NativeId::unicode("~u:eda080"));
    let call_fact = output
        .delta
        .contributions
        .iter()
        .flat_map(|(_, _, c)| &c.projection.facts)
        .find(|f| {
            matches!(
                f.field,
                polylogue_sources::claude_code::projection::Field::Block {
                    field: polylogue_model::merge::BlockField::CallId,
                    ..
                }
            )
        })
        .unwrap();
    let mut encoded = Vec::new();
    call_fact
        .selection
        .as_ref()
        .unwrap()
        .write_value(
            |range| Ok::<_, ()>(bytes[range.start as usize..range.end as usize].to_vec()),
            bytes.len(),
            &mut encoded,
        )
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<NativeId>(&encoded).unwrap(),
        *tool.call_id.as_ref().unwrap()
    );
}

#[test]
fn excessive_depth_is_owed_and_does_not_become_an_accepted_undecodable_record() {
    let mut bytes = b"{\"type\":\"user\",\"uuid\":\"u\",\"message\":{\"content\":".to_vec();
    bytes.extend(std::iter::repeat_n(b'[', 65));
    bytes.extend_from_slice(b"0");
    bytes.extend(std::iter::repeat_n(b']', 65));
    bytes.extend_from_slice(b"}}\n");
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        &bytes,
        0,
    );
    assert_eq!(framed.parsed_upto, 0);
    assert_eq!(framed.counts.framed, 0);
    let failure = framed.output.unwrap_err();
    assert_eq!(failure.phase, FailurePhase::Decoding);
    assert!(matches!(failure.error, ParseError::Decoding(_)));
}

#[test]
fn encoded_structural_key_and_non_scalar_source_session_are_explicit_refusals() {
    for bytes in [
        br#"{"type":"user","uuid":"u","sessionId":"\ud800","message":{"content":"x"}}
"#
        .as_slice(),
        br#"{"type":"user","uuid":"u","\ud800":1,"message":{"content":"x"}}
"#
        .as_slice(),
    ] {
        let memory = MemoryReadView::default();
        let framed = drive_jsonl(
            KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
            bytes,
            0,
        );
        assert!(framed.output.is_err());
        assert_eq!(framed.parsed_upto, 0);
        assert_eq!(framed.counts.framed, 0);
    }
}
