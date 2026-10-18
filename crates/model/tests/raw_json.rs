use polylogue_model::*;

#[test]
fn references_match_the_spec_corpus() {
    // Expected strings come from the spec's native-identity corpus, not from
    // the framing code under test.
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("native_identity_goldens.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let session = SessionId::new(
            &Origin::new("claude-code").unwrap(),
            case["session_native"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(session.as_str(), case["session_ref"].as_str().unwrap());
        let units: Vec<u16> = case["message_native_utf16"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as u16)
            .collect();
        let message =
            MessageId::from_native(session.clone(), &NativeId::from_utf16(&units)).unwrap();
        assert_eq!(message.as_str(), case["message_ref"].as_str().unwrap());
        assert_eq!(message.session(), &session);
    }
}

#[test]
fn block_decode_preserves_opaque_units_numbers_and_call_identity() {
    let id = BlockId::new(
        MessageId::new(
            SessionId::new(&Origin::new("claude-code").unwrap(), "test").unwrap(),
            "m",
        )
        .unwrap(),
        0,
    );
    let id_json = serde_json::to_string(&id).unwrap();
    let body = r#""kind":"tool_call","call_id":{"value":"~u:eda080","encoding":"wtf8"},"input":{"\ud800":"\udfff","n":1e+02,"large":9007199254740993}"#;
    let wire = format!("{{\"id\":{id_json},{body}}}");
    let block: Block = serde_json::from_str(&wire).unwrap();
    assert_eq!(block.id, id);
    let BlockContent::ToolCall(call) = &block.content else {
        panic!()
    };
    assert_eq!(call.call_id.as_ref().unwrap().utf16(), [0xd800]);
    assert_eq!(
        call.input.as_ref().unwrap().canonical(),
        br#"{"large":9007199254740993,"n":1e+02,"\ud800":"\udfff"}"#
    );
    let serialized = serde_json::to_vec(&block).unwrap();
    let restored: Block = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(restored, block);
}

#[test]
fn block_content_direct_decode_preserves_opaque_null_and_token_spelling() {
    for raw in [
        r#"{"kind":"other","type":"provider","data":null}"#,
        r#"{"kind":"other","type":"provider","data":{"n":1.00,"s":"\ud800"}}"#,
    ] {
        let body: BlockContent = serde_json::from_str(raw).unwrap();
        let serialized = serde_json::to_vec(&body).unwrap();
        let restored: BlockContent = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(restored, body);
        assert_eq!(serialized, raw.as_bytes());
    }
}

#[test]
fn declared_operation_ids_keep_lone_surrogates_apart_from_literal_escapes() {
    let session = SessionId::new(&Origin::new("claude-code").unwrap(), "calls").unwrap();
    let lone = NativeId::from_utf16(&[0xd800]);
    let literal = NativeId::unicode("~u:eda080");
    let replacement = NativeId::unicode("\u{fffd}");
    let identities: std::collections::HashSet<_> = [&lone, &literal, &replacement]
        .into_iter()
        .map(|id| DeclaredOperationId::from_native(&session, id).unwrap())
        .collect();
    assert_eq!(identities.len(), 3);
    let mut independent = b"polylogue-declared-operation-v1\0".to_vec();
    independent.extend_from_slice(&17u64.to_le_bytes());
    independent.extend_from_slice(b"claude-code:calls");
    independent.extend_from_slice(&3u64.to_le_bytes());
    independent.extend_from_slice(&[0xed, 0xa0, 0x80]);
    assert_eq!(
        DeclaredOperationId::from_native(&session, &lone)
            .unwrap()
            .as_str(),
        format!("declared-v1:{}", blake3::hash(&independent).to_hex())
    );
}

#[test]
fn exact_block_addresses_preserve_independent_message_references() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("native_identity_goldens.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let session = SessionId::new(
            &Origin::new("claude-code").unwrap(),
            case["session_native"].as_str().unwrap(),
        )
        .unwrap();
        let units: Vec<u16> = case["message_native_utf16"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as u16)
            .collect();
        let message = MessageId::from_native(session, &NativeId::from_utf16(&units)).unwrap();
        for index in [0, 7, u32::MAX] {
            let block = BlockId::new(message.clone(), index);
            let (address, slot) = BlockId::address(block.as_str()).unwrap();
            assert_eq!(address, case["message_ref"].as_str().unwrap());
            assert_eq!(slot, BlockSlot::Item(index));
            let reference = format!("block:{}", block.as_str());
            let (address, slot) = authored::target_address(&reference).unwrap();
            assert_eq!(address, message.as_str());
            assert_eq!(slot, Some(BlockSlot::Item(index)));
        }
    }
    assert_eq!(
        BlockId::address("claude-code:s:m:r7").unwrap(),
        ("claude-code:s:m".into(), BlockSlot::Reasoning(7))
    );
}

#[test]
fn fallback_blocks_resolve_the_full_digest_message_reference() {
    let session = SessionId::new(&Origin::new("claude-code").unwrap(), "fallback").unwrap();
    let raw = br#"{"body":"retained"}"#;
    let digest = polylogue_model::identity::fallback_hash(raw, &[]).unwrap();
    assert_eq!(digest.len(), 64);
    for occurrence in [0, 1, u64::MAX] {
        let message = polylogue_model::identity::provisional_message_id(
            session.clone(),
            raw,
            &[],
            occurrence,
        )
        .unwrap();
        let block = BlockId::new(message.clone(), 9);
        assert_eq!(
            BlockId::address(block.as_str()).unwrap(),
            (message.as_str().into(), BlockSlot::Item(9))
        );
    }
}

#[test]
fn malformed_block_references_are_refused_not_split_on_colons() {
    let session = SessionId::new(&Origin::new("claude-code").unwrap(), "a:b").unwrap();
    let message = MessageId::new(session, "c").unwrap();
    let block = BlockId::new(message.clone(), 0);
    let valid = block.as_str();
    assert!(valid.starts_with("~i1:"));
    let changed_count = valid.replacen("~i1:010502", "~i1:010503", 1);
    let changed_tag = valid.replacen("~i1:0105", "~i1:0106", 1);
    let trailing = format!("{valid}00");
    let truncated = &valid[..valid.len() - 2];
    let uppercase = format!("~i1:{}", valid.strip_prefix("~i1:").unwrap().to_uppercase());
    for bad in [
        changed_count.as_str(),
        changed_tag.as_str(),
        trailing.as_str(),
        truncated,
        uppercase.as_str(),
        "s:m:00",
        "s:m:+1",
        "s:m:r01",
        "s:m:4294967296",
        "s:m:attachment",
        "~i1:zz:0",
    ] {
        assert!(BlockId::address(bad).is_none(), "accepted {bad}");
        assert!(authored::target_address(&format!("block:{bad}")).is_err());
    }
    // Reasoning and attachment block references are not supported.
    assert!(BlockId::address(BlockId::reasoning(message.clone(), 1).as_str()).is_none());
    assert!(BlockId::address(BlockId::attachment(message, "asset").unwrap().as_str()).is_none());
}
