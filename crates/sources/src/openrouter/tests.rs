use super::*;
fn parse(bytes: &[u8]) -> crate::Step {
    crate::export::fixture::parse(&mut open(Path::new("room.json")).unwrap(), bytes).unwrap()
}
use serde_json::json;

/// The rule as a file found at a location meets it: its first bytes alone.
fn recognize(path: &std::path::Path, head: &[u8]) -> bool {
    super::recognize(path, &crate::document::Head::new(head))
}

fn room(messages: Json) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "version": "orpg.2.0", "title": "Nix help",
        "characters": {"char-1": {"id": "char-1", "model": "openrouter/horizon-beta",
                                  "description": "d", "samplingParameters": {}}},
        "messages": messages,
        "artifacts": {}, "artifactFiles": {}, "artifactVersions": {}, "artifactFileContents": {}
    }))
    .unwrap()
}

#[test]
fn a_room_becomes_one_session_named_by_its_first_message() {
    let bytes = room(json!({
        "msg-2-bbb": {"id": "msg-2-bbb", "type": "assistant", "characterId": "char-1",
                      "contentType": "text", "context": "main-chat", "parentMessageId": "msg-1-aaa",
                      "citations": [], "files": [], "createdAt": "2025-08-04T20:01:00.000Z",
                      "updatedAt": "2025-08-04T20:01:00.000Z", "isGenerating": false,
                      "isRetrying": false, "isEdited": false, "isCollapsed": false,
                      "content": "Use flakes.",
                      "metadata": {"plugins": [], "generateDuration": 41006, "latencyDuration": 5889,
                                   "tokensCount": 2394, "tokensPerSecond": 58.3,
                                   "variantSlug": "openrouter/horizon-beta", "provider": "Stealth",
                                   "cost": "0.0012"}},
        "msg-1-aaa": {"id": "msg-1-aaa", "type": "user", "characterId": "USER",
                      "contentType": "text", "context": "main-chat",
                      "attachments": [{"type": "text", "content": "my config"}],
                      "createdAt": "2025-08-04T20:00:00.000Z", "updatedAt": "2025-08-04T20:00:00.000Z",
                      "isGenerating": false, "isRetrying": false, "isEdited": false,
                      "isCollapsed": false, "content": "How do I pin nixpkgs?"}
    }));
    let step = parse(&bytes);
    let s = &step.output.sessions[0];
    assert_eq!(s.id.native().as_str(), "msg-1-aaa");
    assert_eq!(s.title.as_ref().unwrap().text, "Nix help");
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs[0].text(), "How do I pin nixpkgs?");
    match &msgs[0].blocks[1] {
        Block::File(a) => {
            assert_eq!(a.name, None);
            assert_eq!(a.media_type.as_deref(), Some("text/plain"));
            assert_eq!(a.size, Some(9));
            assert_eq!(a.blob, Some(BlobHash::of(b"my config")));
        }
        other => panic!("expected a file block, got {other:?}"),
    }
    assert_eq!(step.output.blobs[0].bytes, b"my config");
    let a = msgs[1];
    assert_eq!(a.parent.as_ref().unwrap().native().as_str(), "msg-1-aaa");
    assert_eq!(a.model.as_deref(), Some("openrouter/horizon-beta"));
    assert_eq!(a.usage.output, Some(2394));
    assert_eq!(a.duration_ms, Some(41006));
    assert_eq!(a.production.first_token_ms, Some(5889));
    assert_eq!(a.production.requested_model, None);
    assert_eq!(a.reported_cost_usd.as_ref().unwrap().to_string(), "0.0012");
    let models: Vec<_> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind == "room_models")
        .collect();
    assert_eq!(models.len(), 1);
    assert_eq!(
        models[0].data,
        json!({"char-1": {"id": "char-1", "model": "openrouter/horizon-beta",
                          "description": "d", "samplingParameters": {}}})
    );
    let expected: std::collections::BTreeMap<String, u64> = [
        "assistant.context",
        "assistant.isCollapsed",
        "assistant.isGenerating",
        "assistant.isRetrying",
        "assistant.metadata.tokensPerSecond",
        "assistant.updatedAt",
        "room.version",
        "user.context",
        "user.isCollapsed",
        "user.isGenerating",
        "user.isRetrying",
        "user.updatedAt",
    ]
    .into_iter()
    .map(|path| (path.to_owned(), 1))
    .collect();
    assert_eq!(step.counts.unread_fields, expected);
}

#[test]
fn reasoning_comes_first_and_a_different_variant_is_the_requested_model() {
    let bytes = room(json!({
        "msg-1-aaa": {"id": "msg-1-aaa", "type": "assistant", "characterId": "char-1",
                      "contentType": "text", "context": "main-chat",
                      "createdAt": "2025-08-04T20:00:00.000Z",
                      "content": "Use flakes.",
                      "metadata": {"reasoning": "pinning needs a lock file",
                                   "variantSlug": "openrouter/other-variant"}}
    }));
    let step = parse(&bytes);
    let message = &step.output.messages[0].message;
    assert!(matches!(
        &message.blocks[0],
        Block::Thinking { text: Some(text), .. } if text == "pinning needs a lock file"
    ));
    assert!(matches!(&message.blocks[1], Block::Text { text } if text == "Use flakes."));
    assert_eq!(
        message.production.requested_model.as_deref(),
        Some("openrouter/other-variant")
    );
    assert_eq!(message.model.as_deref(), Some("openrouter/horizon-beta"));
}

#[test]
fn recognizes_room_exports_by_name_and_first_key() {
    assert!(recognize(
        Path::new("/a/OpenRouter Chat Mon Aug 04 2025 (1).json"),
        br#"{"version": "orpg"#
    ));
    assert!(!recognize(
        Path::new("/a/chat.json"),
        br#"{"version": "orpg"#
    ));
    let named = Path::new("/a/OpenRouter Chat Mon Aug 04 2025 (1).json");
    assert!(!recognize(named, br#"{"title": "x"}"#));
    assert!(!recognize(named, br#"["version"]"#));
}

#[test]
fn integer_time_id_and_object_key_order_matches_optional_tuple_comparison() {
    let times = [
        None,
        Some(-9_999_999),
        Some(-1),
        Some(0),
        Some(1),
        Some(
            polylogue_model::Timestamp::parse_rfc3339("9000-01-01T00:00:00Z")
                .unwrap()
                .unix_ms(),
        ),
    ];
    let ids = [
        None,
        Some("a"),
        Some("a\0"),
        Some("a\0z"),
        Some("z"),
        Some("é"),
    ];
    let mut cases = Vec::new();
    for time in times {
        for id in ids {
            for key in ["a", "a\0", "é"] {
                cases.push((time, id, key));
            }
        }
    }
    for a in &cases {
        for b in &cases {
            let encode = |(time, id, key): &(Option<i64>, Option<&str>, &str)| {
                room_order(
                    time.map(|ms| {
                        polylogue_model::Timestamp::from_unix_ms(ms)
                            .expect("test time is representable")
                    }),
                    *id,
                    key,
                )
            };
            assert_eq!(encode(a).cmp(&encode(b)), a.cmp(b), "{a:?} / {b:?}");
        }
    }
}

#[test]
fn ordering_reads_last_header_values_without_parsing_the_message_body() {
    let raw=serde_json::value::RawValue::from_string(r#"{"id":"old","createdAt":"2026-01-02T00:00:00Z","id":"new","createdAt":"1969-12-31T23:59:59.999Z","content":{"nested":[1,2,3]}}"#.into()).unwrap();
    // -1 ms is 0xffff_ffff_ffff_ffff with the sign bit flipped.
    let expected = vec![
        1, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, b'n', b'e', b'w', 0, 0, b'k', b'e',
        b'y',
    ];
    assert_eq!(message_order(&raw, "key"), expected);
}

#[cfg(target_os = "linux")]
#[test]
fn content_parts_become_the_message_text() {
    let bytes = room(json!({
        "msg-1-aaa": {"id": "msg-1-aaa", "type": "user", "characterId": "USER",
                      "contentType": "text", "context": "main-chat",
                      "createdAt": "2025-08-04T20:00:00.000Z",
                      "content": [{"type": "text", "text": "Use"}, {"type": "text", "text": "flakes."}]}
    }));
    let step = parse(&bytes);
    assert_eq!(step.output.messages[0].message.text(), "Use\nflakes.");
}
