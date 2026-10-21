use super::*;
fn parse(bytes: &[u8]) -> crate::Step {
    crate::export::fixture::parse(&mut open(Path::new("room.json")).unwrap(), bytes).unwrap()
}
use serde_json::json;

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
    assert!(matches!(&msgs[0].blocks[1], Block::File(a) if a.blob.is_some()));
    assert_eq!(step.output.blobs[0].bytes, b"my config");
    let a = msgs[1];
    assert_eq!(a.parent.as_ref().unwrap().native().as_str(), "msg-1-aaa");
    assert_eq!(a.model.as_deref(), Some("openrouter/horizon-beta"));
    assert_eq!(a.usage.output, Some(2394));
    assert_eq!(a.duration_ms, Some(41006));
    assert_eq!(a.production.first_token_ms, Some(5889));
    assert_eq!(a.reported_cost_usd.as_ref().unwrap().to_string(), "0.0012");
    assert!(step.output.events.iter().any(|e| e.kind == "room_models"));
    assert!(
        step.accounting.unread_fields.is_empty(),
        "{:?}",
        step.accounting.unread_fields
    );
    assert_eq!(step.accounting.framed, 3);
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
fn ordering_reads_last_header_values_without_interpreting_the_message_body() {
    let raw=serde_json::value::RawValue::from_string(r#"{"id":"old","createdAt":"2026-01-02T00:00:00Z","id":"new","createdAt":"1969-12-31T23:59:59.999Z","content":{"nested":[1,2,3]}}"#.into()).unwrap();
    assert_eq!(
        message_order(&raw, "key"),
        room_order(
            polylogue_model::Timestamp::from_unix_ms(-1),
            Some("new"),
            "key"
        )
    );
}

#[cfg(target_os = "linux")]
#[test]
fn many_room_entries_scan_with_one_message_buffer() {
    use std::io::Read;
    struct Room {
        part: std::io::Cursor<Vec<u8>>,
        next: usize,
    }
    impl Read for Room {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            loop {
                let count = self.part.read(out)?;
                if count > 0 {
                    return Ok(count);
                }
                let bytes = match self.next {
                    0 => br#"{"version":"example","messages":{"#.to_vec(),
                    1..=12_000 => format!(
                        "{}\"{}\":{{\"id\":\"{}\",\"type\":\"user\",\"content\":\"{}\"}}",
                        if self.next == 1 { "" } else { "," },
                        self.next,
                        self.next,
                        "x".repeat(2048)
                    )
                    .into_bytes(),
                    12_001 => b"}}".to_vec(),
                    _ => return Ok(0),
                };
                self.next += 1;
                self.part = std::io::Cursor::new(bytes);
            }
        }
    }
    fn peak() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let before = peak();
    let mut parser = OpenRouter::default();
    let mut count = 0;
    let stats = parser
        .scan(
            &mut Room {
                part: std::io::Cursor::new(vec![]),
                next: 0,
            },
            &mut |_, _, _, _| {
                count += 1;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(count, 12_000);
    assert_eq!(stats.elements, 12_000);
    assert!(stats.peak_element_bytes < 4500, "{stats:?}");
    let growth = peak().saturating_sub(before);
    eprintln!(
        "12,000 room entries: scan high-water growth {growth}KiB, record buffer {}bytes",
        stats.peak_element_bytes
    );
    assert!(growth < 8 * 1024, "room scan grew by {growth}KiB");
}
