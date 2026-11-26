use super::*;
use proptest::prelude::*;

fn message_id() -> MessageId {
    MessageId::parse("claude-code:745168ba-a1-0").unwrap()
}

fn plain(key: EntryKey, kind: EntryKind) -> IndexEntry {
    IndexEntry {
        key,
        kind,
        written_by: WrittenBy::Model,
        prompt_kind: None,
        active_path: Some(true),
        branch_point: false,
        at: Some(1_788_256_800_000),
        body_bytes: 0,
        tokens_line: 0,
        tokens_full: 0,
        output_lines: 0,
        action: None,
        target: None,
        call_id: None,
        event_kind: None,
        boundary: None,
        compaction: None,
        context_tokens: None,
        cleared: Vec::new(),
    }
}

fn header() -> IndexEntry {
    IndexEntry {
        tokens_line: 20,
        tokens_full: 20,
        context_tokens: Some(1500),
        ..plain(EntryKey::header(&message_id()), EntryKind::Header)
    }
}

fn call() -> IndexEntry {
    IndexEntry {
        body_bytes: 60,
        tokens_line: 12,
        tokens_full: 27,
        action: Some(ToolAction::Shell),
        target: Some("cargo test".into()),
        call_id: Some("toolu_1".into()),
        ..plain(
            EntryKey::Message {
                message: message_id(),
                block: Some(2),
            },
            EntryKind::Call,
        )
    }
}

/// Bytes worked out by hand from the layout in the module documentation.
#[test]
fn header_and_call_store_the_documented_bytes() {
    let header = header();
    assert_eq!(
        encode(&header, None).unwrap(),
        [
            0x80, 0x59, // flags: context tokens, active path known and on it, time
            0x10, // header kind, written by the model
            0x80, 0xb4, 0xca, 0xc6, 0x8b, 0x68, // time, zigzag
            0x00, 0x14, 0x00, // body bytes, line tokens, full minus line
            0xdc, 0x0b, // context tokens
        ]
    );
    let mut expected = vec![
        0x0d, // flags: block, action, target
        0x04, // call kind
        0x02, 0x3c, 0x0c, 0x1e, // block, body bytes, line tokens, full minus line zigzag
        0x00, // shell
        0x0a, // target length
    ];
    expected.extend_from_slice(b"cargo test");
    assert_eq!(encode(&call(), Some(&header)).unwrap(), expected);
}

#[test]
fn a_block_takes_its_message_facts_from_the_header_and_its_call_id_from_the_row() {
    let header = header();
    let stored = encode(&call(), Some(&header)).unwrap();
    let decoded_header = decode(
        &encode(&header, None).unwrap(),
        EntryOf::Message(message_id()),
        None,
    )
    .unwrap();
    assert_eq!(decoded_header, header);
    let decoded = decode(
        &stored,
        EntryOf::Header(&decoded_header),
        Some("toolu_1".into()),
    )
    .unwrap();
    assert_eq!(decoded, call());
}

#[test]
fn a_block_that_differs_from_its_header_is_refused() {
    let header = header();
    for change in 0..6 {
        let mut block = call();
        match change {
            0 => block.written_by = WrittenBy::Harness,
            1 => block.prompt_kind = Some(PromptKind::Prose),
            2 => block.active_path = None,
            3 => block.branch_point = true,
            4 => block.at = None,
            _ => {
                block.key = EntryKey::Message {
                    message: MessageId::parse("claude-code:other").unwrap(),
                    block: Some(2),
                }
            }
        }
        assert!(
            matches!(
                encode(&block, Some(&header)),
                Err(StoreError::StoredValue { column: None, .. })
            ),
            "change {change}"
        );
    }
}

#[test]
fn an_event_entry_round_trips_with_its_typed_facts() {
    let mut event = plain(
        EntryKey::Event {
            event: "claude-code:event-1".into(),
        },
        EntryKind::Event,
    );
    event.written_by = WrittenBy::Harness;
    event.active_path = None;
    event.at = Some(-5);
    event.body_bytes = 300;
    event.tokens_line = 8;
    event.tokens_full = 8;
    event.event_kind = Some("compaction".into());
    event.boundary = Some(Boundary::Compaction);
    event.compaction = Some(polylogue_model::CompactionBoundary {
        pre_tokens: Some(150_000),
        trigger: Some("auto".into()),
        replacement_history: serde_json::json!([{"role": "user"}]),
        ..Default::default()
    });
    event.cleared = vec!["call-a".into(), "call-b".into()];
    let stored = encode(&event, None).unwrap();
    assert_eq!(
        decode(&stored, EntryOf::Event("claude-code:event-1".into()), None).unwrap(),
        event
    );
}

#[test]
fn an_event_entry_is_never_a_later_entry_of_a_message() {
    let event = plain(
        EntryKey::Event {
            event: "claude-code:event-1".into(),
        },
        EntryKind::Event,
    );
    assert!(matches!(
        encode(&event, Some(&header())),
        Err(StoreError::StoredValue { column: None, .. })
    ));
    // Bytes of a message's block entry carry a block number, which an event
    // row never has.
    let block_bytes = encode(&call(), None).unwrap();
    assert!(
        decode(
            &block_bytes,
            EntryOf::Event("claude-code:event-1".into()),
            None
        )
        .is_err()
    );
}

#[test]
fn trailing_or_missing_bytes_are_refused() {
    let mut stored = encode(&call(), Some(&header())).unwrap();
    stored.push(0);
    assert!(decode(&stored, EntryOf::Header(&header()), None).is_err());
    stored.truncate(stored.len() - 3);
    assert!(decode(&stored, EntryOf::Header(&header()), None).is_err());
}

fn kinds() -> impl Strategy<Value = EntryKind> {
    prop::sample::select(vec![
        EntryKind::Text,
        EntryKind::Thinking,
        EntryKind::ThinkingEncrypted,
        EntryKind::Call,
        EntryKind::Result,
        EntryKind::Attachment,
        EntryKind::Other,
    ])
}

prop_compose! {
    fn message_entries()(
        written_by in prop::sample::select(WrittenBy::ALL.to_vec()),
        prompt_kind in prop::option::of(prop::sample::select(PromptKind::ALL.to_vec())),
        active_path in prop::option::of(any::<bool>()),
        branch_point in any::<bool>(),
        at in prop::option::of(any::<i64>()),
        context_tokens in prop::option::of(any::<u64>()),
        header_tokens in any::<u64>(),
        blocks in prop::collection::vec((
            kinds(),
            prop::option::of(any::<u64>()),
            any::<u64>(),
            any::<u64>(),
            any::<u64>(),
            any::<u64>(),
            prop::option::of(prop::sample::select(ToolAction::ALL.to_vec())),
            prop::option::of(".{0,40}"),
            prop::option::of("[a-z_0-9]{1,20}"),
        ), 0..6),
    ) -> Vec<IndexEntry> {
        let id = message_id();
        let mut header = plain(EntryKey::header(&id), EntryKind::Header);
        header.written_by = written_by;
        header.prompt_kind = prompt_kind;
        header.active_path = active_path;
        header.branch_point = branch_point;
        header.at = at;
        header.tokens_line = header_tokens;
        header.tokens_full = header_tokens;
        header.context_tokens = context_tokens;
        header.boundary = (prompt_kind == Some(PromptKind::CompactionSummary))
            .then_some(Boundary::CompactionSummary);
        let mut entries = vec![header.clone()];
        for (kind, block, body, line, full, lines, action, target, call_id) in blocks {
            entries.push(IndexEntry {
                key: EntryKey::Message { message: id.clone(), block },
                kind,
                body_bytes: body,
                tokens_line: line,
                tokens_full: full,
                output_lines: lines,
                action,
                target,
                call_id,
                context_tokens: None,
                boundary: None,
                ..header.clone()
            });
        }
        entries
    }
}

proptest! {
    #[test]
    fn every_message_unit_round_trips(entries in message_entries()) {
        let (header, blocks) = entries.split_first().unwrap();
        let decoded_header = decode(
            &encode(header, None).unwrap(),
            EntryOf::Message(message_id()),
            header.call_id.clone(),
        )
        .unwrap();
        prop_assert_eq!(&decoded_header, header);
        for block in blocks {
            let stored = encode(block, Some(header)).unwrap();
            let decoded = decode(&stored, EntryOf::Header(&decoded_header), block.call_id.clone())
                .unwrap();
            prop_assert_eq!(&decoded, block);
        }
    }

}

#[test]
fn unknown_codes_are_refused() {
    // Flags 0, then a kind code past the last kind (8).
    assert!(decode(&[0, 9], EntryOf::Message(message_id()), None).is_err());
    // A valid kind (Event, 8) with an author code past the last author.
    let event = EntryOf::Event("claude-code:event-1".into());
    assert!(decode(&[0, 8 | (15 << 4)], event, None).is_err());
}
