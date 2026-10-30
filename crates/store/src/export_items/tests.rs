use super::*;

#[test]
fn retained_items_reopen_payload_and_order_metadata_after_disconnect() {
    let directory = tempfile::tempdir().unwrap();
    {
        let mut items = ExportItems::in_directory(directory.path()).unwrap();
        items
            .push_ordered(10..20, vec![2], vec![1], Some("second"), b"second bytes")
            .unwrap();
        items
            .push_ordered(30..40, vec![1], vec![2], Some("first"), b"first bytes")
            .unwrap();
    }
    let payload_size = fs::metadata(directory.path().join("payload.zst"))
        .unwrap()
        .len();
    assert!(payload_size > 0);
    let mut items = ExportItems::in_directory(directory.path()).unwrap();
    let mut bytes = String::new();
    items
        .next_item()
        .unwrap()
        .unwrap()
        .reader
        .read_to_string(&mut bytes)
        .unwrap();
    assert_eq!(bytes, "first bytes");
    items.reset(Order::Alternate).unwrap();
    bytes.clear();
    items
        .next_item()
        .unwrap()
        .unwrap()
        .reader
        .read_to_string(&mut bytes)
        .unwrap();
    assert_eq!(bytes, "second bytes");
    items
        .push(50..60, vec![3], Some("third"), b"third bytes")
        .unwrap();
    items.close().unwrap();
    assert!(
        fs::metadata(directory.path().join("payload.zst"))
            .unwrap()
            .len()
            > payload_size
    );
    let items = ExportItems::in_directory(directory.path()).unwrap();
    bytes.clear();
    items
        .source_item("third")
        .unwrap()
        .unwrap()
        .reader
        .read_to_string(&mut bytes)
        .unwrap();
    assert_eq!(bytes, "third bytes");
}

#[test]
fn retained_drop_discards_unaccepted_scan_group_and_preserves_prior_work() {
    let directory = tempfile::tempdir().unwrap();
    {
        let mut items = ExportItems::in_directory(directory.path()).unwrap();
        items
            .push(0..1, vec![0], Some("kept"), b"accepted")
            .unwrap();
        items.begin_scan().unwrap();
        items
            .push(1..2, vec![1], Some("rejected"), b"unaccepted")
            .unwrap();
    }
    let items = ExportItems::in_directory(directory.path()).unwrap();
    assert!(items.source_item("kept").unwrap().is_some());
    assert!(items.source_item("rejected").unwrap().is_none());
}
#[test]
fn spilled_items_follow_byte_keys_and_claim_owners_once() {
    let mut scratch = ExportItems::new().unwrap();
    for (i, key) in [9, 6, 2, 3, 0, 6].into_iter().enumerate() {
        scratch
            .push(
                i as u64..i as u64 + 1,
                vec![key],
                None,
                i.to_string().as_bytes(),
            )
            .unwrap();
    }
    let mut order = Vec::new();
    while let Some(item) = scratch.next_item().unwrap() {
        let mut text = String::new();
        item.reader.take(100).read_to_string(&mut text).unwrap();
        order.push(text);
    }
    assert_eq!(order, ["4", "2", "3", "1", "5", "0"]);
    let parent = SessionId::parse("chatgpt:parent").unwrap();
    let child = SessionId::parse("chatgpt:child").unwrap();
    assert_eq!(scratch.claim_owner("message", &parent).unwrap(), None);
    assert_eq!(
        scratch.claim_owner("message", &child).unwrap(),
        Some(parent)
    );
}

#[test]
fn a_keyed_entry_replaces_its_old_sort_position_bytes_and_source_range() {
    let mut scratch = ExportItems::new().unwrap();
    scratch.push(0..3, vec![0], Some("a\0"), b"old").unwrap();
    scratch.push(10..14, vec![1], None, b"kept").unwrap();
    scratch.push(20..23, vec![2], Some("a\0"), b"new").unwrap();
    for (range, expected) in [(10..14, "kept"), (20..23, "new")] {
        let item = scratch.next_item().unwrap().unwrap();
        assert_eq!(item.range, range);
        let mut value = String::new();
        item.reader.take(100).read_to_string(&mut value).unwrap();
        assert_eq!(value, expected);
    }
    assert!(scratch.next_item().unwrap().is_none());
}

#[test]
fn rejected_scan_groups_restore_prior_values_ranges_and_order() {
    let mut items = ExportItems::new().unwrap();
    items.push(1..4, vec![1], Some("a"), b"old").unwrap();
    items.begin_scan().unwrap();
    items.push(5..8, vec![0], Some("a"), b"bad").unwrap();
    items.push(10..13, vec![2], None, b"bad").unwrap();
    items.rollback_scan().unwrap();
    items.begin_scan().unwrap();
    items.push(20..24, vec![2], None, b"kept").unwrap();
    items.commit_scan().unwrap();
    for (range, expected) in [(1..4, "old"), (20..24, "kept")] {
        let item = items.next_item().unwrap().unwrap();
        assert_eq!(item.range, range);
        let mut value = String::new();
        item.reader.take(100).read_to_string(&mut value).unwrap();
        assert_eq!(value, expected);
    }
    assert!(items.next_item().unwrap().is_none());
}

#[test]
fn previous_messages_are_independent_for_interleaved_conversations() {
    let mut items = ExportItems::new().unwrap();
    let first = SessionId::parse("gemini-app:first").unwrap();
    let second = SessionId::parse("gemini-app:second").unwrap();
    let a = MessageId::parse("gemini-app:first#1-response").unwrap();
    let b = MessageId::parse("gemini-app:second#2-response").unwrap();
    let c = MessageId::parse("gemini-app:first#3-response").unwrap();
    assert_eq!(items.last_message(&first).unwrap(), None);
    items.set_last_message(&first, &a).unwrap();
    assert_eq!(items.last_message(&second).unwrap(), None);
    items.set_last_message(&second, &b).unwrap();
    assert_eq!(items.last_message(&first).unwrap(), Some(a));
    items.set_last_message(&first, &c).unwrap();
    assert_eq!(items.last_message(&second).unwrap(), Some(b));
    assert_eq!(items.last_message(&first).unwrap(), Some(c));
    assert_eq!(
        ExportItems::new().unwrap().last_message(&first).unwrap(),
        None
    );
}

#[test]
fn staged_payloads_share_bytes_across_restartable_orders_and_independent_readers() {
    let mut items = ExportItems::new().unwrap();
    items
        .push_ordered(10..11, vec![2], vec![1], Some("second"), b"second")
        .unwrap();
    items
        .push_ordered(20..21, vec![1], vec![2], Some("first"), b"first")
        .unwrap();
    let mut a = items.next_item().unwrap().unwrap();
    let mut b = items.next_item().unwrap().unwrap();
    let mut text = String::new();
    b.reader.read_to_string(&mut text).unwrap();
    assert_eq!(text, "second");
    text.clear();
    a.reader.read_to_string(&mut text).unwrap();
    assert_eq!(text, "first");
    items.reset(Order::Alternate).unwrap();
    let mut first = items.next_item().unwrap().unwrap();
    text.clear();
    first.reader.read_to_string(&mut text).unwrap();
    assert_eq!(text, "second");
    let mut found = items.source_item("first").unwrap().unwrap();
    text.clear();
    found.reader.read_to_string(&mut text).unwrap();
    assert_eq!(text, "first");
    items
        .push_ordered(30..31, vec![3], vec![3], Some("third"), b"third")
        .unwrap();
    items.reset(Order::Primary).unwrap();
    let mut got = Vec::new();
    while let Some(mut item) = items.next_item().unwrap() {
        let mut value = String::new();
        item.reader.read_to_string(&mut value).unwrap();
        got.push(value);
    }
    assert_eq!(got, vec!["first", "second", "third"]);
}
