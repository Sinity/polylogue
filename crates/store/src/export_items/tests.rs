use super::*;
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
