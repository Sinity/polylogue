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
    assert_eq!(scratch.claim_message("message", &parent).unwrap(), parent);
    assert_eq!(scratch.claim_message("message", &child).unwrap(), parent);
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
