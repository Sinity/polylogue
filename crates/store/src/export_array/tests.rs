use super::*;
#[test]
fn spilled_items_follow_total_time_order_and_claim_owners_once() {
    let mut scratch = ExportArray::new().unwrap();
    for (i, key) in [
        None,
        Some(2.0),
        Some(-0.0),
        Some(0.0),
        Some(-2.0),
        Some(2.0),
    ]
    .into_iter()
    .enumerate()
    {
        scratch
            .push(i as u64..i as u64 + 1, key, i.to_string().as_bytes())
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
