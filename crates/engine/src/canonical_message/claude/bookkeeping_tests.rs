use super::*;
use polylogue_sources::{
    claude_code::{FileCtx, KeyedRecords, keyed::MemoryReadView},
    frame::drive_jsonl,
};
use serde_json::json;

fn response(extra: serde_json::Value) -> ResponseCell {
    let mut row = json!({"type":"assistant","uuid":"record","sessionId":"neutral","parentUuid":null,"isSidechain":false,"message":{"id":"response","role":"assistant","content":[{"type":"text","text":"neutral"}]}});
    for (key, value) in extra.as_object().unwrap() {
        row["message"][key] = value.clone();
    }
    let view = MemoryReadView::default();
    let raw = row.to_string() + "\n";
    drive_jsonl(
        KeyedRecords::new(FileCtx::main("neutral"), &view).unwrap(),
        raw.as_bytes(),
        0,
    )
    .output
    .unwrap()
    .delta
    .responses
    .pop()
    .unwrap()
    .2
}
#[test]
fn absent_slots_are_bounded_but_supplied_null_empty_and_later_facts_stop_skipping() {
    let empty = response(json!({}));
    let flag = AtomicBool::new(false);
    for old_index in 0..=8 {
        assert_eq!(selected_index(&empty, old_index, &flag).unwrap(), 8);
    }
    for supplied in [json!(null), json!(""), json!({})] {
        let r = response(json!({"stop_details":supplied}));
        assert_eq!(selected_index(&r, 0, &flag).unwrap(), 7);
        assert_eq!(selected_index(&r, 7, &flag).unwrap(), 7);
        assert_eq!(selected_index(&r, 8, &flag).unwrap(), 8);
    }
    let r = response(json!({"stop_sequence":null,"stop_details":"later"}));
    assert_eq!(selected_index(&r, 0, &flag).unwrap(), 6);
    assert_eq!(selected_index(&r, 7, &flag).unwrap(), 7);
    assert!(matches!(
        selected_index(&empty, 0, &AtomicBool::new(true)),
        Err(Error::Cancelled)
    ));
}
