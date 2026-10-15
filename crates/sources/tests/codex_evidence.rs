//! Raw source evidence and unknown ownership, independent of writer finalization.
use polylogue_model::{BlockContent, BlockSlot, ToolOutcome};
use polylogue_sources::{
    codex::keyed::{Cell, EvidenceCursor, EvidenceHead, Key, KeyedRecords, MemoryReadView, Resume},
    frame::drive_jsonl,
};
use serde_json::{Value, json};
fn line(kind: &str, payload: Value) -> String {
    json!({"type":kind,"payload":payload}).to_string() + "\n"
}
fn header() -> String {
    line(
        "session_meta",
        json!({"id":"session:opaque","source":"cli"}),
    )
}
fn run(view: MemoryReadView, resume: Resume, bytes: &str) -> (MemoryReadView, Resume) {
    let start = resume.parsed_upto;
    let framed = drive_jsonl(KeyedRecords::new(&view, resume), bytes.as_bytes(), start);
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    let mut view = view;
    view.apply(&delta);
    (view, delta.resume)
}
fn head(view: &MemoryReadView, message: usize) -> EvidenceHead {
    match view.cells.get(&Key::EvidenceHead(message)).unwrap() {
        Cell::EvidenceHead(head) => head.clone(),
        _ => panic!("shape"),
    }
}
fn units(view: &MemoryReadView, message: usize) -> Vec<(String, Vec<u8>)> {
    let head = head(view, message);
    let mut cursor = EvidenceCursor::default();
    let mut out = Vec::new();
    while let Some(key) = head.next_slot(message, &mut cursor) {
        if let Some(Cell::Evidence(unit)) = view.cells.get(&key) {
            out.push((unit.tag.clone(), unit.canonical.clone()));
        }
    }
    assert_eq!(out.len() as u64, head.units);
    out
}
#[test]
fn reasoning_reemission_replaces_one_slot_and_reanchor_keeps_reasoning_identity() {
    let r = line(
        "response_item",
        json!({"type":"reasoning","id":"r:1","summary":[{"text":"first"}]}),
    );
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &r),
    );
    let original = head(&view, 0);
    let original_units = units(&view, 0);
    let (view, resume) = run(view, resume, &r);
    assert_eq!(head(&view, 0), original);
    assert_eq!(units(&view, 0), original_units);
    let replacement = line(
        "response_item",
        json!({"type":"reasoning","id":"r:1","summary":[{"text":"longer"}]}),
    );
    let (view, resume) = run(view, resume, &replacement);
    assert_eq!(head(&view, 0).items, 1);
    assert!(head(&view, 0).revision > original.revision);
    let anchor = line(
        "response_item",
        json!({"type":"function_call","id":"a:1","call_id":"c:1","name":"exec","arguments":"{}"}),
    );
    let (view, resume) = run(view, resume, &anchor);
    let message = view.message(0).unwrap().unwrap();
    assert_eq!(message.id.session().as_str(), "codex:session:opaque");
    assert_eq!(message.id.as_str(), "codex:session:opaque:a:1");
    assert!(matches!(
        message.blocks[0].id.slot(),
        BlockSlot::Reasoning(0)
    ));
    assert!(matches!(message.blocks[1].id.slot(), BlockSlot::Item(0)));
    assert!(message.version_hash.is_none());
    assert_eq!(resume.message_count(), 1);
    assert_eq!(head(&view, 0).items, 2);
}
#[test]
fn repeat_usage_null_close_and_rate_limit_do_not_change_effective_membership() {
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"tool","arguments":"{}"}),
    );
    let info =
        json!({"total_token_usage":{"input_tokens":2},"last_token_usage":{"input_tokens":2}});
    let usage = line("event_msg", json!({"type":"token_count","info":info}));
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &call + &usage),
    );
    let before = units(&view, 0);
    let before_head = head(&view, 0);
    let repeated = line(
        "event_msg",
        json!({"type":"token_count","info":info,"rate_limits":{"remaining":5}}),
    );
    let null = line("event_msg", json!({"type":"token_count","info":null}));
    let (view, _) = run(view, resume, &(repeated + &null));
    assert_eq!(units(&view, 0), before);
    assert_eq!(head(&view, 0), before_head);
    assert_eq!(before.iter().filter(|(tag, _)| tag == "usage").count(), 1);
}
#[test]
fn unowned_child_never_enters_latest_wrapper_message() {
    let w1 = line(
        "response_item",
        json!({"type":"function_call","id":"a1","call_id":"w1","name":"exec","arguments":"{}"}),
    );
    let close = line("event_msg", json!({"type":"token_count","info":null}));
    let w2 = line(
        "response_item",
        json!({"type":"function_call","id":"a2","call_id":"w2","name":"exec","arguments":"{}"}),
    );
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &w1 + &close + &w2),
    );
    let before = [units(&view, 0), units(&view, 1)];
    let child = line(
        "event_msg",
        json!({"type":"item_completed","item":{"type":"CommandExecution","id":"child","source":"agent","status":"failed","exit_code":1,"aggregated_output":""}}),
    );
    let framed = drive_jsonl(KeyedRecords::new(&view, resume), child.as_bytes(), 0);
    let delta = framed.output.unwrap();
    assert!(delta.resume.normalization_incomplete());
    assert_eq!(delta.events[0].kind, "unowned_tool_item");
    assert!(delta.pending_versions.is_empty());
    let mut after = view;
    after.apply(&delta);
    assert_eq!([units(&after, 0), units(&after, 1)], before);
    for i in 0..2 {
        assert_eq!(after.message(i).unwrap().unwrap().blocks.len(), 1);
    }
}
#[test]
fn exact_numeric_fallback_and_structured_error_are_not_normalized_away() {
    let a = header()
        + "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"text\":\"hello\"}],\"n\":1E2}}\n";
    let b = a.replace("1E2", "1e2");
    let (a, _) = run(MemoryReadView::default(), Resume::default(), &a);
    let (b, _) = run(MemoryReadView::default(), Resume::default(), &b);
    assert_ne!(
        a.message(0).unwrap().unwrap().id,
        b.message(0).unwrap().unwrap().id
    );
    assert!(units(&a, 0)[0].1.windows(3).any(|s| s == b"1E2"));
    let output = line(
        "response_item",
        json!({"type":"function_call_output","call_id":"unpaired","output":{"is_error":true,"output":""}}),
    );
    let (view, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &output),
    );
    let message = view.message(0).unwrap().unwrap();
    assert!(
        matches!(&message.blocks[0].content,BlockContent::ToolResult(result) if matches!(result.outcome,ToolOutcome::Error{..}))
    );
}
