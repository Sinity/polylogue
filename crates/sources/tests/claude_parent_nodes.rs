//! Claude parent nodes for non-Message records.
use polylogue_sources::{
    claude_code::{
        FileCtx, KeyedRecords,
        keyed::{MemoryReadView, ReadView},
    },
    frame::drive_jsonl,
};
fn key() -> String {
    "\"event\"".into()
}
fn step(
    memory: &mut MemoryReadView,
    raw: &str,
    base: u64,
) -> polylogue_sources::claude_code::parent::Node {
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &*memory).unwrap(),
        raw.as_bytes(),
        base,
    );
    let out = framed.output.unwrap();
    memory.apply(out.header, out.delta);
    memory.parent_node("s", &key()).unwrap().unwrap()
}
#[test]
fn supplied_link_corrections_omission_and_identical_replay_have_deterministic_origins() {
    let a = "{\"type\":\"attachment\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"a\",\"attachment\":{\"type\":\"synthetic\"}}\n";
    let b =
        "{\"type\":\"attachment\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"b\"}\n";
    let omitted = "{\"type\":\"attachment\",\"uuid\":\"event\",\"sessionId\":\"s\"}\n";
    let changed = "{\"type\":\"attachment\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"b\",\"attachment\":null}\n";
    let mut m = MemoryReadView::default();
    let first = step(&mut m, a, 0);
    let second = step(&mut m, b, a.len() as u64);
    assert_eq!(second.parent.as_ref().unwrap().value.as_str(), Some("b"));
    let retained = step(&mut m, omitted, (a.len() + b.len()) as u64);
    assert_eq!(retained.parent, second.parent);
    assert_eq!(retained.fingerprint, second.fingerprint);
    let same = step(&mut m, b, (a.len() + b.len() + omitted.len()) as u64);
    assert_eq!(
        same, retained,
        "effective duplicate does not invent a supplier"
    );
    let selected = step(
        &mut m,
        changed,
        (a.len() + b.len() + omitted.len() + b.len()) as u64,
    );
    assert_ne!(
        selected.parent.as_ref().unwrap().origin,
        second.parent.as_ref().unwrap().origin
    );
    assert_ne!(selected.fingerprint, first.fingerprint);
    let restored: polylogue_sources::claude_code::parent::Node =
        serde_json::from_slice(&serde_json::to_vec(&selected).unwrap()).unwrap();
    assert_eq!(selected, restored);
    let mut whole = MemoryReadView::default();
    assert_eq!(
        step(&mut whole, &[a, b, omitted, b, changed].concat(), 0),
        selected
    );
}
#[test]
fn omitted_system_subtype_keeps_its_origin_and_unknown_correction_marks_conflict() {
    let a = "{\"type\":\"system\",\"subtype\":\"compact_boundary\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":null,\"logicalParentUuid\":\"a\"}\n";
    let omitted = "{\"type\":\"system\",\"uuid\":\"event\",\"sessionId\":\"s\",\"logicalParentUuid\":\"b\"}\n";
    let mut m = MemoryReadView::default();
    let first = step(&mut m, a, 0);
    let second = step(&mut m, omitted, a.len() as u64);
    assert_eq!(first.subtype_origin, second.subtype_origin);
    assert_eq!(
        second.predecessor().unwrap().unwrap().value.as_str(),
        Some("b")
    );
    let unknown = "{\"type\":\"future_record\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"other\"}\n";
    let poisoned = step(&mut m, unknown, (a.len() + omitted.len()) as u64);
    assert!(poisoned.conflicted);
    assert!(poisoned.predecessor().is_err());
}
#[test]
fn progress_supplement_keeps_supplied_kind_and_link_origins() {
    let a = "{\"type\":\"progress\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"a\",\"data\":{\"type\":\"mcp_progress\",\"payload\":{\"kept_only_in_source\":true}}}\n";
    let b = "{\"type\":\"progress\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"b\"}\n";
    let mut m = MemoryReadView::default();
    let first = step(&mut m, a, 0);
    let changed = step(&mut m, b, a.len() as u64);
    assert_eq!(first.progress_kind, changed.progress_kind);
    assert!(!changed.conflicted);
    assert_eq!(changed.parent.as_ref().unwrap().value.as_str(), Some("b"));
    assert!(changed.verify(5, a.trim_end().as_bytes()).unwrap());
}
#[test]
fn event_data_is_not_progress_classification_and_literal_metadata_does_not_change_record_kind() {
    let raw = "{\"type\":\"attachment\",\"uuid\":\"event\",\"sessionId\":\"s\",\"parentUuid\":\"a\",\"data\":{\"opaque\":null},\"subtype\":\"opaque subtype\",\"isCompactSummary\":false}\n";
    let mut m = MemoryReadView::default();
    let node = step(&mut m, raw, 0);
    assert!(node.progress_kind.is_none());
    assert!(node.summary.is_none());
    assert!(node.subtype.is_none());
    assert_eq!(
        node.predecessor().unwrap().unwrap().value.as_str(),
        Some("a")
    );
    assert!(node.verify(0, raw.trim_end().as_bytes()).unwrap());
}
