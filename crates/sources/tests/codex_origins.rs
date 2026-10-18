//! Exact current original-record contributions, independent of hash bytes.
use polylogue_model::{
    BlockSlot,
    merge::{BlockField, MessageField, Slot},
};
use polylogue_sources::{
    codex::{
        keyed::{Cell, Key, KeyedRecords, MemoryReadView, Resume},
        origins::{Origins, RecordRange},
        projection::{Cursor, Mode, Step, Unit},
    },
    frame::drive_jsonl,
};
use serde_json::{Value, json};
fn line(kind: &str, payload: Value) -> String {
    json!({"timestamp":"2026-10-09T14:30:00Z","type":kind,"payload":payload}).to_string() + "\n"
}
fn header() -> String {
    line(
        "session_meta",
        json!({"id":"synthetic-origins","source":"cli"}),
    )
}
fn run(
    view: MemoryReadView,
    resume: Resume,
    bytes: &str,
) -> (MemoryReadView, Resume, Vec<(usize, u64)>) {
    let framed = drive_jsonl(
        KeyedRecords::new(&view, resume.clone()),
        bytes.as_bytes(),
        resume.parsed_upto,
    );
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    let versions = delta
        .pending_versions
        .iter()
        .map(|v| (v.message, v.evidence_revision))
        .collect();
    let resume = delta.resume.clone();
    let mut view = view;
    view.apply(&delta);
    (view, resume, versions)
}
fn ranges(origins: &Origins) -> Vec<RecordRange> {
    let mut cursor = origins.cursor();
    let mut out = Vec::new();
    while let Some(r) = origins.advance(&mut cursor).unwrap() {
        out.push(r);
    }
    out
}
fn record(start: usize, line: &str) -> RecordRange {
    RecordRange {
        start: start as u64,
        end: (start + line.len() - 1) as u64,
    }
}
fn witness(view: &MemoryReadView, message: usize, block: usize, field: BlockField) -> Origins {
    match view.cells.get(&Key::PresenceBlock(message, block)).unwrap() {
        Cell::Presence(group) => group
            .fields
            .iter()
            .find(|w| {
                w.slot
                    == Slot::Block {
                        block: BlockSlot::Item(0),
                        field,
                    }
            })
            .unwrap()
            .origins
            .clone(),
        _ => panic!("shape"),
    }
}
#[test]
fn missing_field_overlay_pins_old_name_and_new_arguments_without_history_chain() {
    let h = header();
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"exec_command","arguments":"{\"cmd\":\"first\"}"}),
    );
    let change = line(
        "response_item",
        json!({"id":"a","arguments":"{\"cmd\":\"last\"}"}),
    );
    let a = record(h.len(), &call);
    let b = record(h.len() + call.len(), &change);
    let (view, _, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(h + &call + &change),
    );
    assert_eq!(ranges(&witness(&view, 0, 0, BlockField::Tool)), vec![a]);
    assert_eq!(ranges(&witness(&view, 0, 0, BlockField::Input)), vec![b]);
    assert_eq!(ranges(&witness(&view, 0, 0, BlockField::Command)), vec![b]);
    let Cell::Evidence(unit) = view.cells.get(&Key::Evidence(0, 0, 0)).unwrap() else {
        panic!("shape")
    };
    let Origins::Overlay { envelope, fields } = &unit.origins else {
        panic!("not overlay")
    };
    assert_eq!(*envelope, b);
    assert_eq!(fields["name"].record, a);
    assert_eq!(fields["arguments"].record, b);
}
#[test]
fn identical_reemission_retains_selected_origin_without_changing_revision() {
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"tool","arguments":"{}"}),
    );
    let (view, resume, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &call),
    );
    let old = view.cells.get(&Key::Evidence(0, 0, 0)).unwrap().clone();
    let head = view.cells.get(&Key::EvidenceHead(0)).unwrap().clone();
    let (view, _, versions) = run(view, resume, &call);
    assert!(versions.is_empty());
    assert_eq!(
        serde_json::to_value(&head).unwrap(),
        serde_json::to_value(view.cells.get(&Key::EvidenceHead(0)).unwrap()).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&old).unwrap(),
        serde_json::to_value(view.cells.get(&Key::Evidence(0, 0, 0)).unwrap()).unwrap()
    );
}
#[test]
fn model_context_origins_freeze_and_timestamp_origin_is_outside_digest() {
    let h = header();
    let ca = line("turn_context", json!({"model":"model-a"}));
    let a = line(
        "response_item",
        json!({"type":"message","id":"a","role":"assistant","content":[{"text":"a"}]}),
    );
    let close = line("event_msg", json!({"type":"token_count","info":null}));
    let cb = line("turn_context", json!({"model":"model-b"}));
    let b = line(
        "response_item",
        json!({"type":"message","id":"b","role":"assistant","content":[{"text":"b"}]}),
    );
    let ar = record(h.len(), &ca);
    let br = record(h.len() + ca.len() + a.len() + close.len(), &cb);
    let at = record(h.len() + ca.len(), &a);
    let bytes = h + &ca + &a + &close + &cb + &b;
    let (view, _, _) = run(MemoryReadView::default(), Resume::default(), &bytes);
    for (message, expected) in [(0, ar), (1, br)] {
        let Cell::Evidence(context) = view.cells.get(&Key::Evidence(message, 1, 2)).unwrap() else {
            panic!("shape")
        };
        assert_eq!(ranges(&context.origins), vec![expected]);
        let Cell::SlotValue(w) = view
            .cells
            .get(&Key::Presence(message, "model".into()))
            .unwrap()
        else {
            panic!("shape")
        };
        assert_eq!(ranges(&w.origins), vec![expected]);
    }
    let Cell::SlotValue(w) = view.cells.get(&Key::Presence(0, "at".into())).unwrap() else {
        panic!("shape")
    };
    assert_eq!(w.slot, Slot::Message(MessageField::At));
    assert_eq!(ranges(&w.origins), vec![at]);
    let shifted = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes.as_bytes(),
        100,
    )
    .output
    .unwrap();
    let Cell::Evidence(original) = view.cells.get(&Key::Evidence(0, 0, 0)).unwrap() else {
        panic!("shape")
    };
    let Cell::Evidence(moved) = shifted
        .cells
        .get(&Key::Evidence(0, 0, 0))
        .unwrap()
        .as_ref()
        .unwrap()
    else {
        panic!("shape")
    };
    assert_eq!(original, moved);
    assert_ne!(original.origins, moved.origins);
    assert_eq!(original.canonical, moved.canonical);
}
#[test]
fn status_overlay_preserves_old_exit_and_duration_and_replaces_status_origin() {
    let h = header();
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"exec_command","arguments":"{}"}),
    );
    let status = line(
        "event_msg",
        json!({"type":"item_completed","turn_id":"t","item":{"type":"CommandExecution","id":"c","status":"failed","exit_code":1,"duration_ms":3}}),
    );
    let output = line(
        "response_item",
        json!({"type":"function_call_output","id":"out","call_id":"c","output":"text"}),
    );
    let changed = line(
        "event_msg",
        json!({"type":"item_completed","turn_id":"t","item":{"type":"CommandExecution","id":"c","status":"completed"}}),
    );
    let old = record(h.len() + call.len(), &status);
    let new = record(h.len() + call.len() + status.len() + output.len(), &changed);
    let (view, _, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(h + &call + &status + &output + &changed),
    );
    assert_eq!(
        ranges(&witness(&view, 1, 0, BlockField::Outcome)),
        vec![new]
    );
    assert_eq!(
        ranges(&witness(&view, 1, 0, BlockField::ExitCode)),
        vec![old]
    );
    assert_eq!(
        ranges(&witness(&view, 1, 0, BlockField::DurationMs)),
        vec![old]
    );
}
#[test]
fn projection_preserves_origins_and_whole_versus_record_steps_converge() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"message","id":"u","role":"user","content":[{"text":"hello"}]}),
        );
    let (whole, _, _) = run(MemoryReadView::default(), Resume::default(), &bytes);
    let mut view = MemoryReadView::default();
    let mut resume = Resume::default();
    for record in bytes.split_inclusive('\n') {
        (view, resume, _) = run(view, resume, record);
    }
    let whole_cells: Vec<_> = whole
        .cells
        .iter()
        .map(|entry| serde_json::to_vec(&entry).unwrap())
        .collect();
    let stepped_cells: Vec<_> = view
        .cells
        .iter()
        .map(|entry| serde_json::to_vec(&entry).unwrap())
        .collect();
    assert_eq!(whole_cells, stepped_cells);
    let mut cursor = Cursor::block(0, 0, Mode::Annotated);
    let mut witnessed = false;
    loop {
        match cursor.advance(&view).unwrap() {
            Step::Unit(Unit::BlockPresence {
                group: Some(group), ..
            }) => {
                assert!(group.fields.iter().all(|w| !ranges(&w.origins).is_empty()));
                witnessed = true;
            }
            Step::Done => break,
            _ => {}
        }
    }
    assert!(witnessed);
}
#[test]
fn missing_or_corrupt_origin_cannot_become_a_completed_source_list() {
    let mut cursor = Origins::Unknown.cursor();
    assert!(Origins::Unknown.advance(&mut cursor).is_err());
    assert!(Origins::Unknown.advance(&mut cursor).is_err());
    let bad = Origins::Record(RecordRange { start: 8, end: 2 });
    let mut cursor = bad.cursor();
    assert!(bad.advance(&mut cursor).is_err());
    assert!(bad.advance(&mut cursor).is_err());
    let origin = Origins::Record(RecordRange { start: 0, end: 10 });
    let mut cursor = origin.cursor();
    assert_eq!(
        origin.advance(&mut cursor).unwrap(),
        Some(RecordRange { start: 0, end: 10 })
    );
    cursor = serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
    assert!(origin.advance(&mut cursor).unwrap().is_none());
}

#[test]
fn full_replacement_drops_old_contributors_without_mutating_retained_descriptor() {
    let h = header();
    let first = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"tool","arguments":"{}"}),
    );
    let (view, resume, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(h.clone() + &first),
    );
    let Cell::Evidence(old) = view.cells.get(&Key::Evidence(0, 0, 0)).unwrap() else {
        panic!("shape")
    };
    let retained = old.origins.clone();
    let next = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"tool","arguments":"{\"changed\":true}"}),
    );
    let expected = record(h.len() + first.len(), &next);
    let (view, _, _) = run(view, resume, &next);
    let Cell::Evidence(current) = view.cells.get(&Key::Evidence(0, 0, 0)).unwrap() else {
        panic!("shape")
    };
    assert_eq!(ranges(&current.origins), vec![expected]);
    assert_eq!(ranges(&retained), vec![record(h.len(), &first)]);
}
