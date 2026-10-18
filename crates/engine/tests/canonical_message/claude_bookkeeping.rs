use super::*;
use cc::keyed::{ReadView as _, ResponseCell};

fn fixture(details: Option<Value>) -> Vec<u8> {
    let mut rows = Vec::new();
    for (uuid, text) in [("first", "first text"), ("later", "later text")] {
        let mut row = json!({"type":"assistant","uuid":uuid,"sessionId":SID,"requestId":"request","parentUuid":null,"isSidechain":false,"message":{"id":"response","role":"assistant","content":[{"type":"text","text":text}]}});
        if uuid == "later" {
            if let Some(value) = &details {
                row["message"]["stop_details"] = value.clone();
            }
        }
        rows.push(row.to_string() + "\n");
    }
    rows.concat().into_bytes()
}
fn absent(r: &ResponseCell, index: usize) -> bool {
    match index {
        0..6 => {
            r.selected[index].is_none()
                && r.selected_origins[index].is_none()
                && r.selected_from[index].is_none()
                && r.selected_facts[index].is_empty()
        }
        6 => r.stop_sequence.is_none(),
        7 => r.stop_details.is_none(),
        _ => false,
    }
}
// Replay only the replaced old no-output step through the real typed cursor
// mutation. All source/field/proof work uses the current materializer in both.
fn complete(
    w: &mut World,
    h: &m::Handle,
    response: &ResponseCell,
    legacy: bool,
    chunk: usize,
) -> (SourceProjection, usize, u64) {
    for steps in 1..5000 {
        let projection = w.writer.message_projection(h.build(), 65536).unwrap();
        if legacy {
            if let Some(raw) = projection.cursor.as_ref() {
                let mut cursor: Value = serde_json::from_slice(raw).unwrap();
                let stage = &mut cursor["stage"]["Claude"];
                for name in ["SelectedCitations", "Selected"] {
                    if let Some(index) = stage.get(name).and_then(|v| v["index"].as_u64()) {
                        if absent(response, index as usize) {
                            stage[name]["index"] = json!(index + 1);
                            if name == "Selected" {
                                stage[name]["fact"] = json!(0);
                            }
                            let next = serde_json::to_vec(&cursor).unwrap();
                            w.writer
                                .advance_message_projection(
                                    h.build(),
                                    &ProjectionStep {
                                        expected_sequence: projection.sequence,
                                        next_cursor: &next,
                                    },
                                    65536,
                                )
                                .unwrap();
                            break;
                        }
                    }
                }
                if w.writer
                    .message_projection(h.build(), 65536)
                    .unwrap()
                    .sequence
                    != projection.sequence
                {
                    continue;
                }
            }
        }
        let mut b = work_budget();
        b.value_chunk = chunk;
        let a = w.compute_with(h, b).unwrap();
        if let m::Applied::Source(s) = m::apply(&mut w.writer, a).unwrap() {
            return (s, steps, projection.sequence);
        }
    }
    panic!("finite fixture did not finish")
}
fn complete_value(view: &MessageView, slot: &Slot) -> (Value, bool, Value) {
    let FieldRead::Value(field) = view.field(slot, 65536).unwrap().unwrap() else {
        panic!("conflict")
    };
    let mut all = [Vec::new(), Vec::new()];
    for (origin, bytes) in all.iter_mut().enumerate() {
        let mut offset = 0;
        loop {
            let range = if origin == 0 {
                field.range(offset, 65536).unwrap()
            } else {
                field.origin_range(offset, 65536).unwrap().unwrap()
            };
            bytes.extend(range.bytes);
            if range.eof {
                break;
            }
            assert!(range.next_offset > offset);
            offset = range.next_offset;
        }
    }
    (
        serde_json::from_slice(&all[0]).unwrap(),
        field.supplied,
        serde_json::from_slice(&all[1]).unwrap(),
    )
}
fn source_ranges(v: &MessageView) -> Vec<(u64, u64)> {
    use polylogue_store::contributions::SourceStep;
    let mut after = None;
    let mut ranges = Vec::new();
    loop {
        match v.next_source(after.as_ref(), 65536).unwrap() {
            SourceStep::Span { cursor, source } => {
                ranges.push((source.start, source.end));
                after = Some(cursor);
            }
            SourceStep::Continue { cursor } => after = Some(cursor),
            SourceStep::End => return ranges,
        }
    }
}
#[test]
fn bookkeeping_removes_committed_steps_with_exact_values_origins_citations_and_versions() {
    for (details, chunk) in [
        (None, 65536),
        (Some(Value::Null), 65536),
        (Some(json!("")), 65536),
        (Some(json!({})), 65536),
        (Some(Value::Null), 64),
        (Some(json!("λ".repeat(1024))), 64),
    ] {
        let raw = fixture(details.clone());
        let mut results = Vec::new();
        let mut turns = Vec::new();
        for legacy in [false, true] {
            let (mut w, p) = make_world(&raw, &raw);
            let pending = p.last().unwrap();
            let snapshot = w.writer.parser_snapshot(&w.head.token).unwrap();
            let view = ca::View::new(&snapshot, budget()).unwrap();
            let response = view
                .response(&pending.session_native, &pending.response)
                .unwrap()
                .unwrap();
            drop(view);
            drop(snapshot);
            let req = request(&w, pending);
            let h = begin(&mut w, &req);
            let (s, n, committed) = complete(&mut w, &h, &response, legacy, chunk);
            turns.push((n, committed));
            let view = selected(&mut w, &s);
            let mut text = Vec::new();
            let mut index = 0;
            while let Some(block) = view.block_at(index).unwrap() {
                text.push(block.text_range(0, 65536).unwrap().unwrap().bytes);
                index += 1;
            }
            let slot = Slot::Extra {
                block: None,
                key: "stop_details".into(),
            };
            let metadata = if details.is_some() {
                Some(complete_value(&view, &slot))
            } else {
                assert!(view.field(&slot, 65536).unwrap().is_none());
                None
            };
            if let Some((actual, supplied, origin)) = &metadata {
                assert_eq!(Some(actual), details.as_ref());
                assert!(*supplied);
                assert_eq!(
                    origin["selection"]["inputs"][0]["path"],
                    json!([{"Key":"message"},{"Key":"stop_details"}])
                );
            }
            results.push((s.version_hash(), source_ranges(&view), text, metadata));
        }
        assert_eq!(results[0], results[1]);
        assert!(turns[0].0 < turns[1].0, "{turns:?}");
        assert!(turns[1].1 - turns[0].1 >= 8, "{turns:?}");
        assert_eq!((turns[1].0 - turns[0].0) as u64, turns[1].1 - turns[0].1);
        println!(
            "bookkeeping chunk={chunk} supplied={} actions/committed old={:?} new={:?}",
            details.is_some(),
            turns[1],
            turns[0]
        );
    }
}
#[test]
fn old_absent_cursor_resume_keeps_cancellation_budget_stale_cas_and_restart() {
    let raw = fixture(Some(Value::Null));
    let (mut w, p) = make_world(&raw, &raw);
    let req = request(&w, p.last().unwrap());
    let mut h = begin(&mut w, &req);
    loop {
        let projection = w.writer.message_projection(h.build(), 65536).unwrap();
        if let Some(bytes) = projection.cursor {
            let mut cursor: Value = serde_json::from_slice(&bytes).unwrap();
            if cursor["stage"]["Claude"]["SelectedCitations"]["index"] == 0 {
                cursor["stage"]["Claude"]["SelectedCitations"]["index"] = json!(2);
                w.writer
                    .advance_message_projection(
                        h.build(),
                        &ProjectionStep {
                            expected_sequence: projection.sequence,
                            next_cursor: &serde_json::to_vec(&cursor).unwrap(),
                        },
                        65536,
                    )
                    .unwrap();
                break;
            }
        }
        let a = w.compute(&h);
        m::apply(&mut w.writer, a).unwrap();
    }
    let before = w.writer.message_projection(h.build(), 65536).unwrap();
    let state = m::inspect(&mut w.writer, &h, 65536).unwrap();
    let snapshot = w.writer.parser_snapshot(&w.head.token).unwrap();
    let reader = w.writer.read_handle().unwrap();
    assert!(matches!(
        m::compute(
            &snapshot,
            &reader,
            &state,
            work_budget(),
            &AtomicBool::new(true)
        ),
        Err(m::Error::Cancelled)
    ));
    let mut tiny = work_budget();
    tiny.read.cells = 0;
    assert!(m::compute(&snapshot, &reader, &state, tiny, &AtomicBool::new(false)).is_err());
    drop(reader);
    drop(snapshot);
    drop(state);
    let after = w.writer.message_projection(h.build(), 65536).unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.cursor, after.cursor);
    let action = w.compute(&h);
    let stale = w.compute(&h);
    m::apply(&mut w.writer, action).unwrap();
    assert!(matches!(
        m::apply(&mut w.writer, stale),
        Err(m::Error::Store(
            polylogue_store::Error::MessagePredecessorChanged
        ))
    ));
    w = w.reopen();
    h = begin(&mut w, &req);
    let source = w.complete(&h);
    let view = selected(&mut w, &source);
    assert_eq!(
        value(
            &view,
            &Slot::Extra {
                block: None,
                key: "stop_details".into()
            }
        )
        .0,
        Value::Null
    );
}
