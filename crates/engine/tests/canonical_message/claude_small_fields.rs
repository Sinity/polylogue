use super::*;
use std::collections::BTreeMap;
fn complete_field(view: &MessageView, slot: &Slot) -> (Vec<u8>, bool, Vec<u8>) {
    let FieldRead::Value(field) = view.field(slot, 65536).unwrap().unwrap() else {
        panic!("conflict")
    };
    let mut value = Vec::new();
    let mut origin = Vec::new();
    let mut offset = 0;
    loop {
        let range = field.range(offset, 65536).unwrap();
        value.extend(range.bytes);
        if range.eof {
            break;
        }
        assert!(range.next_offset > offset);
        offset = range.next_offset;
    }
    offset = 0;
    loop {
        let range = field.origin_range(offset, 65536).unwrap().unwrap();
        origin.extend(range.bytes);
        if range.eof {
            break;
        }
        assert!(range.next_offset > offset);
        offset = range.next_offset;
    }
    (value, field.supplied, origin)
}
fn sources(view: &MessageView) -> Vec<(u64, u64)> {
    use polylogue_store::contributions::SourceStep;
    let mut ranges = Vec::new();
    let mut cursor = None;
    loop {
        match view.next_source(cursor.as_ref(), 65536).unwrap() {
            SourceStep::Span {
                cursor: next,
                source,
            } => {
                ranges.push((source.start, source.end));
                cursor = Some(next);
            }
            SourceStep::Continue { cursor: next } => cursor = Some(next),
            SourceStep::End => return ranges,
        }
    }
}
fn bytes() -> Vec<u8> {
    let mut row = json!({"type":"user","uuid":"small","sessionId":SID,"parentUuid":null,"isSidechain":false,"promptSource":"typed","timestamp":"2026-10-01T00:00:00Z","message":{"role":"user","content":"small exact text"}});
    for n in 0..8 {
        row[format!("metadata_{n}")] =
            json!({"number":9007199254740993u64,"null":null,"label":"λ"});
    }
    (row.to_string() + "\n").into_bytes()
}
fn complete_with(
    w: &mut World,
    h: &m::Handle,
    chunk: usize,
) -> (SourceProjection, BTreeMap<String, usize>) {
    let mut counts = BTreeMap::new();
    for _ in 0..5000 {
        let mut b = work_budget();
        b.value_chunk = chunk;
        let a = w.compute_with(h, b).unwrap();
        *counts.entry(a.operation().to_owned()).or_default() += 1;
        if let m::Applied::Source(s) = m::apply(&mut w.writer, a).unwrap() {
            return (s, counts);
        }
    }
    panic!("small field fixture did not converge")
}
#[test]
fn atomic_small_fields_match_streamed_values_origins_evidence_and_citation_set() {
    let raw = bytes();
    let (mut fast, p) = make_world(&raw, &raw);
    let r = request(&fast, &p[0]);
    let h = begin(&mut fast, &r);
    let (s, counts) = complete_with(&mut fast, &h, 65536);
    assert!(counts.get("small_field").copied().unwrap_or(0) > 8);
    // One envelope group, eight selected extras, selected time and author.
    assert_eq!(counts.get("source_range"), Some(&11));
    let fv = selected(&mut fast, &s);
    let (mut slow, p) = make_world(&raw, &raw);
    let r = request(&slow, &p[0]);
    let h = begin(&mut slow, &r);
    let (t, stream) = complete_with(&mut slow, &h, 64);
    assert_eq!(stream.get("small_field"), None);
    assert!(stream.get("value_append").unwrap() > &0);
    let sv = selected(&mut slow, &t);
    assert_eq!(s.version_hash(), t.version_hash());
    assert_eq!(sources(&fv), sources(&sv));
    assert_eq!(sources(&fv), vec![(0, raw.len() as u64 - 1)]);
    assert_eq!(
        fv.block_at(0)
            .unwrap()
            .unwrap()
            .text_range(0, 65536)
            .unwrap()
            .unwrap()
            .bytes,
        sv.block_at(0)
            .unwrap()
            .unwrap()
            .text_range(0, 65536)
            .unwrap()
            .unwrap()
            .bytes
    );
    for n in 0..8 {
        let slot = Slot::Extra {
            block: None,
            key: format!("metadata_{n}"),
        };
        assert_eq!(complete_field(&fv, &slot), complete_field(&sv, &slot));
    }
}
#[test]
fn cancelled_small_field_computation_and_stale_atomic_action_publish_nothing() {
    let raw = bytes();
    let (mut w, p) = make_world(&raw, &raw);
    let r = request(&w, &p[0]);
    let h = begin(&mut w, &r);
    let prepared = loop {
        let a = w.compute(&h);
        if a.operation() == "small_field" {
            break a;
        }
        m::apply(&mut w.writer, a).unwrap();
    };
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
    drop(snapshot);
    drop(reader);
    drop(state);
    assert!(
        w.writer
            .current_source_projection(&w.head.token, &r.logical_key, 65536)
            .unwrap()
            .is_none()
    );
    let peer = w.compute(&h);
    assert_eq!(peer.operation(), "small_field");
    m::apply(&mut w.writer, peer).unwrap();
    assert!(matches!(
        m::apply(&mut w.writer, prepared),
        Err(m::Error::Store(
            polylogue_store::Error::MessagePredecessorChanged
        ))
    ));
    let source = w.complete(&h);
    assert_eq!(
        selected(&mut w, &source)
            .block_at(0)
            .unwrap()
            .unwrap()
            .text_range(0, 65536)
            .unwrap()
            .unwrap()
            .bytes,
        b"small exact text"
    );
}
#[test]
fn small_field_restart_after_atomic_install_keeps_exact_supplier() {
    let raw = bytes();
    let (mut w, p) = make_world(&raw, &raw);
    let r = request(&w, &p[0]);
    let mut h = begin(&mut w, &r);
    let mut restarted = false;
    let source = loop {
        let a = w.compute(&h);
        let small = a.operation() == "small_field";
        if let m::Applied::Source(s) = m::apply(&mut w.writer, a).unwrap() {
            break s;
        }
        if small && !restarted {
            w = w.reopen();
            h = begin(&mut w, &r);
            restarted = true;
        }
    };
    assert!(restarted);
    let view = selected(&mut w, &source);
    let (_, supplied, origin) = value(
        &view,
        &Slot::Extra {
            block: None,
            key: "metadata_0".into(),
        },
    );
    assert!(supplied);
    assert_eq!(
        origin["selection"]["inputs"][0]["path"],
        json!([{"Key":"metadata_0"}])
    );
}
