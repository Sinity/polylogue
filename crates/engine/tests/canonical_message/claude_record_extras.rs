//! Source-selected immediate record extras, never contribution-order overwrites.
use super::*;
use polylogue_model::RetainedJson;
fn row(uuid: &str, text: &str, extras: &str) -> String {
    format!(
        r#"{{"type":"assistant","uuid":"{uuid}","requestId":"r",{extras}"message":{{"id":"response","content":[{{"type":"text","text":"{text}"}}]}}}}
"#
    )
}
fn materialize(raw: &str) -> (World, SourceProjection, MessageView) {
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[0]);
    let h = begin(&mut w, &req);
    let source = w.complete(&h);
    let view = selected(&mut w, &source);
    (w, source, view)
}
fn extra(view: &MessageView, key: &str) -> (RetainedJson, Value) {
    let Some(FieldRead::Value(f)) = view
        .field(
            &Slot::Extra {
                block: None,
                key: key.into(),
            },
            65536,
        )
        .unwrap()
    else {
        panic!("missing extra")
    };
    assert!(f.supplied);
    (
        RetainedJson::parse(&f.range(0, 65536).unwrap().bytes).unwrap(),
        serde_json::from_slice(&f.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap(),
    )
}
fn start(origin: &Value) -> u64 {
    origin["selection"]["inputs"][0]["record"]["start"]
        .as_u64()
        .unwrap()
}
#[test]
fn grouped_repetition_chooses_actual_supplier_even_for_equal_values_and_omissions() {
    let a = row(
        "a",
        "a",
        r#""effort":"same","independent":null,"literal.dot":1.2300,"#,
    );
    let b = row("b", "b", r#""effort":"same","#);
    let c = row("c", "c", r#""effort":"same","#);
    let omit = row("a", "changed", "");
    let raw = a.clone() + &b + &c + &omit;
    let (_w, _, v) = materialize(&raw);
    let (value, origin) = extra(&v, "effort");
    assert_eq!(value, RetainedJson::parse(br#""same""#).unwrap());
    assert_eq!(start(&origin), (a.len() + b.len()) as u64);
    assert_eq!(
        origin["selection"]["inputs"][0]["path"],
        json!([{"Key":"effort"}])
    );
    assert_eq!(start(&extra(&v, "independent").1), 0);
    assert_eq!(extra(&v, "independent").0, RetainedJson::Null);
    assert_eq!(
        extra(&v, "literal.dot").0,
        RetainedJson::parse(b"1.2300").unwrap()
    );
    let equal_update = row("a", "different", r#""effort":"same","#);
    let raw = raw + &equal_update;
    let (_w2, _, v2) = materialize(&raw);
    assert_eq!(
        start(&extra(&v2, "effort").1),
        (raw.len() - equal_update.len()) as u64
    );
    let (_w3, source, v3) = materialize(&(raw.clone() + &equal_update));
    assert_eq!(extra(&v2, "effort").0, extra(&v3, "effort").0);
    // Input bindings differ after appending bytes; selected original recipes do not.
    assert_eq!(
        extra(&v2, "effort").1["selection"],
        extra(&v3, "effort").1["selection"]
    );
    let (_w4, base, _) = materialize(&raw);
    assert_eq!(source.version_hash(), base.version_hash());
}
#[test]
fn earlier_item_correction_changes_selected_raw_evidence_and_reopens_deterministically() {
    let a = row("a", "a", r#""effort":1,"λ/key":1,"#);
    let b = row("b", "b", r#""effort":2,"λ/key":2,"#);
    let update = row(
        "a",
        "updated",
        r#""effort":{"number":9007199254740993,"text":"\ud800","none":null},"λ/key":3,"#,
    );
    let raw = a + &b + &update;
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[0]);
    let mut h = begin(&mut w, &req);
    let mut restarts = 0;
    let source = loop {
        let action = w.compute(&h);
        let field = matches!(action.operation(), "field" | "small_field");
        if let m::Applied::Source(source) = m::apply(&mut w.writer, action).unwrap() {
            break source;
        }
        if field {
            w = w.reopen();
            h = begin(&mut w, &req);
            restarts += 1;
        }
        assert!(restarts < 30);
    };
    assert!(restarts > 0);
    let v = selected(&mut w, &source);
    assert_eq!(
        extra(&v, "effort").0,
        RetainedJson::parse(br#"{"number":9007199254740993,"text":"\ud800","none":null}"#).unwrap()
    );
    assert_eq!(
        start(&extra(&v, "effort").1),
        (raw.len() - update.len()) as u64
    );
    assert_eq!(extra(&v, "λ/key").0, RetainedJson::parse(b"3").unwrap());
    assert_eq!(
        extra(&v, "λ/key").1["selection"]["inputs"][0]["path"],
        json!([{"Key":"λ/key"}])
    );
    let (_bw, base, bv) = materialize(&(update + &b));
    assert_eq!(extra(&bv, "effort").0, RetainedJson::parse(b"2").unwrap());
    assert_ne!(source.version_hash(), base.version_hash());
    let (_rw, replay, rv) = materialize(&raw);
    assert_eq!(source.version_hash(), replay.version_hash());
    assert_eq!(extra(&v, "effort"), extra(&rv, "effort"));
}
#[test]
fn selected_key_collisions_and_unknown_nested_paths_still_refuse() {
    for raw in [
        row("a", "a", r#""provider_usage":null,"#)
            .replace(r#""content":["#, r#""usage":{"speed":null},"content":["#),
        row("a", "a", r#""stop_details":null,"#)
            .replace(r#""content":["#, r#""stop_details":null,"content":["#),
        row("a", "a", r#""stop_sequence":null,"#)
            .replace(r#""content":["#, r#""stop_sequence":null,"content":["#),
        row("a", "a", r#""future":null,"#)
            .replace(r#""content":["#, r#""future":null,"content":["#),
    ] {
        let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
        let req = request(&w, &p[0]);
        let h = begin(&mut w, &req);
        let mut refused = false;
        for _ in 0..2000 {
            match w.compute_with(&h, work_budget()) {
                Err(m::Error::UnsupportedField { .. }) => {
                    refused = true;
                    break;
                }
                Err(e) => panic!("unexpected {e:?}"),
                Ok(a) => {
                    assert!(!matches!(
                        m::apply(&mut w.writer, a).unwrap(),
                        m::Applied::Source(_)
                    ));
                }
            }
        }
        assert!(refused);
    }
}
#[test]
fn proof_scratch_admission_is_typed_and_does_not_publish_partial_selection() {
    let a = row("a", "a", r#""effort":1,"#);
    let b = row("b", "b", r#""effort":2,"#);
    let keys = (0..24)
        .map(|n| format!("long_key_{n:02}_{}", "k".repeat(70)))
        .collect::<Vec<_>>();
    let update = row(
        "a",
        "changed",
        &(keys
            .iter()
            .map(|k| format!("\"{k}\":null,"))
            .collect::<String>()
            + "\"effort\":3,"),
    );
    let raw = a + &b + &update;
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[0]);
    let h = begin(&mut w, &req);
    let mut small = work_budget();
    small.metadata = 1024;
    let mut installed = 0;
    let mut refused = false;
    for _ in 0..2000 {
        match w.compute_with(&h, small) {
            Err(m::Error::Store(polylogue_store::Error::ParserAdmission { resource, .. })) => {
                assert_eq!(resource, "Message materializer metadata");
                refused = true;
                break;
            }
            Err(e) => panic!("unexpected {e:?}"),
            Ok(action) => {
                installed += usize::from(matches!(action.operation(), "field" | "small_field"));
                assert!(!matches!(
                    m::apply(&mut w.writer, action).unwrap(),
                    m::Applied::Source(_)
                ));
            }
        }
    }
    assert!(refused && installed > 0);
    assert!(
        w.writer
            .current_source_projection(&w.head.token, &req.logical_key, 65536)
            .unwrap()
            .is_none()
    );
    let source = w.complete(&h);
    let v = selected(&mut w, &source);
    assert_eq!(extra(&v, "effort").0, RetainedJson::parse(b"3").unwrap());
}
#[test]
fn unchanged_large_extra_keeps_single_whole_record_version_without_metadata_value_copy() {
    let raw = row("a", "a", &format!("\"effort\":\"{}\",", "x".repeat(30000)));
    let (_world, source, view) = materialize(&raw);
    let canonical =
        polylogue_model::identity::canonical_record(raw.trim_end().as_bytes(), &[]).unwrap();
    assert_eq!(
        source.version_hash(),
        polylogue_model::BlobHash::of(&canonical)
    );
    assert_eq!(
        extra(&view, "effort").0,
        RetainedJson::parse(format!("\"{}\"", "x".repeat(30000)).as_bytes()).unwrap()
    );
}

#[test]
fn selected_donor_is_cited_independently_of_current_effective_contribution_origins() {
    let a = row("a", "first", "");
    let b = row("b", "second", r#""effort":null,"#);
    let raw = a.clone() + &b + &b;
    let original = a.len() as u64;
    let replacement = polylogue_sources::selection::RecordRange {
        start: (a.len() + b.len()) as u64,
        end: (raw.len() - 1) as u64,
    };
    let (mut w, p) = make_world_with(raw.as_bytes(), raw.as_bytes(), |output| {
        // Isolate the source contract: the selected supplier remains original,
        // while the effective record's direct donors all name an equal later
        // record. Values and source paths remain unchanged and independently
        // verifiable; no incidental contribution citation can cover the winner.
        let c = &mut output
            .delta
            .contributions
            .iter_mut()
            .find(|(_, item, _)| *item == 1)
            .unwrap()
            .2;
        c.origins.envelope = Some(replacement);
        for origin in c.origins.fields.values_mut() {
            origin.record = replacement;
        }
        for fact in c
            .projection
            .facts
            .iter_mut()
            .chain(&mut c.projection.retained)
        {
            if let Some(selection) = &mut fact.selection {
                for input in &mut selection.inputs {
                    input.record = replacement;
                }
            }
        }
    });
    let req = request(&w, &p[0]);
    let h = begin(&mut w, &req);
    let source = w.complete(&h);
    let v = selected(&mut w, &source);
    let (value, origin) = extra(&v, "effort");
    assert_eq!(value, RetainedJson::Null);
    assert_eq!(start(&origin), original);
}
