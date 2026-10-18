//! The actual canonical materializer verifies exact result data and source recipes.
use super::*;
use polylogue_model::RawJson;
fn row(meta: &str, content: &str, extra: &str) -> String {
    format!(
        r#"{{"type":"user","uuid":"result","toolUseResult":{meta},"message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"call","is_error":false,"content":{content}{extra}}}]}}}}
"#
    )
}
fn materialize(raw: &str) -> (World, SourceProjection, MessageView) {
    let (mut world, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let request = request(&world, &p[0]);
    let h = begin(&mut world, &request);
    world = world.reopen();
    let source = world.complete(&h);
    let view = selected(&mut world, &source);
    (world, source, view)
}
fn leaf(view: &MessageView, key: &str) -> (RawJson, Value) {
    let FieldRead::Value(f) = view
        .field(
            &Slot::Extra {
                block: Some(BlockSlot::Item(0)),
                key: key.into(),
            },
            65536,
        )
        .unwrap()
        .unwrap()
    else {
        panic!("conflict")
    };
    assert!(f.supplied);
    (
        RawJson::parse(&f.range(0, 65536).unwrap().bytes).unwrap(),
        serde_json::from_slice(&f.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap(),
    )
}
#[test]
fn first_sidecar_failure_and_string_carrier_have_exact_original_selectors_after_restart() {
    for (meta, key, path) in [
        (
            r#"{"gitOperation":{"pr":{"action":"create","number":9007199254740993,"url":"https://example.invalid/7","raw":[1.2300,"\ud800",null]}}}"#,
            "gitOperation",
            json!([{"Key":"toolUseResult"},{"Key":"gitOperation"}]),
        ),
        (
            r#""supplied\ud800""#,
            "toolUseResult",
            json!([{"Key":"toolUseResult"}]),
        ),
    ] {
        let raw = row(meta, r#""real output""#, "");
        let (_w, _, view) = materialize(&raw);
        let (v, o) = leaf(&view, key);
        let expected = RawJson::parse(meta.as_bytes()).unwrap();
        assert_eq!(
            v,
            if key == "toolUseResult" {
                expected
            } else {
                expected.get(key).unwrap().clone()
            }
        );
        assert_eq!(o["selection"]["inputs"][0]["path"], path);
        assert_eq!(
            o["selection"]["inputs"][0]["record"],
            json!({"start":0,"end":raw.len()-1})
        );
        assert_eq!(
            value(
                &view,
                &Slot::Block {
                    block: BlockSlot::Item(0),
                    field: BlockField::Output
                }
            )
            .0,
            json!({"Text":"real output"})
        );
    }
}
#[test]
fn reference_arrays_stay_structured_content_not_output_or_sidecar_content() {
    for content in [
        r#"[{"type":"tool_reference","tool_name":"synthetic"}]"#,
        r#"[{"type":"text","text":"supplied text"},{"type":"tool_reference","tool_name":"synthetic"},{"type":"text","text":"\ud800"}]"#,
    ] {
        let raw = row(r#"{"content":"independent sidecar"}"#, content, "");
        let (_w, _, view) = materialize(&raw);
        assert_eq!(
            leaf(&view, "content_items").0,
            RawJson::parse(content.as_bytes()).unwrap()
        );
        assert_eq!(
            leaf(&view, "content").0,
            RawJson::parse(br#""independent sidecar""#).unwrap()
        );
        assert_eq!(
            leaf(&view, "content_items").1["selection"]["inputs"][0]["path"],
            json!([{"Key":"message"},{"Key":"content"},{"Index":0},{"Key":"content"}])
        );
        assert!(
            view.field(
                &Slot::Block {
                    block: BlockSlot::Item(0),
                    field: BlockField::Output
                },
                65536
            )
            .unwrap()
            .is_none()
        );
    }
}
fn refuse(raw: &str) {
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[0]);
    let h = begin(&mut w, &req);
    for _ in 0..4000 {
        match w.compute_with(&h, work_budget()) {
            Ok(action) => assert!(!matches!(
                m::apply(&mut w.writer, action).unwrap(),
                m::Applied::Source(_)
            )),
            Err(m::Error::Unsupported(_) | m::Error::UnsupportedField { .. }) => return,
            Err(e) => panic!("{e}"),
        }
    }
    panic!("unsupported certified")
}
#[test]
fn reference_shape_unknowns_and_selected_data_collisions_are_not_waived() {
    for content in [
        r#"[{"type":"tool_reference","tool_name":null}]"#,
        r#"[{"type":"tool_reference","tool_name":""}]"#,
        r#"[{"type":"tool_reference","tool_name":"t","unknown":null}]"#,
        r#"[{"type":"tool_reference","tool_name":"t"},{"type":"image","source":{}}]"#,
        r#"[{"type":"text","text":"t","unknown":1},{"type":"tool_reference","tool_name":"t"}]"#,
    ] {
        refuse(&row("{}", content, ""));
    }
    refuse(&row(
        r#"{"gitOperation":null}"#,
        r#""text""#,
        r#", "gitOperation":null"#,
    ));
    refuse(&row(
        r#""sidecar""#,
        r#""text""#,
        r#", "toolUseResult":"sidecar""#,
    ));
    refuse(&row(
        "{}",
        r#"[{"type":"tool_reference","tool_name":"t"}]"#,
        r#", "content_items":[]"#,
    ));
    refuse(&row(r#"{"unknown":null}"#, r#""text""#, ""));
}
