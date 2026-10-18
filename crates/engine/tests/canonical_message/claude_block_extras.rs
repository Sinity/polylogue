//! Immediate block members keep opaque values and exact original selectors.
use super::*;
use polylogue_model::RawJson;

#[test]
fn caller_and_text_metadata_are_opaque_source_backed_block_leaves() {
    let raw=br#"{"type":"assistant","uuid":"a","requestId":"r","message":{"id":"response","content":[{"type":"tool_use","id":"call","name":"synthetic","input":{},"caller":{"type":"direct","n":1.2300,"s":"\ud800","nothing":null}},{"type":"text","text":"text","metadata":null}]}}
"#;
    let (mut world, pending) = make_world(raw, raw);
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    let source = world.complete(&handle);
    let view = selected(&mut world, &source);
    for (index, key, expected) in [
        (
            0,
            "caller",
            br#"{"type":"direct","n":1.2300,"s":"\ud800","nothing":null}"#.as_slice(),
        ),
        (1, "metadata", b"null".as_slice()),
    ] {
        let FieldRead::Value(field) = view
            .field(
                &Slot::Extra {
                    block: Some(BlockSlot::Item(index)),
                    key: key.into(),
                },
                65536,
            )
            .unwrap()
            .unwrap()
        else {
            panic!("conflict")
        };
        assert!(field.supplied);
        assert_eq!(
            RawJson::parse(&field.range(0, 65536).unwrap().bytes).unwrap(),
            RawJson::parse(expected).unwrap()
        );
        let origin: Value =
            serde_json::from_slice(&field.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap();
        assert_eq!(
            origin["selection"]["inputs"][0]["path"],
            json!([{"Key":"message"},{"Key":"content"},{"Index":index},{"Key":key}])
        );
        assert_eq!(
            origin["selection"]["inputs"][0]["record"],
            json!({"start":0,"end":raw.len()-1})
        );
    }
}
#[test]
fn block_extra_cannot_alias_selected_result_channels() {
    let raw=br#"{"type":"user","uuid":"a","toolUseResult":{"stdout":"selected"},"message":{"content":[{"type":"tool_result","tool_use_id":"call","content":"text","is_error":false,"stdout":"other"}]}}
"#;
    let (mut world, pending) = make_world(raw, raw);
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    for _ in 0..4000 {
        match world.compute_with(&handle, work_budget()) {
            Ok(action) => {
                assert!(!matches!(
                    m::apply(&mut world.writer, action).unwrap(),
                    m::Applied::Source(_)
                ));
            }
            Err(m::Error::UnsupportedField {
                reason: "Claude block extra conflicts with selected structured data",
                ..
            }) => return,
            Err(error) => panic!("{error}"),
        }
    }
    panic!("collision certified");
}
