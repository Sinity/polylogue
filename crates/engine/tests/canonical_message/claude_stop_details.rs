//! Named opaque response metadata through the actual canonical worker/store path.
use super::*;
use polylogue_sources::claude_code::selection::LeafSelection;

fn row(uuid: &str, text: &str, details: Option<Value>) -> String {
    let mut raw = json!({"type":"assistant","uuid":uuid,"requestId":"request","sessionId":SID,
        "message":{"id":"response","role":"assistant","content":[{"type":"text","text":text}]}});
    if let Some(value) = details {
        raw["message"]["stop_details"] = value;
    }
    raw.to_string() + "\n"
}
fn slot() -> Slot {
    Slot::Extra {
        block: None,
        key: "stop_details".into(),
    }
}
fn materialize(raw: &str) -> (World, SourceProjection, MessageView) {
    let (mut world, pending) = make_world(raw.as_bytes(), raw.as_bytes());
    assert_eq!(pending.len(), 1);
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    let source = world.complete(&handle);
    let view = selected(&mut world, &source);
    (world, source, view)
}
fn assert_value(view: &MessageView, raw: &str, expected: Value, start: usize, end: usize) {
    let (actual, supplied, origin) = value(view, &slot());
    assert_eq!(actual, expected);
    assert!(supplied);
    assert_eq!(origin["format"], "claude-code-field-origin");
    let recipe: LeafSelection = serde_json::from_value(origin["selection"].clone()).unwrap();
    assert_eq!(recipe.transform, cc::selection::Transform::JsonValueV1);
    assert_eq!(recipe.inputs.len(), 1);
    assert_eq!(
        recipe.inputs[0].record,
        polylogue_sources::selection::RecordRange {
            start: start as u64,
            end: end as u64
        }
    );
    assert_eq!(
        recipe.inputs[0].path,
        vec![
            polylogue_sources::selection::PathPart::Key("message".into()),
            polylogue_sources::selection::PathPart::Key("stop_details".into())
        ]
    );
    let mut selected = Vec::new();
    polylogue_sources::selection::write_value(
        &recipe,
        |r| -> Result<Vec<u8>, std::io::Error> {
            Ok(raw.as_bytes()[r.start as usize..r.end as usize].to_vec())
        },
        65536,
        &mut selected,
    )
    .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&selected).unwrap(),
        expected
    );
}
#[test]
fn supplied_null_and_opaque_object_have_exact_origins_without_default_or_finish_inference() {
    let absent = row("a", "text", None);
    let (_world, absent_source, absent_view) = materialize(&absent);
    assert!(absent_view.field(&slot(), 65536).unwrap().is_none());
    let finish = absent_view.header().unwrap().in_progress;
    let mut versions = vec![absent_source.version_hash()];
    for details in [
        Value::Null,
        json!({"kind":"provider detail","nested":{"flag":false},"empty":[]}),
    ] {
        let raw = row("a", "text", Some(details.clone()));
        let (_world, source, view) = materialize(&raw);
        assert_value(&view, &raw, details, 0, raw.len() - 1);
        assert_eq!(view.header().unwrap().in_progress, finish);
        assert!(
            view.field(&Slot::Message(MessageField::StopReason), 65536)
                .unwrap()
                .is_none()
        );
        assert_eq!(view.version_hash().unwrap(), source.version_hash());
        versions.push(source.version_hash());
    }
    assert_ne!(versions[0], versions[1]);
    assert_ne!(versions[1], versions[2]);
}
#[test]
fn last_actual_supplier_and_same_uuid_omission_keep_the_original_donor() {
    let a = row("a", "a", Some(Value::Null));
    let b = row("b", "b", Some(json!({"winner":"b"})));
    let omitted_a = row("a", "changed", None);
    let later = row("c", "later without metadata", None);
    for raw in [
        a.clone() + &b + &later,
        a.clone() + &b + &omitted_a,
        a.clone() + &b + &a,
    ] {
        let (_world, _, view) = materialize(&raw);
        assert_value(
            &view,
            &raw,
            json!({"winner":"b"}),
            a.len(),
            a.len() + b.len() - 1,
        );
    }
    let raw = a.clone() + &omitted_a;
    let (_world, _, view) = materialize(&raw);
    assert_value(&view, &raw, Value::Null, 0, a.len() - 1);
}
#[test]
fn replacement_of_earlier_contribution_changes_selected_metadata_and_exact_version() {
    let a = row("a", "a", Some(Value::Null));
    let b = row("b", "b", Some(json!({"winner":"b"})));
    let replaced_a = row("a", "changed", Some(json!({"winner":"a"})));
    let corrected = a.clone() + &b + &replaced_a;
    let (_world, corrected_source, view) = materialize(&corrected);
    assert_value(
        &view,
        &corrected,
        json!({"winner":"a"}),
        a.len() + b.len(),
        corrected.len() - 1,
    );
    // Same effective contribution records/order, different actual last supplier.
    // The response correction unit must enter version evidence.
    let reordered = replaced_a.clone() + &b;
    let (_world, reordered_source, view) = materialize(&reordered);
    assert_value(
        &view,
        &reordered,
        json!({"winner":"b"}),
        replaced_a.len(),
        reordered.len() - 1,
    );
    assert_ne!(
        corrected_source.version_hash(),
        reordered_source.version_hash()
    );
}
#[test]
fn unknown_nested_extra_and_literal_selected_key_collision_still_refuse_certification() {
    for raw in [
        json!({"type":"assistant","uuid":"a","requestId":"r","message":{"id":"response","content":[{"type":"text","text":"text"}],"unknown_details":null}}),
        json!({"type":"assistant","uuid":"a","requestId":"r","stop_details":1,"message":{"id":"response","content":[{"type":"text","text":"text"}],"stop_details":null}}),
        json!({"type":"assistant","uuid":"a","requestId":"r","message.unknown_details":1,"message":{"id":"response","content":[{"type":"text","text":"text"}],"unknown_details":null}}),
    ] {
        let raw = raw.to_string() + "\n";
        let (mut world, pending) = make_world(raw.as_bytes(), raw.as_bytes());
        let req = request(&world, &pending[0]);
        let handle = begin(&mut world, &req);
        let mut refused = false;
        for _ in 0..4000 {
            match world.compute_with(&handle, work_budget()) {
                Ok(action) => assert!(!matches!(
                    m::apply(&mut world.writer, action).unwrap(),
                    m::Applied::Source(_)
                )),
                Err(m::Error::UnsupportedField { .. }) => {
                    refused = true;
                    break;
                }
                Err(error) => panic!("unexpected refusal: {error}"),
            }
        }
        assert!(refused);
        assert!(
            world
                .writer
                .current_source_projection(&world.head.token, &req.logical_key, 65536)
                .unwrap()
                .is_none()
        );
    }
}
