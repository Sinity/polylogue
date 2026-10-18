//! Block provider metadata is kept opaque, with checked origins.
use super::*;
use polylogue_store::contributions::FieldRead;

const KEY: &str = "internal_chat_message_metadata_passthrough";
const OPAQUE: &[u8] = br#"{"decimal":1.2300,"large":123456789012345678901234567890,"nullable":null,"units":"neutral"}"#;
fn input() -> Vec<u8> {
    let mut bytes = line(
        "session_meta",
        json!({"id":"synthetic-session","source":"cli"}),
    )
    .into_bytes();
    bytes.extend_from_slice(br#"{"type":"response_item","payload":{"type":"message","id":"metadata-message","role":"assistant","content":[{"type":"output_text","text":"Neutral metadata carrier."}],"internal_chat_message_metadata_passthrough":"#);
    bytes.extend_from_slice(OPAQUE);
    bytes.extend_from_slice(b"}}\n");
    bytes
}
#[test]
fn block_extra_keeps_opaque_metadata_and_original_field_selection() {
    let mut world = World::new(&input());
    let handle = world.begin();
    let source = world.complete(&handle);
    world
        .writer
        .select_compatible_source_message_for_test(&source, 65536)
        .unwrap();
    let reader = world.writer.read_handle().unwrap();
    let view = reader.selected_message(source.message()).unwrap().unwrap();
    let slot = Slot::Extra {
        block: Some(BlockSlot::Item(0)),
        key: KEY.into(),
    };
    let FieldRead::Value(field) = view.field(&slot, 65536).unwrap().unwrap() else {
        panic!("opaque field conflict")
    };
    assert!(field.supplied);
    let bytes = field.range(0, 65536).unwrap().bytes;
    assert_eq!(
        polylogue_model::RawJson::parse(&bytes).unwrap(),
        polylogue_model::RawJson::parse(OPAQUE).unwrap()
    );
    assert!(bytes.windows(6).any(|part| part == b"1.2300"));
    let origin: Value =
        serde_json::from_slice(&field.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap();
    assert_eq!(origin["format"], "codex-field-origin");
    assert_eq!(
        origin["binding"]["input"],
        polylogue_model::InputId::imported(&world.source_binding, "codex-rollout", &[]).as_str()
    );
    assert_eq!(
        origin["selection"]["inputs"][0]["path"],
        json!([{"Key":"payload"},{"Key":KEY}])
    );
}
#[test]
fn block_extra_with_a_changed_normalized_value_cannot_get_a_source_certificate() {
    use codex::keyed::{Cell, Key};
    let bytes = input();
    let mut world = World::with_mutation(&bytes, &bytes, |delta| {
        let Some(Some(Cell::Presence(group))) = delta.cells.get_mut(&Key::PresenceBlock(0, 0))
        else {
            panic!("block witnesses")
        };
        let witness = group
            .fields
            .iter_mut()
            .find(|w| matches!(&w.slot,Slot::Extra{key,..} if key==KEY))
            .unwrap();
        witness.canonical = b"null".to_vec();
    });
    let handle = world.begin();
    for _ in 0..1000 {
        match world.compute_with(&handle, work_budget()) {
            Ok(action) => {
                m::apply(&mut world.writer, action).unwrap();
            }
            Err(m::Error::Invariant(
                "source selector output differs from normalized field value",
            )) => return,
            Err(error) => panic!("unexpected refusal: {error}"),
        }
    }
    panic!("changed opaque metadata acquired a source certificate");
}
