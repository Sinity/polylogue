//! Real source -> installed selected fields -> direct normalized output.
use super::*;
use polylogue_model::RawJson;

fn write_document(world: &World) -> Vec<u8> {
    let selected = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let mut bytes = Vec::new();
    write(&selected, &mut bytes, Instant::now() + WAIT).unwrap();
    bytes
}

#[test]
fn canonical_opaque_source_assembled_attrs_sort_keys_without_changing_leaf_values() {
    let mut raw = titled("canonical-root");
    raw.extend(format!(r#"{{"type":"assistant","uuid":"canonical-answer","parentUuid":"canonical-root","sessionId":"{SID}","timestamp":"2026-10-01T08:01:00Z","requestId":"canonical-request","a_extra":null,"z_extra":{{"z":-0,"n":9007199254740993,"e":1e+02,"\ud800":"lone","\ue000":"bmp","\ud800\udc00":"scalar","empty":"","object":{{}},"array":[]}},"message":{{"id":"canonical-response","role":"assistant","model":"synthetic-model","content":[{{"type":"text","text":"Synthetic answer"}}],"stop_reason":"end_turn","stop_sequence":null}}}}
"#).as_bytes());
    let world = World::new();
    let input = world.import(&raw, "canonical-attrs");
    world.wait_rows(&input, false);
    let bytes = write_document(&world);
    let expected = RawJson::parse(br#"{"z_extra":{"z":-0,"n":9007199254740993,"e":1e+02,"\ud800":"lone","\ue000":"bmp","\ud800\udc00":"scalar","empty":"","object":{},"array":[]},"stop_sequence":null,"a_extra":null}"#).unwrap();
    let needle = format!(
        "\"attrs\":{}",
        String::from_utf8(expected.canonical()).unwrap()
    );
    assert!(
        String::from_utf8(bytes).unwrap().contains(&needle),
        "assembled attrs must use retained canonical order: {needle}"
    );
}

fn result_source() -> Vec<u8> {
    let mut raw = titled("canonical-root");
    let metadata = r#"{"agentId":{"z":1E2,"\ud800":null,"\ufffd":"scalar","array":[],"object":{},"empty":""},"totalTokens":9007199254740993,"stdout":""}"#;
    raw.extend(format!(r#"{{"type":"user","uuid":"canonical-result","parentUuid":"canonical-root","sessionId":"{SID}","timestamp":"2026-10-01T08:01:00Z","toolUseResult":{metadata},"message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"synthetic-call","content":"Synthetic result","is_error":false}}]}}}}
"#).as_bytes());
    raw
}

#[test]
fn canonical_opaque_source_assembled_data_preserves_exact_nested_values() {
    let world = World::new();
    let input = world.import(&result_source(), "canonical-data-later");
    world.wait_rows(&input, false);
    let bytes = write_document(&world);
    let expected = RawJson::parse(br#"{"totalTokens":9007199254740993,"agentId":{"z":1E2,"\ud800":null,"\ufffd":"scalar","array":[],"object":{},"empty":""},"stdout":""}"#).unwrap();
    let needle = format!(
        "\"data\":{}",
        String::from_utf8(expected.canonical()).unwrap()
    );
    assert!(
        String::from_utf8(bytes).unwrap().contains(&needle),
        "assembled data must use retained canonical order: {needle}"
    );
}
