use super::*;
use polylogue_model::{NativeId, RetainedJson};

fn tool_input(call: &NativeId) -> Vec<u8> {
    let mut raw = titled("privacy-root");
    let call = String::from_utf8(RetainedJson::from(call).canonical()).unwrap();
    raw.extend(format!(r#"{{"type":"assistant","uuid":"privacy-tool","parentUuid":"privacy-root","sessionId":"{SID}","timestamp":"2026-10-01T08:01:00Z","requestId":"privacy-request","message":{{"id":"privacy-response","role":"assistant","model":"synthetic-model","content":[{{"type":"tool_use","id":{call},"name":"Synthetic","input":{{}}}}],"stop_reason":"end_turn"}}}}
"#).as_bytes());
    raw
}

#[test]
fn native_identity_scalar_and_surrogate_tool_call_ids_require_logical_privacy_before_wire_encoding()
{
    let token = format!("ghp_{}", "A".repeat(36));
    for surrogate in [false, true] {
        let mut units: Vec<u16> = token.encode_utf16().collect();
        if surrogate {
            units.push(0xd800);
        }
        let call = NativeId::from_utf16(&units);
        let world = World::new();
        let input = world.import(&tool_input(&call), "native-call-privacy");
        world.wait_rows(&input, false);
        let session = world
            .reader
            .selected_session(&world.session(), 65536)
            .unwrap()
            .unwrap();
        let mut document = Vec::new();
        let direct = write(&session, &mut document, Instant::now() + WAIT);
        let message = world
            .reader
            .selected_message(&MessageId::new(world.session(), "privacy-tool").unwrap())
            .unwrap()
            .unwrap();
        let block = message.block_at(0).unwrap().unwrap();
        let model = block_metadata(&block, Instant::now() + WAIT);
        assert!(
            direct.is_err() && model.is_err(),
            "surrogate={surrogate}: direct accepted={}, model accepted={}",
            direct.is_ok(),
            model.is_ok()
        );
        assert!(!String::from_utf8(document).unwrap().contains(&call.wire()));
    }
}

#[test]
fn native_identity_ordinary_surrogate_and_split_token_calls_keep_exact_wire_values() {
    let split = NativeId::from_utf16(
        &"ghp_"
            .encode_utf16()
            .chain([0xd800])
            .chain("A".repeat(36).encode_utf16())
            .collect::<Vec<_>>(),
    );
    for call in [
        NativeId::unicode("ordinary-😀"),
        NativeId::from_utf16(&[b'x' as u16, 0xd800]),
        split,
    ] {
        let world = World::new();
        let input = world.import(&tool_input(&call), "native-call-control");
        world.wait_rows(&input, false);
        let session = world
            .reader
            .selected_session(&world.session(), 65536)
            .unwrap()
            .unwrap();
        let mut bytes = Vec::new();
        write(&session, &mut bytes, Instant::now() + WAIT).unwrap();
        let document: Value = serde_json::from_slice(&bytes).unwrap();
        let shown = &document["messages"][1]["blocks"][0]["call_id"];
        assert_eq!(
            serde_json::from_value::<NativeId>(shown.clone()).unwrap(),
            call
        );
        let message = world
            .reader
            .selected_message(&MessageId::new(world.session(), "privacy-tool").unwrap())
            .unwrap()
            .unwrap();
        let block = message.block_at(0).unwrap().unwrap();
        let metadata = block_metadata(&block, Instant::now() + WAIT).unwrap();
        assert_eq!(
            serde_json::from_value::<NativeId>(metadata["call_id"].clone()).unwrap(),
            call
        );
    }
}
