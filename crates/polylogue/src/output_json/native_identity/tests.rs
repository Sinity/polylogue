use crate::output_json::*;
use polylogue_model::{BlockId, MessageId, NativeId, Origin, RawJson, SessionId};
use serde_json::Value;
use std::time::Duration;

fn until() -> Instant {
    Instant::now() + Duration::from_secs(20)
}
fn native(text: &str, suffix: &[u16]) -> NativeId {
    NativeId::from_utf16(
        &text
            .encode_utf16()
            .chain(suffix.iter().copied())
            .collect::<Vec<_>>(),
    )
}
fn secret() -> String {
    format!("ghp_{}", "A".repeat(36))
}
fn origin(id: &NativeId, resolved: &str) -> RawJson {
    RawJson::parse(format!(r#"{{"format":"claude-code-field-origin","selection":{{"transform":{{"ParentReferenceV1":{{"native_uuid":{},"resolved":{{"id":{},"session":"claude:ordinary"}}}}}}}}}}"#,
        serde_json::to_string(id).unwrap(), serde_json::to_string(resolved).unwrap()).as_bytes()).unwrap()
}

#[test]
fn native_id_is_checked_on_its_utf16_units_and_written_unchanged() {
    for prefix in [String::new(), "x".repeat(PIECE - 7) + " "] {
        for suffix in [&[][..], &[0xd800][..]] {
            let id = native(&(prefix.clone() + &secret()), suffix);
            let mut bytes = Vec::new();
            assert!(Encoder::new(&mut bytes, until()).native_id(&id).is_err());
            assert!(bytes.is_empty());
        }
    }
    for id in [
        native("ordinary", &[0xd800]),
        native("ordinary", &[0xd83d, 0xde00]),
        NativeId::from_utf16(
            &"ghp_"
                .encode_utf16()
                .chain([0xd800])
                .chain("A".repeat(36).encode_utf16())
                .collect::<Vec<_>>(),
        ),
        NativeId::unicode("~u:6768705f"),
    ] {
        let mut bytes = Vec::new();
        Encoder::new(&mut bytes, until()).native_id(&id).unwrap();
        let decoded: NativeId = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, id);
    }
}

#[test]
fn framed_references_check_each_native_component() {
    let safe = SessionId::new(&Origin::new("claude").unwrap(), "safe").unwrap();
    let private = native(&secret(), &[0xd800]);
    let private_session =
        SessionId::from_native(&Origin::new("claude").unwrap(), &private).unwrap();
    let private_origin =
        SessionId::from_native(&Origin::new(secret()).unwrap(), &native("safe", &[0xd800]))
            .unwrap();
    let message = MessageId::from_native(safe, &private).unwrap();
    let block = BlockId::new(message.clone(), 0);
    for reference in [
        private_session.as_str(),
        private_origin.as_str(),
        message.as_str(),
        block.as_str(),
    ] {
        // The same bytes in an arbitrary author/name are not interpreted as a reference.
        assert!(check_identity(reference, until()).is_ok());
        assert!(check_reference(reference, until()).is_err());
        let mut bytes = Vec::new();
        assert!(
            Encoder::new(&mut bytes, until())
                .reference(reference)
                .is_err()
        );
        assert!(bytes.is_empty());
    }
    let clean = MessageId::from_native(
        SessionId::new(&Origin::new("claude").unwrap(), "safe").unwrap(),
        &native("safe", &[0xd800]),
    )
    .unwrap();
    assert!(check_reference(clean.as_str(), until()).is_ok());
    assert!(check_reference(&format!("message:{}", clean.as_str()), until()).is_ok());
    for bad in ["~i1:", "~i1:ffff", "~i1:0101"] {
        assert!(check_reference(bad, until()).is_err());
    }
}

#[test]
fn only_the_field_origin_parent_reference_is_decoded_as_native_id() {
    let private = native(&secret(), &[0xd800]);
    let descriptor = origin(&private, "claude:ordinary:message");
    let mut bytes = Vec::new();
    assert!(
        Encoder::new(&mut bytes, until())
            .origin(&descriptor)
            .is_err()
    );
    assert!(bytes.is_empty());
    assert!(check_source_origin(&descriptor, until()).is_err());
    let resolved = MessageId::from_native(
        SessionId::new(&Origin::new("claude").unwrap(), "safe").unwrap(),
        &private,
    )
    .unwrap();
    assert!(
        check_source_origin(
            &origin(&NativeId::unicode("safe"), resolved.as_str()),
            until()
        )
        .is_err()
    );
    let malformed = RawJson::parse(br#"{"format":"claude-code-field-origin","selection":{"transform":{"ParentReferenceV1":{"native_uuid":{"value":"~u:00","encoding":"unknown"},"resolved":{"id":"safe","session":"safe"}}}}}"#).unwrap();
    assert!(check_source_origin(&malformed, until()).is_err());
    let clean = origin(&native("safe", &[0xd800]), "claude:ordinary:message");
    let mut bytes = Vec::new();
    Encoder::new(&mut bytes, until()).origin(&clean).unwrap();
    assert_eq!(RawJson::parse(&bytes).unwrap(), clean);
    let lookalike = RawJson::parse(&serde_json::to_vec(&private).unwrap()).unwrap();
    let mut bytes = Vec::new();
    Encoder::new(&mut bytes, until())
        .raw_json(&lookalike, true)
        .unwrap();
    assert_eq!(RawJson::parse(&bytes).unwrap(), lookalike);
    let mut bytes = Vec::new();
    Encoder::new(&mut bytes, until())
        .opaque(&lookalike)
        .unwrap();
    let envelope: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        RawJson::parse(envelope["json"].as_str().unwrap().as_bytes()).unwrap(),
        lookalike
    );
}
