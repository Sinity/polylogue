use super::*;

#[test]
fn exact_utf16_and_explicit_wire_are_injective() {
    for units in [
        vec![],
        vec![0xd800],
        vec![0xdfff],
        vec![0xd800, 0xdc00],
        vec![0xdfff, 0xd800],
        vec![97, 0, 0xd800, 98],
    ] {
        let id = NativeId::from_utf16(&units);
        assert_eq!(id.utf16(), units);
        assert_eq!(NativeId::from_wtf8(&id.bytes).unwrap(), id);
        assert_eq!(NativeId::from_wire(&id.wire(), id.encoding()).unwrap(), id);
        let wire = serde_json::to_vec(&id).unwrap();
        assert_eq!(serde_json::from_slice::<NativeId>(&wire).unwrap(), id);
    }
    let lone = NativeId::from_utf16(&[0xd800]);
    let literal = NativeId::unicode(lone.as_str());
    assert_ne!(lone, literal);
    assert_eq!(literal.encoding(), None);
    assert_eq!(
        serde_json::from_str::<NativeId>(r#""~u:eda080""#).unwrap(),
        literal
    );
}

#[test]
fn malformed_noncanonical_and_wrong_discriminants_refuse() {
    for bytes in [
        &[0xc0, 0x80][..],
        &[0xf4, 0x90, 0x80, 0x80],
        &[0xed, 0xa0, 0x80, 0xed, 0xb0, 0x80],
        &[0xe0, 0x80],
        &[0x80],
    ] {
        assert!(NativeId::from_wtf8(bytes).is_err());
    }
    for (value, encoding) in [
        ("~u:61", "wtf8"),
        ("~u:EDA080", "wtf8"),
        ("eda080", "wtf8"),
        ("~u:eda080", "unknown"),
    ] {
        assert!(NativeId::from_wire(value, Some(encoding)).is_err());
    }
    assert!(
        serde_json::from_str::<NativeId>(r#"{"value":"~u:eda080","encoding":"wtf8","other":0}"#)
            .is_err()
    );
}
