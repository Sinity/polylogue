//! Native session IDs are stored and read back exactly, including their encoding.
use super::*;
use polylogue_model::NativeId;

fn prepare_native(
    f: &mut Fixture,
    native: &NativeId,
    encoding: Option<&str>,
) -> Result<Preparation> {
    let native = native.as_str().to_owned();
    let encoding = encoding.map(str::to_owned);
    f.prepare_with(|identity, _| {
        identity.native_session = &native;
        identity.native_session_encoding = encoding.as_deref();
    })
}

#[test]
fn encoded_and_literal_native_ids_stay_distinct_after_reopen() {
    let encoded = NativeId::from_utf16(&[0xd800]);
    let literal = NativeId::unicode(encoded.as_str());
    let mut f = Fixture::new();
    let mut ids = Vec::new();
    for native in [&encoded, &literal] {
        f.session = SessionId::from_native(&f.origin, native).unwrap();
        f.message = MessageId::new(f.session.clone(), "message").unwrap();
        prepare_native(&mut f, native, native.encoding()).unwrap();
        ids.push(f.session.clone());
    }
    assert_ne!(ids[0], ids[1]);
    let f = f.reopen();
    let reader = f.writer.read_handle().unwrap();
    for (id, native) in ids.iter().zip([&encoded, &literal]) {
        let view = reader.selected_session(id, 4096).unwrap().unwrap();
        let header = view.header().unwrap();
        assert_eq!(header.native_id, native.as_str());
        assert_eq!(header.native_id_encoding.as_deref(), native.encoding());
        assert_eq!(
            NativeId::from_wire(&header.native_id, header.native_id_encoding.as_deref()).unwrap(),
            *native
        );
    }
}

#[test]
fn prepare_rejects_bad_or_changed_native_encoding() {
    let encoded = NativeId::from_utf16(&[0xd800]);
    let mut f = Fixture::new();
    assert!(prepare_native(&mut f, &encoded, Some("unknown")).is_err());
    assert_eq!(f.scalar("SELECT count(*) FROM session"), 0);
    prepare_native(&mut f, &encoded, encoded.encoding()).unwrap();
    assert!(prepare_native(&mut f, &encoded, None).is_err());
    assert_eq!(f.scalar("SELECT count(*) FROM session"), 1);
}

#[test]
fn selected_session_rejects_corrupt_native_encoding() {
    let encoded = NativeId::from_utf16(&[0xd800]);
    let mut f = Fixture::new();
    prepare_native(&mut f, &encoded, encoded.encoding()).unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE session SET native_id='not-an-encoded-native-id'",
            [],
        )
        .unwrap();
    let reader = f.writer.read_handle().unwrap();
    assert!(reader.selected_session(&f.session, 4096).is_err());
}
