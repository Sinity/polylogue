//! Session native facts survive SQLite without interpreting transport spelling.
use super::*;
use polylogue_model::NativeId;

fn prepare_native(
    f: &mut Fixture,
    native: &NativeId,
    encoding: Option<&str>,
) -> Result<Preparation> {
    f.writer.prepare_message(
        Identity {
            session: &f.session,
            origin: &f.origin,
            native_session: native.as_str(),
            native_session_encoding: encoding,
            session_kind: SessionKind::Main,
            message: &f.message,
            sequence: 0,
            role: Role::Assistant,
            written_by: WrittenBy::Model,
        },
        Source {
            predecessor: &f.head.token,
            revision: 1,
            evidence_revision: 1,
            logical_key: [3; 32],
            route: RouteKind::Native,
        },
        4096,
    )
}

#[test]
fn retained_native_session_and_literal_transport_spelling_are_distinct_after_reopen() {
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
fn preparation_refuses_invalid_native_encoding_and_changed_encoding_for_existing_session() {
    let encoded = NativeId::from_utf16(&[0xd800]);
    let mut f = Fixture::new();
    assert!(prepare_native(&mut f, &encoded, Some("unknown")).is_err());
    assert_eq!(f.scalar("SELECT count(*) FROM session"), 0);
    prepare_native(&mut f, &encoded, encoded.encoding()).unwrap();
    assert!(prepare_native(&mut f, &encoded, None).is_err());
    assert_eq!(f.scalar("SELECT count(*) FROM session"), 1);
}

#[test]
fn selected_session_refuses_corrupt_native_encoding_instead_of_claiming_literal_identity() {
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
