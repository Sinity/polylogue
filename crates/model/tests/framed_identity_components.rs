use polylogue_model::identity::{framed_identity_components, provisional_message_id};
use polylogue_model::{BlockId, MessageId, NativeId, Origin, SessionId};

#[test]
fn framed_references_expose_their_origin_session_and_message_parts() {
    let origin = Origin::new("open-product").unwrap();
    let local = NativeId::from_utf16(&[b'x' as u16, 0xd800]);
    let session = SessionId::from_native(&origin, &local).unwrap();
    assert_eq!(
        framed_identity_components(session.as_str()).unwrap(),
        vec![NativeId::unicode("open-product"), local.clone()]
    );
    let message =
        MessageId::from_native(session.clone(), &NativeId::unicode("~i1:not-a-frame")).unwrap();
    let expected = vec![
        NativeId::unicode("open-product"),
        local.clone(),
        NativeId::unicode("~i1:not-a-frame"),
    ];
    assert_eq!(
        framed_identity_components(message.as_str()).unwrap(),
        expected
    );
    assert_eq!(
        framed_identity_components(BlockId::new(message, 7).as_str()).unwrap(),
        expected
    );
    let provisional = provisional_message_id(session, b"{}", &[], 0).unwrap();
    assert_eq!(
        framed_identity_components(provisional.as_str()).unwrap(),
        vec![NativeId::unicode("open-product"), local]
    );
    for bad in [
        "ordinary",
        "claude:ordinary",
        "~i1:",
        "~i1:0101",
        "~i1:FFFF",
    ] {
        assert!(framed_identity_components(bad).is_none());
    }
    let mut malformed = provisional.as_str().to_string();
    malformed.push_str("00");
    assert!(framed_identity_components(&malformed).is_none());
}
