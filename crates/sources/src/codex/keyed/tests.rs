use super::*;

#[test]
fn timed_first_legacy_message_keeps_its_own_time_slot() {
    let memory = MemoryReadView::default();
    let origin = Origin::new("codex").unwrap();
    let session_id = SessionId::new(&origin, "synthetic-time").unwrap();
    let mut session = Session::new(
        session_id.clone(),
        origin,
        "synthetic-time",
        SessionKind::Main,
    )
    .unwrap();
    session.started_at = Some(crate::time::intrinsic(Timestamp(0)));
    let resume = Resume {
        session: Some(session),
        legacy: true,
        ..Default::default()
    };
    let mut parser = KeyedRecords::new(&memory, resume);
    let source_time = b"\"2026-09-02T10:00:01.000Z\"";
    parser.current_time_raw = Some(source_time.to_vec());
    parser.current_origins = Origins::Record(RecordRange::new(100, 200).unwrap());
    let mut message = Message::new(
        MessageId::new(session_id, "first").unwrap(),
        IdKind::Provider,
        Role::User,
        WrittenBy::Human,
    );
    message.at = Some(crate::time::intrinsic(
        crate::time::parse("2026-09-02T10:00:01.000Z").unwrap(),
    ));
    let expected = message.at;
    parser.push_message(message);
    let delta = parser.finish_delta().unwrap();
    let Some(Some(Cell::Header { message, .. })) = delta.cells.get(&Key::Message(0)) else {
        panic!("missing first Message");
    };
    assert_eq!(message.at, expected);
    let Some(Some(Cell::SlotValue(witness))) = delta.cells.get(&Key::Presence(0, "at".into()))
    else {
        panic!("missing first time slot");
    };
    assert_eq!(witness.canonical, source_time);
    assert_eq!(
        witness.origins,
        Origins::Located(super::super::selection::SelectedRaw {
            record: RecordRange::new(100, 200).unwrap(),
            path: vec![super::super::selection::PathPart::Key("timestamp".into())]
        })
    );
}
