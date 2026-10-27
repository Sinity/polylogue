use super::*;

#[test]
fn request_ids_are_uuid_v7() {
    let id = new_request_id();
    assert_eq!(id.len(), 36);
    assert_eq!(&id[14..15], "7");
    assert_ne!(id, new_request_id());
}

#[test]
fn repeated_search_writes_keep_the_first_resolved_rows_and_revision() {
    use crate::fixture::writer::WriterArchive;
    let w = WriterArchive::new();
    let kept = |rows: &str| UserWrite::KeepResult {
        name: Some("example".into()),
        query: "sessions".into(),
        rows_json: rows.into(),
        read_at: rows.into(),
    };
    assert!(
        !w.writer
            .user(UserRequest::now("keep".into(), kept("first")))
            .unwrap()
            .repeat
    );
    assert!(
        w.writer
            .user(UserRequest::now("keep".into(), kept("second")))
            .unwrap()
            .repeat
    );
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let request = reader.user_request("keep").unwrap().unwrap();
    let UserWrite::KeepResult { rows_json, .. } = request.write else {
        panic!()
    };
    assert_eq!(rows_json, "first");
    drop(reader);
    let changed = UserWrite::KeepResult {
        name: None,
        query: "messages".into(),
        rows_json: "first".into(),
        read_at: "first".into(),
    };
    assert!(matches!(
        w.writer.user(UserRequest::now("keep".into(), changed)),
        Err(StoreError::Conflict(_))
    ));
    let note = |revises| UserWrite::SaveQuery {
        note: NewNote {
            kind: NoteKind::SavedQuery,
            key: Some("example".into()),
            value: Some("sessions".into()),
            author: Author::Operator,
            targets: vec![],
            relations: vec![],
            revises,
            valid_until: None,
        },
    };
    assert!(
        !w.writer
            .user(UserRequest::now("save".into(), note(None)))
            .unwrap()
            .repeat
    );
    assert!(
        w.writer
            .user(UserRequest::now("save".into(), note(Some("save".into()))))
            .unwrap()
            .repeat
    );
    let UserWrite::SaveQuery { note: explicit } = note(None) else {
        panic!()
    };
    w.writer
        .user(UserRequest::now("explicit".into(), UserWrite::AddNote { note: explicit.clone() }))
        .unwrap();
    let mut changed = explicit;
    changed.revises = Some("save".into());
    assert!(matches!(
        w.writer.user(UserRequest::now("explicit".into(), UserWrite::AddNote { note: changed })),
        Err(StoreError::Conflict(_))
    ));

    w.writer
        .user(UserRequest::now("revision".into(), note(Some("save".into()))))
        .unwrap();
    assert!(w.writer
        .user(UserRequest::now("revision".into(), note(Some("explicit".into()))))
        .unwrap()
        .repeat);
    let reader = pool.reader().unwrap();
    let saved = reader.note("revision").unwrap().unwrap();
    let requests = ["save", "revision"].map(|id| reader.user_request(id).unwrap().unwrap());
    let replay = WriterArchive::new();
    for request in requests {
        replay.writer.user(request).unwrap();
    }
    let replay_reads = crate::ReadPool::open(&replay.archive).unwrap();
    assert_eq!(replay_reads.reader().unwrap().note("revision").unwrap().unwrap(), saved);
}
