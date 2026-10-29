use super::*;

#[test]
fn request_ids_are_uuid_v7() {
    let id = new_request_id();
    assert_eq!(id.len(), 36);
    assert_eq!(&id[14..15], "7");
    assert_ne!(id, new_request_id());
}

#[test]
fn search_requests_with_resolved_content_hashes_conflict_without_changing_data() {
    let w = crate::fixture::writer::WriterArchive::new();
    let kept = crate::user_writes::request_now(
        "kept".into(),
        UserWrite::KeepResult {
            name: Some("example".into()),
            query: "sessions".into(),
            rows_json: "[]".into(),
            read_at: "{\"now\":1}".into(),
        },
    );
    let saved = crate::user_writes::request_now(
        "saved".into(),
        UserWrite::AddNote {
            note: NewNote {
                kind: NoteKind::SavedQuery,
                key: Some("example".into()),
                value: Some("sessions".into()),
                author: Author::Operator,
                targets: vec![],
                relations: vec![],
                revises: None,
                valid_until: None,
            },
        },
    );
    for request in [&kept, &saved] {
        w.writer.user(request.clone()).unwrap();
        let hash = blake3::hash(&serde_json::to_vec(&request.write).unwrap());
        crate::archive::open_archive_writer(&w.archive)
            .unwrap()
            .execute(
                "UPDATE client_request SET content_hash=?1 WHERE id=?2",
                rusqlite::params![hash.as_bytes().as_slice(), request.id],
            )
            .unwrap();
    }
    let mut retry = saved.clone();
    let UserWrite::AddNote { note } = &saved.write else {
        panic!()
    };
    retry.write = UserWrite::SaveQuery { note: note.clone() };
    for request in [kept.clone(), retry] {
        assert!(matches!(
            w.writer.user(request),
            Err(StoreError::Conflict(_))
        ));
    }
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    for request in [kept, saved] {
        assert_eq!(reader.user_request(&request.id).unwrap(), Some(request));
    }
    assert_eq!(
        reader
            .query("SELECT rows_json FROM a.kept_result WHERE id='kept'", &[])
            .unwrap()[0]["rows_json"],
        "[]"
    );
    assert_eq!(
        reader.note("saved").unwrap().unwrap().value.as_deref(),
        Some("sessions")
    );
}
