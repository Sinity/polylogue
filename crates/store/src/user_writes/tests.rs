use super::*;
use polylogue_model::Author;

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

#[test]
fn typed_setting_catalog_refuses_bad_writes_without_recording_a_request() {
    let w = crate::fixture::writer::WriterArchive::new();
    for (id, key, value) in [
        ("unknown", "day.timezone", "UTC"),
        ("blank", "browser.actions", " "),
        ("bad-type", "signal.interrupts_per_hour", "many"),
    ] {
        let request = UserRequest {
            id: id.into(),
            machine: "synthetic-machine".into(),
            at: Timestamp::from_unix_ms(1).unwrap(),
            write: UserWrite::SetSetting {
                key: key.into(),
                value: value.into(),
                author: Author::Operator,
                machine: None,
            },
        };
        assert!(matches!(
            w.writer.user(request),
            Err(StoreError::Invalid(_))
        ));
        let reads = crate::ReadPool::open(&w.archive).unwrap();
        assert!(reads.reader().unwrap().user_request(id).unwrap().is_none());
    }
}
#[test]
fn typed_setting_answer_and_snapshot_share_defaults_scope_and_authored_value() {
    let w = crate::fixture::writer::WriterArchive::new();
    let request = UserRequest {
        id: "setting-on".into(),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(123).unwrap(),
        write: UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            author: Author::Operator,
            machine: None,
        },
    };
    w.writer.user(request).unwrap();
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    let setting = reader
        .setting_current("browser.actions", "synthetic-machine")
        .unwrap();
    assert_eq!(
        setting.value,
        Some(polylogue_model::settings::Value::Boolean(true))
    );
    assert_eq!(
        setting.default,
        Some(polylogue_model::settings::Value::Boolean(false))
    );
    assert_eq!(setting.authored.unwrap().id, "setting-on");
    let snapshot = reader.settings_snapshot("synthetic-machine").unwrap();
    assert!(snapshot.browser_actions().unwrap());
    assert!(snapshot.reading_variants().unwrap().is_empty());
    assert_eq!(snapshot.standalone_attention().unwrap(), (None, 4));
}

#[test]
fn host_settings_default_to_the_accepting_machine_and_do_not_leak_to_other_hosts() {
    let w = crate::fixture::writer::WriterArchive::new();
    let request = UserRequest {
        id: "host-attention".into(),
        machine: "host-a".into(),
        at: Timestamp::from_unix_ms(123).unwrap(),
        write: UserWrite::SetSetting {
            key: "signal.interrupts_per_hour".into(),
            value: "7".into(),
            author: Author::Operator,
            machine: None,
        },
    };
    w.writer.user(request).unwrap();
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .setting_value("signal.interrupts_per_hour", "host-a")
            .unwrap(),
        Some(polylogue_model::settings::Value::InterruptBudget(7))
    );
    assert_eq!(
        reader
            .setting_value("signal.interrupts_per_hour", "host-b")
            .unwrap(),
        Some(polylogue_model::settings::Value::InterruptBudget(4))
    );
    assert_eq!(
        reader
            .setting_current("signal.interrupts_per_hour", "host-a")
            .unwrap()
            .authored
            .unwrap()
            .machine,
        "host-a"
    );
}
