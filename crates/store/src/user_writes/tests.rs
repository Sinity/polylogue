use super::*;
use polylogue_model::Timestamp;

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
    let kept = crate::fixture::authored::operator_request(
        "kept".into(),
        UserWrite::KeepResult {
            name: Some("example".into()),
            query: "sessions".into(),
            rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
            read_at: "{\"now\":1}".into(),
        },
    );
    let saved = crate::fixture::authored::operator_request(
        "saved".into(),
        UserWrite::SaveQuery {
            name: "example".into(),
            text: "sessions".into(),
            denominator: None,
        },
    );
    for request in [&kept, &saved] {
        w.writer
            .user_preparing(
                request.clone(),
                PreparationContext::new(vec!["codex".into()]),
                None,
            )
            .unwrap();
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
    if let UserWrite::SaveQuery { text, .. } = &mut retry.write {
        *text = "messages".into();
    }
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
    let stored = reader
        .query("SELECT rows_json FROM a.kept_result WHERE id='kept'", &[])
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_str(stored[0]["rows_json"].as_str().unwrap()).unwrap();
    assert_eq!(
        body,
        serde_json::json!({"unit":"sessions","columns":[],"rows":[],})
    );
    assert_eq!(
        reader
            .assertion(&polylogue_model::RevisionReference::exact(
                "saved".into(),
                "saved".into()
            ))
            .unwrap()
            .unwrap()
            .fact
            .value
            .as_ref()
            .and_then(|value| value.get("text"))
            .and_then(serde_json::Value::as_str),
        Some("sessions")
    );
}

#[test]
fn a_kept_body_later_preparations_cannot_decode_is_refused_without_recording_a_request() {
    let w = crate::fixture::writer::WriterArchive::new();
    for (id, rows_json, field) in [
        ("rows-array", "[]", "rows_json"),
        ("not-json", "not JSON", "rows_json"),
        (
            "bad-columns",
            "{\"unit\":\"sessions\",\"columns\":{}}",
            "columns",
        ),
    ] {
        let request = crate::fixture::authored::operator_request(
            id.into(),
            UserWrite::KeepResult {
                name: None,
                query: "sessions".into(),
                rows_json: rows_json.into(),
                read_at: "{\"now\":1}".into(),
            },
        );
        let prefix = format!("kept result {id:?} field {field}: ");
        assert!(
            matches!(
                w.writer.user(request),
                Err(StoreError::Invalid(message)) if message.starts_with(&prefix)
            ),
            "{id}"
        );
        let reads = crate::ReadPool::open(&w.archive).unwrap();
        let reader = reads.reader().unwrap();
        assert!(reader.user_request(id).unwrap().is_none());
        assert_eq!(
            reader
                .query("SELECT count(*) AS n FROM a.kept_result", &[])
                .unwrap()[0]["n"],
            0
        );
    }
    let saved = crate::fixture::authored::operator_request(
        "saved-after-refusal".into(),
        UserWrite::SaveQuery {
            name: "after".into(),
            text: "sessions".into(),
            denominator: None,
        },
    );
    w.writer
        .user_preparing(saved, PreparationContext::new(vec!["codex".into()]), None)
        .unwrap();
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
            authorship: polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
            id: id.into(),
            machine: "synthetic-machine".into(),
            at: Timestamp::from_unix_ms(1).unwrap(),
            write: UserWrite::SetSetting {
                key: key.into(),
                value: value.into(),
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
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: "setting-on".into(),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(123).unwrap(),
        write: UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
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
    assert_eq!(
        setting.authored.unwrap().reference,
        polylogue_model::Reference::parse("assertion:setting-on@setting-on").unwrap()
    );
    let snapshot = reader.settings_snapshot("synthetic-machine").unwrap();
    assert!(snapshot.browser_actions().unwrap());
    assert!(snapshot.reading_variants().unwrap().is_empty());
    assert_eq!(snapshot.standalone_attention().unwrap(), (None, 4));
}

#[test]
fn host_settings_default_to_the_accepting_machine_and_do_not_leak_to_other_hosts() {
    let w = crate::fixture::writer::WriterArchive::new();
    let request = UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: "host-attention".into(),
        machine: "host-a".into(),
        at: Timestamp::from_unix_ms(123).unwrap(),
        write: UserWrite::SetSetting {
            key: "signal.interrupts_per_hour".into(),
            value: "7".into(),
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

#[test]
fn assertion_retry_ignores_object_key_order_and_preserves_array_and_value_identity() {
    let w = crate::fixture::writer::WriterArchive::new();
    let statement = |value| polylogue_model::assertion::NewAssertion {
        kind: polylogue_model::assertion::AssertionKind::Decision,
        key: None,
        value: Some(value),
        valid_until: None,
        links: vec![polylogue_model::assertion::NewAssertionLink {
            role: polylogue_model::assertion::LinkRole::About,
            reference: polylogue_model::Reference::parse("repo:synthetic").unwrap(),
        }],
    };
    let original: serde_json::Value =
        serde_json::from_str(r#"{"text":"choose A","constraints":["one","two"]}"#).unwrap();
    let reordered: serde_json::Value =
        serde_json::from_str(r#"{"constraints":["one","two"],"text":"choose A"}"#).unwrap();
    let request = crate::fixture::authored::operator_request(
        "object-order".into(),
        UserWrite::Assert {
            assertions: vec![statement(original.clone())],
        },
    );
    assert!(!w.writer.user(request.clone()).unwrap().repeat);
    let mut retry = request.clone();
    retry.write = UserWrite::Assert {
        assertions: vec![statement(reordered)],
    };
    assert!(w.writer.user(retry).unwrap().repeat);
    for value in [
        serde_json::json!({"text":"choose A","constraints":["two","one"]}),
        serde_json::json!({"text":"choose B","constraints":["one","two"]}),
    ] {
        let mut changed = request.clone();
        changed.write = UserWrite::Assert {
            assertions: vec![statement(value)],
        };
        assert!(matches!(
            w.writer.user(changed),
            Err(StoreError::Conflict(_))
        ));
    }
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let fact = reads
        .reader()
        .unwrap()
        .assertion(&polylogue_model::RevisionReference::exact(
            "object-order".into(),
            "object-order".into(),
        ))
        .unwrap()
        .unwrap()
        .fact;
    assert_eq!(fact.value, Some(original));
}
