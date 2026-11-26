use super::*;
use polylogue_model::Timestamp;

#[test]
fn request_ids_are_uuid_v7() {
    let id = new_request_id();
    let parsed = uuid::Uuid::parse_str(&id).unwrap();
    assert_eq!(parsed.get_version_num(), 7);
    assert_eq!(parsed.get_variant(), uuid::Variant::RFC4122);
    assert_eq!(id.len(), 36);
    assert_eq!(id.matches('-').count(), 4);
    assert_ne!(id, new_request_id());
}

#[test]
fn search_requests_with_resolved_content_hashes_conflict_without_changing_data() {
    let w = crate::fixture::writer::WriterArchive::new();
    let kept = crate::fixture::authored::operator_request(
        "kept".into(),
        UserWrite::Save(Box::new(polylogue_model::user::Save {
            name: Some("example".into()),
            query: "sessions".into(),
            rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
            read_at: "{\"now\":1}".into(),
            denominator: None,
            watch: None,
            outcomes: None,
        })),
    );
    let saved = crate::fixture::authored::operator_request(
        "saved".into(),
        UserWrite::Save(Box::new(polylogue_model::user::Save {
            name: Some("example".into()),
            query: "sessions".into(),
            denominator: None,
            rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
            read_at: "{}".into(),
            watch: None,
            outcomes: None,
        })),
    );
    for request in [&kept, &saved] {
        w.writer
            .archive()
            .user(crate::writer::UserSubmission {
                request: request.clone(),
                scope: None,
                preparation: Some(PreparationContext::new(vec!["codex".into()])),
                parent: None,
            })
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
    if let UserWrite::Save(save) = &mut retry.write {
        save.query = "messages".into();
    }
    for request in [kept.clone(), retry] {
        assert!(matches!(
            w.writer.archive().user(request),
            Err(StoreError::Conflict(_))
        ));
    }
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    for request in [kept, saved] {
        let mut recorded = reader.user_request(&request.id).unwrap().unwrap();
        if let UserWrite::Save(save) = &mut recorded.write {
            save.outcomes = None;
        }
        assert_eq!(recorded, request);
    }
    let stored = reader
        .query("SELECT rows_json FROM a.saved_rows WHERE id='kept'", &[])
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_str(stored[0]["rows_json"].as_str().unwrap()).unwrap();
    assert_eq!(
        body,
        serde_json::json!({"unit":"sessions","columns":[],"rows":[],})
    );
    assert_eq!(
        reader
            .annotation(&polylogue_model::RevisionReference::exact(
                "kept".into(),
                "kept".into()
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
            UserWrite::Save(Box::new(polylogue_model::user::Save {
                name: None,
                query: "sessions".into(),
                rows_json: rows_json.into(),
                read_at: "{\"now\":1}".into(),
                denominator: None,
                watch: None,
                outcomes: None,
            })),
        );
        assert!(
            matches!(
                w.writer.archive().user(request),
                Err(StoreError::Refused(polylogue_model::Reason::BadKeptResult { id: refused, field: named, .. })) if refused == id && named == field
            ),
            "{id}"
        );
        let reads = crate::ReadPool::open(&w.archive).unwrap();
        let reader = reads.reader().unwrap();
        assert!(reader.user_request(id).unwrap().is_none());
        assert_eq!(
            reader
                .query("SELECT count(*) AS n FROM a.saved_rows", &[])
                .unwrap()[0]["n"],
            0
        );
    }
    let saved = crate::fixture::authored::operator_request(
        "saved-after-refusal".into(),
        UserWrite::Save(Box::new(polylogue_model::user::Save {
            name: Some("after".into()),
            query: "sessions".into(),
            denominator: None,
            rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
            read_at: "{}".into(),
            watch: None,
            outcomes: None,
        })),
    );
    w.writer
        .archive()
        .user(crate::writer::UserSubmission {
            request: saved,
            scope: None,
            preparation: Some(PreparationContext::new(vec!["codex".into()])),
            parent: None,
        })
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
            w.writer.archive().user(request),
            Err(StoreError::Refused(_))
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
    w.writer.archive().user(request).unwrap();
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
        polylogue_model::Reference::parse("annotation:setting-on@setting-on").unwrap()
    );
    let snapshot = reader.settings_snapshot("synthetic-machine").unwrap();
    assert!(snapshot.browser_actions().unwrap());
    assert!(snapshot.reading_alternates().unwrap().is_empty());
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
    w.writer.archive().user(request).unwrap();
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
fn annotation_retry_ignores_object_key_order_and_preserves_array_and_value_identity() {
    let w = crate::fixture::writer::WriterArchive::new();
    let statement = |value| polylogue_model::annotation::NewAnnotation {
        kind: polylogue_model::annotation::AnnotationKind::Decision,
        key: None,
        value: Some(value),
        valid_until: None,
        links: vec![polylogue_model::annotation::NewAnnotationLink {
            role: polylogue_model::annotation::LinkRole::About,
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
            annotations: vec![statement(original.clone())],
        },
    );
    assert!(!w.writer.archive().user(request.clone()).unwrap().repeat);
    let mut retry = request.clone();
    retry.write = UserWrite::Assert {
        annotations: vec![statement(reordered)],
    };
    assert!(w.writer.archive().user(retry).unwrap().repeat);
    for value in [
        serde_json::json!({"text":"choose A","constraints":["two","one"]}),
        serde_json::json!({"text":"choose B","constraints":["one","two"]}),
    ] {
        let mut changed = request.clone();
        changed.write = UserWrite::Assert {
            annotations: vec![statement(value)],
        };
        assert!(matches!(
            w.writer.archive().user(changed),
            Err(StoreError::Conflict(_))
        ));
    }
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let fact = reads
        .reader()
        .unwrap()
        .annotation(&polylogue_model::RevisionReference::exact(
            "object-order".into(),
            "object-order".into(),
        ))
        .unwrap()
        .unwrap()
        .fact;
    assert_eq!(fact.value, Some(original));
}

#[test]
fn save_selection_rolls_back_every_effect_when_the_kept_insert_fails() {
    let w = crate::fixture::writer::WriterArchive::new();
    let conn = crate::archive::open_archive_writer(&w.archive).unwrap();
    conn.execute_batch("CREATE TRIGGER reject_kept BEFORE INSERT ON saved_rows BEGIN SELECT RAISE(ABORT,'synthetic kept insert failure'); END;").unwrap();
    let request = crate::fixture::authored::operator_request(
        "atomic-save".into(),
        UserWrite::Save(Box::new(polylogue_model::user::Save {
            outcomes: Some(polylogue_model::user::SaveReport {
                saved: None,
                request: "forged-request".into(),
                repeat: true,
                pending: true,
                kept: polylogue_model::user::KeptPart {
                    result: "result:caller-cannot-choose-this".into(),
                    rows: 0,
                    at: Timestamp::from_unix_ms(1).unwrap(),
                },
                watch: None,
            }),
            name: Some("atomic-example".into()),
            query: "sessions".into(),
            denominator: None,
            rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
            read_at: r#"{"now":1}"#.into(),
            watch: Some(polylogue_model::user::SaveWatch {
                hold: Some("20m".into()),
                notify: false,
            }),
        })),
    );
    let submitted = || {
        w.writer
            .archive()
            .submit(
                crate::writer::UserSubmission {
                    request: request.clone(),
                    scope: None,
                    preparation: Some(PreparationContext::new(vec!["codex".into()])),
                    parent: None,
                },
                None,
            )
            .map(|committed| committed.outcome)
    };
    let error = submitted().unwrap_err();
    assert!(
        error.to_string().contains("synthetic kept insert failure"),
        "{error:?}"
    );
    for table in ["annotation", "saved_rows", "client_request"] {
        let count: i64 = conn
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0, "{table}");
    }
    conn.execute_batch("DROP TRIGGER reject_kept").unwrap();
    let first = submitted().unwrap();
    assert!(!first.outcome.repeat);
    let repeated = submitted().unwrap();
    assert!(repeated.outcome.repeat);
    assert_eq!(first.save, repeated.save);
    assert_eq!(
        first.save.as_ref().unwrap().kept.result,
        "result:atomic-save"
    );
    assert_eq!(
        first.save.as_ref().unwrap().saved.as_ref().unwrap().id,
        "atomic-save.1"
    );
    assert_eq!(first.outcome.id, "atomic-save");
    let ids: Vec<String> = conn
        .prepare("SELECT id FROM annotation ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(ids, ["atomic-save.1", "atomic-save.2"]);
    let kept: String = conn
        .query_row("SELECT id FROM saved_rows", [], |r| r.get(0))
        .unwrap();
    assert_eq!(kept, "atomic-save");
    let journal: i64 = conn
        .query_row("SELECT count(*) FROM client_request", [], |r| r.get(0))
        .unwrap();
    assert_eq!(journal, 1, "one Save step has one receipt");
}

#[test]
fn authored_text_links_live_in_the_effect_without_rewriting_the_request() {
    use polylogue_model::annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink};
    use polylogue_model::{Reference, RevisionReference};
    let archive = crate::fixture::writer::WriterArchive::new();
    let text = "Read [the source](https://example.test/source).";
    let request = crate::fixture::authored::operator_request(
        "live-text-note".into(),
        UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Note,
                key: None,
                value: Some(serde_json::json!(text)),
                valid_until: None,
                links: vec![NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::parse("session:codex:fixture").unwrap(),
                }],
            }],
        },
    );
    let first = archive.writer.archive().user(request.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let before_repeat = pool.reader().unwrap().change_position().unwrap();
    assert!(
        archive
            .writer
            .archive()
            .user(request.clone())
            .unwrap()
            .repeat
    );
    let reader = pool.reader().unwrap();
    assert_eq!(reader.change_position().unwrap(), before_repeat);
    assert_eq!(reader.user_request(&request.id).unwrap().unwrap(), request);
    let stored = reader
        .annotation(&RevisionReference {
            root: first.id.clone(),
            revision: Some(first.id),
        })
        .unwrap()
        .unwrap()
        .fact;
    assert_eq!(stored.value, Some(serde_json::json!(text)));
    assert_eq!(
        stored
            .links
            .iter()
            .map(|link| (link.role, link.reference.to_string()))
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
            (LinkRole::About, "session:codex:fixture".into()),
            (LinkRole::Mentions, "url:https://example.test/source".into()),
        ])
    );
    let mut prepared = vec![];
    reader
        .visit_authored(&mut |record| {
            prepared.push(record.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].request, request);
    assert_eq!(prepared[0].effect.annotations, vec![stored]);
}

#[test]
fn tag_receipt_reports_original_skipped_targets_on_retry() {
    use polylogue_model::{
        Reference,
        annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink},
    };
    let world = crate::fixture::writer::WriterArchive::new();
    let first = Reference::parse("session:codex:first").unwrap();
    let second = Reference::parse("session:codex:second").unwrap();
    let mut tag = NewAnnotation {
        kind: AnnotationKind::Tag,
        key: Some("status".into()),
        value: Some(serde_json::json!({"value":"ready"})),
        valid_until: None,
        links: vec![NewAnnotationLink {
            role: LinkRole::About,
            reference: first.clone(),
        }],
    };
    let seed = world
        .writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "seed-tag".into(),
            UserWrite::Assert {
                annotations: vec![tag.clone()],
            },
        ))
        .unwrap();
    tag.links.clear();
    let request = crate::fixture::authored::operator_request(
        "bulk-tags".into(),
        UserWrite::AssertQuery {
            query: "sessions".into(),
            annotation: tag,
            resolved: vec![first.clone(), second.clone()],
        },
    );
    let accepted = world
        .writer
        .archive()
        .submit(request.clone(), None)
        .map(|committed| committed.outcome)
        .unwrap();
    assert_eq!(accepted.written.len(), 1);
    assert_eq!(accepted.already, vec![first.clone()]);
    world
        .writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            "withdraw-seed".into(),
            UserWrite::WithdrawAnnotations {
                targets: vec![Reference::Annotation(
                    polylogue_model::RevisionReference::exact(seed.id.clone(), seed.id),
                )],
                reason: None,
            },
        ))
        .unwrap();
    let repeated = world
        .writer
        .archive()
        .submit(request.clone(), None)
        .map(|committed| committed.outcome)
        .unwrap();
    assert!(repeated.outcome.repeat);
    assert_eq!(repeated.written, accepted.written);
    assert_eq!(repeated.already, vec![first]);
    let mut changed = request;
    let UserWrite::AssertQuery { annotation, .. } = &mut changed.write else {
        unreachable!()
    };
    annotation.value = Some(serde_json::json!({"value":"different"}));
    assert!(matches!(
        world
            .writer
            .archive()
            .submit(changed, None)
            .map(|committed| committed.outcome),
        Err(StoreError::Conflict(_))
    ));
}
