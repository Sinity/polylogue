use super::*;
#[test]
fn an_unpublished_old_generation_is_rebuilt_before_the_next_activation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let fixture =
        polylogue_store::fixture::Fixture::new(&directory.path().join("archive")).unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let current = reader.change_position().unwrap();
    let pointer = completion::CatalogPointer::new(path.clone());
    let old = Metadata {
        archive: "synthetic".into(),
        position: ChangePosition {
            generation: "retired-generation".into(),
            index: 0,
            archive: 0,
        },
        written_at: 1,
    };
    let writer = pointer
        .build(&old, |writer| {
            writer.put_catalog(&super::super::base_catalog())?;
            writer.put_values(
                "old",
                &BTreeMap::from([(
                    "repo".into(),
                    BTreeMap::from([("old-scope-value".into(), 1)]),
                )]),
            )
        })
        .unwrap();
    let mut state = State::new(false, super::super::tests::request_metadata());
    // A build may finish before its activation is rejected by the scope fence.
    state.writer = Some((old.position.generation, writer));
    assert!(state.position.is_none());
    state.refresh(&reader, &path, "synthetic").unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.position, current);
    assert!(cache.values("repo", "").unwrap().is_empty());
}

#[test]
fn publication_uses_the_captured_clock_and_machine_scoped_setting() {
    use polylogue_model::{
        Author, WallClock,
        authorship::{Authorship, Channel},
        user::{RequestMetadata, UserWrite},
    };
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let instant: jiff::Timestamp = "2026-02-03T04:05:06Z".parse().unwrap();
    let requests = RequestMetadata::new("captured-machine".into(), WallClock::fixed(instant));
    for (id, machine, value) in [
        ("selected", "captured-machine", "3"),
        ("other", "different-machine", "9"),
    ] {
        archive
            .writer
            .user(requests.request(
                id.into(),
                UserWrite::SetSetting {
                    key: "signal.interrupts_per_hour".into(),
                    value: value.into(),
                    machine: Some(machine.into()),
                },
                Authorship {
                    author: Author::Operator,
                    via: Channel::Cli,
                },
            ))
            .unwrap();
    }
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let mut state = State::new(false, requests);
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.written_at, instant.as_millisecond());
    let settings = cache.settings("signal.interrupts_per_hour").unwrap();
    let [setting] = &settings[..] else {
        panic!("one setting")
    };
    let current = &setting.value;
    assert_eq!(
        current.value,
        Some(polylogue_model::settings::Value::InterruptBudget(3))
    );
    assert!(current.conflicts.is_empty());
}

#[test]
fn explicit_shutdown_reports_a_panicked_publisher_once() {
    let mut publisher = Publisher {
        stop: Arc::new(AtomicBool::new(false)),
        expected_exit: Arc::new(AtomicBool::new(false)),
        wake: None,
        thread: Some(std::thread::spawn(|| panic!("synthetic publisher failure"))),
    };
    let error = publisher.shutdown().unwrap_err();
    assert_eq!(error.to_string(), "completion worker panicked");
    assert!(publisher.thread.is_none());
    publisher.shutdown().unwrap();
}

#[test]
fn explicit_shutdown_accepts_only_a_recorded_expected_return() {
    for (expected, succeeds) in [(true, true), (false, false)] {
        let mut publisher = Publisher {
            stop: Arc::new(AtomicBool::new(false)),
            expected_exit: Arc::new(AtomicBool::new(expected)),
            wake: None,
            thread: Some(std::thread::spawn(|| {})),
        };
        assert_eq!(publisher.shutdown().is_ok(), succeeds);
        assert!(publisher.thread.is_none());
    }
}

#[test]
fn saved_metadata_follows_committed_families_and_late_exact_query_bodies() {
    use polylogue_model::{
        Author, Timestamp,
        assertion::{AssertionFact, AssertionKind},
        authorship::Channel,
        user::UserWrite,
    };
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut state = State::new(false, super::super::tests::request_metadata());
    // This is a genuine historical assertion whose frozen body arrives later.
    let plan = polylogue_query::compile(
        "sessions | group by origin | count",
        &polylogue_query::Options {
            clock: query_language::time::Clock::new(
                "2026-01-01T00:00:00Z".parse().unwrap(),
                query_language::time::DaySetting::new(
                    jiff::tz::TimeZone::get("UTC").unwrap(),
                    jiff::civil::time(4, 0, 0, 0),
                )
                .unwrap(),
            ),
            origins: vec!["codex".into()],
            page_size: None,
            random_seed: None,
            embeddings: false,
            self_session: None,
            saved: Default::default(),
            fields: vec![],
            kept: Default::default(),
            chat_pages: vec![],
        },
    )
    .unwrap();
    let child = plan.query_object().unwrap();
    let mut bindings = polylogue_query::Bindings {
        origins: vec!["codex".into()],
        ..Default::default()
    };
    bindings.saved.insert(
        "child".into(),
        vec![polylogue_query::SavedQuery {
            text: plan.text().into(),
            revision: "frozen-child".into(),
            root: "frozen-child".into(),
            hash: child.id,
            denominator: None,
            conflicts: vec![],
            body: polylogue_query::SavedQueryBody::Frozen {
                syntax: plan.text().into(),
                object: Some(std::sync::Arc::new(child.clone())),
            },
        }],
    );
    let parent = polylogue_query::plan::prepare("query:child", &bindings, |_| Ok(())).unwrap();
    let object = parent.query_object().unwrap();
    assert!(object.edges.iter().any(|edge| matches!(edge,polylogue_model::query_objects::QueryEdge::Saved{query} if *query==child.id)));

    archive.writer.restore_assertion(AssertionFact {
        id:"historical-definition".into(),root:"historical-definition".into(),kind:AssertionKind::SavedQuery,
        key:Some("historical".into()),value:Some(serde_json::json!({"text":parent.text(),"hash":object.id.to_string(),"denominator":null})),
        author:Author::Operator,via:Channel::Import,machine:"synthetic-machine".into(),created_at:Timestamp::from_unix_ms(1).unwrap(),valid_until:None,links:vec![],
    }).unwrap();
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cached = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert!(cached.saved("historical").unwrap()[0].schema.is_none());
    drop(cached);
    archive.writer.query_object(object.clone()).unwrap();
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert!(
        completion::CatalogReader::open(&path, "synthetic")
            .unwrap()
            .unwrap()
            .saved("historical")
            .unwrap()[0]
            .schema
            .is_some(),
        "the parent retains its own immutable dependency closure"
    );
    let before = state.position.clone().unwrap();
    archive.writer.query_object(child.clone()).unwrap();
    let reader = reads.reader().unwrap();
    let changed = reader.completion_changed_queries(before.archive).unwrap();
    assert!(changed.contains(&child.id.to_string()) && changed.contains(&object.id.to_string()));
    state.refresh(&reader, &path, "synthetic").unwrap();
    drop(reader);
    let cached = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    let saved = cached.saved("historical").unwrap();
    assert!(
        saved[0]
            .schema
            .as_ref()
            .unwrap()
            .fields
            .iter()
            .any(|field| field.name == "count")
    );
    drop(cached);
    archive.writer.restore_assertion(AssertionFact {
        id: "colliding-definition".into(), root: "colliding-definition".into(), kind: AssertionKind::SavedQuery,
        key: Some("historical".into()), value: Some(serde_json::json!({"text":parent.text(),"hash":object.id.to_string(),"denominator":null})),
        author: Author::Operator, via: Channel::Import, machine: "synthetic-machine".into(), created_at: Timestamp::from_unix_ms(2).unwrap(), valid_until: None, links: vec![],
    }).unwrap();
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    let collision = cache.saved("historical").unwrap();
    assert_eq!(collision.len(), 2);
    assert!(
        collision
            .iter()
            .all(|saved| saved.schema.is_none() && saved.help.contains("ambiguous"))
    );
    drop(cache);
    archive
        .writer
        .user(polylogue_store::fixture::authored::operator_request(
            "withdraw-collision".into(),
            UserWrite::WithdrawAssertions {
                targets: vec![
                    polylogue_model::Reference::parse("assertion:colliding-definition").unwrap(),
                ],
                reason: None,
            },
        ))
        .unwrap();
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    let recovered = cache.saved("historical").unwrap();
    assert_eq!(recovered.len(), 1);
    assert!(recovered[0].schema.is_some());
    drop(cache);
    archive
        .writer
        .user(polylogue_store::fixture::authored::operator_request(
            "withdraw-historical".into(),
            UserWrite::WithdrawAssertions {
                targets: vec![
                    polylogue_model::Reference::parse("assertion:historical-definition").unwrap(),
                ],
                reason: None,
            },
        ))
        .unwrap();
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert!(
        completion::CatalogReader::open(&path, "synthetic")
            .unwrap()
            .unwrap()
            .saved("historical")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn failed_catalog_transaction_retries_the_same_committed_definition_delta() {
    use polylogue_model::user::UserWrite;
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut state = State::new(false, super::super::tests::request_metadata());
    let save = |id: &str, name: &str| {
        archive
            .writer
            .user_preparing(
                polylogue_store::fixture::authored::operator_request(
                    id.into(),
                    UserWrite::SaveQuery {
                        name: name.into(),
                        text: "sessions | group by origin | count".into(),
                        denominator: None,
                    },
                ),
                crate::daemon::service::authored_preparation(),
                None,
            )
            .unwrap()
    };
    save("definition-a", "alpha");
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let before = state.position.clone().unwrap();
    polylogue_store::fault::arm(
        "completion.before_commit",
        &path.parent().unwrap().to_string_lossy(),
        polylogue_store::fault::Action::Fail,
        0,
        1,
    );
    save("definition-b", "beta");
    let error = state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap_err();
    assert!(
        matches!(&error,polylogue_store::StoreError::Invalid(message) if message.starts_with("injected fault at completion.before_commit")),
        "{error:?}"
    );
    assert_eq!(state.position, Some(before.clone()));
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.position, before);
    assert!(cache.saved("beta").unwrap().is_empty());
    assert_eq!(cache.saved("alpha").unwrap().len(), 1);
    drop(cache);
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert!(cache.metadata.position.archive > before.archive);
    assert_eq!(cache.saved("alpha").unwrap().len(), 1);
    assert!(cache.saved("beta").unwrap()[0].schema.is_some());
}

#[test]
fn late_kept_custody_refreshes_a_saved_inline_dependency_without_a_child_record() {
    use polylogue_model::user::UserWrite;
    let donor = polylogue_store::fixture::writer::WriterArchive::new();
    let rows = serde_json::json!({"unit":"sessions","shape":"rows","columns":[{"name":"title","kind":"text"}],"rows":[],"identities":[]}).to_string();
    let keep = || {
        polylogue_store::fixture::authored::operator_request(
            "late-kept".into(),
            UserWrite::KeepResult {
                name: None,
                query: "sessions".into(),
                rows_json: rows.clone(),
                read_at: "{}".into(),
            },
        )
    };
    donor
        .writer
        .user_preparing(keep(), crate::daemon::service::authored_preparation(), None)
        .unwrap();
    let donor_reads = polylogue_store::ReadPool::open(&donor.archive).unwrap();
    let mut bindings = polylogue_query::Bindings {
        kept: donor_reads
            .reader()
            .unwrap()
            .selection_kept_results()
            .unwrap(),
        ..Default::default()
    };
    let child = polylogue_query::plan::prepare("result:late-kept", &bindings, |_| Ok(())).unwrap();
    let child = child.query_object().unwrap();
    let child_id = child.id;
    bindings.saved.insert(
        "child".into(),
        vec![polylogue_query::SavedQuery {
            text: "result:late-kept".into(),
            root: "child".into(),
            revision: "child".into(),
            hash: child.id,
            denominator: None,
            conflicts: vec![],
            body: polylogue_query::SavedQueryBody::Frozen {
                syntax: "result:late-kept".into(),
                object: Some(std::sync::Arc::new(child)),
            },
        }],
    );
    let parent = polylogue_query::plan::prepare("query:child", &bindings, |_| Ok(())).unwrap();
    let object = parent.query_object().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut archive =
        polylogue_store::fixture::Fixture::new(&directory.path().join("archive")).unwrap();
    let writer = polylogue_store::Writer::start(&archive.archive).unwrap();
    let error = writer.query_object(object).unwrap_err();
    assert!(
        matches!(&error, polylogue_store::StoreError::NotFound(what) if what == "canonical result schema result:late-kept"),
        "{error:?}"
    );
    // A historical/partial read view can retain a formerly valid frozen query
    // whose kept dependency is unavailable. New admission above must refuse it.
    archive.saved_query("dependent", &parent).unwrap();
    // Account for the fixture's seeded source record before measuring deltas.
    writer
        .user_preparing(
            polylogue_store::fixture::authored::operator_request(
                "seed-position".into(),
                UserWrite::SetSetting {
                    key: "browser.actions".into(),
                    value: "on".into(),
                    machine: None,
                },
            ),
            crate::daemon::service::authored_preparation(),
            None,
        )
        .unwrap();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .query_object(&child_id.to_string(), None)
            .unwrap()
            .is_none()
    );
    let path = directory.path().join("catalog");
    let mut state = State::new(false, super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert!(
        completion::CatalogReader::open(&path, "synthetic")
            .unwrap()
            .unwrap()
            .saved("dependent")
            .unwrap()[0]
            .schema
            .is_none()
    );
    let before = state.position.clone().unwrap();
    writer
        .user_preparing(keep(), crate::daemon::service::authored_preparation(), None)
        .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_changed_queries(before.archive)
            .unwrap()
            .is_empty(),
        "kept custody must invalidate this family without another query admission"
    );
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert!(
        cache.saved("dependent").unwrap()[0]
            .schema
            .as_ref()
            .unwrap()
            .fields
            .iter()
            .any(|field| field.name == "title")
    );
}
