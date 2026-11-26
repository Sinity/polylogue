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
    let mut state = State::new(super::super::tests::request_metadata());
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
            .archive()
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
    let mut state = State::new(requests);
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
fn saved_metadata_follows_committed_families_and_late_query_bodies() {
    use polylogue_model::{
        Author, Timestamp,
        annotation::{AnnotationFact, AnnotationKind},
        authorship::Channel,
        user::UserWrite,
    };
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut state = State::new(super::super::tests::request_metadata());
    // This is a genuine historical annotation whose saved body arrives later.
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
            here: None,
            seen_marks_of: None,
            saved: Default::default(),
            fields: vec![],
            kept: Default::default(),
            provider_catalog: None,
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
            body: polylogue_query::SavedQueryBody::Available {
                syntax: plan.text().into(),
                object: Some(std::sync::Arc::new(child.clone())),
            },
        }],
    );
    let parent = polylogue_query::plan::prepare("query:child", &bindings, |_| Ok(())).unwrap();
    let object = parent.query_object().unwrap();
    assert!(object.edges.iter().any(|edge| matches!(edge,polylogue_model::query_objects::QueryEdge::Saved{query} if *query==child.id)));

    archive.writer.archive().restore_annotation(AnnotationFact {
        id:"historical-definition".into(),root:"historical-definition".into(),kind:AnnotationKind::SavedQuery,
        key:Some("historical".into()),value:Some(serde_json::json!({"text":parent.text(),"hash":object.id.to_string(),"denominator":null})),
        author:Author::Operator,via:Channel::Import,machine:"synthetic-machine".into(),created_at: Some(Timestamp::from_unix_ms(1).unwrap()),valid_until:None,links:vec![],
    }, None).unwrap();
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cached = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert!(cached.saved("historical").unwrap()[0].schema.is_none());
    drop(cached);
    drop(state);
    let mut state = State::new(super::super::tests::request_metadata());
    archive
        .writer
        .archive()
        .query_object(object.clone(), None)
        .unwrap();
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
        "the parent keeps its own immutable dependencies"
    );
    let before = state.position.clone().unwrap();
    archive
        .writer
        .archive()
        .query_object(child.clone(), None)
        .unwrap();
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
    archive.writer.archive().restore_annotation(AnnotationFact {
        id: "colliding-definition".into(), root: "colliding-definition".into(), kind: AnnotationKind::SavedQuery,
        key: Some("historical".into()), value: Some(serde_json::json!({"text":parent.text(),"hash":object.id.to_string(),"denominator":null})),
        author: Author::Operator, via: Channel::Import, machine: "synthetic-machine".into(), created_at: Some(Timestamp::from_unix_ms(2).unwrap()), valid_until: None, links: vec![],
    }, None).unwrap();
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
        .archive()
        .user(polylogue_store::fixture::authored::operator_request(
            "withdraw-collision".into(),
            UserWrite::WithdrawAnnotations {
                targets: vec![
                    polylogue_model::Reference::parse("annotation:colliding-definition").unwrap(),
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
        .archive()
        .user(polylogue_store::fixture::authored::operator_request(
            "withdraw-historical".into(),
            UserWrite::WithdrawAnnotations {
                targets: vec![
                    polylogue_model::Reference::parse("annotation:historical-definition").unwrap(),
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
    let mut state = State::new(super::super::tests::request_metadata());
    let save = |id: &str, name: &str| {
        archive
            .writer
            .archive()
            .user(polylogue_store::writer::UserSubmission {
                request: polylogue_store::fixture::authored::operator_request(
                    id.into(),
                    UserWrite::Save(Box::new(polylogue_model::user::Save {
                        name: Some(name.into()),
                        query: "sessions | group by origin | count".into(),
                        denominator: Some("sessions".into()),
                        rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
                        read_at: "{}".into(),
                        watch: None,
                        outcomes: None,
                    })),
                ),
                scope: None,
                preparation: Some(crate::daemon::service::authored_preparation()),
                parent: None,
            })
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
        matches!(&error,polylogue_store::StoreError::Io(io) if io.to_string().starts_with("injected fault at completion.before_commit")),
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
    drop(state);
    let mut state = State::new(super::super::tests::request_metadata());
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
fn late_kept_result_refreshes_inline_dependency() {
    use polylogue_model::user::UserWrite;
    let donor = polylogue_store::fixture::writer::WriterArchive::new();
    let rows = serde_json::json!({"unit":"sessions","shape":"rows","columns":[{"name":"title","kind":"text"}],"rows":[],"identities":[]}).to_string();
    let keep = || {
        polylogue_store::fixture::authored::operator_request(
            "late-kept".into(),
            UserWrite::Save(Box::new(polylogue_model::user::Save {
                name: None,
                query: "sessions".into(),
                rows_json: rows.clone(),
                read_at: "{}".into(),
                denominator: None,
                watch: None,
                outcomes: None,
            })),
        )
    };
    donor
        .writer
        .archive()
        .user(polylogue_store::writer::UserSubmission {
            request: keep(),
            scope: None,
            preparation: Some(crate::daemon::service::authored_preparation()),
            parent: None,
        })
        .unwrap();
    let donor_reads = polylogue_store::ReadPool::open(&donor.archive).unwrap();
    let mut bindings = polylogue_query::Bindings {
        kept: donor_reads
            .reader()
            .unwrap()
            .selection_definitions("result:late-kept", &[])
            .unwrap()
            .kept,
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
            body: polylogue_query::SavedQueryBody::Available {
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
    let error = writer.archive().query_object(object, None).unwrap_err();
    assert!(
        matches!(&error, polylogue_store::StoreError::NotFound(what) if what == "stored result schema result:late-kept"),
        "{error:?}"
    );
    // A historical/partial read view can keep a formerly valid saved query
    // whose kept dependency is unavailable. Saving it again, above, must be refused.
    archive.saved_query("dependent", &parent).unwrap();
    // Account for the fixture's seeded source record before measuring deltas.
    writer
        .archive()
        .user(polylogue_store::writer::UserSubmission {
            request: polylogue_store::fixture::authored::operator_request(
                "seed-position".into(),
                UserWrite::SetSetting {
                    key: "browser.actions".into(),
                    value: "on".into(),
                    machine: None,
                },
            ),
            scope: None,
            preparation: Some(crate::daemon::service::authored_preparation()),
            parent: None,
        })
        .unwrap();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .query_object(&child_id.to_string())
            .unwrap()
            .is_none()
    );
    let path = directory.path().join("catalog");
    let mut state = State::new(super::super::tests::request_metadata());
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
    drop(state);
    let mut state = State::new(super::super::tests::request_metadata());
    writer
        .archive()
        .user(polylogue_store::writer::UserSubmission {
            request: keep(),
            scope: None,
            preparation: Some(crate::daemon::service::authored_preparation()),
            parent: None,
        })
        .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_changed_queries(before.archive)
            .unwrap()
            .is_empty(),
        "the kept result must invalidate this family without another saved query"
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

/// A commit that changes nothing the catalog holds leaves the published file
/// as it was; the next change is published from where the last one left off.
#[test]
fn a_commit_without_catalog_changes_writes_nothing() {
    use polylogue_store::fixture::writer::{WriterArchive, prose};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut archive = WriterArchive::new();
    archive.write("first", prose("first", 1, 100, "The first prompt"));
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let mut state = State::new(super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let published = reads.reader().unwrap().change_position().unwrap();
    // A discovery walk commits to the index and moves its position.
    archive
        .writer
        .coverage(polylogue_model::capture::Coverage {
            version: 1,
            machine: "synthetic-machine".into(),
            observed_at: polylogue_model::Timestamp::from_unix_ms(1_700_000_000_000).unwrap(),
            roots: Vec::new(),
        })
        .unwrap();
    let moved = reads.reader().unwrap().change_position().unwrap();
    assert_ne!(moved, published);
    let files = |dir: &std::path::Path| {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), std::fs::read(entry.path()).ok())
            })
            .collect();
        files.sort();
        files
    };
    let before = files(directory.path());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert_eq!(files(directory.path()), before);
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.position, published);
    drop(cache);
    archive.write("second", prose("second", 1, 200, "Another prompt"));
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(
        cache.metadata.position,
        reads.reader().unwrap().change_position().unwrap()
    );
    for id in ["claude-code:first", "claude-code:second"] {
        assert!(cache.session_details(id).unwrap().is_some(), "{id}");
    }
}

#[test]
fn historical_drive_intent_is_removed_from_initial_and_incremental_completions() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let requests = super::super::tests::request_metadata();
    let set = |id: &str, value: &str| {
        archive
            .writer
            .archive()
            .user(requests.request(
                id.into(),
                polylogue_model::user::UserWrite::SetSetting {
                    key: "drive.ai-studio".into(),
                    value: value.into(),
                    machine: None,
                },
                polylogue_model::authorship::Authorship {
                    author: polylogue_model::Author::Operator,
                    via: polylogue_model::authorship::Channel::Cli,
                },
            ))
            .unwrap();
    };
    set("old-off", "off");
    let mut state = State::new(requests.clone());
    {
        let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
        let reader = reads.reader().unwrap();
        state.refresh(&reader, &path, "synthetic").unwrap();
        let cache = completion::CatalogReader::open(&path, "synthetic")
            .unwrap()
            .unwrap();
        assert!(cache.settings("drive.ai-studio").unwrap().is_empty());
        assert_eq!(cache.settings("drive.client_id").unwrap().len(), 1);
        // Simulate a cache written by the preceding build, then update the
        // retained authored intent through the normal replay owner.
        let old = completion::Setting {
            key: "drive.ai-studio".into(),
            value: reader
                .setting_current("drive.ai-studio", "synthetic-host")
                .unwrap(),
            choices: vec!["on".into(), "off".into()],
        };
        state
            .writer
            .as_mut()
            .unwrap()
            .1
            .update(&cache.metadata, |writer| writer.put_setting(&old))
            .unwrap();
    }
    assert_eq!(
        completion::CatalogReader::open(&path, "synthetic")
            .unwrap()
            .unwrap()
            .settings("drive.ai-studio")
            .unwrap()
            .len(),
        1
    );
    set("old-on", "on");
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert!(reader.change_position().unwrap().archive > state.position.as_ref().unwrap().archive);
    state.refresh(&reader, &path, "synthetic").unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert!(cache.settings("drive.ai-studio").unwrap().is_empty());
    assert_eq!(
        reader
            .setting_value("drive.ai-studio", "synthetic-host")
            .unwrap(),
        Some(polylogue_model::settings::Value::Boolean(true))
    );
}

#[test]
fn reopened_publication_skips_the_source_snapshot_and_keeps_offline_completion_current() {
    use polylogue_store::fixture::writer::{WriterArchive, prose};
    use std::os::unix::fs::MetadataExt;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut archive = WriterArchive::new();
    let mut messages = (1..=3)
        .flat_map(|n| {
            prose(
                "first",
                n,
                100 + i64::from(n),
                &format!("Retained prompt {n}"),
            )
        })
        .collect::<Vec<_>>();
    for message in &mut messages {
        message.model = Some("retained-model".into());
    }
    archive.write("first", messages);
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let mut state = State::new(super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let saved = state.position.clone().unwrap();
    let initial = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(initial.values("model", "retained").unwrap()[0].count, 3);
    drop(initial);
    drop(state);
    drop(reads);
    let target = std::fs::read_link(&path).unwrap();
    let inode = std::fs::metadata(&path).unwrap().ino();
    let before = std::fs::read(&path).unwrap();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    polylogue_store::fault::arm(
        "completion.full_snapshot",
        &saved.generation,
        polylogue_store::fault::Action::Fail,
        0,
        1,
    );
    let mut state = State::new(super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert_eq!(state.position, Some(saved));
    assert_eq!(std::fs::read_link(&path).unwrap(), target);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    // A failed source delta read can drop the taken writer before returning.
    // Its in-memory cursor cannot stand in for the committed catalog connection.
    state.writer = None;
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert_eq!(std::fs::read_link(&path).unwrap(), target);
    let complete = |text: &str| super::super::complete_file(&path, "synthetic", text, text.len());
    assert_eq!(
        complete("model:retained")
            .iter()
            .map(|candidate| candidate.value.as_str())
            .collect::<Vec<_>>(),
        ["retained-model"]
    );
    let mut messages = prose("second", 1, 200, "A newly committed prompt");
    messages[0].model = Some("new-model".into());
    archive.write("second", messages);
    // A restart behind the source uses the committed delta, not another snapshot.
    drop(state);
    let mut state = State::new(super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert_eq!(
        complete("model:new")
            .iter()
            .map(|candidate| candidate.value.as_str())
            .collect::<Vec<_>>(),
        ["new-model"]
    );
    assert_eq!(
        complete("model:retained")
            .iter()
            .map(|candidate| candidate.value.as_str())
            .collect::<Vec<_>>(),
        ["retained-model"]
    );
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.values("model", "retained").unwrap()[0].count, 3);
    assert_eq!(cache.values("model", "new").unwrap()[0].count, 1);
    // Prove the same armed source-read boundary rejects a real fresh publication.
    let error = State::new(super::super::tests::request_metadata())
        .refresh(
            &reads.reader().unwrap(),
            &directory.path().join("fresh"),
            "synthetic",
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("completion.full_snapshot"),
        "{error}"
    );
}

#[test]
fn damaged_or_missing_catalog_is_rebuilt_for_the_offline_consumer() {
    use polylogue_store::fixture::writer::{WriterArchive, prose};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut archive = WriterArchive::new();
    let mut messages = prose("source", 1, 100, "A source prompt");
    messages[0].model = Some("source-model".into());
    archive.write("source", messages);
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let refresh = || {
        State::new(super::super::tests::request_metadata())
            .refresh(&reads.reader().unwrap(), &path, "synthetic")
            .unwrap();
        let text = "model:source";
        assert!(
            super::super::complete_file(&path, "synthetic", text, text.len())
                .iter()
                .any(|candidate| candidate.value == "source-model")
        );
    };
    refresh();
    let first = std::fs::read_link(&path).unwrap();
    std::fs::write(&path, b"damaged derived catalog").unwrap();
    refresh();
    let second = std::fs::read_link(&path).unwrap();
    assert_ne!(first, second);
    let mut permissions = std::fs::metadata(&path).unwrap().permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&path, permissions).unwrap();
    refresh();
    let third = std::fs::read_link(&path).unwrap();
    assert_ne!(second, third);
    let database = std::fs::canonicalize(&path).unwrap();
    std::fs::remove_file(format!("{}-shm", database.display())).unwrap();
    refresh();
    let fourth = std::fs::read_link(&path).unwrap();
    assert_ne!(third, fourth);
    std::fs::remove_file(&path).unwrap();
    refresh();
    assert_ne!(fourth, std::fs::read_link(&path).unwrap());
}

#[test]
fn reopening_keeps_typed_result_bindings_for_new_saved_definitions() {
    use polylogue_model::user::{Save, UserWrite};
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let save = |id: &str, name: Option<&str>, query: &str| {
        archive.writer.archive().user(polylogue_store::writer::UserSubmission {
            request: polylogue_store::fixture::authored::operator_request(id.into(),
                UserWrite::Save(Box::new(Save {
                    name: name.map(str::to_owned), query: query.into(),
                    rows_json: r#"{"unit":"sessions","shape":"rows","columns":[{"name":"title","kind":"text"}],"rows":[],"identities":[]}"#.into(),
                    read_at: "{}".into(), denominator: None, watch: None, outcomes: None,
                }))),
            scope: None, preparation: Some(crate::daemon::service::authored_preparation()), parent: None,
        }).unwrap()
    };
    save("retained-kept", None, "sessions");
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut state = State::new(super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let binding = state.vocabulary.kept["retained-kept"].clone();
    drop(state);
    save("later-definition", Some("later"), "result:retained-kept");
    let mut state = State::new(super::super::tests::request_metadata());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert_eq!(state.vocabulary.kept["retained-kept"], binding);
    let text = "query:later | where ti";
    assert!(
        super::super::complete_file(&path, "synthetic", text, text.len())
            .iter()
            .any(|candidate| candidate.value == "title")
    );
}

#[test]
fn full_standalone_restart_retains_publication_and_applies_deltas_without_a_snapshot() {
    use std::os::unix::fs::MetadataExt;
    use std::time::{Duration, Instant};
    let mut daemon = crate::test_daemon::start("A synthetic prompt for completion restart");
    let path = daemon.daemon.completion_path.clone();
    let key = crate::runtime_dir::archive_key(&daemon.daemon.archive.root);
    let wait = |name: &str| {
        let text = format!("sessions where tag:{}", &name[..name.len() - 1]);
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            if super::super::complete_file(&path, &key, &text, text.len())
                .iter()
                .any(|candidate| candidate.value == name)
            {
                break;
            }
            assert!(Instant::now() < until, "completion never published {name}");
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    let tag = |daemon: &crate::daemon::service::Daemon, name: &str| {
        crate::test_answers::step(crate::test_daemon::call(
            daemon,
            "rows",
            serde_json::json!({
                "selection":"session:claude-code:s1", "steps":[{"verb":"tag", "name":name}]
            }),
        ));
    };
    tag(&daemon.daemon, "restart-before");
    wait("restart-before");
    let saved = completion::CatalogReader::open(&path, &key)
        .unwrap()
        .unwrap()
        .metadata
        .position;
    let target = std::fs::read_link(&path).unwrap();
    let inode = std::fs::metadata(&path).unwrap().ino();
    daemon.restart_after_stop(|| {
        polylogue_store::fault::arm(
            "completion.full_snapshot",
            &saved.generation,
            polylogue_store::fault::Action::Fail,
            0,
            1,
        );
    });
    assert_eq!(std::fs::read_link(&path).unwrap(), target);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    wait("restart-before");
    tag(&daemon.daemon, "restart-after");
    wait("restart-after");
    wait("restart-before");
    assert_eq!(std::fs::read_link(&path).unwrap(), target);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    let current = completion::CatalogReader::open(&path, &key)
        .unwrap()
        .unwrap()
        .metadata
        .position;
    assert_eq!(current.generation, saved.generation);
    assert!(current.index > saved.index || current.archive > saved.archive);
    // The untouched fault proves the real restarted worker never read all sessions.
    let error = State::new(daemon.daemon.config.requests.clone())
        .refresh(
            &daemon.daemon.reads.reader().unwrap(),
            &path.with_file_name("fresh"),
            &key,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("completion.full_snapshot"),
        "{error}"
    );
}
