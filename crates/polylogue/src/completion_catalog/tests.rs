use super::*;
use polylogue_model::ChangePosition;
use polylogue_store::completion::{CatalogWriter, Kept, Metadata, Saved, Session};
use query_language::catalog::{Field, FieldKind, OutputSchema};
use std::collections::BTreeMap;
fn populated(path: &Path) {
    let metadata = Metadata {
        archive: "synthetic".into(),
        position: ChangePosition {
            generation: "g".into(),
            index: 3,
            archive: 2,
        },
        written_at: 100,
    };
    let catalog = polylogue_query::catalog::catalog(&["codex".into()], &["status".into()]);
    CatalogWriter::build(path, &metadata, |writer| {
        writer.put_catalog(&catalog)?;
        writer.put_session(&Session {
            id: "codex:abc123".into(),
            native_id: "abc123".into(),
            prefix: String::new(),
            title: "A neutral test title".into(),
            origin: "codex".into(),
            repo: Some("two words".into()),
            last_activity: Some(100),
            first_prompt: Some("A neutral prompt".into()),
            values: BTreeMap::from([
                ("repo".into(), BTreeMap::from([("two words".into(), 3)])),
                ("model".into(), BTreeMap::from([("alpha".into(), 2)])),
            ]),
        })?;
        writer.put_saved(&Saved {
            family: "saved-family".into(),
            revision: "saved-revision".into(),
            name: "by-action".into(),
            text: "tool_calls | group by action | count".into(),
            parameters: vec![],
            help: "Count native actions".into(),
            schema: Some(OutputSchema {
                unit: Some("tool_calls".into()),
                fields: vec![
                    Field {
                        name: "action".into(),
                        kind: FieldKind::Text,
                        ..Default::default()
                    },
                    Field {
                        name: "count".into(),
                        kind: FieldKind::Number,
                        ..Default::default()
                    },
                ],
                main_time: None,
            }),
        })?;
        writer.put_kept(&Kept {
            id: "kept-id".into(),
            name: Some("Kept calls".into()),
            unit: Some("tool_calls".into()),
            row_count: Some(3),
            rows: Some(OutputSchema {
                unit: Some("tool_calls".into()),
                fields: vec![Field {
                    name: "count".into(),
                    kind: FieldKind::Number,
                    ..Default::default()
                }],
                main_time: None,
            }),
            matches: Some(OutputSchema {
                unit: Some("tool_calls".into()),
                fields: vec![Field {
                    name: "tool".into(),
                    kind: FieldKind::Text,
                    ..Default::default()
                }],
                main_time: None,
            }),
        })
    })
    .unwrap();
}
fn rendered(path: &Path, text: &str) -> Vec<String> {
    complete_file(path, "synthetic", text, text.len())
        .into_iter()
        .map(|candidate| {
            format!(
                "{}{}{}",
                &text[..candidate.replace.start],
                candidate.value,
                &text[candidate.replace.end..]
            )
        })
        .collect()
}
#[test]
fn on_demand_values_remain_open_and_source_schemas_stay_scoped() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    populated(&path);
    assert!(rendered(&path, "repo:two").contains(&"repo:\"two words\"".into()));
    assert!(
        rendered(&path, "query:by-action | where co")
            .iter()
            .any(|value| value.ends_with("count"))
    );
    assert!(
        !rendered(&path, "query:by-action | where ti")
            .iter()
            .any(|value| value.ends_with("title"))
    );
    assert!(
        rendered(&path, "result:kept-id.matches | where to")
            .iter()
            .any(|value| value.ends_with("tool"))
    );
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert!(
        polylogue_query::parse::with_catalog("repo:unobserved", &reader.catalog().unwrap()).is_ok()
    );
}
#[test]
fn session_prefix_title_and_native_reference_completion_share_rich_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    populated(&path);
    for text in ["codex:a", "codex:abc", "session:codex:abc", "abc", "ntrl"] {
        let candidates = complete_file(&path, "synthetic", text, text.len());
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.help.contains("A neutral test title")
                    && candidate.help.contains("two words")),
            "{text}: {candidates:?}"
        );
    }
}
#[test]
fn quoted_middle_token_keeps_utf8_offsets_and_the_query_suffix() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    populated(&path);
    let text = "messages where repo:\"two old\" | count";
    let position = text.find(" old").unwrap();
    let candidate = complete_file(&path, "synthetic", text, position)
        .into_iter()
        .find(|candidate| candidate.value == "\"two words\"")
        .unwrap();
    assert_eq!(
        format!(
            "{}{}{}",
            &text[..candidate.replace.start],
            candidate.value,
            &text[candidate.replace.end..]
        ),
        "messages where repo:\"two words\" | count"
    );
    assert!(complete_file(&path, "synthetic", "repo:λ", 6).is_empty());
}
#[test]
fn obsolete_and_corrupt_catalogs_leave_grammar_without_writing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    for bytes in [b"PLCAT001".as_slice(), b"not a SQLite database".as_slice()] {
        std::fs::write(&path, bytes).unwrap();
        assert!(
            rendered(&path, "messages | gr")
                .iter()
                .any(|value| value.ends_with("group by"))
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}
#[test]
fn real_writer_deltas_publish_source_values_and_native_first_prompt() {
    use polylogue_store::fixture::writer::{WriterArchive, prose};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut archive = WriterArchive::new();
    let subscription = archive.writer.sender().unwrap().completion_commits();
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let mut state = publisher::State::new(None, false);
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let mut messages = prose("first", 1, 100, "The first native prompt");
    messages[0].model = Some("native-model-a".into());
    archive.write("first", messages);
    assert!(subscription.try_recv().is_ok());
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(
        reader.values("model", "native").unwrap()[0].value,
        "native-model-a"
    );
    assert_eq!(
        reader
            .session_details("claude-code:first")
            .unwrap()
            .unwrap()
            .first_prompt
            .as_deref(),
        Some("The first native prompt")
    );
    drop(reader);
    let mut messages = prose("second", 1, 200, "Another prompt");
    messages[0].model = Some("native-model-b".into());
    archive.write("second", messages);
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(reader.values("model", "native").unwrap().len(), 2);
    assert_eq!(
        reader.metadata.position,
        reads.reader().unwrap().change_position().unwrap()
    );
}

#[test]
fn completion_handler_uses_the_catalog_when_archive_reader_slots_are_closed() {
    let daemon = crate::test_daemon::start("A synthetic completion prompt");
    daemon.wait_for(|_| {
        CatalogReader::open(
            &daemon.daemon.completion_path,
            &crate::runtime_dir::archive_key(&daemon.daemon.archive.root),
        )
        .ok()
        .flatten()
        .is_some()
    });
    daemon.daemon.reads.close();
    let text = "tool_calls where outcome:err";
    let response = crate::daemon::service::handle(
        crate::daemon::Request::Complete {
            text: text.into(),
            position: text.len(),
        },
        &daemon.daemon,
    );
    let crate::daemon::Response::Ok { data, .. } = response else {
        panic!("{response:?}")
    };
    assert!(
        data.as_array()
            .unwrap()
            .iter()
            .any(|candidate| candidate["value"] == "error")
    );
}

#[test]
fn acknowledged_commits_refresh_catalog_values_while_keystrokes_continue() {
    let daemon = crate::test_daemon::start("A neutral completion prompt");
    let archive = crate::runtime_dir::archive_key(&daemon.daemon.archive.root);
    let path = &daemon.daemon.completion_path;
    daemon.set("briefing.budget", "2000");
    let read = || settings_file(path, &archive, Some("briefing.budget"), "");
    let wait = |expected: &str, require_nonempty: bool| {
        let start = std::time::Instant::now();
        loop {
            let values = read();
            if require_nonempty {
                assert!(
                    !values.is_empty(),
                    "a committed catalog lost its values during publication"
                );
            }
            if values.iter().any(|(value, _)| value == expected) {
                break;
            }
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "committed catalog did not refresh"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        start.elapsed()
    };
    wait("2000", false);
    let mut visibility = vec![];
    for number in 1..=5 {
        let value = (2000 + number).to_string();
        daemon.set("briefing.budget", &value);
        visibility.push(wait(&value, true));
        if number < 5 {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    eprintln!(
        "acknowledged committed setting to catalog visibility, five one-second batches: {visibility:?}"
    );
}

#[test]
fn unavailable_kept_schema_does_not_prevent_other_vocabulary_publication() {
    use polylogue_store::fixture::{Fixture, SessionFacts};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    fixture
        .session(SessionFacts {
            id: "codex:example",
            repo: Some("neutral-repository"),
            ..Default::default()
        })
        .unwrap();
    fixture
        .kept_result("invalid-json", "sessions", "not JSON")
        .unwrap();
    fixture
        .kept_result("missing-shape", "sessions", "{}")
        .unwrap();
    let reads = fixture.reads().unwrap();
    let mut publisher = publisher::State::new(None, false);
    publisher
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let reader = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(
        reader.values("repo", "neutral").unwrap()[0].value,
        "neutral-repository"
    );
    for id in ["invalid-json", "missing-shape"] {
        let kept = reader.kept(id).unwrap().unwrap();
        assert!(kept.rows.is_none() && kept.unit.is_none() && kept.row_count.is_none());
    }
}

#[test]
fn saved_and_kept_names_are_runnable_source_suggestions_at_the_first_word() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    populated(&path);
    assert!(rendered(&path, "by-").contains(&"query:by-action".into()));
    assert!(rendered(&path, "Kept").contains(&"result:kept-id".into()));
}

#[test]
fn primary_unknown_checkpoint_does_not_activate_an_empty_catalog() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let fixture =
        polylogue_store::fixture::Fixture::new(&directory.path().join("archive")).unwrap();
    let reads = fixture.reads().unwrap();
    let mut publisher = publisher::State::new(None, true);
    publisher
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    assert!(!path.exists());
}

#[test]
fn observation_literals_roundtrip_and_terminal_controls_never_enter_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let hostile = "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}";
    let metadata = Metadata {
        archive: "synthetic".into(),
        position: ChangePosition {
            generation: "g".into(),
            index: 1,
            archive: 1,
        },
        written_at: 100,
    };
    CatalogWriter::build(&path, &metadata, |writer| {
        writer.put_catalog(&base_catalog())?;
        writer.put_values(
            "observed",
            &BTreeMap::from([
                (
                    "repo".into(),
                    BTreeMap::from([("path\\part \"λ\"".into(), 1)]),
                ),
                (
                    "tag".into(),
                    BTreeMap::from([("ordinary".into(), 1), (hostile.into(), 1)]),
                ),
            ]),
        )?;
        writer.put_session(&Session {
            id: "claude-code:abc".into(),
            native_id: "abc".into(),
            prefix: String::new(),
            title: hostile.into(),
            origin: "claude-code".into(),
            repo: None,
            last_activity: Some(100),
            first_prompt: None,
            values: BTreeMap::new(),
        })
    })
    .unwrap();
    let text = rendered(&path, "repo:path").into_iter().next().unwrap();
    let parsed = polylogue_query::parse(&text).unwrap();
    let polylogue_query::ast::Source::Unit {
        condition: Some(condition),
        ..
    } = parsed.source
    else {
        panic!()
    };
    let polylogue_query::ast::ConditionKind::Match {
        value: polylogue_query::ast::Value::Scalar { text, .. },
        ..
    } = condition.kind
    else {
        panic!()
    };
    assert_eq!(text, "path\\part \"λ\"");
    let candidates = complete_file(&path, "synthetic", "tag:", 4);
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.value == "ordinary")
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| !candidate.value.chars().any(char::is_control))
    );
    let candidates = complete_file(&path, "synthetic", "claude-code:", 12);
    assert!(candidates.iter().any(|candidate| candidate.value == "a"));
    assert!(
        candidates
            .iter()
            .all(|candidate| !candidate.help.contains('\x1b')
                && !candidate.help.contains('\u{202e}'))
    );
}
