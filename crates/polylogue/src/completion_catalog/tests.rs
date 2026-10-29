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
            unit: "tool_calls".into(),
            row_count: Some(3),
            rows: OutputSchema {
                unit: Some("tool_calls".into()),
                fields: vec![Field {
                    name: "count".into(),
                    kind: FieldKind::Number,
                    ..Default::default()
                }],
                main_time: None,
            },
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
    for text in ["codex:a", "abc", "ntrl"] {
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
    let mut state = publisher::State::new(None);
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let mut messages = prose("first", 1, 100, "The first native prompt");
    messages[0].model = Some("native-model-a".into());
    archive.write("first", messages);
    subscription.recv().unwrap();
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
