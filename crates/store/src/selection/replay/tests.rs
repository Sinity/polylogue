use super::*;
use crate::fixture::Fixture;
use query_language::cursor::Continuation;

fn options() -> Options {
    Options {
        clock: Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["test".into()],
        command_words: vec![],
        page_size: Some(1),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    }
}

#[test]
fn semantic_preparation_does_not_certify_replay_custody() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader
        .prepare_query_selection("sessions", &options())
        .unwrap();
    assert_eq!(
        selection
            .authenticated_execution(&reader)
            .err()
            .unwrap()
            .kind(),
        "not_yet_available"
    );
}

fn acknowledged_execution() -> (
    tempfile::TempDir,
    crate::ArchiveLock,
    crate::ReadPool,
    Options,
    polylogue_model::query_objects::QueryRun,
    Continuation,
) {
    use polylogue_model::{
        authorship::{CallerEvidence, Channel, compute_authorship},
        query_objects::{ExecutionPresentation, QueryRun},
    };
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    let writer = crate::Writer::start(&archive).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let options = options();
    let object = {
        let reader = pool.reader().unwrap();
        reader
            .prepare_query_selection("sessions | count", &options)
            .unwrap()
            .query_object()
            .unwrap()
    };
    writer.query_object(object).unwrap();
    let (run, continuation) = {
        let reader = pool.reader().unwrap();
        let selection = reader
            .prepare_query_selection("sessions | count", &options)
            .unwrap();
        let binding = selection
            .execution_binding(
                crate::query::execute::options_snapshot(&options, false, None).unwrap(),
                ExecutionPresentation::Query,
                None,
            )
            .unwrap();
        let run = QueryRun {
            id: "00000000-0000-7000-8000-000000000001".parse().unwrap(),
            question: Some(selection.question_id()),
            at: options.clock.now.into(),
            read_at: reader.change_position().unwrap(),
            machine: "synthetic-machine".into(),
            authorship: compute_authorship(
                Channel::Cli,
                &CallerEvidence {
                    controlling_terminal: true,
                    ..Default::default()
                },
                None,
            ),
            binding: polylogue_model::query_objects::RunBinding::Original {
                execution: Box::new(binding),
            },
        };
        let mut context = selection.cursor_context().clone();
        context
            .versions
            .insert("read_root_run".into(), run.id.to_string());
        context.versions.insert(
            "read_binding".into(),
            run.binding_hash().unwrap().to_string(),
        );
        context.versions.insert(
            "read_content".into(),
            reader
                .selection_content_fingerprint(&selection, false, None)
                .unwrap(),
        );
        (
            run,
            Continuation {
                context,
                after: polylogue_model::read::ReadSeek {
                    direction: polylogue_model::read::ReadDirection::After,
                    anchor: polylogue_model::read::ReadPosition {
                        selection: polylogue_model::read::SelectionLocator {
                            row: 0,
                            record: polylogue_model::read::ReadRecordLocator {
                                kind: polylogue_model::read::ReadRecordKind::Presentation,
                                ordinal: 0,
                            },
                        },
                        item: 0,
                        character: 0,
                        item_hash: polylogue_model::BlobHash::of(b"synthetic item"),
                    },
                }
                .to_order_position()
                .unwrap(),
            },
        )
    };
    writer.begin_query_run(run.clone()).unwrap();
    drop(writer);
    (dir, lock, pool, options, run, continuation)
}

#[test]
fn foreign_seals_cannot_authorize_even_an_acknowledged_root_execution() {
    let (_dir, _lock, pool, options, run, continuation) = acknowledged_execution();
    let reader = pool.reader().unwrap();
    let mut key = reader.cursor_key().unwrap();
    key[0] ^= 1;
    let cursor = CursorSeal::new(key).seal(&continuation).unwrap();
    let error = reader
        .prepare_authenticated_run_selection(
            &cursor,
            &run.id,
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: None,
                near: None,
            },
        )
        .err()
        .unwrap();
    assert!(matches!(
        error,
        SelectionError::Language(query_language::Error {
            kind: query_language::ErrorKind::InvalidQuery,
            ..
        })
    ));
}

#[test]
fn an_authentic_cursor_without_root_custody_binding_is_refused() {
    let (_dir, _lock, pool, options, run, mut continuation) = acknowledged_execution();
    let reader = pool.reader().unwrap();
    continuation.context.versions.remove("read_root_run");
    let cursor = CursorSeal::new(reader.cursor_key().unwrap())
        .seal(&continuation)
        .unwrap();
    let error = reader
        .prepare_authenticated_run_selection(
            &cursor,
            &run.id,
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: None,
                near: None,
            },
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), "invalid_cursor");
    assert!(error.to_string().contains("read_root_run"));
}

#[test]
fn acknowledged_run_history_does_not_stale_unchanged_selected_content() {
    let (_dir, _lock, pool, options, run, continuation) = acknowledged_execution();
    let reader = pool.reader().unwrap();
    assert!(reader.change_position().unwrap().archive > run.read_at.archive);
    let cursor = CursorSeal::new(reader.cursor_key().unwrap())
        .seal(&continuation)
        .unwrap();
    let (selection, position) = reader
        .prepare_authenticated_run_selection(
            &cursor,
            &run.id,
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: Some(99),
                near: None,
            },
        )
        .unwrap();
    assert_eq!(
        selection.authenticated_execution(&reader).unwrap().0,
        &run.id
    );
    assert_eq!(position, continuation.after);
}

#[test]
fn an_authentic_root_seal_cannot_authorize_an_unbounded_legacy_position() {
    let (_dir, _lock, pool, options, run, mut continuation) = acknowledged_execution();
    let reader = pool.reader().unwrap();
    continuation.after = OrderPosition {
        values: vec![serde_json::json!("native-id")],
        row_identity: "native-key".into(),
    };
    let cursor = CursorSeal::new(reader.cursor_key().unwrap())
        .seal(&continuation)
        .unwrap();
    let error = reader
        .prepare_authenticated_run_selection(
            &cursor,
            &run.id,
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: None,
                near: None,
            },
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), "invalid_cursor");
}

#[test]
fn authenticated_reverse_intent_is_preserved_and_flat_positions_are_refused() {
    use polylogue_model::read::{ReadDirection, ReadSeek};
    let (_dir, _lock, pool, options, run, mut continuation) = acknowledged_execution();
    let reader = pool.reader().unwrap();
    let mut seek = ReadSeek::from_order_position(&continuation.after).unwrap();
    seek.direction = ReadDirection::Before;
    continuation.after = seek.to_order_position().unwrap();
    let open = |continuation: &Continuation| {
        let cursor = CursorSeal::new(reader.cursor_key().unwrap())
            .seal(continuation)
            .unwrap();
        reader.prepare_authenticated_run_selection(
            &cursor,
            &run.id,
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: None,
                near: None,
            },
        )
    };
    let (selection, position) = open(&continuation).unwrap();
    assert_eq!(ReadSeek::from_order_position(&position).unwrap(), seek);
    assert_eq!(
        selection.authenticated_execution(&reader).unwrap().0,
        &run.id
    );
    continuation.after.values = vec![serde_json::to_value(seek.anchor).unwrap()];
    assert_eq!(open(&continuation).err().unwrap().kind(), "invalid_cursor");
}
