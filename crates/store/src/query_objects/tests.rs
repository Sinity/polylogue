use super::*;
use crate::fixture::writer::WriterArchive;
use polylogue_model::{
    ChangePosition, SessionId, Timestamp,
    authorship::{CallerEvidence, Channel, compute_authorship},
    query_objects::{
        ExecutionBinding, ExecutionPresentation, QueryExecutionOptions, QueryRunOutcome,
        ReplayBinding, RunOutcome,
    },
};
use query_language::{
    cursor::CursorContext,
    time::{Clock, DaySetting},
};

fn question(text: &str) -> QueryObject {
    polylogue_query::plan::prepare(text, &polylogue_query::Bindings::default(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap()
}

fn run_id(label: &str) -> polylogue_model::RunId {
    let hash = polylogue_model::BlobHash::of(label.as_bytes()).to_string();
    polylogue_model::RunId::parse(&format!(
        "{}-{}-{}-{}-{}",
        &hash[..8],
        &hash[8..12],
        &hash[12..16],
        &hash[16..20],
        &hash[20..32]
    ))
    .unwrap()
}

fn run(object: &QueryObject, id: &str, at: i64, position: ChangePosition) -> QueryRun {
    let clock = Clock::new(
        jiff::Timestamp::from_millisecond(at).unwrap(),
        DaySetting::new(
            jiff::tz::TimeZone::get("UTC").unwrap(),
            jiff::civil::time(4, 0, 0, 0),
        )
        .unwrap(),
    );
    let authorship = compute_authorship(
        Channel::Mcp,
        &CallerEvidence {
            session: Some(SessionId::parse("codex:synthetic-query-agent").unwrap()),
            ..Default::default()
        },
        None,
    );
    QueryRun {
        id: run_id(id),
        question: object.id,
        at: Timestamp::from_unix_ms(at).unwrap(),
        read_at: position.clone(),
        machine: "synthetic-machine".into(),
        authorship,
        binding: ExecutionBinding {
            query_options: QueryExecutionOptions {
                fields: vec![],
                self_session: None,
                embeddings: false,
                why: false,
                ranking_identity: None,
            },
            presentation: ExecutionPresentation::Query,
            accepted_request: Some(
                serde_json::json!({"id":"actual-accepted-id","operation":"query","arguments":{"target":"sessions"}}),
            ),
            replay: ReplayBinding {
                bound_ast: object.identity_ast.clone(),
                source_outputs: object.source_outputs.clone(),
                definitions: vec![],
                context: CursorContext {
                    question: object.id,
                    clock: clock.wire(),
                    resolved_times: vec![],
                    random_seed: 7,
                    read_positions: BTreeMap::from([
                        ("generation".into(), position.generation),
                        ("index".into(), position.index.to_string()),
                        ("archive".into(), position.archive.to_string()),
                    ]),
                    versions: BTreeMap::from([("compiler".into(), "shared-query-v1".into())]),
                },
                scope_fingerprint: None,
                original_text: "sessions".into(),
            },
        },
    }
}

fn outcome(run: &QueryRun, count: u64) -> QueryRunOutcome {
    QueryRunOutcome {
        run: run.id.clone(),
        binding_hash: run.binding.hash().unwrap(),
        at: run.at,
        outcome: RunOutcome::Ok,
        count: Some(count),
        exact: true,
    }
}

#[test]
fn query_objects_keep_one_semantic_body_and_reject_forged_identity() {
    let archive = WriterArchive::new();
    let first = question("sessions title:focus | limit 5");
    assert_eq!(
        archive.writer.query_object(first.clone()).unwrap(),
        QueryCustody::Canonical
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    assert_eq!(
        archive.writer.query_object(first.clone()).unwrap(),
        QueryCustody::Canonical
    );
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), position);
    let read = pool
        .reader()
        .unwrap()
        .query_object(&first.id.to_string()[..12], None)
        .unwrap()
        .unwrap();
    assert_eq!(read.value, first);
    assert_eq!(read.custody, QueryCustody::Canonical);
    let equivalent = question("sessions  title:focus | limit 5");
    assert_eq!(equivalent.id, first.id);
    archive.writer.query_object(equivalent).unwrap();
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), position);
    let mut forged = question("sessions title:focus | limit 6");
    forged.id = first.id;
    assert!(archive.writer.query_object(forged).is_err());
    assert_eq!(
        pool.reader()
            .unwrap()
            .query_object(&first.id.to_string(), None)
            .unwrap()
            .unwrap()
            .value,
        first
    );
}

#[test]
fn run_begin_is_visible_before_completion_and_both_facts_are_immutable() {
    let archive = WriterArchive::new();
    let object = question("sessions since:7d");
    archive.writer.query_object(object.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let expected = run(
        &object,
        "run:opaque:one",
        123_456,
        pool.reader().unwrap().change_position().unwrap(),
    );
    assert_eq!(
        archive.writer.begin_query_run(expected.clone()).unwrap(),
        QueryCustody::Canonical
    );
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.query_run(&expected.id, None).unwrap().unwrap().value,
        expected
    );
    assert!(
        reader
            .query_run_outcome(&expected.id, None)
            .unwrap()
            .is_none()
    );
    let position = reader.change_position().unwrap();
    drop(reader);
    archive.writer.begin_query_run(expected.clone()).unwrap();
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), position);
    let mut reordered = expected.binding.clone();
    let mut envelope = serde_json::Map::new();
    envelope.insert("operation".into(), serde_json::json!("query"));
    envelope.insert("arguments".into(), serde_json::json!({"target":"sessions"}));
    envelope.insert("id".into(), serde_json::json!("actual-accepted-id"));
    reordered.accepted_request = Some(serde_json::Value::Object(envelope));
    assert_eq!(reordered.hash().unwrap(), expected.binding.hash().unwrap());
    let completed = outcome(&expected, 3);
    drop(archive.writer.finish_query_run(completed.clone()).unwrap());
    assert!(archive.writer.flush_rows().unwrap().errors.is_empty());
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query_run_outcome(&expected.id, None)
            .unwrap()
            .unwrap()
            .value,
        completed
    );
    assert_eq!(
        reader.query_run(&expected.id, None).unwrap().unwrap().value,
        expected
    );
    drop(reader);
    let mut contradictory = expected.clone();
    contradictory.id = run_id("contradictory-context");
    contradictory
        .binding
        .replay
        .context
        .read_positions
        .insert("archive".into(), "999".into());
    assert!(archive.writer.begin_query_run(contradictory).is_err());
    let mut conflicting = expected.clone();
    conflicting
        .binding
        .query_options
        .fields
        .push("changed".into());
    assert!(archive.writer.begin_query_run(conflicting).is_err());
    let mut conflicting = completed.clone();
    conflicting.count = Some(4);
    assert!(
        archive
            .writer
            .finish_query_run(conflicting)
            .unwrap()
            .recv()
            .unwrap()
            .is_err()
    );
    let mut mismatched = completed.clone();
    mismatched.binding_hash = polylogue_model::BlobHash::of(b"another-binding");
    assert!(
        archive
            .writer
            .finish_query_run(mismatched)
            .unwrap()
            .recv()
            .unwrap()
            .is_err()
    );
}

#[test]
fn first_run_follows_admission_order_instead_of_supplied_clock() {
    let archive = WriterArchive::new();
    let object = question("sessions");
    archive.writer.query_object(object.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    for expected in [
        run(&object, "first", 10_000, position.clone()),
        run(&object, "second", 1, position),
    ] {
        archive.writer.begin_query_run(expected).unwrap();
    }
    assert_eq!(
        pool.reader()
            .unwrap()
            .query_first_run(object.id)
            .unwrap()
            .as_deref(),
        Some(run_id("first").as_str())
    );
}

#[test]
fn a_large_question_and_replay_body_survive_writer_restart_whole() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("archive");
    let text = format!("sessions title:\"{}\"", "synthetic-word ".repeat(3_000));
    assert!(text.len() > 25_000);
    let object = question(&text);
    let expected;
    {
        let (archive, _lock) = crate::Archive::open_or_create(&path).unwrap();
        let writer = crate::Writer::start(&archive).unwrap();
        writer.query_object(object.clone()).unwrap();
        let pool = crate::ReadPool::open(&archive).unwrap();
        let mut binding = run(
            &object,
            "large-question",
            42,
            pool.reader().unwrap().change_position().unwrap(),
        );
        binding.binding.replay.original_text = text.clone();
        binding.binding.query_options.self_session =
            Some(SessionId::parse(&format!("codex:{}", "synthetic-session".repeat(3000))).unwrap());
        binding.binding.presentation = ExecutionPresentation::Read {
            options: polylogue_model::read::ReadOptions::default(),
            delivery: polylogue_model::read::ReadDelivery::Page,
        };
        binding.binding.accepted_request = Some(
            serde_json::json!({"id":"caller".repeat(10000),"operation":"read","arguments":{"target":text}}),
        );
        writer.begin_query_run(binding.clone()).unwrap();
        expected = binding;
    }
    let (archive, _lock) = crate::Archive::open_or_create(&path).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query_object(&object.id.to_string(), None)
            .unwrap()
            .unwrap()
            .value,
        object
    );
    assert_eq!(
        reader.query_run(&expected.id, None).unwrap().unwrap().value,
        expected
    );
}

#[test]
fn composition_edges_are_read_from_the_canonical_source() {
    let archive = WriterArchive::new();
    let mut bindings = polylogue_query::Bindings::default();
    bindings.saved.insert(
        "focus".into(),
        vec![polylogue_query::SavedQuery {
            text: "sessions title:focus".into(),
            revision: "revision-one".into(),
            first_note: "definition-root".into(),
        }],
    );
    let object = polylogue_query::plan::prepare("query:focus", &bindings, |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    assert!(
        object
            .edges
            .iter()
            .any(|edge| matches!(edge, QueryEdge::Saved { .. }))
    );
    archive.writer.query_object(object.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let mut expected = object.edges.clone();
    expected.sort_by_key(|edge| serde_json::to_string(edge).unwrap());
    let mut stored = pool.reader().unwrap().query_uses(object.id).unwrap();
    stored.sort_by_key(|edge| serde_json::to_string(edge).unwrap());
    assert_eq!(stored, expected);
}

mod migrations;

mod pending;

#[test]
fn result_output_schemas_require_the_canonical_kept_columns() {
    let archive = WriterArchive::new();
    let columns = vec![polylogue_query::Column {
        name: "title".into(),
        kind: polylogue_query::ColKind::Text,
        reference: None,
    }];
    archive.writer.user(polylogue_model::user::UserRequest {
        id: "kept".into(), machine: "synthetic-machine".into(), at: Timestamp::from_unix_ms(1).unwrap(),
        authorship:compute_authorship(Channel::Cli,&CallerEvidence::default(),None),
        write: polylogue_model::user::UserWrite::KeepResult {
            name: None, query: "sessions | select title".into(), read_at: "synthetic-read".into(),
            rows_json: serde_json::json!({"unit":"sessions","columns":columns,"rows":[{"title":"synthetic"}]}).to_string(),
        },
    }).unwrap();
    let mut bindings = polylogue_query::Bindings::default();
    bindings.kept.insert(
        "kept".into(),
        polylogue_query::Kept {
            unit: "sessions".into(),
            columns: columns.clone(),
            match_columns: vec![],
        },
    );
    let actual = polylogue_query::plan::prepare("result:kept", &bindings, |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    bindings.kept.get_mut("kept").unwrap().columns[0].name = "invented".into();
    let forged = polylogue_query::plan::prepare("result:kept", &bindings, |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    assert_eq!(forged.id, actual.id);
    assert!(archive.writer.query_object(forged.clone()).is_err());
    archive.writer.query_object(actual.clone()).unwrap();
    assert!(archive.writer.query_object(forged).is_err());
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .query_object(&actual.id.to_string(), None)
            .unwrap()
            .unwrap()
            .value,
        actual
    );
}

#[test]
fn executed_selection_records_its_actual_clock_resolution_and_read_context() {
    let archive = WriterArchive::new();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let expected;
    {
        let reader = pool.reader().unwrap();
        let options = polylogue_query::Options {
            clock: Clock::new(
                "2026-10-29T12:00:00.123456789Z".parse().unwrap(),
                DaySetting::new(
                    jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                    jiff::civil::time(4, 0, 0, 0),
                )
                .unwrap(),
            ),
            origins: vec![],
            command_words: vec![],
            page_size: Some(20),
            random_seed: Some(91),
            embeddings: false,
            self_session: None,
            saved: BTreeMap::new(),
            fields: vec![],
            kept: BTreeMap::new(),
            chat_pages: vec![],
        };
        let text = "sessions since:7d";
        let selection = reader.prepare_selection(text, &options).unwrap();
        let object = selection.query_object().unwrap();
        let read_at = reader.change_position().unwrap();
        archive.writer.query_object(object.clone()).unwrap();
        let mut count = 0;
        reader
            .visit_selection::<crate::selection::SelectionError>(&selection, None, None, |_| {
                count += 1;
                Ok(std::ops::ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(count, 0);
        let context = selection.cursor_context();
        assert!(!context.resolved_times.is_empty());
        let mut execution = run(&object, "actual-execution", 1, read_at);
        execution.at = Timestamp::from_unix_ns(options.clock.now.as_nanosecond()).unwrap();
        execution.binding.replay.bound_ast = selection.bound_query().query().clone();
        execution.binding.replay.context = CursorContext {
            question: context.question,
            clock: context.clock.wire(),
            resolved_times: context.resolved_times.clone(),
            random_seed: context.random_seed,
            read_positions: context.read_positions.clone(),
            versions: context.versions.clone(),
        };
        execution.binding.replay.scope_fingerprint = reader.query_scope_fingerprint().unwrap();
        execution.binding.replay.original_text = text.into();
        archive.writer.begin_query_run(execution.clone()).unwrap();
        archive
            .writer
            .finish_query_run(outcome(&execution, count))
            .unwrap()
            .recv()
            .unwrap()
            .unwrap();
        expected = execution;
    }
    let stored = pool
        .reader()
        .unwrap()
        .query_run(&expected.id, None)
        .unwrap()
        .unwrap();
    assert_eq!(stored.value, expected);
}
