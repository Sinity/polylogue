use super::*;
use crate::fixture::writer::WriterArchive;
use polylogue_model::{
    ChangePosition, SessionId, Timestamp,
    authorship::{CallerEvidence, Channel, compute_authorship},
    query_objects::{ReplayBinding, RunOutcome},
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
        id: id.into(),
        question: object.id,
        at: Timestamp::from_unix_ms(at).unwrap(),
        read_at: position.clone(),
        machine: "synthetic-machine".into(),
        authorship,
        outcome: RunOutcome::Ok,
        count: Some(3),
        exact: true,
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
fn every_run_is_recorded_once_with_computed_authorship_and_frozen_context() {
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
    // A read need not wait for its SQL history row, and dropping its receipt
    // must not cancel accepted writer work.
    drop(archive.writer.record_query_run(expected.clone()).unwrap());
    assert!(archive.writer.flush_rows().unwrap().errors.is_empty());
    let reader = pool.reader().unwrap();
    let stored = reader.query_run(&expected.id, None).unwrap().unwrap();
    assert_eq!(stored.value, expected);
    assert_eq!(stored.custody, QueryCustody::Canonical);
    let position = reader.change_position().unwrap();
    drop(reader);
    archive
        .writer
        .record_query_run(expected.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), position);
    let mut conflicting = expected.clone();
    conflicting.count = Some(4);
    assert!(
        archive
            .writer
            .record_query_run(conflicting)
            .unwrap()
            .recv()
            .unwrap()
            .is_err()
    );
    assert_eq!(
        pool.reader()
            .unwrap()
            .query_run(&expected.id, None)
            .unwrap()
            .unwrap()
            .value,
        expected
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
        archive
            .writer
            .record_query_run(expected)
            .unwrap()
            .recv()
            .unwrap()
            .unwrap();
    }
    assert_eq!(
        pool.reader()
            .unwrap()
            .query_first_run(object.id)
            .unwrap()
            .as_deref(),
        Some("first")
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
        binding.replay.original_text = text.clone();
        writer
            .record_query_run(binding.clone())
            .unwrap()
            .recv()
            .unwrap()
            .unwrap();
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
