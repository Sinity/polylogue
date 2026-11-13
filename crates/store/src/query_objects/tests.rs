mod history;
mod parent_admission;

use super::*;
use crate::fixture::writer::WriterArchive;
use polylogue_model::{
    ChangePosition, SessionId, Timestamp,
    authorship::{CallerEvidence, Channel, compute_authorship},
    query_objects::{
        ExecutionBinding, ExecutionPresentation, QueryExecutionOptions, QueryRunOutcome,
        ReadSpanBinding, ReplayBinding, RunBinding, RunOutcome,
    },
};
use query_language::{
    cursor::CursorContext,
    time::{Clock, DaySetting},
};

pub(super) fn question(text: &str) -> QueryObject {
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

pub(super) fn run(object: &QueryObject, id: &str, at: i64, position: ChangePosition) -> QueryRun {
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
        question: Some(object.id),
        at: Timestamp::from_unix_ms(at).unwrap(),
        read_at: position.clone(),
        machine: "synthetic-machine".into(),
        authorship,
        binding: RunBinding::Original {
            execution: Box::new(ExecutionBinding {
                query_options: QueryExecutionOptions {
                    direct_reference: None,
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
                    bound_ast: object.query.clone(),
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
            }),
        },
    }
}

fn outcome(run: &QueryRun, count: u64) -> QueryRunOutcome {
    QueryRunOutcome {
        run: run.id.clone(),
        binding_hash: run.binding_hash().unwrap(),
        run_hash: run.record_hash().unwrap(),
        at: run.at,
        outcome: RunOutcome::Ok,
        count: Some(count),
        exact: true,
        next: None,
    }
}

#[test]
fn query_objects_keep_one_semantic_body_and_reject_forged_identity() {
    let archive = WriterArchive::new();
    let first = question("sessions where title:focus | limit 5");
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
    let equivalent = question("sessions  where title:focus | limit 5");
    assert_eq!(equivalent.id, first.id);
    archive.writer.query_object(equivalent).unwrap();
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), position);
    let mut forged = question("sessions where title:focus | limit 6");
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
    let object = question("sessions where since:7d");
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
    let mut reordered = expected.binding.original().unwrap().clone();
    let mut envelope = serde_json::Map::new();
    envelope.insert("operation".into(), serde_json::json!("query"));
    envelope.insert("arguments".into(), serde_json::json!({"target":"sessions"}));
    envelope.insert("id".into(), serde_json::json!("actual-accepted-id"));
    reordered.accepted_request = Some(serde_json::Value::Object(envelope));
    assert_eq!(reordered.hash().unwrap(), expected.binding_hash().unwrap());
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
        .original_mut()
        .unwrap()
        .replay
        .context
        .read_positions
        .insert("archive".into(), "999".into());
    assert!(archive.writer.begin_query_run(contradictory).is_err());
    let mut conflicting = expected.clone();
    conflicting
        .binding
        .original_mut()
        .unwrap()
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
    let text = format!(
        "sessions where title:\"{}\"",
        "synthetic-word ".repeat(3_000)
    );
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
        binding.binding.original_mut().unwrap().replay.original_text = text.clone();
        binding
            .binding
            .original_mut()
            .unwrap()
            .query_options
            .self_session =
            Some(SessionId::parse(&format!("codex:{}", "synthetic-session".repeat(3000))).unwrap());
        binding.binding.original_mut().unwrap().presentation = ExecutionPresentation::Read {
            options: polylogue_model::read::ReadOptions::default(),
            delivery: polylogue_model::read::ReadDelivery::Page,
            format: polylogue_model::read::Format::Json,
            page_budget: Some(Default::default()),
            direct_source: None,
        };
        binding.binding.original_mut().unwrap().accepted_request = Some(
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
    let definition =
        polylogue_query::plan::prepare("sessions where title:focus", &bindings, |_| Ok(()))
            .unwrap();
    bindings
        .add_saved("focus", &definition, "revision-one", "definition-root")
        .unwrap();
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
            output_metadata: Default::default(),
            shape: None,
            native_identity: false,
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
            page_size: Some(20),
            random_seed: Some(91),
            embeddings: false,
            self_session: None,
            saved: BTreeMap::new(),
            fields: vec![],
            kept: BTreeMap::new(),
            provider_catalog: None,
        };
        let text = "sessions where since:7d";
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
        assert!(context.resolved_times.is_empty());
        assert!(context.versions.contains_key("resolved_times"));
        let mut execution = run(&object, "actual-execution", 1, read_at);
        execution.at = Timestamp::from_unix_ns(options.clock.now.as_nanosecond()).unwrap();
        let original = execution.binding.original().unwrap();
        let binding = selection
            .execution_binding(
                original.query_options.clone(),
                ExecutionPresentation::Query,
                original.accepted_request.clone(),
            )
            .unwrap();
        assert!(!binding.replay.context.resolved_times.is_empty());
        assert_eq!(binding.replay.context.clock, options.clock.wire());
        assert_eq!(binding.replay.context.random_seed, 91);
        assert_eq!(
            binding.replay.context.read_positions,
            context.read_positions
        );
        assert_eq!(
            binding.replay.scope_fingerprint,
            reader.query_scope_fingerprint().unwrap()
        );
        assert_eq!(binding.replay.original_text, text);
        crate::query::continuation::check_resolved_times(
            context,
            &binding.replay.context.resolved_times,
        )
        .unwrap();
        execution.binding = RunBinding::Original {
            execution: Box::new(binding),
        };
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

#[test]
fn writer_keeps_normalized_span_metadata_bound_to_the_accepted_direct_target() {
    use polylogue_model::bundle::{
        TextSourceVersion, TextSpanOrigin,
        fragments::seek::{
            NativeSpanSeek, NativeTextBody, NativeTextPiece, NativeTextPosition, TextField,
        },
    };
    use polylogue_model::query_objects::NormalizedReadSpan;
    let archive = WriterArchive::new();
    let object = question("messages");
    archive.writer.query_object(object.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let original = polylogue_model::Reference::parse("span:message:codex:synthetic#c=1-3").unwrap();
    let span = NormalizedReadSpan::new(
        TextSpanOrigin {
            reference: original.clone(),
            start: 4,
            end: 9,
        },
        TextSourceVersion::ContentHash {
            record: polylogue_model::Reference::parse("message:codex:synthetic").unwrap(),
            hash: polylogue_model::BlobHash::of(b"retained version metadata"),
        },
        NativeSpanSeek {
            start: NativeTextPosition {
                piece: NativeTextPiece::Body {
                    body: NativeTextBody {
                        block: 7,
                        field: TextField::Text,
                    },
                },
                byte: 1,
            },
            end: NativeTextPosition {
                piece: NativeTextPiece::Body {
                    body: NativeTextBody {
                        block: 7,
                        field: TextField::Text,
                    },
                },
                byte: 6,
            },
        },
    )
    .unwrap();
    let mut execution = run(
        &object,
        "normalized-span-storage",
        1,
        pool.reader().unwrap().change_position().unwrap(),
    );
    execution
        .binding
        .original_mut()
        .unwrap()
        .query_options
        .direct_reference = Some(original);
    execution.binding.original_mut().unwrap().presentation = ExecutionPresentation::Read {
        options: Default::default(),
        delivery: Default::default(),
        format: polylogue_model::read::Format::Json,
        page_budget: Some(Default::default()),
        direct_source: Some(Box::new(ReadSpanBinding::Native { span })),
    };
    let hash = execution.binding_hash().unwrap();
    archive.writer.begin_query_run(execution.clone()).unwrap();
    let stored = pool
        .reader()
        .unwrap()
        .query_run(&execution.id, None)
        .unwrap()
        .unwrap()
        .value;
    assert_eq!(stored, execution);
    assert_eq!(stored.binding_hash().unwrap(), hash);
    let mut conflicting = execution.clone();
    let ExecutionPresentation::Read {
        direct_source: Some(source),
        ..
    } = &mut conflicting.binding.original_mut().unwrap().presentation
    else {
        unreachable!()
    };
    let ReadSpanBinding::Native { span } = source.as_mut() else {
        unreachable!()
    };
    span.seek.start.byte += 1;
    assert_ne!(conflicting.binding_hash().unwrap(), hash);
    assert!(archive.writer.begin_query_run(conflicting).is_err());
    let mut detached = execution;
    detached.id = run_id("detached-span-storage");
    detached
        .binding
        .original_mut()
        .unwrap()
        .query_options
        .direct_reference = None;
    assert!(archive.writer.begin_query_run(detached.clone()).is_err());
    assert!(
        pool.reader()
            .unwrap()
            .query_run(&detached.id, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn object_run_has_no_question_and_query_or_continuation_cannot_omit_one() {
    use polylogue_model::query_objects::ObjectExecutionBinding;
    let object = question("sessions");
    let position = ChangePosition {
        generation: "synthetic".into(),
        index: 1,
        archive: 1,
    };
    let mut original = run(&object, "query", 1, position.clone());
    assert!(validate_run(&original).is_ok());
    original.question = None;
    assert!(validate_run(&original).is_err());
    let mut exact = run(&object, "object", 1, position);
    exact.question = None;
    exact.binding = RunBinding::OriginalObject {
        execution: Box::new(ObjectExecutionBinding {
            reference: polylogue_model::Reference::Request("accepted:opaque@#..id".into()),
            content_hash: polylogue_model::BlobHash::of(b"retained original record and provenance"),
            scope_fingerprint: None,
            versions: Default::default(),
            presentation: ExecutionPresentation::Read {
                options: Default::default(),
                delivery: polylogue_model::read::ReadDelivery::Stream,
                format: polylogue_model::read::Format::Jsonl,
                page_budget: None,
                direct_source: None,
            },
            accepted_request: None,
        }),
    };
    validate_run(&exact).unwrap();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::QUERY_OBJECTS).unwrap();
    apply_run(&conn, &exact, 1, 0).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM query", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let stored: (Option<String>, String) = conn
        .query_row("SELECT query,object_ref FROM query_run", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(stored, (None, "request:accepted:opaque@#..id".into()));
    let mut invalid = exact.clone();
    invalid.question = Some(object.id);
    assert!(validate_run(&invalid).is_err());
    let fetched = lookup_run(&conn, Source::Ledger, &exact.id, None)
        .unwrap()
        .unwrap();
    assert_eq!(fetched.value, exact);
}

#[test]
fn query_admission_positions_share_the_source_cursor_and_rollback_without_cues() {
    let archive = WriterArchive::new();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let subscription = archive.writer.sender().unwrap().completion_commits();
    let first = question("sessions");
    let before = reads.reader().unwrap().change_position().unwrap();
    archive.writer.query_object(first.clone()).unwrap();
    subscription.try_recv().unwrap();
    let reader = reads.reader().unwrap();
    let after = reader.change_position().unwrap();
    assert!(after.archive > before.archive);
    assert_eq!(
        reader.completion_changed_queries(before.archive).unwrap(),
        std::collections::BTreeSet::from([first.id.to_string()])
    );
    drop(reader);
    // An exact repeat retains its first position and emits no new commit cue.
    archive.writer.query_object(first.clone()).unwrap();
    assert!(matches!(
        subscription.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    assert_eq!(reads.reader().unwrap().change_position().unwrap(), after);
    let conn = Connection::open(archive.archive.archive_db()).unwrap();
    let original: Vec<u8> = conn
        .query_row(
            "SELECT object FROM query WHERE hash=?1",
            [first.id.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    // A true old NULL row remains unknown even if the same body is offered again.
    conn.execute(
        "UPDATE query SET commit_seq=NULL WHERE hash=?1",
        [first.id.to_string()],
    )
    .unwrap();
    archive.writer.query_object(first.clone()).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT commit_seq FROM query WHERE hash=?1",
            [first.id.to_string()],
            |row| row.get::<_, Option<i64>>(0)
        )
        .unwrap(),
        None
    );
    assert_eq!(
        conn.query_row(
            "SELECT object FROM query WHERE hash=?1",
            [first.id.to_string()],
            |row| row.get::<_, Vec<u8>>(0)
        )
        .unwrap(),
        original
    );
    conn.execute_batch("CREATE TRIGGER stop_query BEFORE INSERT ON query BEGIN SELECT raise(ABORT,'stop query'); END").unwrap();
    let second = question("messages");
    let error = archive.writer.query_object(second.clone()).unwrap_err();
    assert!(
        matches!(&error,StoreError::Sqlite(rusqlite::Error::SqliteFailure(code,Some(message))) if code.extended_code==rusqlite::ffi::SQLITE_CONSTRAINT_TRIGGER && message=="stop query"),
        "{error:?}"
    );
    assert_eq!(reads.reader().unwrap().change_position().unwrap(), after);
    assert!(matches!(
        subscription.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    conn.execute_batch("DROP TRIGGER stop_query").unwrap();
    archive.writer.query_object(second.clone()).unwrap();
    subscription.try_recv().unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader.completion_changed_queries(after.archive).unwrap(),
        std::collections::BTreeSet::from([second.id.to_string()])
    );
    assert_eq!(
        conn.query_row(
            "SELECT commit_seq FROM query WHERE hash=?1",
            [second.id.to_string()],
            |row| row.get::<_, Option<i64>>(0)
        )
        .unwrap(),
        Some(reader.change_position().unwrap().archive)
    );
}
