use super::*;
use polylogue_model::{Output, RecordCounts};
fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE input_ref(pk INTEGER PRIMARY KEY,id TEXT NOT NULL,format TEXT NOT NULL); CREATE TABLE parse_state(input_pk INTEGER,chain INTEGER,parse_version TEXT,status TEXT,parsed_upto INTEGER); INSERT INTO input_ref VALUES(1,'first','fixture'),(2,'second','fixture'),(3,'third','other');").unwrap();
    conn.execute_batch(SCHEMA).unwrap();
    conn
}
fn source(input_pk: i64, chain: i64, length: u64, occurrence: i64) -> Source<'static> {
    Source {
        input_pk,
        input: match input_pk {
            1 => "first",
            2 => "second",
            _ => "third",
        },
        format: if input_pk == 3 { "other" } else { "fixture" },
        chain,
        length,
        commit_sequence: occurrence,
        commit_id: Some("reading-occurrence"),
        observation: Some(match occurrence {
            1 => "observation-first",
            2 => "observation-second",
            _ => "observation-later",
        }),
        observed_at_ms: Some(1000 - occurrence),
    }
}
fn shape(path: &str, kind: &str, expected: &[&str]) -> WireShape {
    WireShape {
        record_kind: "message".into(),
        path: path.into(),
        types: BTreeMap::from([(kind.into(), 1)]),
        expected: expected.iter().map(|s| (*s).into()).collect(),
        ..WireShape::default()
    }
}
fn step(
    version: &str,
    range: std::ops::Range<u64>,
    at_end: bool,
    shapes: Vec<WireShape>,
) -> ParseStep {
    let mut counts = RecordCounts {
        records: 1,
        used: BTreeMap::from([("message".into(), 1)]),
        ..RecordCounts::default()
    };
    for shape in shapes {
        counts.wire_shapes.insert(
            if let Some(table) = &shape.table {
                serde_json::to_string(&(&shape.record_kind, table, &shape.path)).unwrap()
            } else {
                serde_json::to_string(&(&shape.record_kind, &shape.path)).unwrap()
            },
            shape,
        );
    }
    let mut step = ParseStep::new(Output::default(), counts, range, at_end);
    step.parse_version = version.into();
    step
}
fn root() -> WireShape {
    shape("", "object", &["object"])
}
fn store(conn: &mut Connection, source: Source<'_>, step: &ParseStep) {
    let tx = conn.transaction().unwrap();
    apply(&tx, source, step).unwrap();
    tx.commit().unwrap();
}
fn at<'a>(report: &'a Report, path: &str) -> &'a PathShape {
    report
        .shapes
        .iter()
        .find(|shape| shape.format == "fixture" && shape.path == path)
        .unwrap()
}
#[test]
fn partial_retry_is_idempotent_and_keeps_first_last() {
    let mut conn = database();
    let first = step(
        "v1",
        0..10,
        false,
        vec![root(), shape("/body", "string", &["string"])],
    );
    store(&mut conn, source(1, 1, 20, 1), &first);
    let before = read(&conn, Scope::current(None)).unwrap();
    store(&mut conn, source(1, 1, 20, 99), &first);
    assert_eq!(read(&conn, Scope::current(None)).unwrap(), before);
    store(
        &mut conn,
        source(1, 1, 20, 2),
        &step(
            "v1",
            10..20,
            true,
            vec![root(), shape("/body", "number", &["string"])],
        ),
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    let body = at(&report, "/body");
    assert_eq!(body.types["string"].count, 1);
    assert_eq!(body.types["number"].count, 1);
    assert_eq!(body.first.sequence, 1);
    assert_eq!(body.last.sequence, 2);
    assert!(body.first.observed_at_ms > body.last.observed_at_ms);
    assert_eq!(report.coverage[0].status(), CoverageStatus::Complete);
    assert!(
        report
            .drift
            .iter()
            .any(|row| row.kind == Kind::TypeChanged && row.shape.path == "/body")
    );
}
#[test]
fn wire_shape_completed_chain_includes_earlier_chunks_in_presence() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 20, 1),
        &step(
            "v1",
            0..10,
            false,
            vec![root(), shape("/earlier", "string", &["string"])],
        ),
    );
    store(
        &mut conn,
        source(1, 1, 20, 2),
        &step(
            "v1",
            10..20,
            true,
            vec![root(), shape("/later", "number", &["number"])],
        ),
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
    assert_eq!(
        at(&report, "/earlier")
            .complete_presence
            .as_ref()
            .unwrap()
            .sequence,
        2
    );
}
#[test]
fn drift_reported_new_path_changed_type_and_stopped_optional_field() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 10, 1),
        &step(
            "v1",
            0..10,
            true,
            vec![
                root(),
                shape("/gone", "string", &["string"]),
                shape("/body", "string", &["string"]),
            ],
        ),
    );
    store(
        &mut conn,
        source(2, 1, 10, 2),
        &step(
            "v1",
            0..10,
            true,
            vec![
                root(),
                shape("/body", "number", &["string"]),
                shape("/new", "boolean", &[]),
            ],
        ),
    );
    let report = read(&conn, Scope::current(Some("fixture"))).unwrap();
    assert_eq!(
        report
            .drift
            .iter()
            .map(|row| (row.kind, row.shape.path.clone()))
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
            (Kind::NewPath, "/new".to_string()),
            (Kind::TypeChanged, "/body".to_string()),
            (Kind::StoppedField, "/gone".to_string()),
        ])
    );
    assert_eq!(report.drift.len(), 3);
    assert!(
        report
            .drift
            .iter()
            .any(|row| row.kind == Kind::NewPath && row.shape.path == "/new")
    );
    assert!(
        report
            .drift
            .iter()
            .any(|row| row.kind == Kind::TypeChanged && row.shape.path == "/body")
    );
    let stopped = report
        .drift
        .iter()
        .find(|row| row.kind == Kind::StoppedField && row.shape.path == "/gone")
        .unwrap();
    assert_eq!(stopped.reason, DriftReason::MaybeOptional);
    assert_eq!(stopped.compared_with.len(), 1);
    assert_eq!(
        stopped.compared_with[0].last.as_ref().unwrap().input,
        "second"
    );
    assert_eq!(stopped.coverage, CoverageStatus::Complete);
}
#[test]
fn wire_shape_incomplete_or_unreadable_population_never_establishes_absence() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 10, 1),
        &step(
            "v1",
            0..10,
            true,
            vec![root(), shape("/optional", "string", &["string"])],
        ),
    );
    let mut later = step("v1", 0..10, false, vec![root()]);
    store(&mut conn, source(2, 1, 20, 2), &later);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.states.contains_key(&CoverageStatus::Partial))
    );
    later = step("v1", 10..20, true, vec![root()]);
    later.counts.used.clear();
    later
        .counts
        .unreadable
        .insert("message: invalid record".into(), 1);
    store(&mut conn, source(2, 1, 20, 3), &later);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.states.contains_key(&CoverageStatus::Unreadable))
    );
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
}
#[test]
fn wire_shape_parse_versions_and_input_chains_keep_their_history() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 10, 1),
        &step(
            "v1",
            0..10,
            true,
            vec![root(), shape("/old", "number", &["string"])],
        ),
    );
    store(
        &mut conn,
        source(1, 2, 10, 2),
        &step(
            "v1",
            0..10,
            true,
            vec![root(), shape("/chain-two", "string", &["string"])],
        ),
    );
    store(
        &mut conn,
        source(1, 1, 10, 3),
        &step(
            "v2",
            0..10,
            true,
            vec![root(), shape("/old", "number", &["number"])],
        ),
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    assert!(
        !report
            .shapes
            .iter()
            .any(|row| row.parse_version == "v1" && row.path == "/old")
    );
    assert!(
        report
            .shapes
            .iter()
            .any(|row| row.parse_version == "v1" && row.path == "/chain-two")
    );
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::TypeChanged && row.shape.path == "/old")
    );
    let history = read(&conn, Scope::history(Some("fixture"), Some("v1"))).unwrap();
    assert_eq!(at(&history, "/old").types.len(), 1);
    assert_eq!(at(&history, "/old").types["number"].count, 1);
    assert_eq!(history.coverage[0].readings, 2);
    assert_eq!(history.coverage[0].current_readings, 1);
}
#[test]
fn reset_replaces_counts_keeping_complete_presence() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 10, 1),
        &step(
            "v1",
            0..10,
            true,
            vec![root(), shape("/gone", "string", &["string"])],
        ),
    );
    let mut later = step(
        "v1",
        10..20,
        true,
        vec![root(), shape("/new", "string", &["string"])],
    );
    later.reset = true;
    store(&mut conn, source(1, 1, 20, 2), &later);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(at(&report, "").types["object"].count, 1);
    let gone = at(&report, "/gone");
    assert_eq!(gone.types["string"].count, 0);
    assert_eq!(gone.last.sequence, 1);
    assert_eq!(gone.complete_presence.as_ref().unwrap().sequence, 1);
    assert!(
        report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField && row.shape.path == "/gone")
    );
    store(&mut conn, source(1, 1, 20, 3), &later);
    assert_eq!(read(&conn, Scope::current(None)).unwrap(), report);
}
/// A file tailed to each end it had, then read again from its first byte in
/// ranges that end elsewhere, starts its observations again with its rows:
/// the row writer takes the chain's parse position back to the first range's
/// end, so the reading's position goes back with it and the range that
/// passes the earlier end continues the reading.
#[test]
fn a_chain_read_again_from_its_start_in_other_ranges_starts_its_observations_again() {
    let mut conn = database();
    let seen = |path| vec![root(), shape(path, "string", &["string"])];
    store(
        &mut conn,
        source(1, 1, 30, 1),
        &step("v1", 0..30, true, seen("/first")),
    );
    store(
        &mut conn,
        source(1, 1, 50, 2),
        &step("v1", 30..50, true, seen("/second")),
    );
    store(
        &mut conn,
        source(1, 1, 50, 3),
        &step("v1", 0..40, false, seen("/first")),
    );
    store(
        &mut conn,
        source(1, 1, 70, 4),
        &step("v1", 40..70, true, seen("/second")),
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(report.coverage[0].parsed_bytes, 70);
    assert_eq!(report.coverage[0].status(), CoverageStatus::Complete);
    assert_eq!(at(&report, "").types["object"].count, 2);
    assert_eq!(at(&report, "/first").types["string"].count, 1);
    assert_eq!(at(&report, "/second").types["string"].count, 1);
    // Where each path was first seen stays through the restart.
    assert_eq!(at(&report, "/first").first.sequence, 1);
}

/// The row writer decides whether a step continues its chain's reading; the
/// observations never refuse a step it took. Where they already reach past
/// the step's start (an index whose observations were left ahead of its
/// rows), the step is counted and the chain's coverage says its earlier part
/// is not what these counts describe, as for a step that starts after them.
#[test]
fn a_step_that_starts_inside_what_was_observed_is_counted_with_partial_coverage() {
    let mut conn = database();
    let seen = |path| vec![root(), shape(path, "string", &["string"])];
    store(
        &mut conn,
        source(1, 1, 50, 1),
        &step("v1", 0..50, true, seen("/first")),
    );
    let mut later = step("v1", 40..70, true, seen("/second"));
    later.reset = false;
    store(&mut conn, source(1, 1, 70, 2), &later);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(report.coverage[0].parsed_bytes, 70);
    assert_eq!(report.coverage[0].status(), CoverageStatus::Partial);
    assert_eq!(at(&report, "/first").types["string"].count, 1);
    assert_eq!(at(&report, "/second").types["string"].count, 1);
}

#[test]
fn wire_shape_empty_eof_finishes_coverage_without_recounting() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 10, 1),
        &step("v1", 0..10, false, vec![root()]),
    );
    let mut eof = step("v1", 10..10, true, vec![]);
    eof.counts = RecordCounts::default();
    store(&mut conn, source(1, 1, 10, 2), &eof);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(report.coverage[0].status(), CoverageStatus::Complete);
    assert_eq!(at(&report, "").types["object"].count, 1);
}
#[test]
fn wire_shape_missing_prefix_remains_incomplete_after_later_eof() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 30, 1),
        &step("v1", 10..20, false, vec![root()]),
    );
    store(
        &mut conn,
        source(1, 1, 30, 2),
        &step("v1", 20..30, true, vec![root()]),
    );
    assert_eq!(
        read(&conn, Scope::current(None)).unwrap().coverage[0].status(),
        CoverageStatus::Partial
    );
}
#[test]
fn wire_shape_unknown_kinds_and_tables_are_counted() {
    let mut conn = database();
    let mut kind = shape("/type", "string", &["string"]);
    kind.known_kinds = vec!["message".into()];
    kind.unknown_kind_count = 1;
    let mut table = root();
    table.table = Some("new_schema_table".into());
    table.unknown_table_count = 1;
    let parsed = step("v1", 0..10, true, vec![table, kind]);
    store(&mut conn, source(1, 1, 10, 1), &parsed);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(report.drift.len(), 2);
    assert!(
        report
            .drift
            .iter()
            .any(|row| row.kind == Kind::UnknownKind && row.shape.unknown_kind_count == 1)
    );
    assert!(report.drift.iter().any(|row| row.kind == Kind::UnknownTable
        && row.shape.table.as_deref() == Some("new_schema_table")));
}
#[test]
fn wire_shape_no_parser_and_failed_refused_coverage_are_visible() {
    let mut conn = database();
    let mut marker = step("", 0..0, true, vec![]);
    marker.counts = RecordCounts::default();
    marker.progress = polylogue_model::batch::StepProgress::NoParserYet { at_end: true };
    store(&mut conn, source(1, 1, 10, 1), &marker);
    conn.execute_batch(
        "INSERT INTO parse_state VALUES(2,1,'v1','failed',0),(3,1,'v1','refused',0);",
    )
    .unwrap();
    let report = read(&conn, Scope::current(None)).unwrap();
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.states.contains_key(&CoverageStatus::NoParserYet))
    );
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.states.contains_key(&CoverageStatus::Failed))
    );
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.states.contains_key(&CoverageStatus::Refused))
    );
    assert!(report.drift.is_empty());
}
#[test]
fn wire_shape_tuple_paths_remain_distinct_and_transaction_rollback_is_atomic() {
    let mut conn = database();
    let parsed = step(
        "v1",
        0..10,
        true,
        vec![
            root(),
            shape("/a~1b/~2/~3/~0", "string", &["string"]),
            shape("/a/b", "number", &["number"]),
        ],
    );
    {
        let tx = conn.transaction().unwrap();
        apply(&tx, source(1, 1, 10, 1), &parsed).unwrap();
    }
    assert!(read(&conn, Scope::current(None)).unwrap().shapes.is_empty());
    store(&mut conn, source(1, 1, 10, 1), &parsed);
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(report.shapes.len(), 3);
    assert_eq!(at(&report, "/a~1b/~2/~3/~0").types.len(), 1);
    assert_eq!(at(&report, "/a~1b/~2/~3/~0").types["string"].count, 1);
    assert_eq!(at(&report, "/a/b").types.len(), 1);
    assert_eq!(at(&report, "/a/b").types["number"].count, 1);
    let mut conflict = parsed;
    conflict
        .counts
        .wire_shapes
        .values_mut()
        .next()
        .unwrap()
        .dynamic_map = true;
    let tx = conn.transaction().unwrap();
    assert!(apply(&tx, source(1, 1, 10, 2), &conflict).is_err());
    drop(tx);
    assert_eq!(read(&conn, Scope::current(None)).unwrap(), report);
}

#[test]
fn wire_shape_unknown_observation_times_remain_unknown_and_formats_stay_distinct() {
    let mut conn = database();
    let mut identity = source(1, 1, 10, 1);
    identity.observation = None;
    identity.observed_at_ms = None;
    store(
        &mut conn,
        identity,
        &step(
            "v1",
            0..10,
            true,
            vec![root(), shape("/old", "string", &["string"])],
        ),
    );
    store(
        &mut conn,
        source(3, 1, 10, 9),
        &step(
            "v1",
            0..10,
            true,
            vec![root(), shape("/new", "boolean", &[])],
        ),
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(at(&report, "/old").first.observed_at_ms, None);
    assert_eq!(at(&report, "/old").first.observation, None);
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
    assert_eq!(report.drift.len(), 1);
    assert_eq!(report.drift[0].kind, Kind::NewPath);
    assert_eq!(report.drift[0].shape.path, "/new");
    assert_eq!(report.drift[0].shape.format, "other");
    assert!(
        read(&conn, Scope::current(Some("fixture")))
            .unwrap()
            .drift
            .is_empty()
    );
}

#[test]
fn wire_shape_sql_aggregation_returns_schema_rows_and_exact_counts() {
    let mut conn = database();
    for index in 10..210 {
        let input = format!("input-{index}");
        conn.execute(
            "INSERT INTO input_ref VALUES(?1,?2,'fixture')",
            params![index, input],
        )
        .unwrap();
        let identity = Source {
            input_pk: index,
            input: &input,
            format: "fixture",
            chain: 1,
            length: 10,
            commit_sequence: index,
            commit_id: None,
            observation: None,
            observed_at_ms: None,
        };
        store(
            &mut conn,
            identity,
            &step(
                "v1",
                0..10,
                true,
                vec![root(), shape("/unrecognized", "string", &[])],
            ),
        );
    }
    let report = read(&conn, Scope::current(Some("fixture"))).unwrap();
    assert_eq!(report.shapes.len(), 2);
    assert_eq!(report.coverage[0].readings, 200);
    assert_eq!(report.coverage[0].inputs, 200);
    assert_eq!(report.coverage[0].parsed_bytes, 2000);
    assert_eq!(report.drift.len(), 1);
    let row = &report.drift[0];
    assert_eq!(row.shape.types["string"].count, 200);
    assert_eq!(row.shape.first.sequence, 10);
    assert_eq!(row.shape.last.sequence, 209);
}
#[test]
fn source_tables_stay_distinct_and_unused_counted() {
    let mut conn = database();
    let mut shapes = Vec::new();
    for table in ["table_a", "table_b"] {
        let mut header = root();
        header.record_kind = "row".into();
        header.table = Some(table.into());
        header.unknown_table_count = 1;
        shapes.push(header);
        let mut column = shape("/value", "string", &[]);
        column.record_kind = "row".into();
        column.table = Some(table.into());
        column.unused_count = 1;
        shapes.push(column);
    }
    store(
        &mut conn,
        source(1, 1, 10, 1),
        &step("v1", 0..10, true, shapes),
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(report.shapes.len(), 4);
    assert_eq!(
        report
            .drift
            .iter()
            .filter(|row| row.kind == Kind::UnknownTable)
            .count(),
        2
    );
    assert_eq!(
        report
            .shapes
            .iter()
            .filter(|row| row.unused_count == 1)
            .count(),
        2
    );
}

#[test]
fn writer_stores_occurrences_and_reparse_versions() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let mut batch = crate::fixture::writer::step(
        "wire-input",
        "fixture",
        "wire.json",
        Output::default(),
        0,
        1,
    );
    batch.parse = vec![step(
        "v1",
        0..2,
        true,
        vec![root(), shape("/new", "number", &[])],
    )];
    let mut bytes = batch.clone();
    bytes.parse.clear();
    let prepared = archive.archive.prepare_bytes(&b"{}"[..]).unwrap();
    let stored = archive.writer.store_bytes(prepared, bytes).unwrap();
    batch.change = crate::Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    archive.writer.send(batch.clone()).unwrap();
    let report = archive.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let before = pool
        .reader()
        .unwrap()
        .wire_shapes(Scope::current(Some("fixture")))
        .unwrap();
    assert_eq!(before.shapes.len(), 2);
    let observed = at(&before, "/new");
    assert_eq!(observed.types["number"].count, 1);
    assert!(observed.last.id.is_none());
    assert!(observed.last.observation.is_some());
    assert_eq!(observed.last.observed_at_ms, Some(1));
    let source = Connection::open(archive.archive.archive_db()).unwrap();
    let commit_seq: i64 = source
        .query_row(
            "SELECT commit_seq FROM revision WHERE input='wire-input'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(observed.last.sequence, commit_seq);
    archive.writer.send(batch.clone()).unwrap();
    let report = archive.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(
        pool.reader()
            .unwrap()
            .wire_shapes(Scope::current(Some("fixture")))
            .unwrap(),
        before
    );
    let chain = pool
        .reader()
        .unwrap()
        .input_progress("wire-input")
        .unwrap()
        .unwrap()
        .chain;
    batch.change = crate::Change::Reparse { chain };
    batch.observation = None;
    batch.parse[0] = step(
        "v2",
        0..2,
        true,
        vec![root(), shape("/new", "number", &["number"])],
    );
    archive.writer.send(batch).unwrap();
    let report = archive.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let current = pool
        .reader()
        .unwrap()
        .wire_shapes(Scope::current(Some("fixture")))
        .unwrap();
    assert_eq!(current.shapes.len(), 2);
    assert!(current.drift.is_empty());
    assert_eq!(current.shapes[0].parse_version, "v2");
    assert_eq!(current.shapes[0].last.sequence, commit_seq);
    let history = pool
        .reader()
        .unwrap()
        .wire_shapes(Scope::history(Some("fixture"), None))
        .unwrap();
    assert_eq!(history.shapes.len(), 4);
    assert!(
        history
            .drift
            .iter()
            .any(|drift| drift.kind == Kind::NewPath && drift.shape.parse_version == "v1")
    );
}

#[test]
fn complete_chain_replacement_rebuilds_chunk_counts_and_rolls_back_overlaps() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let mut batch = crate::fixture::writer::step(
        "replacement-input",
        "fixture",
        "harness.json",
        Output::default(),
        0,
        3,
    );
    batch.parse.clear();
    let prepared = archive.archive.prepare_bytes(&b"{}{}"[..]).unwrap();
    let stored = archive.writer.store_bytes(prepared, batch.clone()).unwrap();
    batch.change = crate::Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    batch.parse = vec![step(
        "v1",
        0..4,
        true,
        vec![root(), shape("/gone", "string", &["string"])],
    )];
    archive.writer.send(batch.clone()).unwrap();
    assert!(archive.writer.flush_rows().unwrap().errors.is_empty());
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let report = || {
        pool.reader()
            .unwrap()
            .wire_shapes(Scope::current(Some("fixture")))
            .unwrap()
    };
    let before = report();
    let original_occurrence = at(&before, "").first.clone();
    let replacement = |tail_start, at_end| {
        let mut reading = crate::writer::ChainParse::new().unwrap();
        let mut first = step(
            "v1",
            0..2,
            false,
            vec![root(), shape("/kept", "number", &["number"])],
        );
        let mut tail = step(
            "v1",
            tail_start..4,
            at_end,
            vec![root(), shape("/new", "boolean", &["boolean"])],
        );
        reading.push(&mut first).unwrap();
        reading.push(&mut tail).unwrap();
        reading
    };
    let sender = archive.writer.sender();
    sender
        .replace_chain(batch.input.clone(), stored.chain, 4, replacement(2, true))
        .unwrap();
    let after = report();
    assert_eq!(at(&after, "").types["object"].count, 2);
    assert_eq!(at(&after, "/kept").types["number"].count, 1);
    assert_eq!(at(&after, "/new").types["boolean"].count, 1);
    assert_eq!(at(&after, "/gone").types["string"].count, 0);
    assert_eq!(at(&after, "").first, original_occurrence);
    assert_eq!(at(&after, "/gone").last, at(&before, "/gone").last);
    assert_eq!(
        at(&after, "/gone").complete_presence,
        at(&before, "/gone").complete_presence
    );
    assert_eq!(after.coverage[0].status(), CoverageStatus::Complete);
    assert_eq!(after.coverage[0].parsed_bytes, 4);
    assert_eq!(after.coverage[0].known_source_bytes, 4);
    sender
        .replace_chain(batch.input.clone(), stored.chain, 4, replacement(2, true))
        .unwrap();
    assert_eq!(report(), after);
    assert!(
        sender
            .replace_chain(batch.input.clone(), stored.chain, 4, replacement(1, true))
            .is_err()
    );
    assert_eq!(report(), after);
    assert!(
        sender
            .replace_chain(batch.input, stored.chain, 4, replacement(2, false))
            .is_err()
    );
    assert_eq!(report(), after);
}

#[test]
fn wire_coverage_survives_parser_status_changes() {
    // `Unknown` came only from the retired export reading marker.
    for wire in [
        CoverageStatus::Unreadable,
        CoverageStatus::NoParserYet,
        CoverageStatus::Complete,
        CoverageStatus::Partial,
    ] {
        let mut conn = database();
        let version = "v1";
        let end = if matches!(wire, CoverageStatus::Partial | CoverageStatus::NoParserYet) {
            0
        } else {
            10
        };
        let mut reading = step(version, 0..end, end == 10, vec![]);
        reading.counts = RecordCounts::default();
        if wire == CoverageStatus::Unreadable {
            reading.counts.records = 1;
            reading.counts.add(
                "record",
                &polylogue_model::RecordOutcome::unreadable("stored source diagnostic"),
            );
        }
        if wire == CoverageStatus::NoParserYet {
            reading.progress = polylogue_model::batch::StepProgress::NoParserYet {
                at_end: reading.progress.reached_end(),
            };
        }
        store(&mut conn, source(1, 1, 10, 1), &reading);
        let original: String = conn
            .query_row("SELECT state FROM wire_reading", [], |row| row.get(0))
            .unwrap();
        for status in ["parsing", "waiting", "done", "failed", "refused"] {
            conn.execute("DELETE FROM parse_state", []).unwrap();
            conn.execute(
                "INSERT INTO parse_state VALUES(1,1,?1,?2,?3)",
                params![version, status, i64::try_from(end).unwrap()],
            )
            .unwrap();
            let expected = match status {
                "failed" => CoverageStatus::Failed,
                "refused" => CoverageStatus::Refused,
                "parsing" | "waiting" if wire == CoverageStatus::Complete => {
                    CoverageStatus::Partial
                }
                _ => wire,
            };
            let report = read(&conn, Scope::current(Some("fixture"))).unwrap();
            assert_eq!(
                report.coverage[0].states,
                BTreeMap::from([(expected, 1)]),
                "{wire:?} with parser {status}"
            );
            assert_eq!(report.coverage[0].status(), expected);
            assert_eq!(report.coverage[0].parsed_bytes, end);
            assert_eq!(report.coverage[0].known_source_bytes, 10);
            assert_eq!(
                conn.query_row("SELECT state FROM wire_reading", [], |row| row
                    .get::<_, String>(0))
                    .unwrap(),
                original
            );
        }
    }
}
/// A reading's shapes cite its occurrences by number, each occurrence stored once
/// (docs/archive.md, format drift): two ranges of one chain over five paths
/// store the two ranges' occurrences and the whole chain's, and a read gives
/// every shape its own occurrences back.
#[test]
fn occurrences_are_stored_once_and_read_back_whole() {
    let mut conn = database();
    let paths = ["/a", "/b", "/c", "/d"];
    let shapes = || {
        std::iter::once(root())
            .chain(paths.iter().map(|path| shape(path, "string", &["string"])))
            .collect::<Vec<_>>()
    };
    store(
        &mut conn,
        source(1, 1, 20, 1),
        &step("v1", 0..10, false, shapes()),
    );
    store(
        &mut conn,
        source(1, 1, 20, 2),
        &step("v1", 10..20, true, shapes()),
    );
    let occurrences: Vec<String> = conn
        .prepare("SELECT occurrence FROM wire_occurrence ORDER BY number")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let occurrences: Vec<Occurrence> = occurrences
        .iter()
        .map(|json| serde_json::from_str(json).unwrap())
        .collect();
    let range = |occurrence: i64, start: u64, end: u64| Occurrence {
        covered: ReadRange::Bytes { start, end },
        ..Occurrence::from(source(1, 1, 20, occurrence))
    };
    assert_eq!(occurrences, [range(1, 0, 10), range(2, 10, 20)]);
    let stored: i64 = conn
        .query_row("SELECT count(*) FROM wire_shape", [], |row| row.get(0))
        .unwrap();
    assert_eq!(stored, 5);
    let report = read(&conn, Scope::current(Some("fixture"))).unwrap();
    for path in paths {
        let shape = at(&report, path);
        assert_eq!(shape.first, range(1, 0, 10));
        assert_eq!(shape.last, range(2, 10, 20));
        assert_eq!(
            shape.complete_presence,
            Some(Occurrence::from(source(1, 1, 20, 2)))
        );
        assert_eq!(shape.types["string"].count, 2);
        assert_eq!(shape.types["string"].first, range(1, 0, 10));
        assert_eq!(shape.types["string"].last, range(2, 10, 20));
    }
}

/// A live append writes shape and type rows only for the paths its records
/// held; the reading's other paths keep their rows untouched.
#[test]
fn live_append_writes_only_observed_paths() {
    let mut conn = database();
    store(
        &mut conn,
        source(1, 1, 30, 1),
        &step(
            "v1",
            0..10,
            false,
            vec![
                root(),
                shape("/a", "string", &["string"]),
                shape("/b", "number", &["number"]),
                shape("/c", "boolean", &["boolean"]),
            ],
        ),
    );
    conn.execute_batch(
        "CREATE TEMP TABLE written(path TEXT NOT NULL);
         CREATE TEMP TRIGGER shape_insert AFTER INSERT ON main.wire_shape BEGIN INSERT INTO written VALUES (new.path); END;
         CREATE TEMP TRIGGER shape_update AFTER UPDATE ON main.wire_shape BEGIN INSERT INTO written VALUES (new.path); END;
         CREATE TEMP TRIGGER shape_delete AFTER DELETE ON main.wire_shape BEGIN INSERT INTO written VALUES (old.path); END;
         CREATE TEMP TRIGGER type_insert AFTER INSERT ON main.wire_type BEGIN INSERT INTO written VALUES (new.path); END;
         CREATE TEMP TRIGGER type_update AFTER UPDATE ON main.wire_type BEGIN INSERT INTO written VALUES (new.path); END;
         CREATE TEMP TRIGGER type_delete AFTER DELETE ON main.wire_type BEGIN INSERT INTO written VALUES (old.path); END;",
    )
    .unwrap();
    store(
        &mut conn,
        source(1, 1, 30, 2),
        &step(
            "v1",
            10..20,
            false,
            vec![root(), shape("/a", "string", &["string"])],
        ),
    );
    let written: std::collections::BTreeSet<String> = conn
        .prepare("SELECT path FROM written")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        written,
        std::collections::BTreeSet::from(["".to_owned(), "/a".to_owned()])
    );
    let report = read(&conn, Scope::current(None)).unwrap();
    assert_eq!(at(&report, "/a").types["string"].count, 2);
    assert_eq!(at(&report, "/b").types["number"].count, 1);
}
