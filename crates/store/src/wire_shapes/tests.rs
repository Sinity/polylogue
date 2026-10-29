use super::*;
use polylogue_model::{Output, RecordCounts};
fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE input_ref(pk INTEGER PRIMARY KEY,id TEXT NOT NULL,format TEXT NOT NULL); CREATE TABLE parse_state(input_pk INTEGER,chain INTEGER,reading_version TEXT,status TEXT,parsed_upto INTEGER); INSERT INTO input_ref VALUES(1,'first','fixture'),(2,'second','fixture'),(3,'third','other');").unwrap();
    conn.execute_batch(SCHEMA).unwrap();
    conn
}
fn source(input_pk: i64, chain: i64, length: u64, receipt: i64) -> Source<'static> {
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
        receipt,
        observation: Some(match receipt {
            1 => "observation-first",
            2 => "observation-second",
            _ => "observation-later",
        }),
        observed_at_ms: Some(1000 - receipt),
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
            serde_json::to_string(&(&shape.record_kind, &shape.path)).unwrap(),
            shape,
        );
    }
    let mut step = ParseStep::new(Output::default(), counts, range, at_end);
    step.reading_version = version.into();
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
fn at<'a>(report: &'a Report, input: &str, path: &str) -> &'a Observation {
    report
        .shapes
        .iter()
        .find(|shape| shape.input == input && shape.path == path && shape.active)
        .unwrap()
}
#[test]
fn wire_shape_partial_retry_is_idempotent_and_receipt_order_owns_first_last() {
    let mut conn = database();
    let first = step(
        "v1",
        0..10,
        false,
        vec![root(), shape("/body", "string", &["string"])],
    );
    store(&mut conn, source(1, 1, 20, 1), &first);
    let before = read(&conn, None).unwrap();
    store(&mut conn, source(1, 1, 20, 99), &first);
    assert_eq!(read(&conn, None).unwrap(), before);
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
    let report = read(&conn, None).unwrap();
    let body = at(&report, "first", "/body");
    assert_eq!(body.types["string"].count, 1);
    assert_eq!(body.types["number"].count, 1);
    assert_eq!(body.first.sequence, 1);
    assert_eq!(body.last.sequence, 2);
    assert!(body.first.observed_at_ms > body.last.observed_at_ms);
    assert_eq!(report.coverage[0].status, CoverageStatus::Complete);
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
    let report = read(&conn, None).unwrap();
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
    assert_eq!(
        at(&report, "first", "/earlier")
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
    let report = read(&conn, Some("fixture")).unwrap();
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
    assert_eq!(stopped.certainty, Certainty::OptionalPresence);
    assert_eq!(stopped.compared_with.len(), 1);
    assert_eq!(stopped.compared_with[0].input, "second");
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
    let report = read(&conn, None).unwrap();
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
            .any(|row| row.status == CoverageStatus::Partial)
    );
    later = step("v1", 10..20, true, vec![root()]);
    later.counts.used.clear();
    later
        .counts
        .unreadable
        .insert("message: invalid record".into(), 1);
    store(&mut conn, source(2, 1, 20, 3), &later);
    let report = read(&conn, None).unwrap();
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.status == CoverageStatus::Unreadable)
    );
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
}
#[test]
fn wire_shape_reading_versions_and_input_chains_keep_their_history() {
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
    let report = read(&conn, None).unwrap();
    assert!(report.shapes.iter().any(|row| row.reading_version == "v1"
        && row.chain == 1
        && !row.active
        && row.path == "/old"));
    assert!(report.shapes.iter().any(|row| row.reading_version == "v1"
        && row.chain == 2
        && row.active
        && row.path == "/chain-two"));
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::TypeChanged && row.shape.path == "/old")
    );
}
#[test]
fn wire_shape_reset_replaces_counts_retaining_prior_complete_presence() {
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
    let report = read(&conn, None).unwrap();
    assert_eq!(at(&report, "first", "").types["object"].count, 1);
    let gone = at(&report, "first", "/gone");
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
    assert_eq!(read(&conn, None).unwrap(), report);
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
    let report = read(&conn, None).unwrap();
    assert_eq!(report.coverage[0].status, CoverageStatus::Complete);
    assert_eq!(at(&report, "first", "").types["object"].count, 1);
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
        read(&conn, None).unwrap().coverage[0].status,
        CoverageStatus::Partial
    );
}
#[test]
fn wire_shape_unknown_discriminators_and_tables_use_explicit_metadata_without_values() {
    let mut conn = database();
    let mut kind = shape("/type", "string", &["string"]);
    kind.known_kinds = vec!["message".into()];
    kind.unknown_kind_count = 1;
    let mut table = root();
    table.table = Some("new_schema_table".into());
    table.unknown_table_count = 1;
    let mut parsed = step("v1", 0..10, true, vec![table, kind]);
    parsed.output.events.push(polylogue_model::Event {
        session: None,
        kind: "fixture".into(),
        at: None,
        after: None,
        range: Some((0, 10)),
        data: serde_json::json!({"secret":"distinctive-source-value-123"}),
    });
    store(&mut conn, source(1, 1, 10, 1), &parsed);
    let report = read(&conn, None).unwrap();
    assert!(
        report
            .drift
            .iter()
            .any(|row| row.kind == Kind::UnknownKind && row.shape.unknown_kind_count == 1)
    );
    assert!(report.drift.iter().any(|row| row.kind == Kind::UnknownTable
        && row.shape.table.as_deref() == Some("new_schema_table")));
    let output = serde_json::to_string(&report).unwrap();
    assert!(!output.contains("distinctive-source-value-123"));
    let stored: String = conn
        .query_row("SELECT group_concat(shape) FROM wire_shape", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(!stored.contains("distinctive-source-value-123"));
}
#[test]
fn wire_shape_no_parser_and_failed_refused_coverage_are_visible() {
    let mut conn = database();
    let mut marker = step("", 0..0, true, vec![]);
    marker.counts = RecordCounts::default();
    marker.no_parser_yet = true;
    store(&mut conn, source(1, 1, 10, 1), &marker);
    conn.execute_batch(
        "INSERT INTO parse_state VALUES(2,1,'v1','failed',0),(3,1,'v1','refused',0);",
    )
    .unwrap();
    let report = read(&conn, None).unwrap();
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.status == CoverageStatus::NoParserYet)
    );
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.status == CoverageStatus::Failed)
    );
    assert!(
        report
            .coverage
            .iter()
            .any(|row| row.status == CoverageStatus::Refused)
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
    assert!(read(&conn, None).unwrap().shapes.is_empty());
    store(&mut conn, source(1, 1, 10, 1), &parsed);
    let report = read(&conn, None).unwrap();
    assert!(report.shapes.iter().any(|row| row.path == "/a~1b/~2/~3/~0"));
    assert!(report.shapes.iter().any(|row| row.path == "/a/b"));
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
    assert_eq!(read(&conn, None).unwrap(), report);
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
    let report = read(&conn, None).unwrap();
    assert_eq!(at(&report, "first", "/old").first.observed_at_ms, None);
    assert_eq!(at(&report, "first", "/old").first.observation, None);
    assert!(
        !report
            .drift
            .iter()
            .any(|row| row.kind == Kind::StoppedField)
    );
    assert_eq!(report.drift[0].shape.format, "other");
    assert!(read(&conn, Some("fixture")).unwrap().drift.is_empty());
}
