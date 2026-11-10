use super::*;
use polylogue_model::batch::{ExportDelivery, ExportDeliverySource};
use polylogue_model::{BlobHash, Output, RecordCounts, RecordOutcome};

fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE input_ref(pk INTEGER PRIMARY KEY,id TEXT,format TEXT); INSERT INTO input_ref VALUES(1,'input','fixture');").unwrap();
    conn.execute_batch(SCHEMA).unwrap();
    conn
}
fn source(length: u64) -> Source<'static> {
    Source {
        input_pk: 1,
        input: "input",
        format: "fixture",
        chain: 1,
        length,
        receipt: 7,
        receipt_id: Some("receipt"),
        observation: Some("observation"),
        observed_at_ms: Some(100),
    }
}
fn callback(ordinal: u64, footprint: ExportDeliverySource, previous: BlobHash) -> ParseStep {
    let complete = matches!(footprint, ExportDeliverySource::Complete { .. });
    let end = match footprint {
        ExportDeliverySource::Complete { length, .. } => length,
        _ => 0,
    };
    let mut step = ParseStep::new(Output::default(), RecordCounts::default(), 0..end, complete);
    step.reading_version = "crate::export::VERSION=2".into();
    step.reset = ordinal == 0;
    if matches!(footprint, ExportDeliverySource::Item { .. }) {
        step.counts.records = 1;
        step.counts.add("item", &RecordOutcome::Used);
        step.counts.wire_shapes.insert(
            "item".into(),
            WireShape {
                record_kind: "item".into(),
                path: "/value".into(),
                types: BTreeMap::from([("string".into(), 1)]),
                expected: vec!["string".into()],
                ..Default::default()
            },
        );
    }
    step.seal_export(ordinal, footprint, previous).unwrap();
    step
}
fn apply_step(conn: &mut Connection, length: u64, step: &ParseStep) -> Result<()> {
    let tx = conn.transaction()?;
    apply(&tx, source(length), step)?;
    tx.commit()?;
    Ok(())
}
fn state(conn: &Connection) -> String {
    conn.query_row("SELECT state FROM wire_reading", [], |row| row.get(0))
        .unwrap()
}

#[test]
fn callback_prefix_retries_are_read_only_and_completion_waits_for_every_prior_callback() {
    let mut conn = database();
    let header = callback(
        0,
        ExportDeliverySource::ContainerHeader,
        ExportDelivery::seed(),
    );
    apply_step(&mut conn, 2, &header).unwrap();
    let mut item = callback(
        1,
        ExportDeliverySource::Item { start: 0, end: 1 },
        header.delivery.unwrap().prefix_hash,
    );
    apply_step(&mut conn, 2, &item).unwrap();
    let saved = state(&conn);
    assert_eq!(
        serde_json::from_str::<Reading>(&saved).unwrap().status(),
        CoverageStatus::Partial
    );
    apply_step(&mut conn, 2, &header).unwrap();
    apply_step(&mut conn, 2, &item).unwrap();
    assert_eq!(state(&conn), saved);
    let gap = callback(
        3,
        ExportDeliverySource::Item { start: 1, end: 2 },
        item.delivery.unwrap().prefix_hash,
    );
    assert!(apply_step(&mut conn, 2, &gap).is_err());
    assert_eq!(state(&conn), saved);
    let mut conflict = item.clone();
    conflict.counts.used.clear();
    conflict.counts.skipped.insert("item".into(), 1);
    conflict
        .seal_export(
            1,
            ExportDeliverySource::Item { start: 0, end: 1 },
            header.delivery.unwrap().prefix_hash,
        )
        .unwrap();
    assert!(apply_step(&mut conn, 2, &conflict).is_err());
    let wrong = callback(
        2,
        ExportDeliverySource::Complete {
            length: 2,
            item_count: 2,
            delivery_count: 2,
        },
        item.delivery.unwrap().prefix_hash,
    );
    assert!(apply_step(&mut conn, 2, &wrong).is_err());
    assert_eq!(state(&conn), saved);
    let complete = callback(
        2,
        ExportDeliverySource::Complete {
            length: 2,
            item_count: 1,
            delivery_count: 2,
        },
        item.delivery.unwrap().prefix_hash,
    );
    apply_step(&mut conn, 2, &complete).unwrap();
    let reading: Reading = serde_json::from_str(&state(&conn)).unwrap();
    assert_eq!(reading.status(), CoverageStatus::Complete);
    assert_eq!(reading.parsed_upto, 2);
    assert!(matches!(
        reading.last.footprint,
        ReadingFootprint::Export {
            source: ExportDeliverySource::Complete { item_count: 1, .. }
        }
    ));
    let shape = stored_rows(&conn);
    assert_eq!(shape.1, vec![("string".to_owned(), 1, 1, 1)]);
    let done = state(&conn);
    apply_step(&mut conn, 2, &complete).unwrap();
    assert_eq!(state(&conn), done);
    // A single retained head cannot authenticate each older callback. Older
    // callbacks are read-only; the candidate prefix must meet the saved head
    // before it can extend or acknowledge completion.
    apply_step(&mut conn, 2, &header).unwrap();
    apply_step(&mut conn, 2, &item).unwrap();
    apply_step(&mut conn, 2, &conflict).unwrap();
    assert_eq!(state(&conn), done);
    let forked_completion = callback(
        2,
        ExportDeliverySource::Complete {
            length: 2,
            item_count: 1,
            delivery_count: 2,
        },
        conflict.delivery.unwrap().prefix_hash,
    );
    assert!(apply_step(&mut conn, 2, &forked_completion).is_err());
    let after_complete = callback(
        3,
        ExportDeliverySource::Item { start: 1, end: 2 },
        complete.delivery.unwrap().prefix_hash,
    );
    assert!(apply_step(&mut conn, 2, &after_complete).is_err());
    assert_eq!(state(&conn), done);
    assert_eq!(stored_rows(&conn), shape);
    item.delivery = None;
    assert!(apply_step(&mut conn, 2, &item).is_err());
}

#[test]
fn unreadable_header_keeps_diagnostics_without_a_cursor_and_new_actual_length_can_restart() {
    let mut conn = database();
    let mut refused = callback(
        0,
        ExportDeliverySource::ContainerHeader,
        ExportDelivery::seed(),
    );
    refused.counts.records = 1;
    refused
        .counts
        .add("export", &RecordOutcome::unreadable("truncated export"));
    refused.progress = polylogue_model::batch::StepProgress::Refused {
        reason: "truncated export".into(),
    };
    refused
        .seal_export(
            0,
            ExportDeliverySource::ContainerHeader,
            ExportDelivery::seed(),
        )
        .unwrap();
    apply_step(&mut conn, 1, &refused).unwrap();
    let original = state(&conn);
    let reading: Reading = serde_json::from_str(&original).unwrap();
    assert!(
        reading.refused_export
            && reading.export_cursor.is_none()
            && !reading.progress.reached_end()
    );
    assert_eq!(reading.status(), CoverageStatus::Unreadable);
    assert_eq!(reading.parsed_upto, 0);
    conn.execute_batch("CREATE TABLE parse_state(input_pk INTEGER,chain INTEGER,reading_version TEXT,status TEXT,parsed_upto INTEGER); INSERT INTO parse_state VALUES(1,1,'crate::export::VERSION=2','parsing',0);").unwrap();
    let report = read(&conn, Scope::current(Some("fixture"))).unwrap();
    assert_eq!(report.coverage.len(), 1);
    assert_eq!(report.coverage[0].status(), CoverageStatus::Unreadable);
    assert_eq!(
        report.coverage[0].states,
        BTreeMap::from([(CoverageStatus::Unreadable, 1)])
    );
    assert_eq!(report.coverage[0].parsed_bytes, 0);
    assert_eq!(report.coverage[0].known_source_bytes, 1);
    assert_eq!(state(&conn), original);
    apply_step(&mut conn, 1, &refused).unwrap();
    assert_eq!(state(&conn), original);
    let header = callback(
        0,
        ExportDeliverySource::ContainerHeader,
        ExportDelivery::seed(),
    );
    assert!(apply_step(&mut conn, 1, &header).is_err());
    apply_step(&mut conn, 2, &header).unwrap();
    let complete = callback(
        1,
        ExportDeliverySource::Complete {
            length: 2,
            item_count: 0,
            delivery_count: 1,
        },
        header.delivery.unwrap().prefix_hash,
    );
    apply_step(&mut conn, 2, &complete).unwrap();
    let reading: Reading = serde_json::from_str(&state(&conn)).unwrap();
    assert_eq!(reading.status(), CoverageStatus::Complete);
    assert_eq!(reading.export_refusals.unwrap().count, 1);
}

#[test]
fn semantic_seals_ignore_receiver_receipts_but_bind_original_event_identities_and_counts() {
    let mut step = callback(
        0,
        ExportDeliverySource::ContainerHeader,
        ExportDelivery::seed(),
    );
    step.output.message_receipts.insert(0, 130);
    step.output.event_receipts.insert(0, 140);
    step.validate_export().unwrap();
    step.output.event_ids.insert(0, "original-event".into());
    assert!(step.validate_export().is_err());
}

#[test]

fn header_line_failures_preserve_counts_and_allow_valid_items_and_complete() {
    let mut conn = database();
    let mut header = callback(
        0,
        ExportDeliverySource::ContainerHeader,
        ExportDelivery::seed(),
    );
    header.counts.records = 1;
    header
        .counts
        .add("line", &RecordOutcome::unreadable("malformed JSON line"));
    header
        .seal_export(
            0,
            ExportDeliverySource::ContainerHeader,
            ExportDelivery::seed(),
        )
        .unwrap();
    assert!(!is_refusal(&header));
    apply_step(&mut conn, 2, &header).unwrap();
    let started: Reading = serde_json::from_str(&state(&conn)).unwrap();
    assert!(!started.refused_export);
    assert_eq!(started.export_cursor.unwrap().last_ordinal, 0);
    let item = callback(
        1,
        ExportDeliverySource::Item { start: 0, end: 2 },
        header.delivery.unwrap().prefix_hash,
    );
    apply_step(&mut conn, 2, &item).unwrap();
    let complete = callback(
        2,
        ExportDeliverySource::Complete {
            length: 2,
            item_count: 1,
            delivery_count: 2,
        },
        item.delivery.unwrap().prefix_hash,
    );
    apply_step(&mut conn, 2, &complete).unwrap();
    let saved = state(&conn);
    let finished: Reading = serde_json::from_str(&saved).unwrap();
    assert!(!finished.refused_export);
    assert!(finished.progress.reached_end());
    assert_eq!(finished.parsed_upto, 2);
    assert_eq!(finished.export_cursor.unwrap().last_ordinal, 2);
    assert_eq!(finished.unreadable, 1);
    assert_eq!(finished.status(), CoverageStatus::Unreadable);
    for step in [&header, &item, &complete] {
        apply_step(&mut conn, 2, step).unwrap();
    }
    assert_eq!(state(&conn), saved);
    let count: i64 = conn
        .query_row(
            "SELECT count FROM wire_type WHERE json_type='string'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

/// The stored shape rows: (path, unknown kinds, unknown tables, unused,
/// first, last) and (type, count, first, last).
#[allow(clippy::type_complexity)]
fn stored_rows(
    conn: &Connection,
) -> (
    Vec<(String, i64, i64, i64, i64, i64)>,
    Vec<(String, i64, i64, i64)>,
) {
    let shapes = conn
        .prepare("SELECT path,unknown_kinds,unknown_tables,unused,first,last FROM wire_shape ORDER BY path")
        .unwrap()
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let types = conn
        .prepare("SELECT json_type,count,first,last FROM wire_type ORDER BY path,json_type")
        .unwrap()
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    (shapes, types)
}
