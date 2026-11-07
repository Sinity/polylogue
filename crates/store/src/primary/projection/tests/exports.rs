//! Rematerializing an export uses its retained callbacks and source receipts.
use super::*;
use polylogue_model::batch::{ExportDelivery, ExportDeliverySource};
use polylogue_model::{BlobHash, Output, RecordCounts, RecordOutcome, WireShape};

fn source(id: &str) -> Product {
    let hash = BlobHash::of(b"{}");
    let extent = hashed_blobs::Extent {
        segment: "segments/confirmed.hb".into(),
        offset: 0,
        len: 100,
        decoded_len: 2,
        hash: hashed_blobs::Hash(hash.0),
        kind: hashed_blobs::Kind::Plain,
        index_hash: hashed_blobs::Hash([1; 32]),
        encoded_hash: hashed_blobs::Hash([2; 32]),
    };
    Product::Input {
        rows: Box::new(crate::input_rows::InputRows {
            acquisition: None,
            input: id.into(),
            format: "synthetic".into(),
            observation: None,
            revision: Some(crate::input_rows::Revision {
                number: 1,
                produced: "imported".into(),
                chain: 1,
                length: 2,
            }),
            extent: Some(crate::input_rows::Extent {
                chain: 1,
                offset: 0,
                length: 2,
                blob: hash,
                blob_offset: 0,
            }),
        }),
        material: Some(Material {
            id: format!("material-{id}"),
            hash,
            length: 2,
            extents: vec![extent.encode()],
        }),
    }
}

fn callbacks(version: &str) -> (crate::ParseStep, crate::ParseStep) {
    let mut counts = RecordCounts::default();
    counts.records = 1;
    counts.add("item", &RecordOutcome::Used);
    counts.wire_shapes.insert(
        "item".into(),
        WireShape {
            record_kind: "item".into(),
            path: "/value".into(),
            types: BTreeMap::from([("string".into(), 1)]),
            expected: vec!["string".into()],
            ..Default::default()
        },
    );
    let mut item = crate::ParseStep::new(Output::default(), counts, 0..0, false);
    item.reading_version = version.into();
    item.reset = true;
    item.order_from = Some(0);
    let prefix = item
        .seal_export(
            0,
            ExportDeliverySource::Item { start: 0, end: 2 },
            ExportDelivery::seed(),
        )
        .unwrap();
    let mut complete =
        crate::ParseStep::new(Output::default(), RecordCounts::default(), 0..2, true);
    // The producer overrides the byte-range reset default: only the first
    // emitted export callback resets, while this EOF continues its prefix.
    complete.reset = false;
    complete.reading_version = version.into();
    complete.order_from = Some(1);
    complete.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        version.into(),
        2,
        None,
    ));
    complete
        .seal_export(
            1,
            ExportDeliverySource::Complete {
                length: 2,
                item_count: 1,
                delivery_count: 1,
            },
            prefix.prefix_hash,
        )
        .unwrap();
    (item, complete)
}

fn callback(position: i64, id: &str, step: &crate::ParseStep) -> Vec<Confirmed> {
    let key = crate::InputKey {
        id: id.into(),
        format: "synthetic".into(),
        arrival_rank: 0,
    };
    let mut step = step.clone();
    step.prepare();
    step.validate_export().unwrap();
    crate::primary::products::parse(&key, 1, Some(2), &step)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(n, product)| Confirmed {
            position,
            event: format!("{id}-{}-{position}-{n}", step.reading_version),
            output_index: n as u64,
            native: None,
            change: ConfirmedChange::Live(Box::new(product)),
        })
        .chain([barrier(position)])
        .collect()
}

fn shapes(index: &Connection, id: &str, version: &str) -> Vec<String> {
    index.prepare("SELECT shape FROM wire_shape w JOIN input_ref i ON i.pk=w.input_pk WHERE i.id=?1 AND w.chain=1 AND w.reading_version=?2 ORDER BY record_kind,path,table_key").unwrap()
        .query_map(params![id,version],|row| row.get(0)).unwrap()
        .collect::<rusqlite::Result<Vec<_>>>().unwrap()
}

#[test]
fn export_completion_rebuilds_shapes_with_foreign_keys_and_original_receipts() {
    let root = tempfile::tempdir().unwrap();
    let mut index = cache(&root.path().join("index.db"));
    index.pragma_update(None, "foreign_keys", true).unwrap();
    let archive = crate::Archive::on_sinex(root.path(), &root.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    for (position, id) in [(1, "target"), (2, "other")] {
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position,
                    event: format!("source-{id}"),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Live(Box::new(source(id))),
                },
                barrier(position),
            ],
        )
        .unwrap();
    }
    let (item, complete) = callbacks("v2");
    apply(&mut index, &resolver, &callback(3, "other", &item)).unwrap();
    let independent = shapes(&index, "other", "v2");
    assert_eq!(independent.len(), 1);
    apply(&mut index, &resolver, &callback(4, "target", &item)).unwrap();
    let before = shapes(&index, "target", "v2");
    assert_eq!(before.len(), 1);
    // The separate EOF control rematerializes a parent which already has
    // shape children. FK checks stay immediate throughout the transaction.
    index.execute_batch("CREATE TRIGGER stop_export BEFORE INSERT ON wire_reading WHEN new.reading_version='v2' BEGIN SELECT raise(ABORT,'stop'); END").unwrap();
    let error = apply(&mut index, &resolver, &callback(5, "target", &complete)).unwrap_err();
    assert!(
        matches!(&error,
        StoreError::Sqlite(rusqlite::Error::SqliteFailure(code, Some(message)))
            if code.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_TRIGGER && message == "stop"),
        "the injected rebuild failure must reach its SQL trigger: {error:?}"
    );
    assert_eq!(checkpoint(&index).unwrap(), Some(4));
    assert_eq!(
        shapes(&index, "target", "v2"),
        before,
        "a failed rebuild rolls back both parent and dependent shape removal"
    );
    index.execute_batch("DROP TRIGGER stop_export").unwrap();
    assert_eq!(apply(&mut index, &resolver, &[]).unwrap(), 5);
    assert_eq!(checkpoint(&index).unwrap(), Some(5));
    assert_eq!(shapes(&index, "other", "v2"), independent);
    let completed = shapes(&index, "target", "v2");
    let old: serde_json::Value = serde_json::from_str(&before[0]).unwrap();
    let current: serde_json::Value = serde_json::from_str(&completed[0]).unwrap();
    assert_eq!(current["types"], old["types"]);
    assert_eq!(current["first"], old["first"]);
    assert_eq!(current["last"], old["last"]);
    assert!(!current["complete_presence"].is_null());
    let status:String=index.query_row("SELECT p.status FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id='target' AND p.chain=1",[],|row| row.get(0)).unwrap();
    assert_eq!(status, "done");
    // Redelivery of the original admitted callback and completion is read-only.
    apply(&mut index, &resolver, &callback(4, "target", &item)).unwrap();
    apply(&mut index, &resolver, &callback(5, "target", &complete)).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(5));
    assert_eq!(shapes(&index, "target", "v2"), completed);
    assert_eq!(shapes(&index, "other", "v2"), independent);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );
}
