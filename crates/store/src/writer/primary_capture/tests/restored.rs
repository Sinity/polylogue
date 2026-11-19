use super::*;
use crate::primary::{Confirmed, ConfirmedChange, InputProvenance, ReplayControl};

#[test]
fn confirmed_restored_history_keeps_coordinates_without_a_current_source() {
    let directory = tempfile::tempdir().unwrap();
    let producer_root = directory.path().join("producer");
    std::fs::create_dir(&producer_root).unwrap();
    let producer = Arc::new(OfflineProducer {
        root: producer_root.clone(),
    });
    let (archive, _lock) =
        Archive::open_primary(&directory.path().join("cache"), &producer_root).unwrap();
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    writer
        .apply_confirmed(vec![read_control(ReplayControl::Reset {
            scope: "scope".into(),
        })])
        .unwrap();
    writer
        .apply_confirmed(vec![read_control(ReplayControl::Completion {
            position: 1,
            delivery: None,
        })])
        .unwrap();
    let rows = InputRows {
        acquisition: None,
        input: "historical".into(),
        format: "codex".into(),
        revision: Some(Revision {
            number: 7,
            produced: "rewritten".into(),
            chain: 4,
            length: 0,
        }),
        extents: Vec::new(),
        observation: None,
    };
    let bytes = serde_json::to_vec(&rows).unwrap();
    let prepared = archive.prepare_bytes(bytes.as_slice()).unwrap();
    let captured = CapturedInput {
        provenance: InputProvenance::Restored,
        input: crate::primary::InputName {
            origin: "codex".into(),
            path: "/historical/deleted.jsonl".into(),
            host: "historical-host".into(),
        },
        rows: rows.clone(),
        index: 0,
        previous: None,
        hash: prepared.hash(),
        length: prepared.len(),
        group: None,
        route: polylogue_model::Route::Copy,
        raw_range: None,
        generation: None,
        raw_total: None,
        raw_predecessor: None,
        basis: None,
    };
    producer
        .prepare_input(&captured, prepared.body().unwrap())
        .unwrap();
    std::fs::write(producer_root.join("ready"), []).unwrap();
    writer.synchronize_primary().unwrap();
    let writes: Vec<ProductWrite> =
        serde_json::from_slice(&std::fs::read(producer_root.join("products.json")).unwrap())
            .unwrap();
    assert_eq!(writes.len(), 1);
    assert!(
        matches!(&writes[0].product,Product::Input{rows:actual,material:None} if **actual==rows)
    );
    writer
        .apply_confirmed(vec![
            Confirmed {
                position: 2,
                event: "restored-history".into(),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Live(Box::new(writes[0].product.clone())),
            },
            read_control(ReplayControl::Completion {
                position: 2,
                delivery: None,
            }),
        ])
        .unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let actual = reader
        .conn()
        .unwrap()
        .query_row(
            "SELECT number,produced,chain,length FROM source_revision WHERE input='historical'",
            [],
            |row| <(i64, String, i64, i64)>::try_from(row),
        )
        .unwrap();
    assert_eq!(actual, (7, "rewritten".into(), 4, 0));
    // A conflicting retry cannot rewrite the confirmed historical revision.
    let mut conflict = captured;
    conflict.rows.revision.as_mut().unwrap().chain = 9;
    producer
        .prepare_input(&conflict, prepared.body().unwrap())
        .unwrap();
    assert!(writer.synchronize_primary().is_err());
    assert_eq!(
        reader
            .conn()
            .unwrap()
            .query_row(
                "SELECT chain FROM source_revision WHERE input='historical'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        4
    );
}
