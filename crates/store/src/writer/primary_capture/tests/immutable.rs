use super::*;
use crate::primary::{
    AcquisitionReceipt, AcquisitionRoute, Confirmed, ConfirmedChange, RawRange, ReplayControl,
    SourceSelector,
};

fn imported(archive: &Archive, bytes: &[u8]) -> PreparedBatch {
    let hash = BlobHash::of(bytes);
    let mut item = source(&hash.to_string(), bytes);
    item.batch.input.route = polylogue_model::Route::Copy;
    item.batch.acquisition = Some(AcquisitionReceipt {
        version: AcquisitionReceipt::VERSION,
        route: AcquisitionRoute::ExplicitImport,
        source: SourceSelector {
            origin: "codex".into(),
            machine: "machine".into(),
            path_bytes: format!("/home/op/{hash}").into_bytes(),
        },
        range: RawRange {
            start: 0,
            end: bytes.len() as u64,
        },
        hash,
        observed_at: polylogue_model::Timestamp::parse_rfc3339("2026-01-01T00:00:00Z").unwrap(),
        generation: None,
        raw_total: None,
        raw_predecessor: None,
        chunks: Vec::new(),
    });
    item.files.push(archive.prepare_bytes(bytes).unwrap());
    item
}

#[test]
fn completed_imports_keep_their_receipt_when_repeated_or_mixed_with_new_members() {
    let directory = tempfile::tempdir().unwrap();
    let producer_root = directory.path().join("producer");
    std::fs::create_dir(&producer_root).unwrap();
    let (archive, _lock) =
        Archive::open_primary(&directory.path().join("cache"), &producer_root).unwrap();
    let producer = Arc::new(OfflineProducer {
        root: producer_root.clone(),
    });
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    writer
        .apply_confirmed(vec![read_control(ReplayControl::Reset {
            scope: "scope".into(),
        })])
        .unwrap();
    writer
        .apply_confirmed(vec![read_control(ReplayControl::Completion {
            position: 0,
            delivery: None,
        })])
        .unwrap();
    let bytes = b"immutable original";
    let store = |items| {
        writer
            .store_group(items, Arc::new(archive.blobs().unwrap()))
            .unwrap()
    };
    assert!(matches!(
        store(vec![imported(&archive, bytes)]).remove(0),
        Err(StoreError::PrimaryPending { .. })
    ));
    let captured = producer.captured();
    let prepared = hashed_blobs::Preparation::new(directory.path().join("scratch"))
        .unwrap()
        .plain(&bytes[..], 1)
        .unwrap();
    let mut segment =
        hashed_blobs::SegmentWriter::open(&producer_root, "segments/original.hb", 0, &[], 0o600)
            .unwrap();
    let group = segment
        .append_group([Ok(prepared)], |_| Ok(None))
        .unwrap()
        .wait()
        .unwrap();
    segment.acknowledge(&group).unwrap();
    let material = Material {
        id: "original".into(),
        hash: BlobHash::of(bytes),
        length: bytes.len() as u64,
        extents: vec![group.extents[0].encode()],
    };
    writer
        .apply_confirmed(vec![
            Confirmed {
                position: 1,
                event: "original-event".into(),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Live(Box::new(Product::Input {
                    rows: Box::new(captured.rows.clone()),
                    material: Some(material),
                })),
            },
            read_control(ReplayControl::Completion {
                position: 1,
                delivery: None,
            }),
        ])
        .unwrap();
    // The completed spool entry is retired, as it is by the real consumer.
    std::fs::remove_file(producer_root.join("capture.json")).unwrap();
    let repeated = store(vec![imported(&archive, bytes)]).remove(0).unwrap();
    assert!(repeated.already_stored);
    assert_eq!(repeated.bytes_written, 0);
    assert_eq!(repeated.chain, 1);
    assert!(
        !producer_root.join("capture.json").exists(),
        "a confirmed repeat must not offer regenerated frame bytes"
    );
    let fresh = b"new original";
    let mut mixed = store(vec![imported(&archive, bytes), imported(&archive, fresh)]);
    assert!(mixed.remove(0).unwrap().already_stored);
    assert!(matches!(
        mixed.remove(0),
        Err(StoreError::PrimaryPending { .. })
    ));
    let pending = producer.captured();
    assert_eq!(pending.hash, BlobHash::of(fresh));
    let pending_intent = pending.group.as_ref().unwrap().intent.clone();
    let mut retry = store(vec![imported(&archive, bytes), imported(&archive, fresh)]);
    assert!(retry.remove(0).unwrap().already_stored);
    assert!(matches!(
        retry.remove(0),
        Err(StoreError::PrimaryPending { .. })
    ));
    assert_eq!(
        producer.captured().group.as_ref().unwrap().intent,
        pending_intent
    );
    // A reused content identity with different confirmed semantics is a conflict.
    let mut wrong_format = imported(&archive, bytes);
    wrong_format.batch.input.format = "other".into();
    wrong_format
        .batch
        .acquisition
        .as_mut()
        .unwrap()
        .source
        .origin = "other".into();
    assert!(matches!(
        store(vec![wrong_format]).remove(0),
        Err(StoreError::Conflict(_))
    ));
    let mut other_receiver = imported(&archive, bytes);
    other_receiver.batch.observation.as_mut().unwrap().machine = "second-machine".into();
    other_receiver
        .batch
        .acquisition
        .as_mut()
        .unwrap()
        .source
        .machine = "second-machine".into();
    assert!(
        matches!(
            store(vec![other_receiver]).remove(0),
            Err(StoreError::PrimaryPending { .. })
        ),
        "a different receiver must still capture its own acquisition"
    );
    let index = crate::archive::open_read(&archive.index_db()).unwrap();
    assert_eq!(
        index
            .scalar::<i64>("SELECT count(*) FROM source_revision", [])
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .scalar::<i64>("SELECT count(*) FROM source_material", [])
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .scalar::<i64>("SELECT count(*) FROM source_observation", [])
            .unwrap(),
        1
    );
    drop(index);
    crate::fault::clear();
    crate::fault::arm(
        "primary.scope_reset_commit",
        "",
        crate::fault::Action::Fail,
        0,
        1,
    );
    assert!(
        writer
            .apply_confirmed(vec![read_control(ReplayControl::Reset {
                scope: "new-scope".into()
            })])
            .is_err()
    );
    crate::fault::clear();
    assert!(matches!(
        store(vec![imported(&archive, bytes)]).remove(0),
        Err(StoreError::ScopeChanged)
    ));
}
