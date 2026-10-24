use super::*;

pub(super) async fn parent_retirement(sinex: &mut Sinex, store: &Path, root: &Path) {
    use polylogue_store::{Delivery, DeliveryOutcome, UserRequest, UserWrite};
    let (archive, _lock) = Archive::open_or_create(&root.join("legacy-source")).unwrap();
    let writer = polylogue_store::Writer::start(&archive).unwrap();
    writer
        .user(UserRequest::now(
            "legacy:[opaque]".into(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "cli".into(),
                event: "advisory".into(),
                cwd: None,
                text: "Synthetic legacy body".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ))
        .unwrap();
    writer
        .user(UserRequest::now(
            "legacy-event".into(),
            UserWrite::DeliveryEvent {
                delivery: "legacy:[opaque]".into(),
                what: DeliveryOutcome::Failed,
                detail: Some("legacy private evidence".into()),
            },
        ))
        .unwrap();
    drop(writer);
    let mut db = polylogue_store::archive::open_write(
        &archive.archive_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    db.execute("UPDATE delivery_event SET commit_seq=0", [])
        .unwrap();
    let source = crate::ArchiveRows::open(&archive.root).unwrap();
    let id = source.archive_id().unwrap();
    let unit = inventory_units(&source)
        .into_iter()
        .find(|u| u.kind == "delivery")
        .unwrap();
    let batch = source.read(&unit).unwrap();
    let row_count = batch.rows;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut &batch.file, &mut bytes).unwrap();
    let mut legacy = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|b| !b.is_empty()) {
        let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if value["table"] == "delivery_event" {
            value["row"].as_object_mut().unwrap().remove("commit_seq");
        }
        serde_json::to_writer(&mut legacy, &value).unwrap();
        legacy.push(b'\n');
    }
    drop(source);
    assert_eq!(
        db.query_row("SELECT count(*) FROM vector_deletion_policy", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    db.execute_batch("ALTER TABLE machine DROP COLUMN first_commit_seq; DROP TABLE extracted_blob; DROP TABLE excision_document; DROP TABLE vector_deletion_policy;
        DELETE FROM meta WHERE key='vector_policy_revision';
        DROP INDEX observation_commit_seq; DROP INDEX delivery_event_commit_seq;
        ALTER TABLE observation DROP COLUMN commit_seq; ALTER TABLE delivery_event DROP COLUMN commit_seq;
        PRAGMA user_version=4;").unwrap();
    // Publish the exact schema-4 closure, keeping its original frame and hash.
    let mut legacy_file = tempfile::tempfile().unwrap();
    std::io::Write::write_all(&mut legacy_file, &legacy).unwrap();
    std::io::Seek::rewind(&mut legacy_file).unwrap();
    let (material, old_hash) = sinex.place_batch(&id, &mut legacy_file).await.unwrap();
    let event = crate::PolylogueRows {
        archive: id.clone(),
        position: 1,
        local_seq: unit.commit_seq,
        unit: unit.kind.clone(),
        key: unit.key.clone(),
        batch: format!("blake3:{}", old_hash.hex()),
        rows: row_count,
        blobs: vec![],
        excision: None,
    };
    sinex
        .send(&crate::Ingress::Events {
            frame: format!("rows:{id}:1"),
            failures: vec![],
            events: vec![crate::ProducerEvent {
                event_type: "polylogue.rows".into(),
                payload: serde_json::to_value(event).unwrap(),
                location: Some(crate::RecordLocation {
                    material,
                    anchor: 0,
                }),
                ordinal: 0,
                parents: vec![],
                reading_order: 1,
                class: crate::Class::Open,
            }],
        })
        .await
        .unwrap();
    polylogue_store::schema::migrate_archive(&mut db).unwrap();
    let sent = sinex.publish(&archive.root).await.unwrap();
    assert_eq!(
        (sent.units, sent.held),
        (1, 1),
        "backfill must append after the held legacy parent"
    );
    let writer = polylogue_store::Writer::start(&archive).unwrap();
    writer
        .excise(
            polylogue_store::excise::ExciseRequest {
                id: "legacy-redact".into(),
                scope: polylogue_store::excise::Scope::Text("private".into()),
                reason: None,
                dry_run: false,
            },
            std::sync::Arc::new(
                hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
            ),
        )
        .unwrap();
    drop(writer);
    assert_eq!(sinex.publish(&archive.root).await.unwrap().units, 1);
    let (target, _lock) = Archive::open_or_create(&root.join("legacy-restored")).unwrap();
    let deadline = Instant::now();
    loop {
        if sinex.restore(&target, &id).await.unwrap().units == 3 {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        !hashed_blobs::Store::open(store.to_path_buf(), vec![], 0o600)
            .unwrap()
            .contains(&old_hash)
    );
    let (fresh, _lock) = Archive::open_or_create(&root.join("legacy-after-retirement")).unwrap();
    assert_eq!(sinex.restore(&fresh, &id).await.unwrap().units, 3);
    let restored = polylogue_store::archive::open_read(&fresh.archive_db()).unwrap();
    assert_eq!(
        restored
            .query_row("SELECT detail FROM delivery_event", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "legacy         evidence"
    );
    assert_eq!(sinex.restore(&fresh, &id).await.unwrap().rows, 0);
}
