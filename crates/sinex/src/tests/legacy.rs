use super::*;

pub(super) async fn parent_retirement(sinex: &mut Sinex, store: &Path, root: &Path) {
    let archive = polylogue_store::fixture::archive4::create(&root.join("legacy-source")).unwrap();
    let mut db = polylogue_store::archive::open_write(
        &archive.archive_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    let legacy = polylogue_store::fixture::archive4::delivery(&db).unwrap();
    let id: String = db
        .query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let unit = polylogue_store::sinex_rows::Unit {
        commit_seq: 1,
        kind: "delivery".into(),
        key: "legacy:[opaque]".into(),
    };
    let row_count = legacy
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .count() as u64;
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
    drop(db);
    let (archive, _lock) = Archive::open_or_create(&archive.root).unwrap();
    let source = crate::ArchiveRows::open(&archive.root).unwrap();
    assert_eq!(source.archive_id().unwrap(), id);
    drop(source);
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
