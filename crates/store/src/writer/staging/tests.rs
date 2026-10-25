use super::*;
use std::sync::{Arc, Barrier};

fn batch(id: &str, length: u64) -> Batch {
    Batch {
        input: InputKey {
            id: id.into(),
            format: "export-file".into(),
            arrival_rank: 2,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: "/synthetic/export.bin".into(),
            facts: None,
            seen_at_ms: 1,
            import: None,
        }),
        continues: false,
        new_blob: None,
        blobs: Vec::new(),
        length,
        parse: Vec::new(),
        release: Default::default(),
    }
}

#[test]
fn dropping_a_private_preparation_or_refusing_another_archive_writes_no_blob_file() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(first.path()).unwrap();
    let (other, _other_lock) = Archive::open_or_create(second.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let body = b"synthetic private prepared bytes";
    let prepared = archive.prepare_bytes(body.as_slice()).unwrap();
    let hash = prepared.hash();
    let store = hashed_blobs::Store::open(first.path(), vec![], 0o600).unwrap();
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
    drop(prepared);
    assert_eq!(
        std::fs::read_dir(first.path().join("blobs/tmp"))
            .unwrap()
            .count(),
        0
    );
    let foreign = other.prepare_bytes(body.as_slice()).unwrap();
    assert!(matches!(
        writer.store_bytes(foreign, batch(&hash.to_string(), body.len() as u64)),
        Err(StoreError::Invalid(_))
    ));
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
    assert_eq!(
        std::fs::read_dir(second.path().join("blobs/tmp"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn concurrent_excision_and_an_exact_write_cannot_bring_back_a_deleted_blob() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Arc::new(Writer::start(&archive).unwrap());
    let store = Arc::new(hashed_blobs::Store::open(directory.path(), vec![], 0o600).unwrap());
    let body = vec![b'x'; 256 * 1024];
    let prepared = archive.prepare_bytes(body.as_slice()).unwrap();
    let hash = prepared.hash();
    let id = hash.to_string();
    writer
        .store_bytes(prepared, batch(&id, body.len() as u64))
        .unwrap();
    let prepared = archive.prepare_bytes(body.as_slice()).unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let writing = {
        let writer = Arc::clone(&writer);
        let barrier = Arc::clone(&barrier);
        let id = id.clone();
        std::thread::spawn(move || {
            barrier.wait();
            writer.store_bytes(prepared, batch(&id, body.len() as u64))
        })
    };
    let remove = {
        let writer = Arc::clone(&writer);
        let barrier = Arc::clone(&barrier);
        let store = Arc::clone(&store);
        let id = id.clone();
        std::thread::spawn(move || {
            barrier.wait();
            writer.excise(
                crate::excise::ExciseRequest {
                    id: "concurrent-remove".into(),
                    target: crate::excise::ExciseTarget::Input(id),
                    reason: None,
                    dry_run: false,
                },
                store,
            )
        })
    };
    barrier.wait();
    let stored = writing.join().unwrap();
    assert!(stored.is_ok() || matches!(stored, Err(StoreError::InputRefused { .. })));
    remove.join().unwrap().unwrap();
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
    let retry = archive
        .prepare_bytes(b"x".repeat(256 * 1024).as_slice())
        .unwrap();
    assert!(matches!(
        writer.store_bytes(retry, batch(&id, 256 * 1024)),
        Err(StoreError::InputRefused { .. })
    ));
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
}

#[test]
fn an_unreferenced_private_stage_is_refused_before_any_group_installation() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(dir.path(), Vec::new(), 0o600).unwrap());
    let source = archive
        .prepare_bytes(b"synthetic stored source".as_slice())
        .unwrap();
    let stray = archive
        .prepare_bytes(b"synthetic unreferenced stage".as_slice())
        .unwrap();
    let source_hash = source.hash();
    let stray_hash = stray.hash();
    let mut batch = batch(&source_hash.to_string(), source.len());
    batch.new_blob = Some(NewBlob {
        hash: source_hash,
        size: source.len(),
        stored: Stored::File { stored_size: 0 },
    });
    let result = writer
        .sender()
        .unwrap()
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch,
                files: vec![source, stray],
            }],
            Arc::clone(&store),
        )
        .unwrap();
    assert!(
        result[0].is_err(),
        "a private stage without a source owner was stored"
    );
    assert!(!store.contains(&hashed_blobs::Hash(source_hash.0)));
    assert!(!store.contains(&hashed_blobs::Hash(stray_hash.0)));
    let db = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}
