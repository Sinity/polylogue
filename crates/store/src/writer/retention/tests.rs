use super::*;
use std::sync::{Arc, Barrier};

fn batch(id: &str, length: u64) -> Batch {
    Batch {
        input: InputKey {
            id: id.into(),
            format: "export-file".into(),
            route_rank: 2,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: "/synthetic/export.bin".into(),
            facts: None,
            seen_at_ms: 1,
        }),
        continues: false,
        piece: None,
        blobs: Vec::new(),
        length,
        parse: Vec::new(),
        release: Default::default(),
    }
}

#[test]
fn dropping_private_preparation_and_refusing_another_archive_publish_no_cas_name() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(first.path()).unwrap();
    let (other, _other_lock) = Archive::open_or_create(second.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let body = b"synthetic private prepared bytes";
    let prepared = archive.prepare_bytes(&[], body.as_slice()).unwrap();
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
    let foreign = other.prepare_bytes(&[], body.as_slice()).unwrap();
    assert!(matches!(
        writer.retain_bytes(foreign, batch(&hash.to_string(), body.len() as u64)),
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
fn concurrent_excision_and_exact_admission_cannot_resurrect_a_public_blob() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Arc::new(Writer::start(&archive).unwrap());
    let store = Arc::new(hashed_blobs::Store::open(directory.path(), vec![], 0o600).unwrap());
    let body = vec![b'x'; 256 * 1024];
    let prepared = archive.prepare_bytes(&[], body.as_slice()).unwrap();
    let hash = prepared.hash();
    let id = hash.to_string();
    writer
        .retain_bytes(prepared, batch(&id, body.len() as u64))
        .unwrap();
    let prepared = archive.prepare_bytes(&[], body.as_slice()).unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let admit = {
        let writer = Arc::clone(&writer);
        let barrier = Arc::clone(&barrier);
        let id = id.clone();
        std::thread::spawn(move || {
            barrier.wait();
            writer.retain_bytes(prepared, batch(&id, body.len() as u64))
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
                    scope: crate::excise::Scope::Input(id),
                    reason: None,
                    dry_run: false,
                },
                store,
            )
        })
    };
    barrier.wait();
    let admitted = admit.join().unwrap();
    assert!(admitted.is_ok() || matches!(admitted, Err(StoreError::InputRefused { .. })));
    remove.join().unwrap().unwrap();
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
    let retry = archive
        .prepare_bytes(&[], b"x".repeat(256 * 1024).as_slice())
        .unwrap();
    assert!(matches!(
        writer.retain_bytes(retry, batch(&id, 256 * 1024)),
        Err(StoreError::InputRefused { .. })
    ));
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
}
