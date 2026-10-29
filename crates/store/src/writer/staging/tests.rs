use super::*;
use std::sync::Arc;

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
            container: None,
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
fn dropped_and_foreign_preparations_leave_no_named_parts_or_committed_blobs() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(first.path()).unwrap();
    let (other, _other_lock) = Archive::open_or_create(second.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let body = b"synthetic private prepared bytes";
    let prepared = archive.prepare_bytes(body.as_slice()).unwrap();
    let hash = prepared.hash();
    let store = archive.blobs().unwrap();
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(hash.0)),
        Err(StoreError::NotFound(_))
    ));
    drop(prepared);
    assert_eq!(
        std::fs::read_dir(archive.parts_dir().unwrap())
            .unwrap()
            .count(),
        0
    );
    let foreign = other.prepare_bytes(body.as_slice()).unwrap();
    assert!(matches!(
        writer.store_bytes(foreign, batch(&hash.to_string(), body.len() as u64)),
        Err(StoreError::Invalid(_))
    ));
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(hash.0)),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(
        std::fs::read_dir(other.parts_dir().unwrap())
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn an_unreferenced_private_stage_is_refused_before_any_group_installation() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(archive.blobs().unwrap());
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
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(source_hash.0)),
        Err(StoreError::NotFound(_))
    ));
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(stray_hash.0)),
        Err(StoreError::NotFound(_))
    ));
    let db = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn preparation_encodes_once_and_hands_off_the_same_inode_after_seekable_inspection() {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::fs::MetadataExt;
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let bytes = vec![b'x'; 2 * 1024 * 1024];
    let (prepared, tail) = archive
        .prepare_stream(bytes.as_slice(), |reader| {
            reader.seek(SeekFrom::End(-7))?;
            let mut tail = Vec::new();
            reader.read_to_end(&mut tail)?;
            Ok(tail)
        })
        .unwrap();
    assert_eq!(tail, b"xxxxxxx");
    let body = prepared.body().unwrap();
    let path = body.path.as_ref().unwrap();
    assert_eq!(
        body.file.metadata().unwrap().ino(),
        std::fs::metadata(path).unwrap().ino()
    );
    assert!(body.descriptor.stored_len < bytes.len() as u64 / 10);
    assert_eq!(
        std::fs::read_dir(archive.parts_dir().unwrap())
            .unwrap()
            .count(),
        1
    );
    let descriptor = body.descriptor;
    drop(prepared);
    assert!(!path.exists());
    // The pinned FD remains sufficient after the producer loses the optional path.
    let received =
        hashed_blobs::Prepared::from_encoded(body.file, descriptor, None, Default::default())
            .unwrap();
    let mut restored = Vec::new();
    received
        .reader(None, Default::default())
        .unwrap()
        .read_to_end(&mut restored)
        .unwrap();
    assert_eq!(restored, bytes);
}

#[test]
fn failed_stream_inspection_releases_the_encoded_preparation() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let result: Result<_> = archive.prepare_stream(b"encoded synthetic source".as_slice(), |_| {
        Err::<(), _>(StoreError::Invalid("synthetic inspection refusal".into()))
    });
    assert!(matches!(result, Err(StoreError::Invalid(_))));
    assert_eq!(
        std::fs::read_dir(archive.parts_dir().unwrap())
            .unwrap()
            .count(),
        0
    );
}
