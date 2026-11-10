use super::*;
use std::sync::Arc;

fn batch(id: &str, length: u64) -> Batch {
    Batch {
        required_disposition: None,
        input: InputKey {
            id: id.into(),
            format: "export-file".into(),
            route: polylogue_model::Route::Export,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: "/synthetic/export.bin".into(),
            path_bytes: b"/synthetic/export.bin".to_vec(),
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
        acquisition: None,
        basis: None,
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
    assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
    let foreign = other.prepare_bytes(body.as_slice()).unwrap();
    assert!(matches!(
        writer.store_bytes(foreign, batch(&hash.to_string(), body.len() as u64)),
        Err(StoreError::Invalid(_))
    ));
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(hash.0)),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(std::fs::read_dir(other.staging_dir()).unwrap().count(), 0);
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
    batch.new_blob = Some(NewBlob::whole(source_hash, source.len()));
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
    assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 1);
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
    assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
}

#[test]
fn encoded_preparation_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_ENCODED_PREP_ROOT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let path = root.join("archive");
    let (_lock, _prepared) = if std::env::var_os("POLYLOGUE_ENCODED_PREP_PRIMARY").is_some() {
        let (archive, lock) = Archive::open_primary(&path, &root.join("bytes")).unwrap();
        (
            lock,
            archive
                .prepare_bytes(b"unaccepted encoded source".as_slice())
                .unwrap(),
        )
    } else {
        let (archive, lock) = Archive::open_or_create(&path).unwrap();
        (
            lock,
            archive
                .prepare_bytes(b"unaccepted encoded source".as_slice())
                .unwrap(),
        )
    };
    std::fs::write(root.join("ready"), b"ready").unwrap();
    loop {
        std::thread::park();
    }
}

#[test]
fn ordinary_open_releases_encoded_preparations_after_process_death_in_both_modes() {
    for primary in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "writer::staging::tests::encoded_preparation_crash_child",
                "--nocapture",
            ])
            .env("POLYLOGUE_ENCODED_PREP_ROOT", directory.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        if primary {
            command.env("POLYLOGUE_ENCODED_PREP_PRIMARY", "1");
        }
        let mut child = command.spawn().unwrap();
        let started = std::time::Instant::now();
        while !directory.path().join("ready").exists() {
            if child.try_wait().unwrap().is_some()
                || started.elapsed() > std::time::Duration::from_secs(30)
            {
                let _ = child.kill();
                let _ = child.wait();
                panic!("preparation child did not signal that its bytes were prepared");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let path = directory.path().join("archive");
        assert_eq!(std::fs::read_dir(path.join("staging")).unwrap().count(), 1);
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        let (archive, _lock) = if primary {
            Archive::open_primary(&path, &directory.path().join("bytes")).unwrap()
        } else {
            Archive::open_or_create(&path).unwrap()
        };
        assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
    }
}

#[test]
fn a_late_encoding_refusal_publishes_no_group_and_reopen_can_retry_original_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let bodies = [
        b"first synthetic source".as_slice(),
        b"second synthetic source".as_slice(),
    ];
    {
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let mut group = Vec::new();
        for body in bodies {
            let prepared = archive.prepare_bytes(body).unwrap();
            let mut item = batch(&prepared.hash().to_string(), prepared.len());
            item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
            group.push(crate::staging::PreparedBatch {
                batch: item,
                files: vec![prepared],
            });
        }
        let late = group[1].files[0].body().unwrap().path.unwrap();
        std::fs::write(late, b"synthetic damaged encoding").unwrap();
        let result = writer
            .sender()
            .unwrap()
            .store_group(group, Arc::new(archive.blobs().unwrap()))
            .unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(Result::is_err));
        for body in bodies {
            assert!(matches!(
                archive
                    .blobs()
                    .unwrap()
                    .extent(&hashed_blobs::Hash::of(body)),
                Err(StoreError::NotFound(_))
            ));
        }
    }
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut group = Vec::new();
    for body in bodies {
        let prepared = archive.prepare_bytes(body).unwrap();
        let mut item = batch(&prepared.hash().to_string(), prepared.len());
        item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
        group.push(crate::staging::PreparedBatch {
            batch: item,
            files: vec![prepared],
        });
    }
    let results = writer
        .sender()
        .unwrap()
        .store_group(group, Arc::new(archive.blobs().unwrap()))
        .unwrap();
    assert!(results.iter().all(Result::is_ok));
    for body in bodies {
        let mut reader = archive
            .blobs()
            .unwrap()
            .read(&hashed_blobs::Hash::of(body))
            .unwrap();
        let mut restored = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut restored).unwrap();
        assert_eq!(restored, body);
    }
}

#[test]
fn an_equal_length_change_to_a_staged_file_is_refused_at_append() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let body = b"synthetic staged source".as_slice();
    let prepared = archive.prepare_bytes(body).unwrap();
    let mut item = batch(&prepared.hash().to_string(), prepared.len());
    item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
    // One encoded byte changes and the length stays: only the append's
    // encoded digest can see it.
    let path = prepared.body().unwrap().path.unwrap();
    let mut encoded = std::fs::read(&path).unwrap();
    encoded[0] ^= 1;
    std::fs::write(&path, &encoded).unwrap();
    let results = writer
        .sender()
        .unwrap()
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: item,
                files: vec![prepared],
            }],
            Arc::new(archive.blobs().unwrap()),
        )
        .unwrap();
    assert!(results[0].is_err());
    assert!(matches!(
        archive
            .blobs()
            .unwrap()
            .extent(&hashed_blobs::Hash::of(body)),
        Err(StoreError::NotFound(_))
    ));
}
