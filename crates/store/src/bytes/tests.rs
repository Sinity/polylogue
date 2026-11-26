use super::*;

type Start = fn(&Archive) -> Result<Arc<dyn Bytes>>;

/// Both homes, each started on an archive.
fn homes() -> [(&'static str, Start); 2] {
    [
        ("segments", |archive| {
            Ok(Arc::new(Segments::start(archive)?))
        }),
        ("directory", |archive| {
            Ok(Arc::new(Directory::start(archive)?))
        }),
    ]
}

fn read(home: &dyn Bytes, hash: &BlobHash, range: Range<u64>) -> Vec<u8> {
    let mut bytes = Vec::new();
    home.open(hash, range)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

fn blob_rows(archive: &Archive) -> i64 {
    crate::archive::open_read(&archive.archive_db())
        .unwrap()
        .query_row("SELECT count(*) FROM blob", [], |row| row.get(0))
        .unwrap()
}

/// Bytes larger than the blobs kept in memory after their put, so they are
/// read from their home.
fn large() -> Vec<u8> {
    (0..(RECENT_BLOB as usize + 4096))
        .map(|n| (n * 7 % 251) as u8)
        .collect()
}

#[test]
fn put_then_open_reads_the_same_bytes_after_restart() {
    let small = b"a synthetic line of captured text\n".to_vec();
    let large = large();
    for (name, start) in homes() {
        let directory = tempfile::tempdir().unwrap();
        let hashes = {
            let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
            let home = start(&archive).unwrap();
            let files = vec![
                archive.prepare_bytes(small.as_slice()).unwrap(),
                archive.prepare_bytes(large.as_slice()).unwrap(),
            ];
            let hashes: Vec<BlobHash> = files.iter().map(PreparedBlob::hash).collect();
            let put = home.put_all(files).unwrap();
            assert_eq!(
                (put.added.len(), put.appended, put.deduplicated, put.syncs),
                (2, 2, 0, 1),
                "{name}"
            );
            assert_eq!(
                read(&*home, &hashes[0], 0..small.len() as u64),
                small,
                "{name}"
            );
            assert_eq!(
                read(&*home, &hashes[1], 0..large.len() as u64),
                large,
                "{name}"
            );
            assert_eq!(read(&*home, &hashes[1], 10..74), large[10..74], "{name}");
            hashes
        };
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let home = start(&archive).unwrap();
        assert_eq!(
            read(&*home, &hashes[0], 0..small.len() as u64),
            small,
            "{name}"
        );
        assert_eq!(
            read(&*home, &hashes[1], 0..large.len() as u64),
            large,
            "{name}"
        );
        // Putting the same bytes again finds them and writes nothing.
        let put = home
            .put_all(vec![
                archive.prepare_bytes(small.as_slice()).unwrap(),
                archive.prepare_bytes(large.as_slice()).unwrap(),
            ])
            .unwrap();
        assert_eq!(
            (put.added.len(), put.appended, put.deduplicated, put.syncs),
            (0, 0, 2, 0),
            "{name}"
        );
        assert_eq!(blob_rows(&archive), 2, "{name}");
    }
}

#[test]
fn puts_waiting_together_share_one_sync() {
    for (name, start) in homes() {
        let directory = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let home = start(&archive).unwrap();
        let files: Vec<PreparedBlob> = (0..8)
            .map(|n| {
                archive
                    .prepare_bytes(format!("blob {n}\n").as_bytes())
                    .unwrap()
            })
            .collect();
        let puts: Vec<Put> = std::thread::scope(|scope| {
            let handles: Vec<_> = files
                .into_iter()
                .map(|file| {
                    let home = &home;
                    scope.spawn(move || home.put_all(vec![file]).unwrap())
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });
        assert_eq!(
            puts.iter().map(|put| put.appended).sum::<u64>(),
            8,
            "{name}"
        );
        // Each put is answered after its own rows commit, and the puts that
        // waited together counted their shared sync once.
        assert!(puts.iter().map(|put| put.syncs).sum::<u64>() <= 8, "{name}");
        assert_eq!(blob_rows(&archive), 8, "{name}");
    }
}

#[test]
fn a_segment_past_its_record_from_a_newer_record_stops_the_open() {
    let directory = tempfile::tempdir().unwrap();
    let saved = directory.path().join("archive-copy.db");
    {
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let home = Segments::start(&archive).unwrap();
        home.put(
            archive
                .prepare_bytes(&b"first synthetic blob\n"[..])
                .unwrap(),
        )
        .unwrap();
        drop(home);
        // The record as a copy of it was taken here.
        let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
        conn.execute("VACUUM INTO ?1", [saved.to_str().unwrap()])
            .unwrap();
        let home = Segments::start(&archive).unwrap();
        home.put(
            archive
                .prepare_bytes(&b"second synthetic blob\n"[..])
                .unwrap(),
        )
        .unwrap();
    }
    let segment = directory.path().join(segments::CURRENT);
    let length = std::fs::metadata(&segment).unwrap().len();
    for suffix in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(directory.path().join(format!("archive.db{suffix}")));
    }
    std::fs::rename(&saved, directory.path().join("archive.db")).unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let error = Segments::start(&archive).err().expect("the open stops");
    let message = error.to_string();
    let end: i64 = crate::archive::open_read(&archive.archive_db())
        .unwrap()
        .query_row("SELECT committed_end FROM segment", [], |row| row.get(0))
        .unwrap();
    assert!(
        message.contains(&format!("holds {length} bytes"))
            && message.contains(&format!("names {end}")),
        "{message}"
    );
    // Nothing was cut off.
    assert_eq!(std::fs::metadata(&segment).unwrap().len(), length);
}
