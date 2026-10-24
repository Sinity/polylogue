use super::*;
use crate::{
    Archive,
    archive::{cache, open_write},
};

fn register(
    conn: &Connection,
    hash: polylogue_model::BlobHash,
    size: u64,
    place: &str,
    data: Option<&[u8]>,
) {
    conn.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state,data) VALUES (?1,?2,0,?3,'plain','present',?4)",
        rusqlite::params![&hash.0[..],size as i64,place,data]).unwrap();
}

#[test]
fn corrupt_inline_content_refuses_a_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let hash = polylogue_model::BlobHash::of(b"secret");
    let compressed = zstd::bulk::compress(b"public", 1).unwrap();
    register(&conn, hash, 6, "inline", Some(&compressed));
    drop(conn);
    let publisher = Published::open(dir.path()).unwrap();
    assert!(matches!(
        publisher.blob_snapshot(&hash.0),
        Err(crate::StoreError::BlobCorrupt(_))
    ));
}

#[cfg(target_os = "linux")]
#[test]
fn large_file_and_historical_inline_snapshots_do_not_allocate_whole_bodies() {
    use std::io::Read;
    fn peak_kib() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    const SIZE: u64 = 64 * 1024 * 1024;
    for inline in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let mut digest = blake3::Hasher::new();
        std::io::copy(&mut std::io::repeat(b'x').take(SIZE), &mut digest).unwrap();
        let hash = polylogue_model::BlobHash(*digest.finalize().as_bytes());
        let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        if inline {
            let data = zstd::stream::encode_all(std::io::repeat(b'x').take(SIZE), 1).unwrap();
            register(&conn, hash, SIZE, "inline", Some(&data));
        } else {
            let source = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
            assert_eq!(
                source.put(std::io::repeat(b'x').take(SIZE), 1).unwrap().0,
                hash.0
            );
            register(&conn, hash, SIZE, "file", None);
        }
        drop(conn);
        let publisher = Published::open(dir.path()).unwrap();
        let before = peak_kib();
        let mut snapshot = publisher.blob_snapshot(&hash.0).unwrap();
        assert_eq!(snapshot.length, SIZE);
        let growth = peak_kib().saturating_sub(before);
        eprintln!("64MiB inline={inline}: placement high-water growth {growth}KiB");
        assert!(growth < 32 * 1024, "whole-body memory growth {growth}KiB");
        let mut found = blake3::Hasher::new();
        assert_eq!(std::io::copy(&mut snapshot.file, &mut found).unwrap(), SIZE);
        assert_eq!(found.finalize(), digest.finalize());
    }
}
