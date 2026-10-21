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
fn corrupt_inline_content_is_refused_before_any_destination_blob_is_installed() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let hash = polylogue_model::BlobHash::of(b"secret");
    let compressed = zstd::bulk::compress(b"public", 1).unwrap();
    register(&conn, hash, 6, "inline", Some(&compressed));
    drop(conn);
    let target_dir = tempfile::tempdir().unwrap();
    let target = hashed_blobs::Store::open(target_dir.path(), vec![], 0o600).unwrap();
    let entries = std::fs::read_dir(target_dir.path()).unwrap().count();
    let publisher = Published::open(dir.path()).unwrap();
    assert!(matches!(
        publisher.copy_blob_to(&hash.0, &target),
        Err(crate::StoreError::BlobCorrupt(_))
    ));
    assert_eq!(
        std::fs::read_dir(target_dir.path()).unwrap().count(),
        entries
    );
    assert!(!target.contains(&hashed_blobs::Hash(hash.0)));
}

#[cfg(target_os = "linux")]
#[test]
fn large_file_and_historical_inline_blobs_are_placed_without_whole_body_allocations() {
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
        let target_dir = tempfile::tempdir().unwrap();
        let target = hashed_blobs::Store::open(target_dir.path(), vec![], 0o600).unwrap();
        let publisher = Published::open(dir.path()).unwrap();
        let before = peak_kib();
        assert_eq!(publisher.copy_blob_to(&hash.0, &target).unwrap(), SIZE);
        let growth = peak_kib().saturating_sub(before);
        eprintln!("64MiB inline={inline}: placement high-water growth {growth}KiB");
        assert!(growth < 32 * 1024, "whole-body memory growth {growth}KiB");
        let mut reader = target
            .read_plain(&hashed_blobs::Hash(hash.0), Some(SIZE), || Ok(()))
            .unwrap();
        let mut found = blake3::Hasher::new();
        assert_eq!(std::io::copy(&mut reader, &mut found).unwrap(), SIZE);
        assert_eq!(found.finalize(), digest.finalize());
    }
}

#[test]
fn a_corrupt_existing_destination_refuses_placement_and_preserves_the_source() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let hash = polylogue_model::BlobHash::of(b"retained source");
    let compressed = zstd::bulk::compress(b"retained source", 1).unwrap();
    register(&conn, hash, 15, "inline", Some(&compressed));
    drop(conn);
    let target_dir = tempfile::tempdir().unwrap();
    let target = hashed_blobs::Store::open(target_dir.path(), vec![], 0o600).unwrap();
    let installed = target.put(b"retained source".as_slice(), 1).unwrap();
    let name = installed.hex();
    let path = target_dir
        .path()
        .join("blobs")
        .join(&name[..2])
        .join(format!("{name}.zst"));
    std::fs::write(&path, b"corrupt synthetic destination").unwrap();
    assert!(
        Published::open(dir.path())
            .unwrap()
            .copy_blob_to(&hash.0, &target)
            .is_err()
    );
    assert_eq!(
        std::fs::read(path).unwrap(),
        b"corrupt synthetic destination"
    );
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    let kept: Vec<u8> = conn
        .query_row("SELECT data FROM blob WHERE hash=?1", [&hash.0[..]], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(kept, compressed);
}
