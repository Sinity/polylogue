use super::*;

#[test]
fn detects_a_changed_tail() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("f");
    std::fs::write(&p, b"hello world\nmore\n").unwrap();
    assert!(tail_intact(&p, 12, b"world\n").unwrap());
    assert!(!tail_intact(&p, 12, b"WORLD\n").unwrap());
    assert_eq!(read_range(&p, 6, 11).unwrap(), b"world");
    assert_eq!(tail_of(b"abc"), b"abc");
}

#[test]
fn imported_ranges_verify_once_and_decode_once_even_with_an_oversized_frame() {
    use polylogue_model::BlobHash;
    let dir = tempfile::tempdir().unwrap();
    let size = 10 * 1024 * 1024;
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600)
        .unwrap()
        .with_frame_size(size);
    let hash = store
        .put(std::io::repeat(b'x').take(size as u64), 1)
        .unwrap();
    let mut source = Imported::open(&store, BlobHash(hash.0), size as u64).unwrap();
    let verified = source.reader.stats();
    assert_eq!(verified.decoded_bytes, size as u64);
    let chunk = 1024;
    for start in (0..size).step_by(chunk) {
        let end = (start + chunk).min(size);
        assert_eq!(
            source.read_range(start as u64, end as u64).unwrap(),
            vec![b'x'; end - start]
        );
        source.consume(end as u64);
        assert!(source.pending.is_empty());
        assert!(source.pending.capacity() <= chunk);
    }
    let stats = source.reader.stats();
    assert_eq!(stats.decoded_bytes, 2 * size as u64);
    assert_eq!(stats.frames, verified.frames + 1);
    assert!(stats.compressed_bytes <= 2 * verified.compressed_bytes);
}

#[test]
fn imported_ranges_keep_only_the_incomplete_record_overlap() {
    use polylogue_model::BlobHash;
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let hash = store.put(&b"complete\npartial\n"[..], 1).unwrap();
    let mut source = Imported::open(&store, BlobHash(hash.0), 17).unwrap();
    assert_eq!(source.read_range(0, 12).unwrap(), b"complete\npar");
    source.consume(9);
    assert_eq!(source.pending, b"par");
    assert_eq!(source.read_range(9, 17).unwrap(), b"partial\n");
    source.consume(17);
    assert!(source.pending.is_empty());
    assert_eq!(source.reader.stats().decoded_bytes, 34);
}

#[test]
fn a_smaller_parse_range_can_reuse_already_decoded_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let hash = store.put(&b"0123456789abcdef"[..], 1).unwrap();
    let mut source = Imported::open(&store, polylogue_model::BlobHash(hash.0), 16).unwrap();
    assert_eq!(source.read_range(0, 14).unwrap(), b"0123456789abcd");
    source.consume(4);
    assert_eq!(source.read_range(4, 8).unwrap(), b"4567");
    source.consume(8);
    assert_eq!(source.read_range(8, 16).unwrap(), b"89abcdef");
    assert_eq!(source.reader.stats().decoded_bytes, 32);
}
