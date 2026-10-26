use super::*;

fn stats(source: &Imported) -> hashed_blobs::VerificationStats {
    match &source.reader {
        Decoder::Plain(reader) => reader.stats(),
        Decoder::Chain(_) | Decoder::Unopened(_) => panic!("the plain reader is the counted case"),
    }
}

#[test]
fn stored_bytes_are_opened_at_their_first_read() {
    let home = polylogue_testkit::ClaudeHome::new();
    // Larger than the inline limit, so the bytes are a blob file.
    let body = "synthetic stored body ".repeat(10_000);
    home.append(
        "large.jsonl",
        &polylogue_testkit::claude_user("u1", "s", &body),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let blobs = std::sync::Arc::clone(&engine.blobs);
    let (input, chain, length): (String, i64, i64) =
        rusqlite::Connection::open(home.root.join("archive.db"))
            .unwrap()
            .query_row("SELECT input, chain, length FROM revision", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
    let reads = home.reads();
    let stored = reads
        .reader()
        .unwrap()
        .stored_chain(&input, chain, length as u64)
        .unwrap();
    let hex = hashed_blobs::Hash(stored.single_file_blob().expect("a blob file").0).hex();
    let file = home
        .root
        .join("blobs")
        .join(&hex[..2])
        .join(format!("{hex}.zst"));
    let aside = home.root.join("aside.zst");
    std::fs::rename(&file, &aside).unwrap();
    // A plan holds no open file: making one succeeds, reading it fails.
    let mut missing = Imported::from_chain(stored.clone(), std::sync::Arc::clone(&blobs));
    assert!(missing.read_range(0, 16).is_err());
    std::fs::rename(&aside, &file).unwrap();
    let mut present = Imported::from_chain(stored, blobs);
    assert_eq!(present.read_range(0, 1).unwrap(), b"{");
}

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
    let verified = stats(&source);
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
    let stats = stats(&source);
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
    assert_eq!(stats(&source).decoded_bytes, 34);
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
    assert_eq!(stats(&source).decoded_bytes, 32);
}

#[test]
fn export_scanning_uses_one_forward_decode_after_verification() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600)
        .unwrap()
        .with_frame_size(4 << 20);
    let mut bytes = b"[{}".to_vec();
    for _ in 0..10_000 {
        bytes.extend_from_slice(b",{\"padding\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}");
    }
    bytes.push(b']');
    let hash = store.put(&bytes[..], 1).unwrap();
    let mut source = store
        .read_plain(&hash, Some(bytes.len() as u64), || Ok(()))
        .unwrap();
    let verified = source.stats();
    let scan = polylogue_sources::export::scan(&mut source, |_, _| Ok(())).unwrap();
    assert_eq!(scan.elements, 10_001);
    assert!(scan.peak_element_bytes < 256);
    let stats = source.stats();
    assert_eq!(stats.decoded_bytes, 2 * bytes.len() as u64);
    assert_eq!(stats.frames, verified.frames + 1);
}
