use super::*;

fn facts(inode: u64, size: u64, mtime: i128) -> Option<FileFacts> {
    Some(FileFacts {
        device: 1,
        inode,
        size,
        mtime_ns: mtime,
        ctime_ns: mtime,
    })
}

#[test]
fn classifies_every_kind_of_change() {
    assert_eq!(classify(None, facts(1, 10, 1)), Change::First);
    assert_eq!(
        classify(facts(1, 10, 1), facts(1, 10, 1)),
        Change::Unchanged
    );
    assert_eq!(
        classify(facts(1, 10, 1), facts(1, 25, 2)),
        Change::Appended { from: 10 }
    );
    assert_eq!(classify(facts(1, 10, 1), facts(1, 5, 2)), Change::Rewritten);
    assert_eq!(
        classify(facts(1, 10, 1), facts(2, 30, 2)),
        Change::Rewritten
    );
    assert_eq!(classify(facts(1, 10, 1), None), Change::Disappeared);
    assert_eq!(classify(None, None), Change::Unchanged);
}

#[test]
fn small_pieces_are_inline_and_large_ones_files() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let small = Piece::prepare(&store, b"hello").unwrap();
    assert!(matches!(small.stored, Stored::Inline(_)));
    let big = vec![7u8; INLINE_LIMIT + 1];
    let large = Piece::prepare(&store, &big).unwrap();
    assert!(matches!(large.stored, Stored::File { .. }));
    assert!(store.contains(&hashed_blobs::Hash(large.hash.0)));
}

#[test]
fn sequential_chain_reader_obeys_inline_and_file_extent_boundaries() {
    use std::io::Read;
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap());
    let file = Piece::prepare(&store, &vec![b'x'; INLINE_LIMIT + 1]).unwrap();
    let chain = StoredChain {
        length: 6,
        extents: vec![
            StoredExtent {
                offset: 0,
                length: 3,
                blob: BlobHash::of(b"_abc_"),
                blob_offset: 1,
                inline: Some(Arc::new(b"_abc_".to_vec())),
            },
            StoredExtent {
                offset: 3,
                length: 3,
                blob: file.hash,
                blob_offset: 5,
                inline: None,
            },
        ],
    };
    let mut reader = chain.reader(Arc::clone(&store));
    let mut result = Vec::new();
    for _ in 0..6 {
        let mut byte = [0];
        assert_eq!(reader.read(&mut byte).unwrap(), 1);
        result.push(byte[0]);
    }
    assert_eq!(result, b"abcxxx");
    assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    let mut missing = chain.clone();
    missing.extents[1].offset = 4;
    assert!(missing.reader(store).read_to_end(&mut Vec::new()).is_err());
}
