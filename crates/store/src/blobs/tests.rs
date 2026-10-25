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
    let small = NewBlob::prepare(&store, b"hello").unwrap();
    assert!(matches!(small.stored, Stored::Inline(_)));
    let big = vec![7u8; INLINE_LIMIT + 1];
    let large = NewBlob::prepare(&store, &big).unwrap();
    assert!(matches!(large.stored, Stored::File { .. }));
    assert!(store.contains(&hashed_blobs::Hash(large.hash.0)));
}
