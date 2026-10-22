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
fn streamed_pieces_match_slice_storage_across_the_inline_threshold() {
    use std::io::Read;
    struct Short<'a>(&'a [u8]);
    impl Read for Short<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let take = out.len().min(19);
            self.0.read(&mut out[..take])
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    for length in [
        0,
        1,
        INLINE_LIMIT - 1,
        INLINE_LIMIT,
        INLINE_LIMIT + 1,
        2 << 20,
    ] {
        let bytes: Vec<_> = (0..length).map(|n| (n % 251) as u8).collect();
        let piece = Piece::prepare_reader(&store, Short(&bytes)).unwrap();
        assert_eq!(piece, Piece::prepare(&store, &bytes).unwrap());
    }
}

#[test]
fn a_failed_stream_never_returns_a_successful_piece() {
    use std::io::{self, Read};
    struct Fails(usize);
    impl Read for Fails {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.0 == 0 {
                return Err(io::Error::other("synthetic read failure"));
            }
            let n = out.len().min(self.0);
            out[..n].fill(b'x');
            self.0 -= n;
            Ok(n)
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    for n in [11, INLINE_LIMIT, INLINE_LIMIT + 23] {
        assert!(matches!(
            Piece::prepare_reader(&store, Fails(n)),
            Err(crate::StoreError::Io(_)) | Err(crate::StoreError::Blobs(_))
        ));
        assert!(!store.contains(&hashed_blobs::Hash::of(&vec![b'x'; n])));
    }
}
