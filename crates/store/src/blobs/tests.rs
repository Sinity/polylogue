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

#[test]
fn a_retained_range_decodes_only_intersecting_extents_and_refuses_requested_gaps() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute(
        "INSERT INTO input(id,format) VALUES ('input','claude-code')",
        [],
    )
    .unwrap();
    conn.execute("INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',1,'first',1,6,1)", []).unwrap();
    for (offset, bytes, corrupt) in [(0, b"abc", true), (3, b"def", false)] {
        let hash = BlobHash::of(bytes);
        let data = if corrupt {
            vec![0]
        } else {
            zstd::bulk::compress(bytes, 3).unwrap()
        };
        conn.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state,data) VALUES (?1,3,?2,'inline','plain','present',?3)", rusqlite::params![hash.0.as_slice(), data.len() as i64, data]).unwrap();
        conn.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES ('input',1,?1,3,?2,0)", rusqlite::params![offset,hash.0.as_slice()]).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let tail = StoredChain::load_range(&conn, "", "input", 1, 3, 6).unwrap();
    assert_eq!(tail.extents.len(), 1);
    assert_eq!(tail.read_range(&store, 3, 6).unwrap(), b"def");
    assert!(StoredChain::load_range(&conn, "", "input", 1, 2, 6).is_err());
    conn.execute("DELETE FROM extent WHERE offset=3", [])
        .unwrap();
    let missing = StoredChain::load_range(&conn, "", "input", 1, 3, 6).unwrap();
    assert!(missing.read_range(&store, 3, 6).is_err());
}
