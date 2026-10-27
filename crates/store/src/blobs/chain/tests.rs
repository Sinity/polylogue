use super::*;
use crate::blobs::{INLINE_LIMIT, NewBlob, Stored};
use rusqlite::params;
use std::io::Read;

struct Fixture {
    conn: Connection,
    store: Arc<hashed_blobs::Store>,
    _dir: tempfile::TempDir,
}
impl Fixture {
    fn new(length: u64) -> Self {
        let conn = Connection::open_in_memory().unwrap();
        crate::schema::ensure_archive(&conn).unwrap();
        conn.execute(
            "INSERT INTO input(id,format) VALUES ('input','claude-code')",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',1,'first',1,?1,1)", [length as i64]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap());
        Self {
            conn,
            store,
            _dir: dir,
        }
    }
    fn add(&self, at: u64, length: u64, blob_offset: u64, bytes: &[u8]) -> BlobHash {
        let p = NewBlob::prepare(&self.store, bytes).unwrap();
        let (place, size, data) = match &p.stored {
            Stored::Inline(data) => ("inline", data.len() as u64, Some(data.as_slice())),
            Stored::File { stored_size } => ("file", *stored_size, None),
        };
        self.conn.execute("INSERT OR IGNORE INTO blob(hash,size,stored_size,place,encoding,state,data) VALUES (?1,?2,?3,?4,'plain','present',?5)",params![p.hash.0.as_slice(),p.size as i64,size as i64,place,data]).unwrap();
        self.conn.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES ('input',1,?1,?2,?3,?4)",params![at as i64,length as i64,p.hash.0.as_slice(),blob_offset as i64]).unwrap();
        p.hash
    }
    fn load(&self, length: u64) -> StoredChain {
        StoredChain::load(&self.conn, "", "input", 1, length).unwrap()
    }
}

#[test]
fn independent_sequential_and_range_readers_preserve_shared_slices_and_short_reads() {
    let w = Fixture::new(9);
    let mut file = vec![b'x'; INLINE_LIMIT + 1];
    file[5..8].copy_from_slice(b"def");
    w.add(0, 3, 1, b"_abc_");
    w.add(3, 3, 5, &file);
    w.add(6, 3, 1, b"_abc_");
    let chain = w.load(9);
    for start in 0..=9 {
        for end in start..=9 {
            assert_eq!(
                chain.read_range(&w.store, start, end).unwrap(),
                &b"abcdefabc"[start as usize..end as usize]
            );
        }
    }
    let mut first = chain.reader(Arc::clone(&w.store));
    let mut second = chain.reader(Arc::clone(&w.store));
    let mut actual = Vec::new();
    for _ in 0..9 {
        let mut one = [0];
        assert_eq!(first.read(&mut one).unwrap(), 1);
        actual.push(one[0]);
        assert_eq!(second.read(&mut one).unwrap(), 1);
        assert_eq!(one[0], *actual.last().unwrap());
    }
    assert_eq!(actual, b"abcdefabc");
    std::thread::scope(|scope| {
        let one = scope.spawn(|| chain.read_range(&w.store, 1, 8).unwrap());
        let two = scope.spawn(|| {
            let mut bytes = Vec::new();
            chain
                .reader(Arc::clone(&w.store))
                .read_to_end(&mut bytes)
                .unwrap();
            bytes
        });
        assert_eq!(one.join().unwrap(), b"bcdefab");
        assert_eq!(two.join().unwrap(), b"abcdefabc");
    });
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
}

#[test]
fn many_inline_blobs_stay_compressed_on_disk_and_clones_share_one_selected_view() {
    let w = Fixture::new(512 * 16384);
    for n in 0u64..512 {
        let mut bytes = vec![b'x'; 16384];
        bytes[..8].copy_from_slice(&n.to_le_bytes());
        w.add(n * 16384, 16384, 0, &bytes);
    }
    let chain = w.load(512 * 16384);
    let clone = chain.clone();
    assert!(Arc::ptr_eq(
        chain.extents.as_ref().unwrap(),
        clone.extents.as_ref().unwrap()
    ));
    let extents = chain.extents.as_ref().unwrap();
    let conn = extents.connection();
    let (new_blobs, compressed): (i64, i64) = conn
        .query_row("SELECT count(*),sum(length(data)) FROM blob", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(new_blobs, 512);
    assert!(compressed < (chain.length / 100) as i64);
    drop(conn);
    w.conn.execute("UPDATE blob SET data=x'00'", []).unwrap();
    w.conn.execute("DELETE FROM extent", []).unwrap();
    assert_eq!(
        chain
            .read_range(&w.store, 500 * 16384, 500 * 16384 + 8)
            .unwrap(),
        500u64.to_le_bytes()
    );
    let weak = Arc::downgrade(extents);
    let reader = clone.reader(Arc::clone(&w.store));
    drop(chain);
    drop(clone);
    assert!(weak.upgrade().is_some());
    drop(reader);
    assert!(weak.upgrade().is_none());
}

#[test]
fn unrelated_corruption_is_not_decoded_but_selected_corruption_and_gaps_refuse() {
    let w = Fixture::new(6);
    let bad = w.add(0, 3, 0, b"abc");
    w.add(3, 3, 0, b"def");
    w.conn
        .execute(
            "UPDATE blob SET data=x'00' WHERE hash=?1",
            [bad.0.as_slice()],
        )
        .unwrap();
    let chain = w.load(6);
    assert_eq!(chain.read_range(&w.store, 3, 6).unwrap(), b"def");
    assert!(chain.read_range(&w.store, 2, 6).is_err());
    w.conn
        .execute("DELETE FROM extent WHERE offset=3", [])
        .unwrap();
    let gap = w.load(6);
    assert!(gap.read_range(&w.store, 3, 6).is_err());
    assert!(chain.read_range(&w.store, 6, 2).is_err());
}

#[test]
fn invalid_extent_metadata_refuses_capture_without_panicking() {
    for (column, value) in [
        ("blob_offset", -1),
        ("blob_offset", i64::MAX),
        ("length", 4),
        ("length", 0),
    ] {
        let w = Fixture::new(3);
        w.add(0, 3, 0, b"abc");
        w.conn
            .execute(&format!("UPDATE extent SET {column}=?1"), [value])
            .unwrap();
        // A zero-length extent is outside the requested range and leaves a gap.
        let result = StoredChain::load(&w.conn, "", "input", 1, 3);
        if value == 0 {
            assert!(result.unwrap().read_range(&w.store, 0, 3).is_err());
        } else {
            assert!(result.is_err(), "{column}={value}");
        }
    }
}

#[test]
fn historical_oversized_inline_blobs_are_verified_before_any_range_is_exposed() {
    let w = Fixture::new(2 << 20);
    let bytes = vec![b'z'; 2 << 20];
    let hash = BlobHash::of(&bytes);
    let data = zstd::bulk::compress(&bytes, 3).unwrap();
    w.conn.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state,data) VALUES (?1,?2,?3,'inline','plain','present',?4)",params![hash.0.as_slice(),bytes.len() as i64,data.len() as i64,data]).unwrap();
    w.conn
        .execute(
            "INSERT INTO extent VALUES ('input',1,0,?1,?2,0)",
            params![bytes.len() as i64, hash.0.as_slice()],
        )
        .unwrap();
    let chain = w.load(bytes.len() as u64);
    assert_eq!(
        chain.read_range(&w.store, 100, 117).unwrap(),
        vec![b'z'; 17]
    );
    w.conn.execute("UPDATE blob SET size=size+1", []).unwrap();
    let changed = w.load(bytes.len() as u64);
    assert!(changed.read_range(&w.store, 100, 117).is_err());
}

#[test]
fn only_a_complete_file_blob_uses_the_direct_reader() {
    let w = Fixture::new((INLINE_LIMIT + 1) as u64);
    let bytes = vec![b'x'; INLINE_LIMIT + 1];
    let hash = w.add(0, bytes.len() as u64, 0, &bytes);
    assert_eq!(w.load(bytes.len() as u64).single_file_blob(), Some(hash));
    assert_eq!(w.load(10).single_file_blob(), None);
}

#[test]
fn overlapping_extents_refuse_instead_of_selecting_a_different_source_by_cursor_order() {
    let w = Fixture::new(6);
    w.add(0, 4, 0, b"abcd");
    w.add(3, 3, 0, b"def");
    assert!(StoredChain::load(&w.conn, "", "input", 1, 6).is_err());
}

#[test]
fn a_file_range_uses_one_verification_pass_without_decoding_its_prefix_twice() {
    let w = Fixture::new(2 << 20);
    let bytes: Vec<_> = (0..2 << 20).map(|n| (n % 251) as u8).collect();
    let hash = w.add(0, bytes.len() as u64, 0, &bytes);
    let extent = extents::Extent {
        offset: 0,
        length: bytes.len() as u64,
        hash,
        blob_offset: 0,
        size: bytes.len() as u64,
        inline: false,
    };
    let mut out = [0; 37];
    let start = bytes.len() as u64 - 100;
    let stats = reader::file_range(&w.store, &extent, start, &mut out).unwrap();
    assert_eq!(out, bytes[start as usize..start as usize + 37]);
    assert_eq!(stats.decoded_bytes, bytes.len() as u64);
}

#[test]
fn an_open_file_reader_needs_no_extent_list_lock() {
    let w = Fixture::new((INLINE_LIMIT + 1) as u64);
    let bytes = vec![b'q'; INLINE_LIMIT + 1];
    w.add(0, bytes.len() as u64, 0, &bytes);
    let chain = w.load(bytes.len() as u64);
    let mut reader = chain.reader(Arc::clone(&w.store));
    let mut one = [0];
    reader.read_exact(&mut one).unwrap();
    let _locked = chain.extents.as_ref().unwrap().connection();
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, bytes[1..]);
}

#[test]
fn a_poisoned_extent_list_returns_a_typed_error() {
    let w = Fixture::new(3);
    w.add(0, 3, 0, b"abc");
    let chain = w.load(3);
    let extents = Arc::clone(chain.extents.as_ref().unwrap());
    assert!(
        std::thread::spawn(move || {
            let _locked = extents.connection();
            panic!("synthetic poisoned lock");
        })
        .join()
        .is_err()
    );
    let error = chain.read_range(&w.store, 0, 3).unwrap_err();
    assert!(matches!(error, StoreError::Invalid(text) if text.contains("lock poisoned")));
}

#[cfg(target_os = "linux")]
mod scratch;
