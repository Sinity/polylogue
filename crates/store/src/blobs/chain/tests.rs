use super::*;
use rusqlite::params;
use std::io::{Read, Seek, SeekFrom};

struct Fixture {
    conn: Connection,
    store: Arc<crate::blobs::Blobs>,
    _dir: tempfile::TempDir,
}
impl Fixture {
    fn new(length: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let catalog = dir.path().join("archive.db");
        let conn = Connection::open(&catalog).unwrap();
        conn.pragma_update(None, "journal_mode", "wal").unwrap();
        crate::schema::ensure_archive(&conn).unwrap();
        conn.execute(
            "INSERT INTO input(id,format) VALUES ('input','synthetic')",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',1,'first',1,?1,1)", [length as i64]).unwrap();
        let store = Arc::new(crate::blobs::Blobs::new(
            dir.path().to_owned(),
            catalog,
            crate::blobs::BlobCatalog::Archive,
        ));
        Self {
            conn,
            store,
            _dir: dir,
        }
    }
    fn add(&self, at: u64, length: u64, blob_offset: u64, bytes: &[u8]) -> BlobHash {
        self.add_group([(at, length, blob_offset, bytes)])
            .pop()
            .unwrap()
    }
    fn add_group<'a>(
        &self,
        rows: impl IntoIterator<Item = (u64, u64, u64, &'a [u8])>,
    ) -> Vec<BlobHash> {
        let rows: Vec<_> = rows.into_iter().collect();
        let preparation = hashed_blobs::Preparation::new(self._dir.path().join("parts")).unwrap();
        let prepared = rows
            .iter()
            .map(|(_, _, _, bytes)| preparation.plain(*bytes, crate::blobs::LEVEL).unwrap())
            .collect::<Vec<_>>();
        let mut writer = crate::blobs::open_writer(&self.conn, self._dir.path()).unwrap();
        let group = writer
            .append_group(prepared.into_iter().map(Ok), |hash| {
                crate::blobs::load_extent(&self.conn, "", hash)
                    .map_err(|e| hashed_blobs::Error::Refused(e.to_string()))
            })
            .unwrap()
            .wait()
            .unwrap();
        let tx = self.conn.unchecked_transaction().unwrap();
        for ((at, length, blob_offset, _), extent) in rows.iter().zip(&group.extents) {
            crate::blobs::insert_extent(&tx, extent).unwrap();
            tx.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset,commit_seq) VALUES ('input',1,?1,?2,?3,?4,1)",params![*at as i64,*length as i64,extent.hash.0.as_slice(),*blob_offset as i64]).unwrap();
        }
        tx.execute("INSERT INTO segment(name,committed_end) VALUES ('segments/current.hb',?1) ON CONFLICT(name) DO UPDATE SET committed_end=excluded.committed_end",[group.end as i64]).unwrap();
        tx.commit().unwrap();
        writer.acknowledge(&group).unwrap();
        group
            .extents
            .iter()
            .map(|extent| BlobHash(extent.hash.0))
            .collect()
    }
    fn source(&self) -> ChainSource {
        ChainSource {
            path: self._dir.path().join("archive.db"),
            tables: "",
        }
    }
    fn load(&self, length: u64) -> StoredChain {
        StoredChain::load(&self.source(), "input", 1, length).unwrap()
    }
}

#[test]
fn independent_sequential_and_range_readers_preserve_shared_slices_and_short_reads() {
    let w = Fixture::new(9);
    let mut file = vec![b'x'; 64 << 10];
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
fn many_segment_blobs_stay_compressed_and_clones_share_one_extent_view() {
    let w = Fixture::new(512 * 16384);
    let bytes: Vec<_> = (0u64..512)
        .map(|n| {
            let mut bytes = vec![b'x'; 16384];
            bytes[..8].copy_from_slice(&n.to_le_bytes());
            bytes
        })
        .collect();
    w.add_group(
        bytes
            .iter()
            .enumerate()
            .map(|(n, bytes)| (n as u64 * 16384, 16384, 0, bytes.as_slice())),
    );
    let chain = w.load(512 * 16384);
    let clone = chain.clone();
    assert!(Arc::ptr_eq(
        chain.extents.as_ref().unwrap(),
        clone.extents.as_ref().unwrap()
    ));
    let extents = chain.extents.as_ref().unwrap();
    let (new_blobs, compressed): (i64, i64) = w
        .conn
        .query_row("SELECT count(*),sum(stored_size) FROM blob", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(new_blobs, 512);
    assert!(compressed < (chain.length / 10) as i64);
    assert_eq!(
        chain
            .read_range(&w.store, 500 * 16384, 500 * 16384 + 8)
            .unwrap(),
        500u64.to_le_bytes()
    );
    // Each read checks that the extents are still the ones checked at load.
    w.conn.execute("DELETE FROM extent", []).unwrap();
    let error = chain
        .read_range(&w.store, 500 * 16384, 500 * 16384 + 8)
        .unwrap_err();
    assert!(
        matches!(&error, StoreError::Invalid(text) if text == "stored extents changed since the chain was loaded"),
        "{error:?}"
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
    let extent = w.store.extent(&hashed_blobs::Hash(bad.0)).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(w._dir.path().join(&extent.segment))
        .unwrap();
    use std::os::unix::fs::FileExt;
    let mut byte = [0];
    file.read_exact_at(&mut byte, extent.offset).unwrap();
    byte[0] ^= 1;
    file.write_all_at(&byte, extent.offset).unwrap();
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
        let result = StoredChain::load(&w.source(), "input", 1, 3);
        if value == 0 {
            assert!(result.unwrap().read_range(&w.store, 0, 3).is_err());
        } else {
            assert!(result.is_err(), "{column}={value}");
        }
    }
}

#[test]
fn large_segment_blobs_refuse_changed_catalog_size_before_exposing_ranges() {
    let bytes = vec![b'z'; 2 << 20];
    let w = Fixture::new(bytes.len() as u64);
    w.add(0, bytes.len() as u64, 0, &bytes);
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
fn only_a_complete_single_blob_uses_the_direct_reader() {
    let w = Fixture::new(64 << 10);
    let bytes = vec![b'x'; 64 << 10];
    let hash = w.add(0, bytes.len() as u64, 0, &bytes);
    assert_eq!(w.load(bytes.len() as u64).single_file_blob(), Some(hash));
    assert_eq!(w.load(10).single_file_blob(), None);
}

#[test]
fn overlapping_extents_refuse_instead_of_selecting_a_different_source_by_cursor_order() {
    let w = Fixture::new(6);
    w.add(0, 4, 0, b"abcd");
    w.add(3, 3, 0, b"def");
    assert!(StoredChain::load(&w.source(), "input", 1, 6).is_err());
}

#[test]
fn a_segment_range_decodes_only_its_selected_frame() {
    let size = 2 * hashed_blobs::DEFAULT_FRAME + 1024;
    let bytes: Vec<_> = (0..size).map(|n| (n % 251) as u8).collect();
    let w = Fixture::new(size as u64);
    let hash = w.add(0, size as u64, 0, &bytes);
    let mut reader = w
        .store
        .read_plain(&hashed_blobs::Hash(hash.0), Some(size as u64), || Ok(()))
        .unwrap();
    let start = size - 100;
    reader.seek(SeekFrom::Start(start as u64)).unwrap();
    let mut actual = [0; 37];
    reader.read_exact(&mut actual).unwrap();
    assert_eq!(actual, bytes[start..start + 37]);
    assert_eq!(reader.stats().decoded_bytes, 1024);
    assert_eq!(reader.stats().frames, 1);
}

#[test]
fn an_open_file_reader_needs_no_extent_list_lock() {
    let w = Fixture::new(64 << 10);
    let bytes = vec![b'q'; 64 << 10];
    w.add(0, bytes.len() as u64, 0, &bytes);
    let chain = w.load(bytes.len() as u64);
    let mut reader = chain.reader(Arc::clone(&w.store));
    let mut one = [0];
    reader.read_exact(&mut one).unwrap();
    let locked = chain.extents.as_ref().unwrap().connection();
    // Within the open extent no read takes the lock; reading the last byte
    // ends the chain's read transaction, which does.
    let mut rest = vec![0; bytes.len() - 2];
    reader.read_exact(&mut rest).unwrap();
    assert_eq!(rest, bytes[1..bytes.len() - 1]);
    drop(locked);
    reader.read_exact(&mut one).unwrap();
    assert!(chain.extents.as_ref().unwrap().connection().is_none());
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

#[test]
fn chain_read_creates_no_database() {
    // Enough extents that a copy of the list would spill out of any cache.
    let w = Fixture::new(32768 * 256);
    let bytes: Vec<_> = (0u64..64)
        .flat_map(|n| blake3::hash(&n.to_le_bytes()).as_bytes().to_vec())
        .collect();
    let hash = w.add(0, 256, 0, &bytes[..256]);
    {
        let tx = w.conn.unchecked_transaction().unwrap();
        let mut insert = tx
            .prepare("INSERT INTO extent(input,chain,offset,length,blob,blob_offset,commit_seq) VALUES ('input',1,?1,256,?2,0,1)")
            .unwrap();
        for n in 1u64..32768 {
            insert
                .execute(params![(n * 256) as i64, hash.0.as_slice()])
                .unwrap();
        }
        drop(insert);
        tx.commit().unwrap();
    }
    let chain = w.load(32768 * 256);
    let mut reader = chain.reader(Arc::clone(&w.store));
    let mut all = vec![0; 32768 * 256 - 1];
    reader.read_exact(&mut all).unwrap();
    assert_eq!(&all[256 * 100..256 * 101], &bytes[..256]);
    // The chain reads the archive itself, read-only: no database of its own,
    // and nothing in the temporary one.
    let guard = chain.extents.as_ref().unwrap().connection();
    let handle = guard.as_ref().expect("open while the chain is read");
    let databases: Vec<(String, String)> = handle
        .prepare("PRAGMA database_list")
        .unwrap()
        .query_map([], |r| Ok((r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let archive = w._dir.path().join("archive.db").canonicalize().unwrap();
    assert_eq!(
        databases,
        vec![("main".to_owned(), archive.display().to_string())]
    );
    assert!(handle.is_readonly("main").unwrap());
    let temporary: i64 = handle
        .query_row("PRAGMA temp.page_count", [], |r| r.get(0))
        .unwrap();
    assert_eq!(temporary, 0);
    drop(guard);
    let mut last = Vec::new();
    reader.read_to_end(&mut last).unwrap();
    assert_eq!(last.len(), 1);
    assert!(chain.extents.as_ref().unwrap().connection().is_none());
}

/// The result of a truncating checkpoint on the fixture's archive: whether a
/// reader kept it from completing.
fn checkpoint_blocked(w: &Fixture) -> bool {
    w.conn.busy_timeout(std::time::Duration::ZERO).unwrap();
    let (busy, log): (i64, i64) = w
        .conn
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    busy != 0 || log != 0
}

#[test]
fn loaded_chains_waiting_to_be_read_let_a_truncating_checkpoint_complete() {
    let w = Fixture::new(3 * 64);
    for n in 0..64u64 {
        w.add(n * 3, 3, 0, format!("{n:03}").as_bytes());
    }
    // As a pass's queued plans: every chain loaded, none read yet.
    let queued: Vec<_> = (1..=64).map(|n| w.load(n * 3)).collect();
    w.conn
        .execute("UPDATE revision SET produced='later'", [])
        .unwrap();
    assert!(!checkpoint_blocked(&w));
    assert_eq!(queued[9].read_range(&w.store, 27, 30).unwrap(), b"009");
}

#[test]
fn a_released_chain_lets_a_checkpoint_complete_and_reads_on() {
    let w = Fixture::new(6);
    w.add(0, 3, 0, b"abc");
    w.add(3, 3, 0, b"def");
    let chain = w.load(6);
    let mut reader = chain.reader(Arc::clone(&w.store));
    let mut first = [0; 2];
    reader.read_exact(&mut first).unwrap();
    w.conn
        .execute("UPDATE revision SET produced='later'", [])
        .unwrap();
    assert!(
        checkpoint_blocked(&w),
        "a chain being read holds its snapshot"
    );
    reader.release();
    assert!(!checkpoint_blocked(&w));
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest).unwrap();
    assert_eq!([&first[..], &rest].concat(), b"abcdef");
}
