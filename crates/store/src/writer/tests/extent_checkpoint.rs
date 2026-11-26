use super::*;
use crate::staging::PreparedBatch;
use std::io::Read;
use std::sync::Arc;

fn bytes(archive: &Archive, from: u64, body: &[u8]) -> PreparedBatch {
    let (blob, file) = archive.prepare_blob(body).unwrap();
    let mut batch = batch("input", Route::Live, from, vec![], &[]);
    batch.parse.clear();
    batch.new_blob = Some(blob);
    batch.length = from + body.len() as u64;
    PreparedBatch {
        batch,
        files: file.into_iter().collect(),
    }
}
fn store(w: &TestArchive, bytes: PreparedBatch) -> Result<crate::staging::StoredInput> {
    w.writer.store_group(vec![bytes]).unwrap().pop().unwrap()
}

#[test]
fn extent_writer_refuses_stale_append_without_replacing_prefix() {
    let w = test_archive();
    store(&w, bytes(&w.archive, 0, b"abc")).unwrap();
    store(&w, bytes(&w.archive, 3, b"def")).unwrap();
    // An identical committed retry and a genuine continuation remain valid.
    store(&w, bytes(&w.archive, 3, b"def")).unwrap();
    let mut continuation = bytes(&w.archive, 6, b"ghi");
    continuation.batch.continues = true;
    store(&w, continuation).unwrap();
    let mut stale = bytes(&w.archive, 0, b"abcdefghijkl");
    stale.batch.change = Change::Appended { from: 0 };
    match store(&w, stale) {
        Err(StoreError::Invalid(reason)) => {
            assert_eq!(reason, "an append must start at the stored length");
        }
        other => panic!("expected the stale-append refusal, got {other:?}"),
    }
    assert_eq!(stored_bytes(&w), b"abcdefghi");
}

/// The input's latest revision read back from its stored extents.
fn stored_bytes(w: &TestArchive) -> Vec<u8> {
    let reads = ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    let progress = reader.input_progress("input").unwrap().unwrap();
    let chain = reader
        .stored_chain("input", progress.chain, progress.length)
        .unwrap();
    let mut actual = Vec::new();
    chain
        .reader(Arc::new(w.archive.blobs().unwrap()))
        .read_to_end(&mut actual)
        .unwrap();
    actual
}

fn assert_refused(result: Result<crate::staging::StoredInput>, expected: &str) {
    match result {
        Err(StoreError::Invalid(reason)) => assert_eq!(reason, expected),
        other => panic!("expected the refusal {expected:?}, got {other:?}"),
    }
}

/// A step with no bytes of its own at `length`.
fn without_bytes(from: u64, length: u64) -> PreparedBatch {
    let mut batch = batch("input", Route::Live, from, vec![], &[]);
    batch.parse.clear();
    batch.length = length;
    PreparedBatch {
        batch,
        files: Vec::new(),
    }
}

#[test]
fn a_step_shorter_than_its_stored_chain_is_refused() {
    let w = test_archive();
    store(&w, bytes(&w.archive, 0, b"abc")).unwrap();
    store(&w, bytes(&w.archive, 3, b"def")).unwrap();
    let mut continuing = without_bytes(6, 4);
    continuing.batch.continues = true;
    assert_refused(
        store(&w, continuing),
        "a step cannot make its revision chain shorter",
    );
    assert_refused(
        store(&w, without_bytes(6, 4)),
        "a step cannot make its revision chain shorter",
    );
    assert_eq!(stored_bytes(&w), b"abcdef");
}

#[test]
fn a_step_cannot_place_bytes_over_a_stored_extent() {
    let w = test_archive();
    store(&w, bytes(&w.archive, 0, b"abc")).unwrap();
    store(&w, bytes(&w.archive, 3, b"def")).unwrap();
    store(&w, bytes(&w.archive, 6, b"ghi")).unwrap();
    // Unchanged bytes are named in `blobs`; new bytes at a stored place
    // would replace the extent that is there.
    let mut over = bytes(&w.archive, 3, b"xyzw");
    over.batch.change = Change::Unchanged;
    over.batch.length = 3;
    assert_refused(store(&w, over), "an unchanged step adds no bytes");
    assert_eq!(stored_bytes(&w), b"abcdefghi");
}

/// A location seen holding only the first bytes of a stored chain holds no
/// byte the chain lacks: it is recorded as seen back with what it held, the
/// revision that ends there when one does and otherwise the chain and the
/// length, and the chain stays as it is. Bytes the chain does not hold are
/// refused.
#[test]
fn a_location_holding_the_first_bytes_of_a_chain_is_seen_back_with_what_it_held() {
    let w = test_archive();
    store(&w, bytes(&w.archive, 0, b"abc")).unwrap();
    store(&w, bytes(&w.archive, 3, b"def")).unwrap();
    let seen = |length: u64| {
        let mut step = without_bytes(0, length);
        step.batch.change = Change::ObservedPrefix { chain: 1 };
        store(&w, step)
    };
    assert!(seen(4).unwrap().already_stored);
    assert!(seen(3).unwrap().already_stored);
    let db = crate::archive::open_read(&w.archive.archive_db()).unwrap();
    type Seen = (String, Option<i64>, Option<i64>, Option<i64>);
    let observations = |db: &rusqlite::Connection| -> Vec<Seen> {
        db.prepare("SELECT class,revision,chain,length FROM observation ORDER BY rowid")
            .unwrap()
            .query_map([], |r| <Seen>::try_from(r))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    let recorded = observations(&db);
    assert_eq!(
        recorded[recorded.len() - 2..],
        [
            ("back".to_owned(), None, Some(1), Some(4)),
            ("back".to_owned(), Some(1), None, Some(3)),
        ]
    );
    let revisions: Vec<(i64, String, i64, i64)> = db
        .prepare("SELECT number,produced,chain,length FROM revision ORDER BY number")
        .unwrap()
        .query_map([], |r| <(i64, String, i64, i64)>::try_from(r))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        revisions,
        [
            (1, "first".to_owned(), 1, 3),
            (2, "appended".to_owned(), 1, 6)
        ]
    );
    assert!(seen(9).is_err());
    assert_eq!(observations(&db), recorded);
    assert_eq!(stored_bytes(&w), b"abcdef");
}

/// A step that does not start at the committed reading's position is a
/// stale step from its sender, not a fault of the bytes: the writer keeps
/// the stored bytes and refuses the rows with the refusal's own cause, so
/// the chain is planned again instead of failing until its bytes change
/// (docs/archive.md, parse state).
#[test]
fn a_stale_step_is_refused_for_its_own_cause_not_the_bytes() {
    use polylogue_model::{Cause, Caused as _};
    let w = test_archive();
    let step = |range: std::ops::Range<u64>| {
        let mut step = ParseStep::new(Output::default(), counts(0), range, false);
        step.parse_version = "synthetic=1".into();
        step
    };
    let mut first = bytes(&w.archive, 0, b"abc");
    first.batch.parse = vec![step(0..3)];
    store(&w, first).unwrap();
    let mut stale = bytes(&w.archive, 3, b"def");
    stale.batch.parse = vec![step(1..4)];
    match store(&w, stale) {
        Err(error @ StoreError::RowsRefused { .. }) => {
            assert_eq!(
                error.to_string(),
                "reading starts at 1 after committed position 3"
            );
            assert_eq!(error.cause(), Cause::Request, "{error}");
        }
        other => panic!("expected the stale step's refusal, got {other:?}"),
    }
}
