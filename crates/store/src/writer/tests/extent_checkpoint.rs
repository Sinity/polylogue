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
    w.writer
        .store_group(vec![bytes], Arc::new(w.archive.blobs().unwrap()))
        .unwrap()
        .pop()
        .unwrap()
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
    let reads = ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    let progress = reader.input_progress("input").unwrap().unwrap();
    assert_eq!(progress.length, 9);
    let chain = reader
        .stored_chain("input", progress.chain, progress.length)
        .unwrap();
    let mut actual = Vec::new();
    chain
        .reader(Arc::new(w.archive.blobs().unwrap()))
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, b"abcdefghi");
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
