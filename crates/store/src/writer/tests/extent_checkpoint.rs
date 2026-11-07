use super::*;
use crate::staging::PreparedBatch;
use std::io::Read;
use std::sync::Arc;

fn bytes(archive: &Archive, from: u64, body: &[u8]) -> PreparedBatch {
    let (blob, file) = archive.prepare_blob(body).unwrap();
    let mut batch = batch("input", 0, from, vec![], &[]);
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
        .sender()
        .unwrap()
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
    assert!(
        store(&w, stale).is_err(),
        "a stale append must not replace committed extent zero"
    );
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
