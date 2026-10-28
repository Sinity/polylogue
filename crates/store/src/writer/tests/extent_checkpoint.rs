use super::*;
use std::io::Read;
use std::sync::Arc;

fn bytes(store: &hashed_blobs::Store, from: u64, body: &[u8]) -> Batch {
    let mut b = batch("input", 0, from, vec![], &[]);
    b.parse.clear();
    b.new_blob = Some(NewBlob::prepare(store, body).unwrap());
    b.length = from + body.len() as u64;
    b
}

#[test]
fn extent_writer_refuses_stale_append_without_replacing_prefix() {
    let w = test_archive();
    let store = Arc::new(hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap());
    w.writer.send(bytes(&store, 0, b"abc")).unwrap();
    let append = bytes(&store, 3, b"def");
    w.writer.send(append.clone()).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    // An identical committed retry and a genuine continuation remain valid.
    w.writer.send(append).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut continuation = bytes(&store, 6, b"ghi");
    continuation.continues = true;
    w.writer.send(continuation).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut stale = bytes(&store, 0, b"abcdefghijkl");
    stale.change = Change::Appended { from: 0 };
    w.writer.send(stale).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(
        !report.errors.is_empty(),
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
    chain.reader(store).read_to_end(&mut actual).unwrap();
    assert_eq!(actual, b"abcdefghi");
}
