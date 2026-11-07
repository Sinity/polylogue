use super::*;

#[test]
fn empty_source_keeps_its_physical_blob_and_admitted_revision_without_logical_extents() {
    let w = test_archive();
    let mut raw = crate::fixture::writer::step(
        "empty-source",
        "fixture",
        "empty.jsonl",
        Output::default(),
        0,
        0,
    );
    raw.length = 0;
    raw.parse.clear();
    let prepared = w.archive.prepare_bytes(&b""[..]).unwrap();
    let hash = prepared.hash();
    let stored = w.writer.store_bytes(prepared, raw).unwrap();
    assert_eq!(stored.length, 0);
    for _ in 0..2 {
        let archive = Connection::open(w.archive.archive_db()).unwrap();
        assert_eq!(
            archive
                .query_row("SELECT count(*) FROM extent", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            archive
                .query_row(
                    "SELECT count(*) FROM revision WHERE input='empty-source' AND length=0",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
        assert_eq!(archive.query_row("SELECT count(*) FROM observation WHERE input='empty-source' AND class='first' AND length=0", [], |row| row.get::<_, i64>(0)).unwrap(), 1);
        let descriptor: Vec<u8> = archive
            .query_row(
                "SELECT storage FROM blob WHERE hash=?1 AND size=0",
                [&hash.0[..]],
                |row| row.get(0),
            )
            .unwrap();
        let extent = hashed_blobs::Extent::decode(&descriptor).unwrap();
        assert_eq!(extent.decoded_len, 0);
        assert!(extent.len > 0);
        let committed: i64 = archive
            .query_row(
                "SELECT committed_end FROM segment WHERE name=?1",
                [&extent.segment],
                |row| row.get(0),
            )
            .unwrap();
        assert!(u64::try_from(committed).unwrap() >= extent.offset + extent.len);
        assert_eq!(
            crate::blobs::read(&archive, &w.archive.blobs().unwrap(), &hash).unwrap(),
            b""
        );
        let receipt =
            crate::source::Source::reading_receipt(&archive, "empty-source", stored.chain, 0)
                .unwrap();
        let admitted: i64 = archive
            .query_row(
                "SELECT commit_seq FROM revision WHERE input='empty-source'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(admitted > 0);
        assert_eq!(receipt.sequence, admitted);
        assert!(receipt.observation.is_some());
    }
}
