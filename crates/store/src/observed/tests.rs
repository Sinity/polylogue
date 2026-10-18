use super::*;
use std::{fs::OpenOptions, io::Write};
fn try_acquire(w: &mut Writer, p: &Path, id: &str) -> crate::Result<StoredSource> {
    w.acquire_claude_main(
        p,
        "synthetic-machine",
        "synthetic-host",
        id,
        1,
        &AtomicBool::new(false),
    )
}
fn acquire(w: &mut Writer, p: &Path, id: &str) -> StoredSource {
    try_acquire(w, p, id).unwrap()
}
#[test]
fn corrupt_existing_blob_file_adds_no_revision() {
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("archive");
    let path = root.path().join("source.jsonl");
    let bytes = b"synthetic stored source\n";
    std::fs::write(&path, bytes).unwrap();
    let mut w = Writer::open(&archive).unwrap();
    let blobs = hashed_blobs::Store::open(&archive, vec![], 0o600).unwrap();
    let hash = blobs.put(&bytes[..], 1).unwrap();
    let hex = hash.hex();
    let physical = archive
        .join("blobs")
        .join(&hex[..2])
        .join(format!("{hex}.zst"));
    std::fs::write(&physical, b"invalid existing dependency").unwrap();
    let before = w.status().unwrap().write_counter;
    assert!(try_acquire(&mut w, &path, "refused").is_err());
    assert_eq!(w.status().unwrap().write_counter, before);
    let revisions: i64 = w
        .archive
        .query_row("SELECT count(*) FROM revision", [], |row| row.get(0))
        .unwrap();
    assert_eq!(revisions, 0);
    assert_eq!(
        std::fs::read(physical).unwrap(),
        b"invalid existing dependency"
    );
}
#[test]
fn range_reads_refuse_a_blob_marked_for_deletion() {
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("archive");
    let path = root.path().join("source.jsonl");
    std::fs::write(&path, b"abcdefghij").unwrap();
    let mut w = Writer::open(&archive).unwrap();
    acquire(&mut w, &path, "first");
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"klmnopqr")
        .unwrap();
    let source = acquire(&mut w, &path, "second");
    let reader = w.read_handle().unwrap();
    reader.verify_source(&source, || Ok(())).unwrap();
    assert_eq!(
        reader
            .read_source_range(&source.input.id, 1, source.binding, 8, 6)
            .unwrap(),
        b"ijklmn"
    );
    let hash = BlobHash::of(b"klmnopqr");
    w.archive
        .execute(
            "UPDATE blob SET state='deleting' WHERE hash=?1",
            [hash.bytes().as_slice()],
        )
        .unwrap();
    assert!(
        reader
            .read_source_range(&source.input.id, 1, source.binding, 10, 2)
            .is_err()
    );
}
#[test]
fn first_observed_revision_has_digest_but_no_original_hash() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("session.jsonl");
    let bytes = b"synthetic first observed bytes\n";
    std::fs::write(&path, bytes).unwrap();
    let mut w = Writer::open(&root.path().join("archive")).unwrap();
    let source = acquire(&mut w, &path, "first");
    let (digest, original): (Vec<u8>, Option<Vec<u8>>) = w.archive.query_row("SELECT r.blake3,i.original_hash FROM revision r JOIN input i ON i.id=r.input WHERE r.input=?1 AND r.number=1", [&source.input.id], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert_eq!(digest, blake3::hash(bytes).as_bytes());
    assert!(original.is_none());
    assert!(matches!(source.identity, SourceIdentity::Observed { .. }));
}
#[test]
fn appends_to_one_file_extend_one_input() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("session.jsonl");
    std::fs::write(&path, b"").unwrap();
    let archive = root.path().join("archive");
    let mut w = Writer::open(&archive).unwrap();
    let first = acquire(&mut w, &path, "genesis");
    assert_eq!(first.input.length, 0);
    let digest: Vec<u8> = w
        .archive
        .query_row(
            "SELECT blake3 FROM revision WHERE input=?1 AND number=1",
            [&first.input.id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(digest, blake3::hash(b"").as_bytes());
    let original: Option<Vec<u8>> = w
        .archive
        .query_row(
            "SELECT original_hash FROM input WHERE id=?1",
            [&first.input.id],
            |r| r.get(0),
        )
        .unwrap();
    assert!(original.is_none());
    assert_eq!(first.context, ["claude-main", "session"]);
    let mut file = OpenOptions::new().append(true).open(&path).unwrap();
    let parts = [
        b"{\"a\":1}\n".as_slice(),
        b"\n \r\n",
        b"{\"partial\":",
        b"true}\n",
    ];
    let mut whole = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        file.write_all(part).unwrap();
        file.sync_all().unwrap();
        whole.extend_from_slice(part);
        let next = acquire(&mut w, &path, &format!("step-{i}"));
        assert_eq!(next.input.id, first.input.id);
        assert_eq!(next.input.chain, 1);
        assert_eq!(next.binding, first.binding);
        assert_eq!(next.revision, i as u64 + 2);
        let digest: Option<Vec<u8>> = w
            .archive
            .query_row(
                "SELECT blake3 FROM revision WHERE input=?1 AND number=?2",
                params![next.input.id, i64::try_from(next.revision).unwrap()],
                |r| r.get(0),
            )
            .unwrap();
        assert!(digest.is_none());
        assert_eq!(
            w.read_handle()
                .unwrap()
                .read_source_range(&next.input.id, 1, next.binding, 0, next.input.length)
                .unwrap(),
            whole
        );
        drop(w);
        w = crate::tests::reopen_after_drop(&archive);
    }
    let before = w.status().unwrap().write_counter;
    let retry = acquire(&mut w, &path, "unchanged");
    assert_eq!(retry.revision, 5);
    assert_eq!(w.status().unwrap().write_counter, before);
    assert_eq!(
        w.archive
            .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        4
    );
}
#[test]
fn changed_or_replaced_file_and_bad_extents_are_refused() {
    let root = tempfile::tempdir().unwrap();
    let p = root.path().join("s.jsonl");
    std::fs::write(&p, b"first\n").unwrap();
    let mut w = Writer::open(&root.path().join("archive")).unwrap();
    let s = acquire(&mut w, &p, "first");
    let counter = w.status().unwrap().write_counter;
    std::fs::write(&p, b"wrong\nmore\n").unwrap();
    assert!(try_acquire(&mut w, &p, "bad").is_err());
    assert_eq!(w.status().unwrap().write_counter, counter);
    let replacement = root.path().join("replacement");
    std::fs::write(&replacement, b"first\nmore\n").unwrap();
    std::fs::rename(&replacement, &p).unwrap();
    assert!(try_acquire(&mut w, &p, "replaced").is_err());
    assert_eq!(w.status().unwrap().write_counter, counter);
    w.archive.execute("UPDATE extent SET offset=1", []).unwrap();
    assert!(
        w.read_handle()
            .unwrap()
            .read_source_range(&s.input.id, 1, s.binding, 0, 6)
            .is_err()
    );
    w.archive.execute("UPDATE extent SET offset=0", []).unwrap();
    w.archive.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) SELECT input,chain,2,2,blob,0 FROM extent",[]).unwrap();
    assert!(
        w.read_handle()
            .unwrap()
            .read_source_range(&s.input.id, 1, s.binding, 0, 6)
            .is_err()
    );
    w.archive
        .execute("DELETE FROM extent WHERE offset=2", [])
        .unwrap();
    w.archive
        .execute("UPDATE blob SET state='deleting'", [])
        .unwrap();
    assert!(
        w.read_handle()
            .unwrap()
            .read_source_range(&s.input.id, 1, s.binding, 0, 6)
            .is_err()
    );
}
#[test]
fn index_failure_recovers_from_archive_without_the_file() {
    let root = tempfile::tempdir().unwrap();
    let p = root.path().join("s.jsonl");
    std::fs::write(&p, b"stored\n").unwrap();
    let archive = root.path().join("archive");
    let mut w = Writer::open(&archive).unwrap();
    w._index.execute_batch("CREATE TRIGGER refuse_ref BEFORE INSERT ON input_ref BEGIN SELECT RAISE(ABORT,'registration interrupted'); END;").unwrap();
    assert!(try_acquire(&mut w, &p, "id").is_err());
    let before = w.status().unwrap().write_counter;
    assert!(before > 0);
    w._index.execute_batch("DROP TRIGGER refuse_ref;").unwrap();
    drop(w);
    std::fs::remove_file(p).unwrap();
    let mut w = Writer::open(&archive).unwrap();
    let recovered = w.next_stored_source("").unwrap().unwrap();
    assert_eq!(recovered.input.id, "id");
    assert_eq!(w.status().unwrap().write_counter, before);
    assert_eq!(
        w.read_handle()
            .unwrap()
            .read_source_range("id", 1, recovered.binding, 0, 7)
            .unwrap(),
        b"stored\n"
    );
}
#[test]
fn exact_reader_fails_on_short_input() {
    let mut bytes = Vec::new();
    let result = Exact {
        reader: b"short".as_slice(),
        remaining: 9,
    }
    .read_to_end(&mut bytes);
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    let mut bounded = Vec::new();
    Exact {
        reader: b"longer".as_slice(),
        remaining: 3,
    }
    .read_to_end(&mut bounded)
    .unwrap();
    assert_eq!(bounded, b"lon");
}
#[test]
fn source_reads_refuse_when_archive_has_excisions() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("s.jsonl");
    std::fs::write(&path, b"stored\n").unwrap();
    let mut writer = Writer::open(&root.path().join("archive")).unwrap();
    let source = acquire(&mut writer, &path, "input");
    writer
        .archive
        .execute(
            "INSERT INTO excision(id,scope,created_at,local_seq) VALUES('e','input:unknown',1,1)",
            [],
        )
        .unwrap();
    let reader = writer.read_handle().unwrap();
    assert!(matches!(
        reader.source(&source.input.id),
        Err(Error::ExcisionCheckRequired)
    ));
    assert!(matches!(
        reader.read_source_range(&source.input.id, 1, source.binding, 0, source.input.length),
        Err(Error::ExcisionCheckRequired)
    ));
}

#[test]
fn main_and_companions_share_one_turn_counter() {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("session.jsonl");
    std::fs::write(&main, b"\n").unwrap();
    let directory = temp.path().join("session/tool-results");
    std::fs::create_dir_all(&directory).unwrap();
    let references = (0..3)
        .map(|index| {
            let path = directory.join(format!("{index}.txt"));
            std::fs::write(&path, vec![b'x'; 32768]).unwrap();
            (path.to_str().unwrap().to_owned(), None)
        })
        .collect();
    let writer = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = writer.blob_capture().unwrap();
    let mut turns = Vec::new();
    let prepared = capture
        .prepare_claude_location_in_turns(
            crate::configured_claude::Location::explicit_main(main).unwrap(),
            None,
            &AtomicBool::new(false),
            crate::history_pages::turns::Budget {
                bytes: 65536,
                checkpoints: 512,
            },
            |progress| {
                turns.push(progress);
                Ok(())
            },
            |_| Ok(()),
            || Ok(references),
        )
        .unwrap();
    assert_eq!(prepared.bodies.len(), 3);
    assert!(turns.len() > 3);
    assert!(turns.iter().all(|p| p.charged_bytes <= 65536));
    assert!(
        turns
            .windows(2)
            .all(|pair| pair[1].completed_turns == pair[0].completed_turns + 1)
    );
    assert!(turns.iter().any(|p| p.phase == "claude_body"));
    let selected: i64 = writer
        .archive
        .query_row("SELECT count(*) FROM input", [], |row| row.get(0))
        .unwrap();
    assert_eq!(selected, 0);
}
