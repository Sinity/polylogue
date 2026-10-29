use super::*;

fn stats(source: &Imported) -> hashed_blobs::VerificationStats {
    match &source.reader {
        Decoder::Plain(reader) => reader.stats(),
        Decoder::Chain(_) | Decoder::Unopened(_) => panic!("the plain reader is the counted case"),
    }
}

fn commit_blob(engine: &crate::Engine, bytes: impl std::io::Read) -> polylogue_model::BlobHash {
    let file = engine.archive.prepare_bytes(bytes).unwrap();
    let blob = polylogue_store::NewBlob {
        hash: file.hash(),
        size: file.len(),
    };
    engine.writer.store_blob(blob, Some(file)).unwrap()
}

#[test]
fn stored_bytes_are_opened_at_their_first_read() {
    let home = polylogue_testkit::ClaudeHome::new();
    let body = "synthetic stored body ".repeat(10_000);
    home.append(
        "large.jsonl",
        &polylogue_testkit::claude_user("u1", "s", &body),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let blobs = std::sync::Arc::clone(&engine.blobs);
    let (input, chain, length): (String, i64, i64) =
        rusqlite::Connection::open(home.root.join("archive.db"))
            .unwrap()
            .query_row("SELECT input, chain, length FROM revision", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
    let reads = home.reads();
    let stored = reads
        .reader()
        .unwrap()
        .stored_chain(&input, chain, length as u64)
        .unwrap();
    let hash = stored
        .single_file_blob()
        .expect("one complete segment blob");
    let extent = blobs.extent(&hashed_blobs::Hash(hash.0)).unwrap();
    let file = home.root.join(&extent.segment);
    let aside = home.root.join("aside.hb");
    std::fs::rename(&file, &aside).unwrap();
    // A plan holds no open file: making one succeeds, reading it fails.
    let mut missing = Imported::from_chain(stored.clone(), std::sync::Arc::clone(&blobs));
    assert!(missing.read_range(0, 16).is_err());
    std::fs::rename(&aside, &file).unwrap();
    let mut present = Imported::from_chain(stored, blobs);
    assert_eq!(present.read_range(0, 1).unwrap(), b"{");
}

#[test]
fn detects_a_changed_tail() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("f");
    std::fs::write(&p, b"hello world\nmore\n").unwrap();
    assert!(tail_intact(&p, 12, b"world\n").unwrap());
    assert!(!tail_intact(&p, 12, b"WORLD\n").unwrap());
    assert_eq!(read_range(&p, 6, 11).unwrap(), b"world");
    assert_eq!(tail_of(b"abc"), b"abc");
}

#[test]
fn imported_ranges_decode_each_checked_frame_once_and_release_consumed_chunks() {
    let home = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 64 << 20);
    let size = 10 * 1024 * 1024;
    let hash = commit_blob(&engine, std::io::repeat(b'x').take(size as u64));
    let mut source = Imported::open(&engine.blobs, hash, size as u64).unwrap();
    let verified = stats(&source);
    assert_eq!(verified.decoded_bytes, 0);
    let chunk = 1024;
    for start in (0..size).step_by(chunk) {
        let end = (start + chunk).min(size);
        assert_eq!(
            source.read_range(start as u64, end as u64).unwrap(),
            vec![b'x'; end - start]
        );
        source.consume(end as u64);
        assert!(source.pending.is_empty());
        assert!(source.pending.capacity() <= chunk);
    }
    let stats = stats(&source);
    assert_eq!(stats.decoded_bytes, size as u64);
    assert_eq!(
        stats.frames,
        size.div_ceil(hashed_blobs::DEFAULT_FRAME) as u64
    );
}

#[test]
fn imported_ranges_keep_only_the_incomplete_record_overlap() {
    let home = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 64 << 20);
    let hash = commit_blob(&engine, b"complete\npartial\n".as_slice());
    let mut source = Imported::open(&engine.blobs, hash, 17).unwrap();
    assert_eq!(source.read_range(0, 12).unwrap(), b"complete\npar");
    source.consume(9);
    assert_eq!(source.pending, b"par");
    assert_eq!(source.read_range(9, 17).unwrap(), b"partial\n");
    source.consume(17);
    assert!(source.pending.is_empty());
    assert_eq!(stats(&source).decoded_bytes, 17);
}

#[test]
fn a_smaller_parse_range_can_reuse_already_decoded_overlap() {
    let home = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 64 << 20);
    let hash = commit_blob(&engine, b"0123456789abcdef".as_slice());
    let mut source = Imported::open(&engine.blobs, hash, 16).unwrap();
    assert_eq!(source.read_range(0, 14).unwrap(), b"0123456789abcd");
    source.consume(0);
    assert_eq!(source.read_range(0, 14).unwrap(), b"0123456789abcd");
    source.consume(4);
    assert_eq!(source.read_range(4, 8).unwrap(), b"4567");
    source.consume(8);
    assert_eq!(source.read_range(8, 16).unwrap(), b"89abcdef");
    assert_eq!(stats(&source).decoded_bytes, 16);
}

#[test]
fn export_scanning_uses_one_forward_pass_over_checked_frames() {
    let home = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 64 << 20);
    let mut bytes = b"[{}".to_vec();
    for _ in 0..10_000 {
        bytes.extend_from_slice(b",{\"padding\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}");
    }
    bytes.push(b']');
    let hash = commit_blob(&engine, bytes.as_slice());
    let mut source = engine
        .blobs
        .read_plain(
            &hashed_blobs::Hash(hash.0),
            Some(bytes.len() as u64),
            || Ok(()),
        )
        .unwrap();
    let verified = source.stats();
    let scan = polylogue_sources::export::scan(&mut source, |_, _| Ok(())).unwrap();
    assert_eq!(scan.elements, 10_001);
    assert!(scan.peak_element_bytes < 256);
    let stats = source.stats();
    assert_eq!(verified.decoded_bytes, 0);
    assert_eq!(stats.decoded_bytes, bytes.len() as u64);
    assert_eq!(stats.frames, verified.frames + 1);
}

#[test]
fn stored_chain_pool_releases_consumed_chunks_before_done_is_drained() {
    use crate::plan::Plan;
    use crate::pool::{Bytes, Pool};
    use polylogue_store::{Change, InputKey};
    use std::sync::{Arc, Mutex, mpsc::channel};

    const CHUNK: usize = 64 << 10;
    const INPUTS: usize = 32;
    let budget = 2 * CHUNK;
    let home = polylogue_testkit::ClaudeHome::new();
    let mut expected = Vec::new();
    for i in 0..INPUTS {
        let session = format!("stored-{i}");
        let first_id = format!("first-{i}");
        let empty = polylogue_testkit::claude_user(&first_id, &session, "");
        let body = "x".repeat(CHUNK - 3 - empty.len());
        let first = polylogue_testkit::claude_user(&first_id, &session, &body);
        assert_eq!(first.len(), CHUNK - 3);
        let last = format!("last complete record {i}");
        let second = polylogue_testkit::claude_user(&format!("last-{i}"), &session, &last);
        let path = home.file(&format!("stored-{i}.jsonl"));
        std::fs::write(&path, first + &second).unwrap();
        expected.push((path, session, body, last, second.as_bytes()[..3].to_vec()));
    }
    {
        let (mut seed, _lock) = crate::import::tests::engine(&home, CHUNK as u64);
        let pass = seed.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    for (path, ..) in &expected {
        std::fs::remove_file(path).unwrap();
    }
    polylogue_testkit::delete_index(&home.root);
    let (engine, _lock) = crate::import::tests::engine(&home, CHUNK as u64);
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let mut plans = Vec::new();
    for (path, ..) in &expected {
        let input = engine.location_id(path);
        let reader = engine.reads.reader().unwrap();
        let progress = reader.input_progress(&input).unwrap().unwrap();
        let stored = reader
            .stored_chain(&input, progress.chain, progress.length)
            .unwrap();
        plans.push(Plan::stored(
            path.clone(),
            origin,
            InputKey {
                id: input,
                format: "claude-code".into(),
                arrival_rank: 0,
            },
            Change::Reparse {
                chain: progress.chain,
            },
            None,
            progress.length,
            0,
            Bytes::Import(Arc::new(Mutex::new(Imported::from_chain(
                stored,
                Arc::clone(&engine.blobs),
            )))),
            CHUNK as u64,
            false,
        ));
    }
    let (done, received) = channel();
    let pool = Pool::start(
        3,
        budget as u64,
        engine.writer.sender().unwrap(),
        Arc::clone(&engine.blobs),
        engine.archive.clone(),
        done,
    );
    // Match the scheduler's read-ahead: retained plans are not advanced while
    // their completed jobs release the byte budget and later inputs enter.
    for plan in &mut plans {
        pool.submit(plan.next_job());
    }
    drop(pool);
    engine.writer.flush().unwrap();
    let retained: usize = plans
        .iter()
        .zip(&expected)
        .map(|(plan, (_, _, _, _, suffix))| {
            let Bytes::Import(source) = &plan.source else {
                unreachable!()
            };
            let source = source.lock().unwrap();
            assert_eq!(&source.pending, suffix);
            source.pending.capacity()
        })
        .sum();
    let completed: Vec<_> = received.into_iter().collect();
    assert_eq!(completed.len(), INPUTS);
    for result in completed {
        assert!(result.error.is_none(), "{:?}", result.error);
        let plan = plans.iter_mut().find(|p| p.path == result.path).unwrap();
        plan.apply(result, true);
    }
    let (done, received) = channel();
    let pool = Pool::start(
        3,
        budget as u64,
        engine.writer.sender().unwrap(),
        Arc::clone(&engine.blobs),
        engine.archive.clone(),
        done,
    );
    for plan in &mut plans {
        pool.submit(plan.next_job());
    }
    drop(pool);
    engine.writer.flush().unwrap();
    for result in received {
        assert!(result.error.is_none(), "{:?}", result.error);
        let plan = plans.iter_mut().find(|p| p.path == result.path).unwrap();
        plan.apply(result, true);
        assert!(plan.finished());
    }
    for (_, session, first, last, _) in &expected {
        assert_eq!(home.texts(session), [first.clone(), last.clone()]);
    }
    assert!(
        retained <= budget,
        "consumed chunks retained {retained} bytes outside the {budget}-byte pool budget"
    );
}
