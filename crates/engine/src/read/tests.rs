use super::*;

fn stats(source: &Imported) -> hashed_blobs::VerificationStats {
    match &source.reader {
        Decoder::Plain(reader) => reader.stats(),
        Decoder::Chain(_) | Decoder::Unopened(_) => panic!("the plain reader is the counted case"),
    }
}

fn commit_blob(engine: &crate::Engine, bytes: impl std::io::Read) -> polylogue_model::BlobHash {
    let file = engine.archive.prepare_bytes(bytes).unwrap();
    let blob = polylogue_store::NewBlob::whole(file.hash(), file.len());
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
                route: polylogue_model::Route::Live,
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
        Default::default(),
    );
    // Match the scheduler's read-ahead: retained plans are not advanced while
    // their completed jobs release the byte budget and later inputs enter.
    for plan in &mut plans {
        assert!(pool.submit(plan.next_job(&polylogue_testkit::stopped_attempt())));
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
        let Change::Reparse { chain } = plan.change else {
            unreachable!()
        };
        assert_eq!(plan.parser_chain, Some(chain));
        assert_eq!(
            result.chain,
            Some(chain),
            "the writer acknowledges the selected stored chain"
        );
        plan.apply(result, true);
        assert_eq!(plan.parser_chain, Some(chain));
    }
    let (done, received) = channel();
    let pool = Pool::start(
        3,
        budget as u64,
        engine.writer.sender().unwrap(),
        Arc::clone(&engine.blobs),
        engine.archive.clone(),
        done,
        Default::default(),
    );
    for plan in &mut plans {
        assert!(pool.submit(plan.next_job(&polylogue_testkit::stopped_attempt())));
    }
    drop(pool);
    engine.writer.flush().unwrap();
    for result in received {
        assert!(result.error.is_none(), "{:?}", result.error);
        let plan = plans.iter_mut().find(|p| p.path == result.path).unwrap();
        let Change::Reparse { chain } = plan.change else {
            unreachable!()
        };
        assert_eq!(plan.parser_chain, Some(chain));
        assert_eq!(
            result.chain,
            Some(chain),
            "the writer acknowledges the selected stored chain"
        );
        plan.apply(result, true);
        assert_eq!(plan.parser_chain, Some(chain));
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

/// The pass can submit a job that does not fit (a continuation enlarged for
/// a long record) while the queue holds the budget. After a stop, workers
/// take no queued job, so that room never comes back: the submit must
/// return on the stop, and the parse already started still commits.
#[test]
fn a_submit_waiting_for_room_returns_when_a_stop_is_requested() {
    use crate::plan::Plan;
    use crate::pool::{Bytes, Pool};
    use polylogue_store::fault::{self, Action};
    use polylogue_store::{Change, InputKey};
    use std::sync::{Arc, Mutex, mpsc::channel};
    use std::time::Duration;

    const CHUNK: usize = 64 << 10;
    let home = polylogue_testkit::ClaudeHome::new();
    let mut paths = Vec::new();
    // The third job reads twice the span, as a continuation enlarged for a
    // long record does, so it and the queued job together exceed the budget
    // even after the started parse gives its bytes back.
    let spans = [CHUNK, CHUNK, 2 * CHUNK];
    for (i, span) in spans.iter().enumerate() {
        let session = format!("room-{i}");
        let body = "x".repeat(*span);
        let records = polylogue_testkit::claude_user(&format!("first-{i}"), &session, &body)
            + &polylogue_testkit::claude_user(&format!("last-{i}"), &session, "last");
        let path = home.file(&format!("room-{i}.jsonl"));
        std::fs::write(&path, records).unwrap();
        paths.push(path);
    }
    {
        let (mut seed, _lock) = crate::import::tests::engine(&home, CHUNK as u64);
        let pass = seed.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    for path in &paths {
        std::fs::remove_file(path).unwrap();
    }
    polylogue_testkit::delete_index(&home.root);
    let (engine, _lock) = crate::import::tests::engine(&home, CHUNK as u64);
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let mut jobs = Vec::new();
    for (path, span) in paths.iter().zip(spans) {
        let input = engine.location_id(path);
        let reader = engine.reads.reader().unwrap();
        let progress = reader.input_progress(&input).unwrap().unwrap();
        let stored = reader
            .stored_chain(&input, progress.chain, progress.length)
            .unwrap();
        let mut plan = Plan::stored(
            path.clone(),
            origin,
            InputKey {
                id: input,
                format: "claude-code".into(),
                route: polylogue_model::Route::Live,
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
            span as u64,
            false,
        );
        jobs.push(plan.next_job(&polylogue_testkit::stopped_attempt()));
    }
    let [started, queued, waiting] = <[_; 3]>::try_from(jobs).ok().unwrap();
    let (done, received) = channel();
    let stop = crate::StopRequest::default();
    let pool = Arc::new(Pool::start(
        1,
        2 * CHUNK as u64,
        engine.writer.sender().unwrap(),
        Arc::clone(&engine.blobs),
        engine.archive.clone(),
        done,
        stop.clone(),
    ));
    fault::clear();
    fault::arm("parse", "room-0", Action::Hold, 0, 1);
    assert!(pool.submit(started));
    while !fault::held() {
        std::thread::sleep(Duration::from_millis(1));
    }
    // With the worker held, this job stays queued and the budget is full.
    assert!(pool.submit(queued));
    let (submitted, answer) = channel();
    let submitter = Arc::clone(&pool);
    let submitter = std::thread::spawn(move || {
        let _ = submitted.send(submitter.submit(waiting));
    });
    stop.request();
    fault::release();
    let queued = answer
        .recv_timeout(Duration::from_secs(5))
        .expect("a submit waiting for room returns once a stop is requested");
    submitter.join().unwrap();
    assert!(!queued, "a job submitted after the stop is not queued");
    assert_eq!(pool.cancel_queued(), 1, "the queued job is taken back");
    drop(Arc::into_inner(pool).expect("the submitter has ended"));
    // Only now has the held worker certainly passed its point.
    fault::clear();
    engine.writer.flush().unwrap();
    let completed: Vec<_> = received.into_iter().collect();
    assert_eq!(completed.len(), 1, "only the started parse reports");
    assert!(completed[0].error.is_none(), "{:?}", completed[0].error);
    assert_eq!(completed[0].path, paths[0]);
    assert!(completed[0].chain.is_some(), "the started parse commits");
}

#[test]
fn native_range_binds_the_opened_file_generation_and_rejects_path_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("native");
    std::fs::write(&path, b"original body").unwrap();
    let metadata = std::fs::metadata(&path).unwrap();
    let facts = polylogue_store::FileFacts::of(&metadata);
    let (bytes, observed) = read_native_range(&path, 0, facts.size, Some(facts)).unwrap();
    assert_eq!(bytes, b"original body");
    assert_eq!(
        observed,
        crate::acquisition::file_snapshot(&metadata).unwrap()
    );
    let replacement = dir.path().join("replacement");
    std::fs::write(&replacement, b"different body").unwrap();
    std::fs::rename(replacement, &path).unwrap();
    assert!(read_native_range(&path, 0, facts.size, Some(facts)).is_err());
}

#[test]
fn a_new_empty_native_file_has_a_real_body_and_known_eof_without_parser_progress() {
    let home = polylogue_testkit::ClaudeHome::new();
    home.append("empty.jsonl", "");
    let path = home.home.join(".claude/projects/-proj/empty.jsonl");
    let facts = polylogue_store::FileFacts::of(&std::fs::metadata(&path).unwrap());
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let input = polylogue_store::InputKey {
        id: "empty-native".into(),
        format: origin.name.into(),
        route: polylogue_model::Route::Live,
    };
    let mut plan = crate::plan::Plan::stored(
        path.clone(),
        origin,
        input.clone(),
        polylogue_store::Change::First,
        Some(facts),
        0,
        0,
        crate::pool::Bytes::File,
        64,
        true,
    );
    plan.observation = Some(engine.observation(&path, Some(facts)));
    plan.parser_deferred = true;
    assert!(
        !plan.finished(),
        "an unrecorded observation is still local acquisition work"
    );
    let mut job = plan.next_job(&polylogue_testkit::stopped_attempt());
    assert_eq!(job.store_from, Some(0));
    let mut context = engine
        .sampling_context(&path, origin.name, &origin.reading_version())
        .unwrap();
    context.facts = Some(facts);
    job.acquisition = Some(context);
    let (prepared, done) = crate::pool::run(job, &engine.archive, &engine.blobs, true);
    assert!(done.error.is_none(), "{:?}", done.error);
    let prepared = prepared.unwrap();
    assert!(prepared.batch.parse.is_empty());
    assert_eq!(prepared.batch.new_blob.as_ref().unwrap().size, 0);
    let receipt = prepared.batch.acquisition.as_ref().unwrap();
    assert_eq!(
        receipt.range,
        polylogue_store::primary::RawRange { start: 0, end: 0 }
    );
    assert_eq!(receipt.raw_total, Some(0));
    assert!(
        matches!(&receipt.generation, Some(polylogue_store::primary::SourceGeneration::File { observed, .. }) if observed.size == 0)
    );
    let results = engine
        .writer
        .sender()
        .unwrap()
        .store_group(vec![prepared], std::sync::Arc::clone(&engine.blobs))
        .unwrap();
    assert!(results.into_iter().all(|result| result.is_ok()));
    let hash = hashed_blobs::Hash(*blake3::hash(b"").as_bytes());
    assert_eq!(engine.blobs.extent(&hash).unwrap().decoded_len, 0);
    let reader = engine.reads.reader().unwrap();
    let progress = reader.input_progress(&input.id).unwrap().unwrap();
    assert_eq!(progress.length, 0);
    assert!(!progress.parsed);
}

#[test]
fn malformed_middle_is_counted_unreadable_and_keeps_later_native_facts_and_raw_eof() {
    let home = polylogue_testkit::ClaudeHome::new();
    let first = polylogue_testkit::claude_user("first", "raw-error", "valid prefix");
    let later = "retained raw tail ".repeat(128);
    let bytes = [
        first.as_bytes(),
        b"not-json\n",
        polylogue_testkit::claude_user("later", "raw-error", &later).as_bytes(),
    ]
    .concat();
    home.append("corrupt.jsonl", std::str::from_utf8(&bytes).unwrap());
    assert!(bytes.len() > 3 * first.len());
    let mut admission = None;
    for attempt in 0..2 {
        let (mut engine, lock) = crate::import::tests::engine(&home, first.len() as u64);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "attempt {attempt}: {pass:?}");
        if attempt == 1 {
            assert_eq!(pass.bytes_stored, 0);
            assert_eq!(pass.bytes_read, 0);
        }
        let reader = engine.reads.reader().unwrap();
        let known = reader
            .known_watched_input_at(
                "test-machine",
                home.file("corrupt.jsonl").as_os_str().as_encoded_bytes(),
            )
            .unwrap()
            .unwrap();
        let progress = reader.input_progress(&known.input_id).unwrap().unwrap();
        assert_eq!(progress.length, bytes.len() as u64);
        assert_eq!(progress.parsed_upto, bytes.len() as u64);
        assert!(progress.parsed);
        assert_eq!(
            reader
                .stored_range(
                    &known.input_id,
                    progress.chain,
                    0,
                    progress.length,
                    &engine.blobs
                )
                .unwrap(),
            bytes
        );
        let version = polylogue_sources::origin::by_name("claude-code")
            .unwrap()
            .reading_version();
        let checkpoint = reader
            .parser_checkpoint(&known.input_id, progress.chain, &version)
            .unwrap()
            .unwrap();
        assert!(
            checkpoint
                .checkpoint
                .unwrap()
                .matches(&version, bytes.len() as u64)
        );
        let session = polylogue_model::SessionId::parse("claude-code:raw-error").unwrap();
        assert_eq!(
            reader
                .messages(&session, false)
                .unwrap()
                .iter()
                .map(polylogue_model::Message::text)
                .collect::<Vec<_>>(),
            ["valid prefix", later.as_str()]
        );
        let current = reader
            .message_admission(&polylogue_model::MessageId::parse("claude-code:first").unwrap())
            .unwrap()
            .unwrap();
        if let Some(admission) = admission {
            assert_eq!(current, admission);
        }
        admission = Some(current);
        drop(reader);
        assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 3);
        assert_eq!(
            home.count("SELECT sum(unreadable) AS n FROM parse_state"),
            1
        );
        drop(engine);
        drop(lock);
    }
}

#[cfg(feature = "faults")]
#[test]
fn parser_error_keeps_last_checkpoint_and_quiet_native_raw_eof_across_reopen() {
    use polylogue_store::fault::{self, Action};
    let home = polylogue_testkit::ClaudeHome::new();
    let first = polylogue_testkit::claude_user("first", "raw-error", "valid prefix");
    let bytes = [
        first.as_bytes(),
        polylogue_testkit::claude_user("later", "raw-error", &"retained raw tail ".repeat(128))
            .as_bytes(),
    ]
    .concat();
    home.append("failed.jsonl", std::str::from_utf8(&bytes).unwrap());
    assert!(bytes.len() > 3 * first.len());
    let mut admission = None;
    let mut retained_checkpoint = None;
    for attempt in 0..2 {
        let (mut engine, lock) = crate::import::tests::engine(&home, first.len() as u64);
        fault::clear();
        // The real pool failure boundary is before provider parsing and after
        // raw preparation. Reopen retries that same failing parser boundary.
        fault::arm(
            "parse",
            &home.file("failed.jsonl").to_string_lossy(),
            Action::Fail,
            u32::from(attempt == 0),
            1,
        );
        let result = engine.run_pass();
        fault::clear();
        let pass = result.unwrap();
        assert!(
            pass.errors
                .iter()
                .any(|error| error.contains("injected fault at parse")),
            "attempt {attempt}: {pass:?}"
        );
        if attempt == 1 {
            assert_eq!(pass.bytes_stored, 0);
        }
        let reader = engine.reads.reader().unwrap();
        let known = reader
            .known_watched_input_at(
                "test-machine",
                home.file("failed.jsonl").as_os_str().as_encoded_bytes(),
            )
            .unwrap()
            .unwrap();
        let progress = reader.input_progress(&known.input_id).unwrap().unwrap();
        assert_eq!(progress.length, bytes.len() as u64);
        assert_eq!(progress.parsed_upto, first.len() as u64);
        assert!(progress.parsed, "the valid prefix retains its reading");
        assert!(!progress.waiting);
        assert_eq!(
            reader
                .stored_range(
                    &known.input_id,
                    progress.chain,
                    0,
                    progress.length,
                    &engine.blobs
                )
                .unwrap(),
            bytes
        );
        let version = polylogue_sources::origin::by_name("claude-code")
            .unwrap()
            .reading_version();
        let checkpoint = reader
            .parser_checkpoint(&known.input_id, progress.chain, &version)
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint.parsed_upto, first.len() as u64);
        assert!(
            checkpoint
                .checkpoint
                .as_ref()
                .unwrap()
                .matches(&version, first.len() as u64)
        );
        if let Some(previous) = &retained_checkpoint {
            assert_eq!(&checkpoint, previous);
        }
        retained_checkpoint = Some(checkpoint);
        let session = polylogue_model::SessionId::parse("claude-code:raw-error").unwrap();
        assert_eq!(
            reader
                .messages(&session, false)
                .unwrap()
                .iter()
                .map(polylogue_model::Message::text)
                .collect::<Vec<_>>(),
            ["valid prefix"]
        );
        let current = reader
            .message_admission(&polylogue_model::MessageId::parse("claude-code:first").unwrap())
            .unwrap()
            .unwrap();
        if let Some(admission) = admission {
            assert_eq!(current, admission);
        }
        admission = Some(current);
        let wire = reader
            .wire_shapes(polylogue_store::wire_shapes::Scope::current(Some(
                "claude-code",
            )))
            .unwrap();
        // The parser failure is the chain's parse state (docs/archive.md).
        assert_eq!(
            wire.coverage[0].status(),
            polylogue_store::wire_shapes::CoverageStatus::Failed
        );
        drop(reader);
        assert_eq!(home.count("SELECT sum(records) AS n FROM parse_state"), 1);
        assert_eq!(
            home.count("SELECT sum(unreadable) AS n FROM parse_state"),
            0
        );
        assert_eq!(
            home.count("SELECT count(*) AS n FROM parse_state WHERE status='failed' AND reason LIKE '%injected fault at parse%'"),
            1
        );
        drop(engine);
        drop(lock);
    }
}
