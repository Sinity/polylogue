use super::*;
use polylogue_store::fault::{self, Action};

/// While a build prepares, its progress counts the chains and bytes it
/// found due a reading and those it has read whole.
#[test]
fn build_progress_counts_chains_read_whole() {
    let home = ClaudeHome::new();
    let first = claude_user("first", "progress", "first chunk");
    let second = claude_user("second", "progress", "second chunk");
    home.append(
        "progress.jsonl",
        &[first.as_str(), second.as_str()].concat(),
    );
    {
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, lock) = home.engine(first.len() as u64);
    engine.config.workers = 1;
    let archive = engine.archive.clone();
    fault::arm("parse", "progress.jsonl", Action::Hold, 1, 1);
    let running = std::thread::spawn(move || {
        let result = engine.run_pass();
        drop(engine);
        drop(lock);
        result
    });
    while !fault::held() && !running.is_finished() {
        std::thread::yield_now();
    }
    let progress = archive.index_build();
    let held = fault::held();
    fault::release();
    let pass = running.join().unwrap().unwrap();
    fault::clear();
    assert!(held, "the chain's second range waits");
    let progress = progress.expect("the index stays private while the build prepares");
    assert_eq!(
        (progress.chains, progress.chains_read),
        (1, 0),
        "{progress:?}"
    );
    assert_eq!(progress.bytes, (first.len() + second.len()) as u64);
    assert_eq!(progress.bytes_read, 0);
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.texts("progress"), ["first chunk", "second chunk"]);
}

/// Reading stored files has the whole build's chains and bytes as its total
/// before it plans the first page of inputs (docs/operations.md, build
/// progress), so the share and estimate status shows are of the whole build.
#[test]
fn build_progress_has_the_whole_total_before_planning_starts() {
    let home = ClaudeHome::new();
    let first = claude_user("first", "one", "first file");
    let second = claude_user("second", "two", "second file");
    home.append("one.jsonl", &first);
    home.append("two.jsonl", &second);
    {
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, lock) = home.engine(1 << 20);
    engine.config.workers = 1;
    let archive = engine.archive.clone();
    // The build's first planning page waits.
    fault::arm("read.chains_behind_index", "All:[", Action::Hold, 0, 1);
    let running = std::thread::spawn(move || {
        let result = engine.run_pass();
        drop(engine);
        drop(lock);
        result
    });
    let held = fault::wait_held_within(Duration::from_secs(30));
    let progress = archive.index_build();
    let count = archive
        .build_progress()
        .and_then(|(_, progress)| progress.current);
    fault::release();
    let pass = running.join().unwrap().unwrap();
    fault::clear();
    assert!(held, "the first planning page waits");
    let progress = progress.expect("the index stays private while the build plans");
    let bytes = (first.len() + second.len()) as u64;
    assert_eq!(
        (
            progress.chains,
            progress.chains_read,
            progress.bytes,
            progress.bytes_read
        ),
        (2, 0, bytes, 0),
        "{progress:?}"
    );
    let count = count
        .expect("reading stored files is the current step")
        .count;
    assert_eq!(
        count.map(|count| (count.done, count.total)),
        Some((0, Some(bytes)))
    );
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.texts("one"), ["first file"]);
    assert_eq!(home.texts("two"), ["second file"]);
}

/// A new archive's first pass is a build (docs/installing.md, replacing an
/// index): the files found are stored first and then read by the build's
/// workers, which find them as stored chains due a reading, and the index
/// installs at the end of the pass.
#[test]
fn a_new_archive_stores_found_files_then_builds_its_index() {
    let home = ClaudeHome::new();
    let first = claude_user("first", "fresh", "first chunk");
    let second = claude_user("second", "fresh", "second chunk");
    home.append("fresh.jsonl", &[first.as_str(), second.as_str()].concat());
    home.append(
        "other.jsonl",
        &claude_user("third", "other", "another file"),
    );
    let (mut engine, lock) = home.engine(first.len() as u64);
    engine.config.workers = 1;
    let archive = engine.archive.clone();
    assert!(
        archive.index_build().is_some(),
        "a new archive's index is built, not created in place"
    );
    // The build reads `fresh.jsonl` in ranges; its second range waits.
    fault::arm("parse", "fresh.jsonl", Action::Hold, 1, 1);
    let running = std::thread::spawn(move || {
        let result = engine.run_pass();
        drop(engine);
        drop(lock);
        result
    });
    while !fault::held() && !running.is_finished() {
        std::thread::yield_now();
    }
    let progress = archive.index_build();
    let held = fault::held();
    fault::release();
    let pass = running.join().unwrap().unwrap();
    fault::clear();
    assert!(held, "the chain's second range waits");
    let progress = progress.expect("the index stays private while the build reads");
    assert_eq!(
        progress.chains, 2,
        "both files were stored before the build planned its chains: {progress:?}"
    );
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(archive.index_build().is_none());
    assert_eq!(home.texts("fresh"), ["first chunk", "second chunk"]);
    assert_eq!(home.texts("other"), ["another file"]);
}

/// Codex rollouts, plain and compressed, are named by the session ID their
/// first record states, so their capture reads that record. A new archive's
/// first pass still only stores them, and every reading of them is the
/// build's, on its workers (docs/installing.md, replacing an index).
#[test]
fn a_new_archive_stores_codex_rollouts_and_its_build_reads_them() {
    let home = ClaudeHome::new();
    let sessions = home.home.join(".codex/sessions");
    let archived = home.home.join(".codex/archived_sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    std::fs::create_dir_all(&archived).unwrap();
    let rollout = |id: &str, text: &str| {
        format!(
            "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{id}\"}}}}\n{}\n",
            serde_json::json!({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":text}]}})
        )
    };
    std::fs::write(
        sessions.join("rollout-firstbuild-plain.jsonl"),
        rollout("firstbuild-plain", "a plain rollout"),
    )
    .unwrap();
    std::fs::write(
        archived.join("rollout-firstbuild-packed.jsonl.zst"),
        zstd::stream::encode_all(
            rollout("firstbuild-packed", "a compressed rollout").as_bytes(),
            1,
        )
        .unwrap(),
    )
    .unwrap();
    let (mut engine, _lock) = home.engine(1 << 20);
    assert!(engine.archive.index_build().is_some());
    fault::arm("parse", "rollout-firstbuild-", Action::Record, 0, u32::MAX);
    let pass = engine.run_pass();
    let reads = fault::recorded("parse");
    fault::clear();
    let pass = pass.unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(engine.archive.index_build().is_none());
    let mut readers: Vec<_> = reads
        .iter()
        .map(|(path, thread)| {
            (
                std::path::Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                thread.starts_with("polylogue-build-"),
            )
        })
        .collect();
    readers.sort();
    readers.dedup();
    assert_eq!(
        readers,
        [
            ("rollout-firstbuild-packed.jsonl.zst".to_owned(), true),
            ("rollout-firstbuild-plain.jsonl".to_owned(), true),
        ],
        "each rollout is read, and only by the build's workers: {reads:?}"
    );
    for (session, text) in [
        ("codex:firstbuild-plain", "a plain rollout"),
        ("codex:firstbuild-packed", "a compressed rollout"),
    ] {
        let session = polylogue_model::SessionId::parse(session).unwrap();
        let messages = home.read(|reader| reader.messages(&session, false));
        assert_eq!(
            messages
                .iter()
                .map(polylogue_model::Message::text)
                .collect::<Vec<_>>(),
            [text]
        );
    }
}

/// With nothing stored and no sources, the empty build installs in the
/// first pass, so a daemon started that way serves reads at once.
#[test]
fn an_empty_archive_without_sources_installs_its_empty_build_in_the_first_pass() {
    let home = ClaudeHome::new();
    let (mut engine, _lock) = home.engine(1 << 20);
    engine.config.reads_sources = false;
    assert!(engine.archive.index_build().is_some());
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(engine.archive.index_build().is_none());
    assert_eq!(home.count("SELECT count(*) AS n FROM a.input"), 0);
}

/// A build reads the record as it stood when it started; the pass that
/// installs it then reads what arrived since, continuing each chain from
/// the build's position (docs/architecture.md, one path).
#[test]
fn a_build_installed_behind_its_record_is_brought_forward_by_upkeep() {
    let home = ClaudeHome::new();
    home.append(
        "forward.jsonl",
        &claude_user("one", "forward", "before the build"),
    );
    {
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    home.append(
        "forward.jsonl",
        &claude_user("two", "forward", "after the position"),
    );
    home.append("later.jsonl", &claude_user("three", "later", "a new input"));
    let (mut engine, _lock) = home.engine(1 << 20);
    assert!(engine.archive.index_build().is_some());
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(engine.archive.index_build().is_none());
    assert_eq!(
        home.texts("forward"),
        ["before the build", "after the position"]
    );
    assert_eq!(home.texts("later"), ["a new input"]);
}

#[test]
fn content_named_messages_keep_their_occurrence_after_ranges_and_restart() {
    use std::io::Write;
    let home = ClaudeHome::new();
    let path = home.home.join(".codex/sessions/rollout-counter.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let message = |text: &str| {
        serde_json::to_string(&serde_json::json!({
        "type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":text}]}
    })).unwrap()+"\n"
    };
    let mut bytes = "{\"type\":\"session_meta\",\"payload\":{\"id\":\"counter\"}}\n".to_owned();
    bytes.push_str(&message("same"));
    for n in 0..5000 {
        bytes.push_str(&message(&format!("distinct-{n}")));
    }
    bytes.push_str(&message("same"));
    std::fs::write(&path, bytes).unwrap();
    {
        let (mut engine, _lock) = home.engine(64 << 10);
        engine.config.workers = 1;
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(message("same").as_bytes())
        .unwrap();
    {
        let (mut engine, _lock) = home.engine(64 << 10);
        engine.config.workers = 1;
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    let session = polylogue_model::SessionId::parse("codex:counter").unwrap();
    let messages = home.read(|reader| reader.messages(&session, false));
    assert_eq!(messages.len(), 5003);
    let hash = blake3::hash(
        b"polylogue-message-content\0[\"user\",[{\"text\":\"same\",\"type\":\"text\"}]]",
    )
    .to_hex();
    let prefix = format!("counter~user-{}-", &hash.as_str()[..16]);
    let repeated: Vec<_> = messages
        .iter()
        .filter(|row| row.id.native().as_str().starts_with(&prefix))
        .map(|row| row.id.native().as_str().to_owned())
        .collect();
    assert_eq!(
        repeated,
        [
            format!("{prefix}0"),
            format!("{prefix}1"),
            format!("{prefix}2")
        ]
    );
}

/// Imports arriving every 100 ms while a large chain is built are each
/// answered while the build runs, the build completes, and it reads every
/// byte of the chain once (docs/architecture.md, one path).
#[test]
fn imports_every_100_ms_are_answered_and_the_build_reads_each_byte_once() {
    use polylogue_model::authorship::{Author, Authorship, Channel};
    use std::sync::mpsc::sync_channel;
    let home = ClaudeHome::new();
    let large: String = (0..12_000)
        .map(|n| {
            claude_user(
                &format!("large-{n}"),
                "large",
                &format!("record {n} {}", "x".repeat(400)),
            )
        })
        .collect();
    home.append("large.jsonl", &large);
    {
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    // Ranges of 256 KiB: the chain is read in many steps.
    let (mut engine, lock) = crate::import::tests::engine(&home, 256 << 10);
    engine.config.workers = 1;
    // Only the stored record is read: no file is looked at again.
    engine.config.reads_sources = false;
    let archive = engine.archive.clone();
    let (tx, rx) = crossbeam_channel::unbounded();
    let worker_tx = tx.clone();
    let worker = std::thread::spawn(move || {
        let status = Arc::new(SharedLoopStatus::default());
        let result = engine.run(worker_tx, rx, &status, Lifetime::UntilIdle, || {});
        drop(engine);
        drop(lock);
        result
    });
    let mut latencies = Vec::new();
    let mut imported_bytes = 0;
    let mut answered_during_build = 0;
    // Imports keep coming until the build is installed: one that waited for
    // a pause in the requests would never end.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut installed_under_requests = false;
    for n in 0.. {
        if Instant::now() > deadline {
            break;
        }
        let path = home.dir.path().join(format!("request-{n}.json"));
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "uuid": format!("request-{n}"), "chat_messages": [{
                    "uuid": format!("answer-{n}"), "sender": "human", "text": "a request"
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        let building = archive.index_build().is_some();
        // An import is read once: during the rebuild, with no installed
        // index, it is stored and read after the install.
        imported_bytes += std::fs::metadata(&path).unwrap().len();
        let (reply, answer) = sync_channel(1);
        let sent = Instant::now();
        if tx
            .send(LoopMsg::Import {
                paths: vec![path],
                format: None,
                dry_run: false,
                authorship: Authorship {
                    author: Author::Operator,
                    via: Channel::Cli,
                },
                parent: None,
                reply,
            })
            .is_err()
        {
            break;
        }
        let report = answer
            .recv_timeout(Duration::from_secs(30))
            .expect("the import is answered")
            .unwrap();
        latencies.push(sent.elapsed());
        if building && !archive.index_files().unwrap().serves() {
            assert!(
                matches!(report.readable, crate::Readable::AfterBuild { .. }),
                "{:?}",
                report.readable
            );
        }
        answered_during_build += u64::from(building && archive.index_build().is_some());
        std::thread::sleep(Duration::from_millis(100));
        if archive.index_build().is_none() {
            installed_under_requests = true;
            break;
        }
    }
    drop(tx);
    let summary = worker.join().unwrap().unwrap();
    assert!(summary.work.errors.is_empty(), "{:?}", summary.work.errors);
    assert!(
        installed_under_requests,
        "the build completed while the imports kept coming"
    );
    assert!(answered_during_build > 0, "{latencies:?}");
    let slowest = latencies.iter().max().unwrap();
    eprintln!(
        "imports answered: {}, {answered_during_build} during the build; slowest {slowest:?}",
        latencies.len()
    );
    // The loop answers an import between the chains it hands the workers
    // and while the rebuild's own thread runs its later stages.
    assert!(*slowest < Duration::from_secs(20), "{latencies:?}");
    // What the parser read, less the imports' own files: the chain once.
    // A range ends inside a record and the next range starts at that
    // record, so a record crossing a range boundary is read twice, under
    // one record per range; no part of the chain is read again otherwise.
    let read = summary.work.bytes_read - imported_bytes;
    let ranges = (large.len() as u64).div_ceil(256 << 10);
    let record = large.len() as u64 / 12_000;
    eprintln!("chain {} bytes; the build read {read}", large.len());
    assert!(read >= large.len() as u64, "{read}");
    assert!(read <= large.len() as u64 + ranges * 2 * record, "{read}");
    assert_eq!(home.texts("large").len(), 12_000);
    // Every import answered, during the build or after it, is read.
    for n in 0..latencies.len() {
        let id = polylogue_model::SessionId::parse(&format!("claude-ai:request-{n}")).unwrap();
        assert_eq!(
            home.read(|reader| reader.messages(&id, false)).len(),
            1,
            "request-{n}"
        );
    }
}

/// A new archive's first pass stores its files, rebuilds the index from
/// them and parses each file exactly once, in the rebuild: with no
/// installed index nothing plans a parse before the install
/// (docs/architecture.md, one path).
#[test]
fn a_first_pass_over_a_new_archive_parses_each_file_once_in_the_rebuild() {
    let home = ClaudeHome::new();
    home.append("first.jsonl", &claude_user("one", "first", "first file"));
    home.append("second.jsonl", &claude_user("two", "second", "second file"));
    fault::clear();
    fault::arm("parse", "", Action::Record, 0, 1_000);
    let (mut engine, _lock) = home.engine(1 << 20);
    assert!(engine.archive.index_build().is_some());
    let pass = engine.run_pass().unwrap();
    let parses = fault::recorded("parse");
    fault::clear();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(engine.archive.index_build().is_none());
    let mut parsed: Vec<String> = parses
        .iter()
        .map(|(subject, thread)| {
            assert!(
                thread.starts_with("polylogue-build-"),
                "{subject} on {thread}"
            );
            std::path::Path::new(subject)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    parsed.sort();
    assert_eq!(parsed, ["first.jsonl", "second.jsonl"]);
    assert_eq!(home.texts("first"), ["first file"]);
    assert_eq!(home.texts("second"), ["second file"]);
}

#[test]
fn a_live_spool_found_before_index_install_is_planned_for_capture_only() {
    let home = ClaudeHome::new();
    let (archive, _lock) = polylogue_store::Archive::open_or_create(&home.root).unwrap();
    let config = Config {
        home: home.home.clone(),
        chunk: 4096,
        ..crate::test_config()
    };
    let mut engine = Engine::new(config, archive).unwrap();
    let spool = engine
        .config
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    std::fs::create_dir_all(&spool).unwrap();
    let path = spool.join("42.ndjson");
    std::fs::write(
        &path,
        "{\"at_ns\":\"1767225600000000000\",\"event\":\"Stop\",\"provider\":\"claude-code\",\"session_id\":\"synthetic\",\"payload\":{}}\n",
    )
    .unwrap();
    let plans = engine
        .live_plans(&std::collections::HashSet::new(), &mut Pass::default())
        .unwrap();
    let [plan] = plans.as_slice() else {
        panic!("expected one new spool path, got {}", plans.len());
    };
    assert_eq!(plan.path, path);
    assert!(plan.storing_only);
}
