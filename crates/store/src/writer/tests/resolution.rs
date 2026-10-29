use super::*;
use std::sync::{Arc, Barrier, mpsc::channel};

#[test]
fn overlapping_workers_read_after_the_preceding_commit() {
    let w = test_archive();
    let first = w
        .commit_source(
            batch(
                "first-route",
                0,
                0,
                vec![row(
                    1,
                    text_message("shared", "s", Role::Assistant, "partial", None),
                )],
                &["s"],
            ),
            &SOURCE_RECORD,
        )
        .unwrap();
    let mut second = batch(
        "second-route",
        2,
        0,
        vec![row(
            1,
            text_message("shared", "s", Role::Assistant, "complete", Some("end_turn")),
        )],
        &["s"],
    );
    second.input.id = blake3::hash(&SOURCE_RECORD).to_hex().to_string();
    let second = w.commit_source(second, &SOURCE_RECORD).unwrap();
    let mut lock = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let blocked = lock
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let sender = w.writer.sender().unwrap();
    sender.send(first).unwrap();
    let (waiting, observed) = channel();
    sender.0.resolver.observe_wait(waiting);
    let next = sender.clone();
    let thread = std::thread::spawn(move || next.send(second));
    observed
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("overlapping worker did not wait for its source snapshot");
    blocked.commit().unwrap();
    thread.join().unwrap().unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(texts(&w, "s"), ["complete"]);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM message WHERE id='claude-code:shared'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message_source", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn disjoint_workers_reserve_unique_rows_before_either_commits() {
    let w = test_archive();
    let readings = [("one", "one", "first"), ("two", "two", "second")]
        .into_iter()
        .map(|(input, session, word)| {
            w.commit_source(
                batch(
                    input,
                    0,
                    0,
                    vec![row(1, text_message(input, session, Role::User, word, None))],
                    &[session],
                ),
                &SOURCE_RECORD,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut lock = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let blocked = lock
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let start = Arc::new(Barrier::new(3));
    let (sent, queued) = channel();
    let mut threads = Vec::new();
    for reading in readings {
        let sender = w.writer.sender().unwrap();
        let start = Arc::clone(&start);
        let sent = sent.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            sent.send(sender.send(reading)).unwrap();
        }));
    }
    start.wait();
    queued
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap()
        .unwrap();
    queued
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap()
        .unwrap();
    blocked.commit().unwrap();
    for thread in threads {
        thread.join().unwrap();
    }
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(texts(&w, "one"), ["first"]);
    assert_eq!(texts(&w, "two"), ["second"]);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    for table in ["input_ref", "session", "message", "text"] {
        assert_eq!(
            index
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            2,
            "{table} reservations collided"
        );
    }
}

#[test]
fn disjoint_workers_share_text_identity_without_waiting_for_a_commit() {
    let w = test_archive();
    let readings = [("one", "one", "shared text"), ("two", "two", "shared text")]
        .into_iter()
        .map(|(input, session, word)| {
            w.commit_source(
                batch(
                    input,
                    0,
                    0,
                    vec![row(1, text_message(input, session, Role::User, word, None))],
                    &[session],
                ),
                &SOURCE_RECORD,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut lock = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let blocked = lock
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let start = Arc::new(Barrier::new(3));
    let (sent, queued) = channel();
    let mut threads = Vec::new();
    for reading in readings {
        let sender = w.writer.sender().unwrap();
        let start = Arc::clone(&start);
        let sent = sent.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            sent.send(sender.send(reading)).unwrap();
        }));
    }
    start.wait();
    queued
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap()
        .unwrap();
    queued
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap()
        .unwrap();
    blocked.commit().unwrap();
    for thread in threads {
        thread.join().unwrap();
    }
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(texts(&w, "one"), ["shared text"]);
    assert_eq!(texts(&w, "two"), ["shared text"]);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    for table in ["input_ref", "session", "message"] {
        assert_eq!(
            index
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            2,
            "{table} reservations collided"
        );
    }
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM text", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1,
        "shared text received two row identities"
    );
}

#[cfg(feature = "faults")]
#[test]
fn refused_commit_releases_reservations_for_an_ordinary_retry() {
    let w = test_archive();
    let sender = w.writer.sender().unwrap();
    let blobs = Arc::new(w.archive.blobs().unwrap());
    let make = || {
        batch(
            "retry-input",
            0,
            0,
            vec![row(
                1,
                text_message("retry-message", "s", Role::User, "kept after retry", None),
            )],
            &["s"],
        )
    };
    let reading = w.commit_source(make(), &SOURCE_RECORD).unwrap();
    crate::fault::arm("write.index_commit", "", crate::fault::Action::Fail, 0, 2);
    let results = sender
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: reading.clone(),
                files: Vec::new(),
            }],
            Arc::clone(&blobs),
        )
        .unwrap();
    assert!(results.iter().any(Result::is_err));
    crate::fault::clear();
    let results = sender
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: reading.clone(),
                files: Vec::new(),
            }],
            blobs,
        )
        .unwrap();
    for result in results {
        result.unwrap();
    }
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(texts(&w, "s"), ["kept after retry"]);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT parsed_upto FROM parse_state", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        100
    );
}

#[test]
fn collection_without_a_parser_commits_its_explicit_status() {
    let w = test_archive();
    let mut input = batch("collected", 0, 0, Vec::new(), &[]);
    input.parse[0].no_parser_yet = true;
    let input = w.commit_source(input, &SOURCE_RECORD).unwrap();
    w.writer.send(input).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let progress: (String, i64, String) = index
        .query_row(
            "SELECT status,parsed_upto,reading_version FROM parse_state",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        progress,
        ("no_parser_yet".into(), 100, "synthetic=1".into())
    );
}

#[test]
fn value_identity_survives_a_commit_newer_than_the_workers_snapshot() {
    let w = test_archive();
    let sender = w.writer.sender().unwrap();
    let captured = w
        .commit_source(batch("value-route", 0, 0, Vec::new(), &[]), &SOURCE_RECORD)
        .unwrap();
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let snapshot = index.unchecked_transaction().unwrap();
    assert_eq!(
        snapshot
            .query_row("SELECT count(*) FROM event_value", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let data = serde_json::json!({"shared":42});
    let mut input = batch("value-route", 0, 0, Vec::new(), &["s"]);
    input.parse[0].output.events.push(Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(sid("s")),
        kind: "value".into(),
        at: None,
        after: None,
        range: Some((0, 1)),
        data: data.clone(),
    });
    input.change = captured.change;
    input.observation = None;
    sender.send(input).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let latest = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let committed: i64 = latest
        .query_row("SELECT pk FROM event_value", [], |row| row.get(0))
        .unwrap();
    let prepared = sender
        .0
        .resolver
        .value(&snapshot, resolve::Kind::EventValue, &data)
        .unwrap();
    assert_eq!(
        prepared.key.pk, committed,
        "an older snapshot assigned a second identity to committed content"
    );
}

#[test]
fn keys_created_after_the_first_build_are_shared_by_following_routes() {
    let w = test_archive();
    w.writer
        .send(
            w.commit_source(
                batch(
                    "initial",
                    0,
                    0,
                    vec![row(
                        1,
                        text_message("initial", "s", Role::User, "initial", None),
                    )],
                    &["s"],
                ),
                &SOURCE_RECORD,
            )
            .unwrap(),
        )
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    for input in ["later-a", "later-b"] {
        w.writer
            .send(
                w.commit_source(
                    batch(
                        input,
                        0,
                        0,
                        vec![row(
                            1,
                            text_message("later", "s", Role::User, "later", None),
                        )],
                        &["s"],
                    ),
                    &SOURCE_RECORD,
                )
                .unwrap(),
            )
            .unwrap();
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{report:?}");
    }
    assert_eq!(texts(&w, "s").len(), 2);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM message WHERE id='claude-code:later'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
}

#[test]
fn a_metadata_only_observation_keeps_the_following_rewrite_chain_number() {
    let w = test_archive();
    w.writer
        .send(
            w.commit_source(
                batch(
                    "observed",
                    0,
                    0,
                    vec![row(
                        1,
                        text_message("same", "s", Role::User, "before", None),
                    )],
                    &["s"],
                ),
                &SOURCE_RECORD,
            )
            .unwrap(),
        )
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let mut metadata = batch("observed", 0, 0, Vec::new(), &[]);
    metadata.change = Change::Unchanged;
    metadata.parse.clear();
    let mut rewritten = batch(
        "observed",
        0,
        0,
        vec![row(1, text_message("same", "s", Role::User, "after", None))],
        &["s"],
    );
    rewritten.change = Change::Rewritten;
    let (blob, file) = w.archive.prepare_blob(&[b'w'; 100]).unwrap();
    rewritten.new_blob = Some(blob);
    let results = w
        .writer
        .sender()
        .unwrap()
        .store_group(
            vec![
                crate::staging::PreparedBatch {
                    batch: metadata,
                    files: Vec::new(),
                },
                crate::staging::PreparedBatch {
                    batch: rewritten,
                    files: file.into_iter().collect(),
                },
            ],
            Arc::new(w.archive.blobs().unwrap()),
        )
        .unwrap();
    for result in results {
        result.unwrap();
    }
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(texts(&w, "s"), ["after"]);
}

#[test]
fn confirmed_readings_resolve_and_apply_without_a_local_source_database() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    crate::schema::create_index(&index).unwrap();
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let blob = BlobHash::of(b"confirmed image");
    let mut message = text_message("confirmed", "s", Role::User, "confirmed text", None);
    message
        .blocks
        .push(Block::Image(polylogue_model::Attachment {
            blob: Some(blob),
            ..Default::default()
        }));
    let mut input = batch("confirmed-input", 0, 0, vec![row(1, message)], &["s"]);
    input.change = Change::Reparse { chain: 41 };
    let available = HashMap::from([(blob, Some(15))]);
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut input.parse.remove(0)).unwrap();
    let mut readings = vec![(input.input.clone(), 41, input.length, reading)];
    resolver
        .confirmed_readings(&mut readings, &available)
        .unwrap();
    assert!(!archive.archive_db().exists());
    let tx = index.transaction().unwrap();
    readings[0]
        .3
        .visit(|step| {
            rows::write_confirmed_rows(&tx, &input, step, 41, 1, None, Some(&fixture_receipt()))
                .map(|_| ())
        })
        .unwrap();
    tx.commit().unwrap();
    let (chain, position): (i64, i64) = index
        .query_row("SELECT chain,parsed_upto FROM parse_state", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!((chain, position), (41, 100));
    let stored: (Vec<u8>, i64) = index
        .query_row(
            "SELECT blob_hash,size FROM block WHERE kind='image'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(stored, (blob.0.to_vec(), 15));
    assert!(!archive.archive_db().exists());
}

#[test]
fn confirmed_readings_require_the_confirmed_chain_identity() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    let index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    crate::schema::create_index(&index).unwrap();
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let mut input = batch("confirmed-input", 0, 0, Vec::new(), &[]);
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut input.parse.remove(0)).unwrap();
    let mut readings = vec![(input.input, 0, input.length, reading)];
    assert!(matches!(
        resolver.confirmed_readings(&mut readings, &HashMap::new()),
        Err(StoreError::Invalid(_))
    ));
    assert!(!archive.archive_db().exists());
}
#[test]
fn continuing_readings_require_the_committed_prefix_and_retry_once() {
    let w = test_archive();
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let tx = conn.transaction().unwrap();
    let first = batch(
        "cursor",
        0,
        0,
        vec![row(
            1,
            text_message("first", "s", Role::User, "first", None),
        )],
        &["s"],
    );
    write_test_rows(&tx, &first, &first.parse[0], 1, 1, None).unwrap();
    let gap = batch(
        "cursor",
        0,
        200,
        vec![row(
            201,
            text_message("last", "s", Role::User, "last", None),
        )],
        &["s"],
    );
    assert!(write_test_rows(&tx, &gap, &gap.parse[0], 1, 2, None).is_err());
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let middle = batch(
        "cursor",
        0,
        100,
        vec![row(
            101,
            text_message("middle", "s", Role::User, "middle", None),
        )],
        &["s"],
    );
    assert_eq!(
        write_test_rows(&tx, &middle, &middle.parse[0], 1, 3, None).unwrap(),
        1
    );
    assert_eq!(
        write_test_rows(&tx, &middle, &middle.parse[0], 1, 4, None).unwrap(),
        0
    );
    assert_eq!(
        write_test_rows(&tx, &gap, &gap.parse[0], 1, 5, None).unwrap(),
        1
    );
    assert_eq!(
        tx.query_row("SELECT records FROM parse_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        3
    );
    tx.commit().unwrap();
}

#[test]
fn no_parser_metadata_waits_at_partial_eof_and_resumes_on_append() {
    let w = test_archive();
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let tx = conn.transaction().unwrap();
    let mut partial = batch("metadata", 0, 0, Vec::new(), &[]);
    partial.parse[0].no_parser_yet = true;
    partial.parse[0].parsed_upto = 40;
    write_test_rows(&tx, &partial, &partial.parse[0], 1, 1, None).unwrap();
    let state: (String, String, String) = tx
        .query_row(
            "SELECT status,detail,reading_version FROM parse_state",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        state,
        ("waiting".into(), "100".into(), "synthetic=1".into())
    );
    let mut append = batch("metadata", 0, 40, Vec::new(), &[]);
    append.parse[0].no_parser_yet = true;
    write_test_rows(&tx, &append, &append.parse[0], 1, 2, None).unwrap();
    let state: (String, i64, Option<String>) = tx
        .query_row(
            "SELECT status,parsed_upto,detail FROM parse_state",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(state, ("no_parser_yet".into(), 140, None));
    tx.commit().unwrap();
}

#[test]
fn imported_message_order_is_independent_of_local_byte_ranges() {
    let w = test_archive();
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let tx = conn.transaction().unwrap();
    let mut input = batch(
        "bundle",
        2,
        0,
        vec![
            row(
                10,
                text_message(
                    "late-commit",
                    "s",
                    Role::Assistant,
                    "first",
                    Some("end_turn"),
                ),
            ),
            row(
                20,
                text_message("early-commit", "s", Role::User, "second", None),
            ),
        ],
        &["s"],
    );
    input.parse[0].order_from = Some(100);
    input.parse[0].output.message_order.extend([(0, 1), (1, 0)]);
    write_test_rows(&tx, &input, &input.parse[0], 1, 1, None).unwrap();
    let rows = tx
        .prepare("SELECT id,seq,source_start FROM message ORDER BY seq")
        .unwrap()
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(
        rows,
        vec![
            ("claude-code:early-commit".into(), 0, 20),
            ("claude-code:late-commit".into(), 1, 10)
        ]
    );
    let mut invalid = input.clone();
    invalid.parse[0].output.message_order.insert(0, u64::MAX);
    assert!(write_test_rows(&tx, &invalid, &invalid.parse[0], 1, 2, None).is_err());
    tx.commit().unwrap();
}

fn overlapping_confirmed_readings() -> Vec<(InputKey, i64, u64, ChainReading)> {
    [
        ("source-a", 1, 0u8, "partial", None, "/first"),
        ("source-b", 7, 2u8, "complete", Some("end_turn"), "/second"),
    ]
    .into_iter()
    .map(|(id, chain, rank, text, stop, cwd)| {
        let mut input = batch(
            id,
            rank,
            0,
            vec![row(
                1,
                text_message("shared-confirmed", "s", Role::Assistant, text, stop),
            )],
            &["s"],
        );
        input.parse[0].output.sessions[0].cwd.push(cwd.into());
        let mut reading = ChainReading::new().unwrap();
        reading.push(&mut input.parse.remove(0)).unwrap();
        (input.input, chain, input.length, reading)
    })
    .collect()
}

fn apply_confirmed_readings(
    index: &mut Connection,
    readings: &mut [(InputKey, i64, u64, ChainReading)],
) -> Result<()> {
    let tx = index.transaction()?;
    for (input, chain, length, reading) in readings {
        let batch = Batch {
            input: input.clone(),
            change: Change::Reparse { chain: *chain },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: *length,
            parse: Vec::new(),
            release: InFlight::default(),
        };
        reading.visit(|step| {
            write_rows(&tx, &batch, step, *chain, 1, None, Some(&fixture_receipt())).map(|_| ())
        })?;
    }
    tx.commit()?;
    Ok(())
}

#[test]
fn a_streamed_position_resolves_overlapping_chains_once() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    crate::schema::create_index(&index).unwrap();
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let mut readings = overlapping_confirmed_readings();
    resolver
        .confirmed_readings(&mut readings, &HashMap::new())
        .unwrap();
    assert!(!archive.archive_db().exists());
    apply_confirmed_readings(&mut index, &mut readings).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message_source", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM parse_state", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    let text: Vec<u8> = index
        .query_row(
            "SELECT t.body FROM block b JOIN text t ON t.id=b.text_id",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(crate::encoding::unpack_text(&text).unwrap(), "complete");
    let mut statement=index.prepare("SELECT i.id,s.facts FROM session_record s JOIN input_ref i ON i.pk=s.input_pk ORDER BY i.id").unwrap();
    let facts = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                serde_json::from_str::<Json>(&row.get::<_, String>(1)?).unwrap(),
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(facts[0].1["cwd"], serde_json::json!(["/first"]));
    assert_eq!(facts[1].1["cwd"], serde_json::json!(["/second"]));
}

#[test]
fn a_refused_streamed_position_keeps_all_rows_and_positions_uncommitted() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    crate::schema::create_index(&index).unwrap();
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let mut readings = overlapping_confirmed_readings();
    resolver
        .confirmed_readings(&mut readings, &HashMap::new())
        .unwrap();
    index.execute_batch("CREATE TRIGGER refuse_second_chain BEFORE INSERT ON parse_state WHEN NEW.chain=7 BEGIN SELECT RAISE(ABORT,'synthetic position refusal'); END;").unwrap();
    assert!(apply_confirmed_readings(&mut index, &mut readings).is_err());
    drop(readings);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM parse_state", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    index
        .execute_batch("DROP TRIGGER refuse_second_chain")
        .unwrap();
    let mut retry = overlapping_confirmed_readings();
    resolver
        .confirmed_readings(&mut retry, &HashMap::new())
        .unwrap();
    apply_confirmed_readings(&mut index, &mut retry).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM parse_state", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

fn commit_event_bytes(w: &TestArchive, mut input: Batch, bytes: &[u8]) -> Batch {
    let (blob, file) = w.archive.prepare_blob(bytes).unwrap();
    if input.input.arrival_rank == 2 {
        input.input.id = blob.hash.to_string();
    }
    input.new_blob = Some(blob);
    input.length = bytes.len() as u64;
    for step in &mut input.parse {
        step.start = 0;
        step.parsed_upto = input.length;
        step.at_end = true;
    }
    let mut retry = input.clone();
    let stored = w
        .writer
        .sender()
        .unwrap()
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: input,
                files: file.into_iter().collect(),
            }],
            Arc::new(w.archive.blobs().unwrap()),
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .unwrap();
    retry.change = Change::Reparse {
        chain: stored.chain,
    };
    retry.new_blob = None;
    retry.observation = None;
    assert!(w.writer.flush().unwrap().errors.is_empty());
    retry
}

#[test]
fn imported_events_preserve_identity_claims_and_canonical_fields_across_hops() {
    let first = test_archive();
    let mut input = batch("original", 0, 0, Vec::new(), &[]);
    let event = Event {
        at_quality: polylogue_model::TimeQuality::Intrinsic,
        at_precision: Some(polylogue_model::Precision::Instant),
        at_until: None,
        session: None,
        kind: "input.report".into(),
        at: Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z"),
        after: None,
        range: Some((0, 1000)),
        data: serde_json::json!({"remaining":42}),
    };
    input.parse[0].output.events.push(event.clone());
    input.parse[0].counts = counts(1);
    let mut native = serde_json::to_vec(&event).unwrap();
    assert!(native.len() <= 1000);
    native.resize(1000, b' ');
    commit_event_bytes(&first, input, &native);
    let conn = crate::archive::open_read(&first.archive.index_db()).unwrap();
    let id: String = conn
        .query_row("SELECT id FROM event", [], |row| row.get(0))
        .unwrap();
    let source = crate::archive::open_read(&first.archive.archive_db()).unwrap();
    let admitted: i64 = source
        .query_row(
            "SELECT commit_seq FROM extent WHERE input='original' AND offset=0",
            [],
            |row| row.get(0),
        )
        .unwrap();
    commit_event_bytes(
        &first,
        batch("unrelated", 0, 0, Vec::new(), &[]),
        b"         ",
    );
    let reads = ReadPool::open(&first.archive).unwrap();
    let reader = reads.reader().unwrap();
    let exported = reader.event_record(&id).unwrap().unwrap();
    assert_eq!(exported.event, event);
    assert_eq!(exported.commit, admitted as u64);
    assert!(reader.change_position().unwrap().index > admitted);
    drop(reader);
    let wire = serde_json::to_vec(&exported).unwrap();
    let imported: polylogue_model::bundle::InputEventRecord =
        serde_json::from_slice(&wire).unwrap();
    imported.validate().unwrap();
    let second = test_archive();
    let mut portable = batch("bundle-local", 2, 0, Vec::new(), &[]);
    portable.input.format = polylogue_model::bundle::FORMAT.into();
    portable.parse[0].output.events.push(imported.event);
    portable.parse[0]
        .output
        .imported_events
        .insert(0, imported.identity);
    portable.parse[0].counts = counts(1);
    let retry = commit_event_bytes(&second, portable, &wire);
    let local_input = retry.input.id.clone();
    second.writer.send(retry).unwrap();
    assert!(second.writer.flush().unwrap().errors.is_empty());
    let reads = ReadPool::open(&second.archive).unwrap();
    let reexported = reads.reader().unwrap().event_record(&id).unwrap().unwrap();
    assert_eq!(reexported.identity.id, id);
    assert_eq!(reexported.event, event);
    assert!(
        exported
            .identity
            .sources
            .iter()
            .all(|claim| reexported.identity.sources.contains(claim))
    );
    assert!(
        reexported
            .identity
            .sources
            .iter()
            .any(|claim| claim.input == local_input
                && claim.source_start == Some(0)
                && claim.source_end == Some(wire.len() as u64))
    );
    let conn = crate::archive::open_read(&second.archive.index_db()).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM input_ref WHERE id='original'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM event", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let mut newer = event.clone();
    newer.kind = "input.revised".into();
    newer.range = Some((500, 999));
    newer.data = serde_json::json!({"remaining":7});
    let identity = polylogue_model::bundle::ImportedEventIdentity {
        id: id.clone(),
        sources: vec![polylogue_model::bundle::EventSourceClaim {
            input: "newer-original".into(),
            source: "native".into(),
            source_chain: 1,
            source_start: Some(500),
            source_end: Some(999),
        }],
    };
    let record = polylogue_model::bundle::InputEventRecord {
        reference: format!("event:{id}"),
        commit: 17,
        identity: identity.clone(),
        event: newer.clone(),
    };
    let wire = serde_json::to_vec(&record).unwrap();
    let mut competing = batch("preferred-local", 0, 0, Vec::new(), &[]);
    competing.input.format = polylogue_model::bundle::FORMAT.into();
    competing.parse[0].output.events.push(newer.clone());
    competing.parse[0]
        .output
        .imported_events
        .insert(0, identity);
    competing.parse[0].counts = counts(1);
    let retry = commit_event_bytes(&second, competing, &wire);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM event_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM event_value", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    let selected = reads.reader().unwrap().event_record(&id).unwrap().unwrap();
    assert_eq!(selected.event, newer);
    assert!(
        selected
            .identity
            .sources
            .iter()
            .all(|claim| claim.input != "original" && claim.input != local_input)
    );
    assert!(
        selected
            .identity
            .sources
            .iter()
            .any(|claim| claim.input == "newer-original")
    );
    let mut empty = retry;
    empty.parse[0].output = Output::default();
    empty.parse[0].reading_version = "replacement-reading".into();
    empty.parse[0].counts = counts(0);
    second.writer.send(empty).unwrap();
    let report = second.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .event_record(&id)
            .unwrap()
            .unwrap()
            .event,
        event
    );
}

#[test]
fn confirmed_projection_applies_live_facts_after_an_archived_byte_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = crate::archive::open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&index).unwrap();
    let tx = index.transaction().unwrap();
    let mut input = batch(
        "confirmed-subset",
        0,
        100,
        vec![row(
            120,
            text_message("live", "s", Role::User, "retained fact", None),
        )],
        &["s"],
    );
    resolve::prepare_index(&tx, &mut input, 1, false).unwrap();
    assert!(
        write_rows(
            &tx,
            &input,
            &input.parse[0],
            1,
            1,
            None,
            Some(&fixture_receipt())
        )
        .is_err()
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    rows::write_confirmed_rows(
        &tx,
        &input,
        &input.parse[0],
        1,
        1,
        None,
        Some(&fixture_receipt()),
    )
    .unwrap();
    assert_eq!(
        tx.query_row("SELECT parsed_upto FROM parse_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        200
    );
    assert_eq!(
        tx.query_row("SELECT source_start FROM message", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        120
    );
    tx.commit().unwrap();
}

#[test]
fn canonical_source_times_keep_nanoseconds_and_evidence_independently_of_query_caches() {
    use polylogue_model::{
        Document, DocumentId, DocumentPurpose, DocumentRow, Precision, TimeEvidence, TimeQuality,
    };
    let w = test_archive();
    let at = Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z");
    let evidence = TimeEvidence {
        quality: TimeQuality::Operator,
        precision: Some(Precision::Bounded),
        until: Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456999Z"),
    };
    let mut message = text_message("precise", "s", Role::User, "precise source time", None);
    message.set_time(at, evidence).unwrap();
    let mut document = Document::new(
        DocumentId::parse("claude-code:precise-document").unwrap(),
        DocumentPurpose::Context,
    );
    document.blocks = vec![Block::Text {
        text: "later canonical document".into(),
    }];
    document.set_time(at, evidence).unwrap();
    let mut input = batch(
        "precise-source",
        0,
        0,
        vec![row(1, message.clone())],
        &["s"],
    );
    input.parse[0].output.sessions[0]
        .set_time(at, evidence)
        .unwrap();
    input.parse[0].output.sessions[0].git_commit = Some("synthetic-revision".into());
    input.parse[0].output.sessions[0].model_provider = Some("synthetic-provider".into());
    input.parse[0].output.sessions[0].instructions =
        Some("Exact native session instructions".into());
    let session = input.parse[0].output.sessions[0].clone();
    input.parse[0].output.documents.push(DocumentRow {
        offset: 2,
        end: 3,
        document: document.clone(),
    });
    let mut event = Event {
        session: Some(sid("s")),
        kind: "precise.event".into(),
        at: None,
        at_quality: TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((4, 5)),
        data: serde_json::json!({"exact":true}),
    };
    event.set_time(at, evidence).unwrap();
    input.parse[0].output.events.push(event.clone());
    let bytes = serde_json::to_vec(&input.parse[0].output).unwrap();
    commit_event_bytes(&w, input, &bytes);
    w.writer.derive().unwrap();
    let conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    conn.execute_batch("UPDATE message SET at=0;UPDATE session SET started_at=0;UPDATE event SET at=0;UPDATE document_source SET source_time=0;").unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    {
        let reader = reads.reader().unwrap();
        assert_eq!(reader.messages(&sid("s"), false).unwrap(), vec![message]);
        assert_eq!(reader.session(&sid("s")).unwrap(), Some(session));
        assert_eq!(
            reader.document(&document.id).unwrap(),
            Some(document.clone())
        );
        assert_eq!(reader.events(&sid("s")).unwrap(), vec![event]);
    }
    let mut earlier = document.clone();
    earlier.blocks = vec![Block::Text {
        text: "earlier canonical document".into(),
    }];
    earlier
        .set_time(
            Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456700Z"),
            TimeEvidence::intrinsic(Precision::Instant),
        )
        .unwrap();
    let mut other = batch("other-source", 0, 0, Vec::new(), &[]);
    other.parse[0].output.documents.push(DocumentRow {
        offset: 2,
        end: 3,
        document: earlier,
    });
    let bytes = serde_json::to_vec(&other.parse[0].output).unwrap();
    commit_event_bytes(&w, other, &bytes);
    assert_eq!(
        reads.reader().unwrap().document(&document.id).unwrap(),
        Some(document)
    );
}

#[test]
fn a_remaining_native_event_keeps_its_original_identity_after_subset_projection() {
    let w = test_archive();
    let event = Event {
        session: None,
        kind: "native.input".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((3, 17)),
        data: serde_json::json!({"same":true}),
    };
    let mut full = batch("native-source", 0, 0, Vec::new(), &[]);
    full.parse[0].output.events = vec![event.clone(), event.clone()];
    full.parse[0].prepare();
    let ids = full.parse[0].event_identities("native-source", 1).unwrap();
    assert_ne!(ids[0], ids[1]);
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let tx = conn.transaction().unwrap();
    write_test_rows(&tx, &full, &full.parse[0], 1, 1, None).unwrap();
    let mut subset = batch("native-source", 0, 0, Vec::new(), &[]);
    subset.parse[0].output.events.push(event);
    subset.parse[0].output.event_ids.insert(0, ids[1].clone());
    write_test_rows(&tx, &subset, &subset.parse[0], 1, 2, None).unwrap();
    let remaining: String = tx
        .query_row("SELECT id FROM event", [], |row| row.get(0))
        .unwrap();
    assert_eq!(remaining, ids[1]);
    subset.parse[0].output.imported_events.insert(
        0,
        polylogue_model::bundle::ImportedEventIdentity {
            id: "contradictory-id".into(),
            sources: vec![polylogue_model::bundle::EventSourceClaim {
                input: "original".into(),
                source: "native".into(),
                source_chain: 1,
                source_start: Some(3),
                source_end: Some(17),
            }],
        },
    );
    assert!(write_test_rows(&tx, &subset, &subset.parse[0], 1, 3, None).is_err());
    assert_eq!(
        tx.query_row("SELECT count(*) FROM event", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1
    );
    tx.commit().unwrap();
}

#[test]
fn retained_event_admission_precedes_raw_receipts_and_survives_missing_source_mappings() {
    let w = test_archive();
    let event = Event {
        session: None,
        kind: "retained.event".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((0, 1000)),
        data: serde_json::json!({"retained":true}),
    };
    let mut input = batch("retained-source", 0, 0, Vec::new(), &[]);
    input.parse[0].output.events.push(event);
    input.parse[0].counts = counts(1);
    let mut bytes = serde_json::to_vec(&input.parse[0].output.events[0]).unwrap();
    assert!(bytes.len() <= 1000);
    bytes.resize(1000, b' ');
    let mut confirmed = commit_event_bytes(&w, input, &bytes);
    confirmed.parse[0].output.event_receipts.insert(0, 41);
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let tx = conn.transaction().unwrap();
    resolve::prepare_index(&tx, &mut confirmed, 1, false).unwrap();
    let control = crate::wire_shapes::ReadingReceipt {
        sequence: 777,
        id: Some("parse-progress-control".into()),
        observation: None,
        observed_at_ms: None,
    };
    rows::write_confirmed_rows(
        &tx,
        &confirmed,
        &confirmed.parse[0],
        1,
        999,
        None,
        Some(&control),
    )
    .unwrap();
    let id: String = tx
        .query_row("SELECT id FROM event", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT admitted FROM event_source", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        41
    );
    tx.commit().unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    {
        let reader = reads.reader().unwrap();
        assert!(
            reader
                .source_commit("retained-source", 1, Some(1000))
                .unwrap()
                .unwrap()
                < 41
        );
        assert_eq!(reader.event_record(&id).unwrap().unwrap().commit, 41);
    }
    let source = crate::archive::open_write(&w.archive.archive_db(), 1).unwrap();
    source
        .execute("DELETE FROM extent WHERE input='retained-source'", [])
        .unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .source_commit("retained-source", 1, Some(1000))
            .unwrap(),
        None
    );
    assert_eq!(reader.event_record(&id).unwrap().unwrap().commit, 41);
}

#[test]
fn checkpoint_admission_stamps_survive_duplicate_resets_and_rollback() {
    let w = test_archive();
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let first = batch("checkpoint", 0, 0, Vec::new(), &[]);
    let tx = conn.transaction().unwrap();
    write_test_rows(&tx, &first, &first.parse[0], 1, 10, None).unwrap();
    write_test_rows(&tx, &first, &first.parse[0], 1, 20, None).unwrap();
    assert_eq!(
        tx.query_row("SELECT commit_seq FROM parse_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        10
    );
    tx.commit().unwrap();
    let continuation = batch("checkpoint", 0, 100, Vec::new(), &[]);
    let tx = conn.transaction().unwrap();
    write_test_rows(&tx, &continuation, &continuation.parse[0], 1, 25, None).unwrap();
    assert_eq!(
        tx.query_row("SELECT commit_seq FROM parse_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        25
    );
    tx.rollback().unwrap();
    assert_eq!(
        conn.query_row("SELECT commit_seq FROM parse_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        10
    );
    let mut waiting = batch("checkpoint", 0, 100, Vec::new(), &[]);
    waiting.parse[0].parsed_upto = 100;
    let tx = conn.transaction().unwrap();
    write_test_rows(&tx, &waiting, &waiting.parse[0], 1, 30, None).unwrap();
    write_test_rows(&tx, &waiting, &waiting.parse[0], 1, 40, None).unwrap();
    let state: (i64, String) = tx
        .query_row("SELECT commit_seq,status FROM parse_state", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(state, (30, "waiting".into()));
    let mut upgraded = first;
    upgraded.parse[0].reading_version = "synthetic=2".into();
    write_test_rows(&tx, &upgraded, &upgraded.parse[0], 1, 50, None).unwrap();
    assert_eq!(
        tx.query_row("SELECT commit_seq FROM parse_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        50
    );
    tx.commit().unwrap();
}

#[test]
fn canonical_fold_receipts_survive_raw_mapping_absence_and_cache_rebuild() {
    use polylogue_model::{Document, DocumentId, DocumentPurpose, DocumentRow};
    let w = test_archive();
    let mut input = batch("fact-source", 0, 0, Vec::new(), &["receipt-session"]);
    input.parse[0].output.sessions[0].offer_cwd("/workspace/first");
    let mut document = Document::new(
        DocumentId::parse("claude-code:receipt-document").unwrap(),
        DocumentPurpose::Context,
    );
    document.blocks = vec![Block::Text {
        text: "retained canonical document".into(),
    }];
    input.parse[0].output.documents.push(DocumentRow {
        offset: 1,
        end: 2,
        document: document.clone(),
    });
    let link = Link {
        kind: LinkKind::Spawn,
        parent: sid("receipt-session"),
        child: sid("receipt-child"),
        at: None,
        call_id: None,
        evidence: "explicit-parent".into(),
    };
    input.parse[0].output.links.push(link.clone());
    let wire = serde_json::to_vec(&input.parse[0].output).unwrap();
    let mut confirmed = commit_event_bytes(&w, input, &wire);
    confirmed.parse[0].output.session_receipts.insert(0, 130);
    confirmed.parse[0].output.document_receipts.insert(0, 140);
    confirmed.parse[0].output.link_receipts.insert(0, 150);
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let control = crate::wire_shapes::ReadingReceipt {
        sequence: 999,
        id: Some("progress-only".into()),
        observation: None,
        observed_at_ms: None,
    };
    let apply = |conn: &mut rusqlite::Connection, batch: &mut Batch| {
        let tx = conn.transaction().unwrap();
        resolve::prepare_index(&tx, batch, 1, false).unwrap();
        rows::write_confirmed_rows(&tx, batch, &batch.parse[0], 1, 2000, None, Some(&control))
            .unwrap();
        tx.commit().unwrap();
    };
    apply(&mut conn, &mut confirmed);
    let mut contributor = batch("later-fact-source", 0, 0, Vec::new(), &["receipt-session"]);
    contributor.parse[0].output.sessions[0].offer_cwd("/workspace/second");
    contributor.parse[0].output.session_receipts.insert(0, 160);
    contributor.parse[0].output.links.push(link.clone());
    contributor.parse[0].output.link_receipts.insert(0, 170);
    apply(&mut conn, &mut contributor);
    let reads = ReadPool::open(&w.archive).unwrap();
    let assert_receipts = || {
        let reader = reads.reader().unwrap();
        assert_eq!(
            reader.session_commit(&sid("receipt-session")).unwrap(),
            Some(160)
        );
        assert_eq!(reader.document_commit(&document.id).unwrap(), Some(140));
        assert_eq!(reader.link_commit(&link).unwrap(), Some(170));
    };
    assert_receipts();
    let source = crate::archive::open_write(&w.archive.archive_db(), 1).unwrap();
    source
        .execute("DELETE FROM extent WHERE input='fact-source'", [])
        .unwrap();
    assert_receipts();
    // Rebuilding the projection uses original fact admissions, not its newer
    // cache transaction or the ParseProgress control's admission.
    apply(&mut conn, &mut confirmed);
    assert_receipts();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .session_commit(&sid("absent"))
            .unwrap(),
        None
    );
    // A historical fold with one unproven contributor cannot borrow the
    // admission of its other contributor or the current cache transaction.
    conn.execute("UPDATE session_record SET admitted=NULL WHERE input_pk=(SELECT pk FROM input_ref WHERE id='later-fact-source')",[]).unwrap();
    conn.execute("UPDATE link_source SET admitted=NULL WHERE input_pk=(SELECT pk FROM input_ref WHERE id='later-fact-source')",[]).unwrap();
    conn.execute("UPDATE document_source SET admitted=NULL", [])
        .unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader.session_commit(&sid("receipt-session")).unwrap(),
        None
    );
    assert_eq!(reader.link_commit(&link).unwrap(), None);
    assert_eq!(reader.document_commit(&document.id).unwrap(), None);
    for invalid in [0, u64::MAX] {
        let mut malformed = confirmed.clone();
        malformed.parse[0].output.link_receipts.insert(0, invalid);
        assert!(
            malformed.parse[0]
                .prepared
                .validate(&malformed.parse[0].output)
                .is_err()
        );
    }
}

#[test]
fn record_visitors_stop_before_decoding_later_session_payloads() {
    use std::ops::ControlFlow;
    let w = test_archive();
    let mut input = batch("visitor-source", 0, 0, Vec::new(), &["visitor-session"]);
    for ordinal in 0..2 {
        input.parse[0].output.events.push(Event {
            session: Some(sid("visitor-session")),
            kind: format!("event-{ordinal}"),
            at: None,
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((ordinal, ordinal + 1)),
            data: serde_json::json!({"ordinal":ordinal}),
        });
        input.parse[0].output.links.push(Link {
            kind: LinkKind::Spawn,
            parent: sid("visitor-session"),
            child: sid(&format!("child-{ordinal}")),
            at: None,
            call_id: None,
            evidence: "explicit-child".into(),
        });
    }
    let wire = serde_json::to_vec(&input.parse[0].output).unwrap();
    commit_event_bytes(&w, input, &wire);
    let conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    conn.execute("UPDATE event SET fields=x'00' WHERE pk=(SELECT pk FROM event ORDER BY seq,id LIMIT 1 OFFSET 1)",[]).unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    let mut events = 0;
    reader
        .visit_event_records(&sid("visitor-session"), |_| {
            events += 1;
            ControlFlow::Break(())
        })
        .unwrap();
    assert_eq!(events, 1);
    assert!(reader.event_records(&sid("visitor-session")).is_err());
    let mut links = 0;
    reader
        .visit_links(&sid("visitor-session"), |_| {
            links += 1;
            ControlFlow::Break(())
        })
        .unwrap();
    assert_eq!(links, 1);
    assert_eq!(reader.links(&sid("visitor-session")).unwrap().len(), 2);
}
