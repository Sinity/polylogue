use super::*;
use std::sync::{Arc, Barrier, mpsc::channel};

#[test]
fn overlapping_workers_read_after_the_preceding_commit() {
    let w = test_archive();
    let mut lock = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let blocked = lock
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let sender = w.writer.sender().unwrap();
    sender
        .send(batch(
            "first-route",
            0,
            0,
            vec![row(
                1,
                text_message("shared", "s", Role::Assistant, "partial", None),
            )],
            &["s"],
        ))
        .unwrap();
    let (waiting, observed) = channel();
    sender.0.resolver.observe_wait(waiting);
    let next = sender.clone();
    let thread = std::thread::spawn(move || {
        next.send(batch(
            "second-route",
            2,
            0,
            vec![row(
                1,
                text_message("shared", "s", Role::Assistant, "complete", Some("end_turn")),
            )],
            &["s"],
        ))
    });
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
    let mut lock = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let blocked = lock
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let start = Arc::new(Barrier::new(3));
    let (sent, queued) = channel();
    let mut threads = Vec::new();
    for (input, session, word) in [("one", "one", "first"), ("two", "two", "second")] {
        let sender = w.writer.sender().unwrap();
        let start = Arc::clone(&start);
        let sent = sent.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            let result = sender.send(batch(
                input,
                0,
                0,
                vec![row(1, text_message(input, session, Role::User, word, None))],
                &[session],
            ));
            sent.send(result).unwrap();
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
    let mut lock = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    let blocked = lock
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let start = Arc::new(Barrier::new(3));
    let (sent, queued) = channel();
    let mut threads = Vec::new();
    for (input, session, word) in [("one", "one", "shared text"), ("two", "two", "shared text")] {
        let sender = w.writer.sender().unwrap();
        let start = Arc::clone(&start);
        let sent = sent.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            let result = sender.send(batch(
                input,
                0,
                0,
                vec![row(1, text_message(input, session, Role::User, word, None))],
                &[session],
            ));
            sent.send(result).unwrap();
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
    crate::fault::arm("write.index_commit", "", crate::fault::Action::Fail, 0, 2);
    let results = sender
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: make(),
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
                batch: make(),
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
        session: Some(sid("s")),
        kind: "value".into(),
        at: None,
        after: None,
        range: Some((0, 1)),
        data: data.clone(),
    });
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
        .send(batch(
            "initial",
            0,
            0,
            vec![row(
                1,
                text_message("initial", "s", Role::User, "initial", None),
            )],
            &["s"],
        ))
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    for input in ["later-a", "later-b"] {
        w.writer
            .send(batch(
                input,
                0,
                0,
                vec![row(
                    1,
                    text_message("later", "s", Role::User, "later", None),
                )],
                &["s"],
            ))
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
        .send(batch(
            "observed",
            0,
            0,
            vec![row(
                1,
                text_message("same", "s", Role::User, "before", None),
            )],
            &["s"],
        ))
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
                    files: Vec::new(),
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
        .visit(|step| rows::write_confirmed_rows(&tx, &input, step, 41, 1, None).map(|_| ()))
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
    assert!(write_rows(&tx, &input, &input.parse[0], 1, 1, None).is_err());
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    rows::write_confirmed_rows(&tx, &input, &input.parse[0], 1, 1, None).unwrap();
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
        reading.visit(|step| write_rows(&tx, &batch, step, *chain, 1, None).map(|_| ()))?;
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

#[test]
fn imported_events_preserve_identity_claims_and_canonical_fields_across_hops() {
    let first = test_archive();
    let mut conn = crate::archive::open_write(&first.archive.index_db(), 1).unwrap();
    let tx = conn.transaction().unwrap();
    let mut input = batch("original", 0, 0, Vec::new(), &[]);
    let event = Event {
        session: None,
        kind: "input.report".into(),
        at: Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z"),
        after: None,
        range: Some((3, 17)),
        data: serde_json::json!({"remaining":42}),
    };
    input.parse[0].output.events.push(event.clone());
    write_test_rows(&tx, &input, &input.parse[0], 1, 1, None).unwrap();
    let id: String = tx
        .query_row("SELECT id FROM event", [], |row| row.get(0))
        .unwrap();
    tx.commit().unwrap();
    let reads = ReadPool::open(&first.archive).unwrap();
    let exported = reads.reader().unwrap().event_record(&id).unwrap().unwrap();
    assert_eq!(exported.event, event);
    let second = test_archive();
    let mut conn = crate::archive::open_write(&second.archive.index_db(), 1).unwrap();
    let mut portable = batch("bundle-local", 2, 0, Vec::new(), &[]);
    portable.parse[0].output.events.push(exported.event.clone());
    portable.parse[0]
        .output
        .imported_events
        .insert(0, exported.identity.clone());
    for counter in [1, 2] {
        let tx = conn.transaction().unwrap();
        write_test_rows(&tx, &portable, &portable.parse[0], 1, counter, None).unwrap();
        tx.commit().unwrap();
    }
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
            .any(|claim| claim.input == "bundle-local"
                && claim.source_start == Some(0)
                && claim.source_end == Some(100))
    );
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
    let tx = conn.transaction().unwrap();
    let mut competing = batch("preferred-local", 0, 0, Vec::new(), &[]);
    let mut newer = event.clone();
    newer.kind = "input.revised".into();
    newer.range = Some((500, 999));
    newer.data = serde_json::json!({"remaining":7});
    competing.parse[0].output.events.push(newer.clone());
    competing.parse[0]
        .output
        .imported_events
        .insert(0, reexported.identity);
    write_test_rows(&tx, &competing, &competing.parse[0], 1, 3, None).unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM event_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM event_value", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    tx.commit().unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .event_record(&id)
            .unwrap()
            .unwrap()
            .event,
        newer
    );
    let tx = conn.transaction().unwrap();
    let empty = batch("preferred-local", 0, 0, Vec::new(), &[]);
    write_test_rows(&tx, &empty, &empty.parse[0], 1, 4, None).unwrap();
    tx.commit().unwrap();
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
