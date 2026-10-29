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
    assert_eq!(progress, ("no_parser_yet".into(), 100, String::new()));
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
