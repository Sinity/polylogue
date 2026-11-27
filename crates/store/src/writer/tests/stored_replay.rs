use super::*;

fn archive_state(w: &TestArchive) -> (i64, i64, i64, String) {
    let conn = crate::archive::open_read(&w.archive.archive_db()).unwrap();
    conn.query_row("SELECT (SELECT count(*) FROM input),(SELECT count(*) FROM revision),(SELECT count(*) FROM observation),(SELECT value FROM meta WHERE key='write_counter')",[],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).unwrap()
}

#[test]
fn queued_stored_reading_changes_only_the_index() {
    let w = test_archive();
    let reading = w
        .commit_source(
            batch(
                "kept",
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message(
                        "message",
                        "session",
                        Role::User,
                        "Kept source reading",
                        None,
                    ),
                )],
                &["session"],
            ),
            &SOURCE_RECORD,
        )
        .unwrap();
    let before = archive_state(&w);
    let stored = w.store_reading(reading, None).unwrap();
    assert!(stored.already_stored);
    assert_eq!(stored.bytes_written, 0);
    assert!(w.writer.flush_rows().unwrap().errors.is_empty());
    assert_eq!(archive_state(&w), before);
    let reads = ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader.query("SELECT id FROM message", &[]).unwrap(),
        vec![serde_json::json!({"id":"claude-code:message"})]
    );
    assert_eq!(
        reader
            .query("SELECT CASE WHEN state='failed' AND failure_kind='refused' THEN 'refused' WHEN state='failed' THEN 'failed' WHEN waiting>0 THEN 'waiting' ELSE 'done' END AS status,parsed_to AS parsed_upto FROM parse_state", &[])
            .unwrap(),
        vec![serde_json::json!({"status":"done","parsed_upto":100})]
    );
}

/// An older reading can arrive after both chains were captured but before
/// the newer chain's reading. Publish the older records until the newer
/// reading commits in their place.
#[test]
fn a_replayed_chain_keeps_records_until_the_newer_reading_commits() {
    let w = test_archive();
    let mut record = row(
        0,
        text_message("message", "session", Role::User, "Shared record", None),
    );
    record.end = 100;
    let older = batch("rewrite", Route::Live, 0, vec![record], &["session"]);
    let mut newer = older.clone();
    newer.change = Change::Rewritten;
    let older = w.commit_source(older, &SOURCE_RECORD).unwrap();
    // Pass through a separate chain so the archive stores the repeated
    // bytes as a new chain rather than observing its existing latest one.
    let mut middle = batch("rewrite", Route::Live, 0, Vec::new(), &[]);
    middle.change = Change::Rewritten;
    w.commit_source(middle, &[b'x'; 100]).unwrap();
    let newer = w.commit_source(newer, &SOURCE_RECORD).unwrap();
    let chain = |batch: &Batch| match batch.change {
        Change::Reparse { chain } => chain,
        _ => panic!("a stored reading names its chain"),
    };
    let reads = ReadPool::open(&w.archive).unwrap();
    let holders = || {
        assert!(w.writer.flush_rows().unwrap().errors.is_empty());
        reads
            .reader()
            .unwrap()
            .query("SELECT chain FROM message_parse ORDER BY chain", &[])
            .unwrap()
    };
    w.store_reading(older.clone(), None).unwrap();
    assert_eq!(holders(), vec![serde_json::json!({"chain": chain(&older)})]);

    w.store_reading(newer.clone(), None).unwrap();
    assert_eq!(holders(), vec![serde_json::json!({"chain": chain(&newer)})]);

    // Another replay of the older reading also leaves the record with the
    // newer chain, now genuinely present in the index.
    w.store_reading(older, None).unwrap();
    assert_eq!(holders(), vec![serde_json::json!({"chain": chain(&newer)})]);
}

#[cfg(feature = "faults")]
#[test]
fn stored_reading_index_failure_keeps_source_and_can_retry() {
    crate::fault::clear();
    let w = test_archive();
    let reading = w
        .commit_source(
            batch(
                "retry",
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message(
                        "message",
                        "session",
                        Role::User,
                        "Retry stored source",
                        None,
                    ),
                )],
                &["session"],
            ),
            &SOURCE_RECORD,
        )
        .unwrap();
    let before = archive_state(&w);
    crate::fault::arm("write.index_commit", "", crate::fault::Action::Fail, 0, 1);
    assert!(w.store_reading(reading.clone(), None).is_err());
    assert_eq!(archive_state(&w), before);
    let reads = ReadPool::open(&w.archive).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .query("SELECT id FROM message", &[])
            .unwrap()
            .is_empty()
    );
    w.store_reading(reading, None).unwrap();
    assert_eq!(archive_state(&w), before);
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .query("SELECT id FROM message", &[])
            .unwrap(),
        vec![serde_json::json!({"id":"claude-code:message"})]
    );
    crate::fault::clear();
}

#[test]
fn sparse_stored_readings_commit_their_diagnosis_during_stop() {
    for missing in ["prefix", "tail"] {
        let w = test_archive();
        let mut reading = w
            .commit_source(
                batch("partial", Route::Live, 0, Vec::new(), &[]),
                &SOURCE_RECORD,
            )
            .unwrap();
        {
            let conn = crate::archive::open_write(&w.archive.archive_db(), 1).unwrap();
            if missing == "prefix" {
                conn.execute(
                    "UPDATE extent SET offset=50,length=50,blob_offset=50 WHERE input='partial'",
                    [],
                )
                .unwrap();
            } else {
                conn.execute("UPDATE extent SET length=50 WHERE input='partial'", [])
                    .unwrap();
            }
        }
        let mut step = ParsePart::new(Default::default(), Default::default(), 0..0, false);
        step.reset = false;
        step.parse_version = "synthetic=1".into();
        step.failure = Some(polylogue_model::ChainFailure::refused(format!(
            "missing source {missing}"
        )));
        reading.parse = vec![step];
        let before = archive_state(&w);
        w.archive.stop().cancel();
        w.store_reading(reading, None).unwrap();
        assert_eq!(archive_state(&w), before);
        let reads = ReadPool::open(&w.archive.reopened()).unwrap();
        assert_eq!(
            reads
                .reader()
                .unwrap()
                .query("SELECT CASE WHEN state='failed' AND failure_kind='refused' THEN 'refused' WHEN state='failed' THEN 'failed' WHEN waiting>0 THEN 'waiting' ELSE 'done' END AS status,failure_detail AS reason,parsed_to AS parsed_upto FROM parse_state", &[])
                .unwrap(),
            vec![
                serde_json::json!({"status":"refused","reason":format!("missing source {missing}"),"parsed_upto":0})
            ]
        );
    }
}
