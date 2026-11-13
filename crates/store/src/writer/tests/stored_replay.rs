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
                "retained",
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message(
                        "message",
                        "session",
                        Role::User,
                        "Retained source reading",
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
            .query("SELECT status,parsed_upto FROM parse_state", &[])
            .unwrap(),
        vec![serde_json::json!({"status":"done","parsed_upto":100})]
    );
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
                        "Retry retained source",
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
fn sparse_retained_readings_commit_their_diagnosis_even_during_stop() {
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
        let mut step = ParseStep::new(Default::default(), Default::default(), 0..0, false);
        step.reset = false;
        step.reading_version = "synthetic=1".into();
        step.progress = polylogue_model::batch::StepProgress::Refused {
            reason: format!("missing source {missing}"),
        };
        reading.parse = vec![step];
        let before = archive_state(&w);
        w.writer.stop_handle().request();
        w.store_reading(reading, None).unwrap();
        assert_eq!(archive_state(&w), before);
        let reads = ReadPool::open(&w.archive).unwrap();
        assert_eq!(
            reads
                .reader()
                .unwrap()
                .query("SELECT status,reason,parsed_upto FROM parse_state", &[])
                .unwrap(),
            vec![
                serde_json::json!({"status":"refused","reason":format!("missing source {missing}"),"parsed_upto":0})
            ]
        );
    }
}
