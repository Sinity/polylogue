use crate::{Archive, Change, ReadPool, Writer};

#[test]
fn checkpoint_commits_with_its_cursor_and_mismatched_state_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut batch = crate::fixture::writer::step(
        "native-checkpoint",
        "claude-code",
        "/synthetic/checkpoint.jsonl",
        Default::default(),
        0,
        1,
    );
    let mut raw = batch.clone();
    raw.parse.clear();
    let stored = writer
        .store_bytes(archive.prepare_bytes(&b"r\n"[..]).unwrap(), raw)
        .unwrap();
    batch.change = Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    let state = polylogue_model::batch::ParserCheckpoint::new(
        "synthetic=1".into(),
        batch.length,
        Some(serde_json::json!({"open":"response"})),
    );
    batch.parse[0].checkpoint = Some(state.clone());
    writer.send(batch.clone()).unwrap();
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let pool = ReadPool::open(&archive).unwrap();
    let row = pool
        .reader()
        .unwrap()
        .parser_checkpoint("native-checkpoint", stored.chain, "synthetic=1")
        .unwrap()
        .unwrap();
    assert_eq!(row.checkpoint, Some(state.clone()));
    assert!(
        pool.reader()
            .unwrap()
            .parser_checkpoint("native-checkpoint", stored.chain, "synthetic=2")
            .unwrap()
            .unwrap()
            .checkpoint
            .is_none()
    );
    // The invalid checkpoint accompanies a distinct cursor and must not publish it.
    batch.parse[0].parsed_upto = 1;
    batch.parse[0].checkpoint = Some(state.clone());
    writer.send(batch).unwrap();
    let report = writer.flush().unwrap();
    assert!(!report.errors.is_empty());
    let row = pool
        .reader()
        .unwrap()
        .parser_checkpoint("native-checkpoint", stored.chain, "synthetic=1")
        .unwrap()
        .unwrap();
    assert_eq!(row.parsed_upto, 2);
    assert_eq!(row.checkpoint, Some(state));
    let conn = rusqlite::Connection::open(archive.index_db()).unwrap();
    conn.execute("UPDATE parse_state SET checkpoint=x'ff'", [])
        .unwrap();
    // A new reading does not even decode the prior reading's opaque envelope.
    assert!(
        pool.reader()
            .unwrap()
            .parser_checkpoint("native-checkpoint", stored.chain, "synthetic=2")
            .unwrap()
            .unwrap()
            .checkpoint
            .is_none()
    );
    assert!(
        pool.reader()
            .unwrap()
            .parser_checkpoint("native-checkpoint", stored.chain, "synthetic=1")
            .is_err()
    );
}

#[test]
fn unchanged_stopped_readings_wait_until_the_source_or_reading_changes() {
    use polylogue_model::batch::StepProgress;
    for progress in [
        StepProgress::AtEnd,
        StepProgress::Refused {
            reason: "unsupported synthetic record".into(),
        },
        StepProgress::Failed {
            error: "synthetic parse failure".into(),
        },
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let path = "/synthetic/stopped.jsonl";
        let mut batch =
            crate::fixture::writer::step("stopped", "claude-code", path, Default::default(), 0, 1);
        let mut raw = batch.clone();
        raw.parse.clear();
        let stored = writer
            .store_bytes(archive.prepare_bytes(&b"r\n"[..]).unwrap(), raw.clone())
            .unwrap();
        batch.change = Change::Reparse {
            chain: stored.chain,
        };
        batch.observation = None;
        batch.parse[0].parsed_upto = 0;
        batch.parse[0].counts = Default::default();
        batch.parse[0].progress = progress;
        batch.parse[0].checkpoint = None;
        writer.send(batch.clone()).unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
        let pool = ReadPool::open(&archive).unwrap();
        let waiting = || {
            let reader = pool.reader().unwrap();
            let progress = reader.input_progress("stopped").unwrap().unwrap();
            let known = reader
                .known_watched_input_at("m", path.as_bytes())
                .unwrap()
                .unwrap();
            assert_eq!(known.waiting, progress.waiting);
            progress.waiting
        };
        assert!(waiting());
        assert!(
            pool.reader()
                .unwrap()
                .parser_checkpoint("stopped", stored.chain, "synthetic=2")
                .unwrap()
                .unwrap()
                .checkpoint
                .is_none()
        );
        raw.change = Change::Appended {
            from: stored.length,
        };
        raw.length = stored.length + 2;
        writer
            .store_bytes(archive.prepare_bytes(&b"s\n"[..]).unwrap(), raw)
            .unwrap();
        assert!(!waiting(), "new bytes invalidate the old stopped length");
        batch.length = stored.length + 2;
        batch.parse[0].parsed_upto = batch.length;
        batch.parse[0].progress = StepProgress::AtEnd;
        batch.parse[0].reading_version = "synthetic=2".into();
        writer.send(batch).unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
        assert!(!waiting());
        assert_eq!(
            pool.reader()
                .unwrap()
                .parser_checkpoint("stopped", stored.chain, "synthetic=2")
                .unwrap()
                .unwrap()
                .parsed_upto,
            stored.length + 2
        );
    }
}

#[test]
fn rejected_context_and_stopped_attempt_roll_back_with_the_reading_cursor() {
    use polylogue_model::{
        batch::{ParserCheckpoint, StepProgress, StoppedAttempt},
        parser_context::{Context, Key, Value},
    };
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut batch = crate::fixture::writer::step(
        "context-rollback",
        "claude-code",
        "/synthetic/context.jsonl",
        Default::default(),
        0,
        1,
    );
    let mut raw = batch.clone();
    raw.parse.clear();
    let stored = writer
        .store_bytes(archive.prepare_bytes(&b"r\n"[..]).unwrap(), raw)
        .unwrap();
    batch.change = Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    let cursor = batch.length;
    let continuation = serde_json::json!({"_polylogue_context":{"version":1,"base":0,"through":cursor,"entries":[{"key":{"kind":"alias","namespace":"","uuid":"event"},"value":{"kind":"alias","target":null}}]}});
    batch.parse[0].checkpoint = Some(ParserCheckpoint::new(
        "synthetic=1".into(),
        cursor,
        Some(continuation),
    ));
    let attempt = StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"real stopped program"),
        parser_settings: None,
    };
    batch.parse[0].progress = StepProgress::Refused {
        reason: "original refusal".into(),
    };
    batch.parse[0].stopped_attempt = Some(attempt);
    writer.send(batch.clone()).unwrap();
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let index = rusqlite::Connection::open(archive.index_db()).unwrap();
    index.execute_batch("CREATE TRIGGER reject_context_checkpoint BEFORE INSERT ON parse_state BEGIN SELECT RAISE(ABORT,'reject entire reading'); END;").unwrap();
    batch.parse[0]
        .checkpoint
        .as_mut()
        .unwrap()
        .parser_state
        .as_mut()
        .unwrap()["_polylogue_context"]["entries"][0]["value"]["target"] =
        "claude-code:speculative".into();
    batch.parse[0].stopped_attempt = Some(StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"rejected program"),
        parser_settings: None,
    });
    writer.send(batch.clone()).unwrap();
    assert!(!writer.flush().unwrap().errors.is_empty());
    let reads = ReadPool::open(&archive).unwrap();
    let selected = reads
        .reader()
        .unwrap()
        .parser_checkpoint("context-rollback", stored.chain, "synthetic=1")
        .unwrap()
        .unwrap();
    assert_eq!(selected.stopped_attempt, Some(attempt));
    let mut overlay = crate::parser_context::Overlay::open(
        &archive,
        "context-rollback",
        Some(stored.chain),
        "synthetic=1",
        cursor,
    )
    .unwrap();
    assert!(matches!(
        overlay
            .get(&Key::Alias {
                namespace: "".into(),
                uuid: "event".into()
            })
            .unwrap(),
        Some(Value::Alias { target: None })
    ));
    drop(overlay);
    assert!(matches!(
        crate::parser_context::Overlay::open(
            &archive,
            "context-rollback",
            Some(stored.chain),
            "synthetic=1",
            u64::MAX,
        ),
        Err(crate::StoreError::Invalid(_))
    ));
    index
        .execute("UPDATE parser_context_state SET through=-1", [])
        .unwrap();
    assert!(matches!(
        crate::parser_context::Overlay::open(
            &archive,
            "context-rollback",
            Some(stored.chain),
            "synthetic=1",
            cursor,
        ),
        Err(crate::StoreError::Invalid(_))
    ));
    let sql_cursor = i64::try_from(cursor).unwrap();
    index
        .execute("UPDATE parser_context_state SET through=?1", [sql_cursor])
        .unwrap();
    index
        .execute("UPDATE parser_context SET cursor=-1", [])
        .unwrap();
    let mut overlay = crate::parser_context::Overlay::open(
        &archive,
        "context-rollback",
        Some(stored.chain),
        "synthetic=1",
        cursor,
    )
    .unwrap();
    assert!(
        overlay
            .get(&Key::Alias {
                namespace: "".into(),
                uuid: "event".into(),
            })
            .is_err()
    );
    drop(overlay);
    index
        .execute("UPDATE parser_context SET cursor=?1", [sql_cursor])
        .unwrap();
    index
        .execute_batch("DROP TRIGGER reject_context_checkpoint")
        .unwrap();
    batch.parse[0].progress = StepProgress::AtEnd;
    batch.parse[0].stopped_attempt = None;
    writer.send(batch).unwrap();
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .parser_checkpoint("context-rollback", stored.chain, "synthetic=1")
            .unwrap()
            .unwrap()
            .stopped_attempt,
        None
    );
}

#[test]
fn keyed_context_reuses_a_retained_message_version_without_changing_source_selection() {
    use polylogue_model::{
        batch::ParserCheckpoint,
        parser_context::{Context, Delta, Entry, Key, Reference, Value},
    };
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let first = crate::fixture::writer::message(
        "one",
        "copies",
        polylogue_model::Role::Assistant,
        0,
        vec![polylogue_model::Block::Text {
            text: "first recorded value".into(),
        }],
    );
    let mut later = first.clone();
    later.blocks = vec![polylogue_model::Block::Text {
        text: "later source copy".into(),
    }];
    let key = Key::Response {
        namespace: "embedded".into(),
        session: first.session.to_string(),
        key: "same-api-response".into(),
    };
    let output = polylogue_model::Output {
        sessions: vec![polylogue_model::Session::new(
            first.session.clone(),
            polylogue_model::SessionKind::Main,
        )],
        messages: vec![
            polylogue_model::MessageRow {
                offset: 0,
                end: 1,
                message: first.clone(),
            },
            polylogue_model::MessageRow {
                offset: 1,
                end: 2,
                message: later.clone(),
            },
        ],
        ..Default::default()
    };
    let mut batch = crate::fixture::writer::step(
        "context-copy",
        "claude-code",
        "/synthetic/copy.jsonl",
        output,
        0,
        2,
    );
    let mut raw = batch.clone();
    raw.parse.clear();
    let stored = writer
        .store_bytes(archive.prepare_bytes(&b"r\n\n"[..]).unwrap(), raw)
        .unwrap();
    batch.change = Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    let delta = Delta {
        version: 1,
        base: 0,
        through: batch.length,
        entries: vec![Entry {
            key: key.clone(),
            value: Reference::Response {
                message: later.id.clone(),
                version: later.content_hash().unwrap(),
                offset: 1,
                end: 2,
            },
        }],
    };
    let state = serde_json::json!({"_polylogue_context":delta});
    batch.parse[0].checkpoint = Some(ParserCheckpoint::new(
        "synthetic=1".into(),
        batch.length,
        Some(state),
    ));
    writer.send(batch).unwrap();
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let reads = ReadPool::open(&archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .message_with_projection(&first.id)
            .unwrap()
            .unwrap()
            .0
            .text(),
        "first recorded value"
    );
    writer.derive().unwrap();
    let mut overlay = crate::parser_context::Overlay::open(
        &archive,
        "context-copy",
        Some(stored.chain),
        "synthetic=1",
        3,
    )
    .unwrap();
    let Some(Value::Response { row }) = overlay.get(&key).unwrap() else {
        panic!("retained keyed response missing")
    };
    assert_eq!(row.message.text(), "later source copy");
    assert_eq!((row.offset, row.end), (1, 2));
    drop(overlay);
    // Retraction of the context releases the body that has no visible source.
    let index = rusqlite::Connection::open(archive.index_db()).unwrap();
    index
        .execute_batch(
            "BEGIN; DELETE FROM parser_context; DELETE FROM parser_context_state; COMMIT;",
        )
        .unwrap();
    writer.derive().unwrap();
    let hash = later.content_hash().unwrap();
    assert!(
        !index
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM message_version WHERE version_hash=?1)",
                [hash.0.as_slice()],
                |row| row.get::<_, bool>(0),
            )
            .unwrap()
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .message_with_projection(&first.id)
            .unwrap()
            .unwrap()
            .0
            .text(),
        "first recorded value"
    );
}
