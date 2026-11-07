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
