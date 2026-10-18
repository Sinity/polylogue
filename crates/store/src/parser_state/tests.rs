use super::*;
use crate::Writer;
mod reused_reader;
pub(super) fn fixture() -> (tempfile::TempDir, std::path::PathBuf, Writer, i64) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let writer = Writer::open(&root).unwrap();
    writer
        ._index
        .execute("INSERT INTO input_ref(id) VALUES('synthetic-state')", [])
        .unwrap();
    let input = writer
        ._index
        .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
        .unwrap();
    (dir, root, writer, input)
}
pub(super) fn parser_input(input_pk: i64, end: u64) -> ParserInput<'static> {
    ParserInput {
        input_pk,
        chain: 1,
        parser_version: 1,
        digest: [31; 32],
        acquired_upto: end,
        companions_json: None,
        excisions_seen: 0,
    }
}
fn budget() -> Budget {
    Budget {
        cells: 16,
        encoded_bytes: 4096,
    }
}
fn change(key: &str, value: &str) -> CellChange {
    CellChange {
        namespace: "test".into(),
        key: key.as_bytes().to_vec(),
        value: Some(value.as_bytes().to_vec()),
    }
}
#[test]
fn source_rows_need_finalizing_state_and_roll_back_on_failure() {
    let (_dir, root, mut writer, input) = fixture();
    let initial = writer.begin_parser(&parser_input(input, 10)).unwrap();
    assert!(matches!(
        writer.publish_source_rows(&initial, b"synthetic producer proof", true),
        Err(StoreError::StaleParserState)
    ));
    let finalizing = write_step(
        &mut writer._index,
        &root.join("index.db"),
        &initial,
        Progress {
            inspected_upto: 10,
            parsed_upto: 10,
            phase: Phase::Finalizing,
            finalization_pending: true,
            counts: Accounting::default(),
        },
        &[],
        budget(),
        |_| Ok(()),
    )
    .unwrap();
    let pinned = writer.parser_snapshot(&finalizing.token).unwrap();
    let mut wrong = finalizing.clone();
    wrong.token.generation += 1;
    assert!(matches!(
        writer.publish_source_rows(&wrong, b"synthetic producer proof", true),
        Err(StoreError::StaleParserState)
    ));
    writer._index.execute_batch("CREATE TRIGGER refuse_done BEFORE UPDATE ON parse_state WHEN NEW.status='done' BEGIN SELECT RAISE(ABORT,'late failure'); END;").unwrap();
    assert!(matches!(
        writer.publish_source_rows(&finalizing, b"synthetic producer proof", true),
        Err(StoreError::Sqlite { .. })
    ));
    assert!(
        writer
            .parser_snapshot(&finalizing.token)
            .unwrap()
            .cell(SOURCE_ROWS_NAMESPACE, b"", 16384)
            .unwrap()
            .is_none()
    );
    assert!(
        writer
            .parser_head(input, 1)
            .unwrap()
            .unwrap()
            .finalization_pending
    );
    writer
        ._index
        .execute_batch("DROP TRIGGER refuse_done")
        .unwrap();
    let done = writer
        .publish_source_rows(&finalizing, b"synthetic producer proof", true)
        .unwrap();
    assert_eq!(done.token, finalizing.token);
    assert!(!done.finalization_pending);
    assert_eq!(done.status, Phase::Done);
    assert!(
        pinned
            .cell(SOURCE_ROWS_NAMESPACE, b"", 16384)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        writer
            .parser_snapshot(&done.token)
            .unwrap()
            .cell(SOURCE_ROWS_NAMESPACE, b"", 16384)
            .unwrap()
            .unwrap(),
        b"synthetic producer proof"
    );
    assert!(matches!(
        writer.publish_source_rows(&finalizing, b"replay", true),
        Err(StoreError::StaleParserState)
    ));
}
#[test]
fn source_rows_refused_after_growth_or_excision() {
    for removed in [false, true] {
        let (_dir, root, mut writer, input) = fixture();
        let initial = writer.begin_parser(&parser_input(input, 10)).unwrap();
        let finalizing = write_step(
            &mut writer._index,
            &root.join("index.db"),
            &initial,
            Progress {
                inspected_upto: 10,
                parsed_upto: 10,
                phase: Phase::Finalizing,
                finalization_pending: true,
                counts: Accounting::default(),
            },
            &[],
            budget(),
            |_| Ok(()),
        )
        .unwrap();
        if removed {
            writer
                ._index
                .execute("UPDATE input_ref SET excised=1 WHERE pk=?1", [input])
                .unwrap();
        } else {
            writer.begin_parser(&parser_input(input, 11)).unwrap();
        }
        assert!(matches!(
            writer.publish_source_rows(&finalizing, b"old position", true),
            Err(StoreError::StaleParserState)
        ));
        assert!(
            writer
                .parser_snapshot(&writer.parser_head(input, 1).unwrap().unwrap().token)
                .unwrap()
                .cell(SOURCE_ROWS_NAMESPACE, b"", 16384)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn failure_detail_is_bounded_and_tied_to_its_token() {
    let (_dir, _root, mut writer, input_pk) = fixture();
    let first = ParserInput {
        input_pk,
        chain: 1,
        parser_version: 1,
        digest: [9; 32],
        acquired_upto: 20,
        companions_json: None,
        excisions_seen: 0,
    };
    let head = writer.begin_parser(&first).unwrap();
    writer
        .record_parser_failure(&head.token, Some("retry required"), 4096)
        .unwrap();
    assert_eq!(
        writer.parser_failure(&head.token).unwrap().as_deref(),
        Some("retry required")
    );
    let after = writer.parser_head(input_pk, 1).unwrap().unwrap();
    assert_eq!(after.token, head.token);
    assert_eq!(after.parsed_upto, 0);
    assert_eq!(after.status, Phase::Parsing);
    assert!(matches!(
        writer.record_parser_failure(&head.token, Some(&"x".repeat(4097)), usize::MAX),
        Err(StoreError::ParserAdmission { .. })
    ));
    writer
        .record_parser_failure(&head.token, None, 4096)
        .unwrap();
    assert_eq!(writer.parser_failure(&head.token).unwrap(), None);
    let newer = writer
        .begin_parser(&ParserInput {
            digest: [8; 32],
            ..first
        })
        .unwrap();
    assert_ne!(newer.token, head.token);
    assert!(matches!(
        writer.record_parser_failure(&head.token, Some("stale"), 4096),
        Err(StoreError::StaleParserState)
    ));
    assert!(writer.parser_failure(&newer.token).unwrap().is_none());
}

#[test]
fn snapshot_holds_archive_lock_after_its_handles_close() {
    let (_dir, root, mut writer, input) = fixture();
    let head = writer.begin_parser(&parser_input(input, 100)).unwrap();
    let reader = writer.read_handle().unwrap();
    let snapshot = reader.parser_snapshot(&head.token).unwrap();
    drop(reader);
    drop(writer);
    assert!(matches!(
        Writer::open(&root),
        Err(StoreError::WriterHeld { .. })
    ));
    assert!(matches!(
        crate::ReadHandle::open(&root),
        Err(StoreError::WriterHeld { .. })
    ));
    assert_eq!(snapshot.cell("test", b"absent", 100).unwrap(), None);
    drop(snapshot);
    assert!(Writer::open(&root).is_ok());
}
fn progress(parsed: u64, inspected: u64, phase: Phase, pending: bool) -> Progress {
    Progress {
        parsed_upto: parsed,
        inspected_upto: inspected,
        phase,
        finalization_pending: pending,
        counts: Accounting::default(),
    }
}
#[test]
fn step_after_growth_returns_to_parsing_and_rejects_stale_heads() {
    let (_dir, root, mut writer, input) = fixture();
    let file = root.join("index.db");
    let head = writer.begin_parser(&parser_input(input, 10)).unwrap();
    let head = write_step(
        &mut writer._index,
        &file,
        &head,
        progress(5, 5, Phase::Parsing, true),
        &[change("a", "old")],
        budget(),
        |_| Ok(()),
    )
    .unwrap();
    let old = writer
        .read_handle()
        .unwrap()
        .parser_snapshot(&head.token)
        .unwrap();
    let extended = writer.begin_parser(&parser_input(input, 20)).unwrap();
    assert_eq!(extended.token, head.token);
    let next = write_step(
        &mut writer._index,
        &file,
        &head,
        progress(10, 10, Phase::Done, false),
        &[change("a", "new")],
        budget(),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(next.acquired_upto, 20);
    assert_eq!(next.status, Phase::Parsing);
    assert_eq!(old.cell("test", b"a", 10).unwrap().unwrap(), b"old");
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(StoreError::StaleParserState)
    ));
    assert!(matches!(
        write_step(
            &mut writer._index,
            &file,
            &head,
            progress(10, 10, Phase::Done, false),
            &[],
            budget(),
            |_| panic!("stale output")
        ),
        Err(StoreError::StaleParserState)
    ));
    let fresh = writer.parser_snapshot(&next.token).unwrap();
    assert_eq!(fresh.cell("test", b"a", 10).unwrap().unwrap(), b"new");
}
#[test]
fn reused_input_key_never_recreates_an_old_token() {
    let (_dir, root, mut writer, input) = fixture();
    let file = root.join("index.db");
    let old = writer.begin_parser(&parser_input(input, 10)).unwrap();
    let retired = retire(&mut writer._index, &file, &old.token).unwrap();
    remove_retired(&mut writer._index, &file, &retired.token).unwrap();
    writer
        ._index
        .execute("DELETE FROM input_ref WHERE pk=?1", [input])
        .unwrap();
    writer
        ._index
        .execute("INSERT INTO input_ref(id) VALUES('replacement')", [])
        .unwrap();
    let reused: i64 = writer
        ._index
        .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
        .unwrap();
    assert_eq!(input, reused);
    let next = writer.begin_parser(&parser_input(reused, 10)).unwrap();
    assert_ne!(next.token, old.token);
    assert!(matches!(
        writer.parser_snapshot(&old.token),
        Err(StoreError::StaleParserState)
    ));
}
#[test]
fn waiting_tail_and_retired_cleanup_respect_budgets() {
    let (_dir, root, mut writer, input) = fixture();
    let file = root.join("index.db");
    let head = writer.begin_parser(&parser_input(input, 10)).unwrap();
    let head = write_step(
        &mut writer._index,
        &file,
        &head,
        progress(5, 10, Phase::WaitingTail, true),
        &[change("a", &"x".repeat(100)), change("b", "second")],
        budget(),
        |_| Ok(()),
    )
    .unwrap();
    let view = writer.parser_snapshot(&head.token).unwrap();
    assert!(matches!(
        view.cell("test", b"a", 10),
        Err(StoreError::ParserAdmission { required: 100, .. })
    ));
    assert!(head.finalization_pending);
    assert_eq!(head.status, Phase::WaitingTail);
    assert_eq!(
        writer
            .begin_parser(&parser_input(input, 10))
            .unwrap()
            .status,
        Phase::WaitingTail
    );
    drop(view);
    let retired = retire(&mut writer._index, &file, &head.token).unwrap();
    assert!(remove_retired(&mut writer._index, &file, &retired.token).is_err());
    let first = collect_retired(
        &mut writer._index,
        &file,
        &retired.token,
        Budget {
            cells: 1,
            encoded_bytes: 200,
        },
    )
    .unwrap();
    assert_eq!(first.deleted, 1);
    assert!(first.remaining);
    let second = collect_retired(
        &mut writer._index,
        &file,
        &retired.token,
        Budget {
            cells: 1,
            encoded_bytes: 200,
        },
    )
    .unwrap();
    assert_eq!(second.deleted, 1);
    assert!(!second.remaining);
    remove_retired(&mut writer._index, &file, &retired.token).unwrap();
}
#[test]
fn process_exit_mid_step_rolls_back_and_releases_lock() {
    let (_dir, root, mut writer, input) = fixture();
    let file = root.join("index.db");
    let head = writer.begin_parser(&parser_input(input, 10)).unwrap();
    drop(writer);
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "parser_state::tests::crash_child", "--nocapture"])
        .env("PARSER_STATE_CRASH_ROOT", &root)
        .output()
        .unwrap();
    assert_eq!(child.status.code(), Some(71), "{child:?}");
    let mut writer = Writer::open(&root).unwrap();
    let restored = writer.parser_head(input, 1).unwrap().unwrap();
    assert_eq!(restored.token, head.token);
    assert!(
        writer
            .parser_snapshot(&head.token)
            .unwrap()
            .cell("test", b"a", 10)
            .unwrap()
            .is_none()
    );
    let count: i64 = writer
        ._index
        .query_row("SELECT count(*) FROM unrecognized", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
    let mut p = progress(10, 10, Phase::Finalizing, true);
    p.counts = Accounting {
        framed: 1,
        unrecognized: 1,
        ..Accounting::default()
    };
    let next=write_step(&mut writer._index,&file,&head,p,&[change("a","new")],budget(), |tx| {
        crate::db(&file,||tx.execute("INSERT INTO unrecognized(input_pk,chain,format,what,count) VALUES(?1,1,'synthetic','record:test',1)",[input]))?; Ok(())
    }).unwrap();
    assert_eq!(next.token.step, 1);
    assert_eq!(next.counts.framed, 1);
    assert!(next.finalization_pending);
    let count: i64 = writer
        ._index
        .query_row("SELECT count(*) FROM unrecognized", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
}
#[test]
fn crash_child() {
    let Some(root) = std::env::var_os("PARSER_STATE_CRASH_ROOT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let file = root.join("index.db");
    let mut writer = Writer::open(&root).unwrap();
    let input = writer
        ._index
        .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
        .unwrap();
    let head = writer.parser_head(input, 1).unwrap().unwrap();
    let _ = write_step(
        &mut writer._index,
        &file,
        &head,
        progress(10, 10, Phase::Finalizing, true),
        &[change("a", "new")],
        budget(),
        |tx| {
            crate::db(&file, || {
                tx.execute("INSERT INTO unrecognized(input_pk,chain,format,what,count) VALUES(?1,1,'synthetic','record:test',1)",[input])
            })?;
            std::process::exit(71)
        },
    );
    panic!("crash callback must exit before commit");
}
