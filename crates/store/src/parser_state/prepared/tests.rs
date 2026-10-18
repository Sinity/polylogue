use super::*;
use crate::parser_state::tests::{basis, fixture as archive};
use crate::tests::reopen_after_drop;
fn budget() -> Budget {
    Budget {
        cells: 1,
        encoded_bytes: 4096,
    }
}
fn progress() -> Progress {
    Progress {
        inspected_upto: 100,
        parsed_upto: 100,
        phase: Phase::Done,
        finalization_pending: false,
        counts: Accounting {
            framed: 1,
            used: 1,
            ..Accounting::default()
        },
    }
}
fn change(key: &str, value: &str) -> CellChange {
    CellChange {
        namespace: "provider".into(),
        key: key.as_bytes().to_vec(),
        value: Some(value.as_bytes().to_vec()),
    }
}
fn fixture() -> (tempfile::TempDir, std::path::PathBuf, crate::Writer, Head) {
    let (dir, root, mut writer, input) = archive();
    let head = writer.begin_parser(&basis(input, 100)).unwrap();
    (dir, root, writer, head)
}
#[test]
fn pending_work_blocks_new_snapshots_and_keeps_later_growth() {
    let (_dir, root, mut writer, head) = fixture();
    let old = writer.parser_snapshot(&head.token).unwrap();
    let mut work = writer.prepare_parser(&head, progress()).unwrap();
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(StoreError::ParserWorkPending)
    ));
    assert!(matches!(
        writer.prepare_parser(&head, progress()),
        Err(StoreError::ParserWorkPending)
    ));
    assert!(matches!(
        write_step(
            &mut writer._index,
            &root.join("index.db"),
            &head,
            progress(),
            &[],
            budget(),
            |_| Ok(())
        ),
        Err(StoreError::ParserWorkPending)
    ));
    work = writer
        .stage_parser(&work.token, &[change("a", "one")], budget())
        .unwrap();
    work = writer
        .stage_parser(&work.token, &[change("b", "two")], budget())
        .unwrap();
    assert!(matches!(
        writer.publish_parser(&work.token),
        Err(StoreError::ParserWorkPending)
    ));
    work = writer.seal_parser(&work.token).unwrap();
    let stale = work.token.clone();
    work = writer.apply_parser(&work.token, budget()).unwrap();
    assert_eq!(work.token.applied, 1);
    assert!(matches!(
        writer.apply_parser(&stale, budget()),
        Err(StoreError::StaleParserState)
    ));
    assert_eq!(old.cell("provider", b"a", 100).unwrap(), None);
    assert_eq!(
        writer
            .parser_head(head.token.input_pk, 1)
            .unwrap()
            .unwrap()
            .parsed_upto,
        0
    );
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(StoreError::ParserWorkPending)
    ));
    writer
        .begin_parser(&basis(head.token.input_pk, 200))
        .unwrap();
    work = writer.apply_parser(&work.token, budget()).unwrap();
    let published = writer.publish_parser(&work.token).unwrap();
    assert_eq!(
        (
            published.parsed_upto,
            published.acquired_upto,
            published.status.as_str()
        ),
        (100, 200, "parsing")
    );
    assert_eq!(published.counts.framed, 1);
    assert!(matches!(
        writer.publish_parser(&work.token),
        Err(StoreError::StaleParserState)
    ));
    let new = writer.parser_snapshot(&published.token).unwrap();
    assert_eq!(new.cell("provider", b"a", 100).unwrap().unwrap(), b"one");
    assert_eq!(
        new.neighbor("provider", b"a", true, 100)
            .unwrap()
            .unwrap()
            .0,
        b"b"
    );
    assert_eq!(old.cell("provider", b"b", 100).unwrap(), None);
}
#[test]
fn restart_resumes_staging_and_partial_apply() {
    let (_dir, root, mut writer, head) = fixture();
    let work = writer.prepare_parser(&head, progress()).unwrap();
    let work = writer
        .stage_parser(&work.token, &[change("a", "one")], budget())
        .unwrap();
    drop(writer);
    let mut writer = reopen_after_drop(&root);
    let resumed = writer.parser_work(&head.token).unwrap().unwrap();
    assert_eq!(resumed.token, work.token);
    let work = writer
        .stage_parser(&resumed.token, &[change("b", "two")], budget())
        .unwrap();
    let work = writer.seal_parser(&work.token).unwrap();
    let work = writer.apply_parser(&work.token, budget()).unwrap();
    drop(writer);
    let mut writer = reopen_after_drop(&root);
    let resumed = writer.parser_work(&head.token).unwrap().unwrap();
    assert_eq!(resumed.token, work.token);
    assert_eq!(resumed.token.applied, 1);
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(StoreError::ParserWorkPending)
    ));
    let work = writer.apply_parser(&resumed.token, budget()).unwrap();
    let published = writer.publish_parser(&work.token).unwrap();
    assert_eq!(published.status, "done");
    assert_eq!(published.counts.used, 1);
    drop(writer);
    let writer = reopen_after_drop(&root);
    assert!(writer.parser_work(&published.token).unwrap().is_none());
    assert_eq!(
        writer
            .parser_snapshot(&published.token)
            .unwrap()
            .cell("provider", b"b", 100)
            .unwrap()
            .unwrap(),
        b"two"
    );
}
#[test]
fn new_digest_invalidates_work_and_cleanup_deletes_staged_cells() {
    let (_dir, root, mut writer, head) = fixture();
    let work = writer.prepare_parser(&head, progress()).unwrap();
    let work = writer
        .stage_parser(&work.token, &[change("a", "one")], budget())
        .unwrap();
    let work = writer.seal_parser(&work.token).unwrap();
    let work = writer.apply_parser(&work.token, budget()).unwrap();
    let mut new_basis = basis(head.token.input_pk, 100);
    new_basis.digest = [8; 32];
    let next = writer.begin_parser(&new_basis).unwrap();
    assert!(matches!(
        writer.publish_parser(&work.token),
        Err(StoreError::StaleParserState)
    ));
    assert_eq!(
        writer
            .parser_snapshot(&next.token)
            .unwrap()
            .cell("provider", b"a", 100)
            .unwrap(),
        None
    );
    let gc = collect_retired(
        &mut writer._index,
        &root.join("index.db"),
        &next.token,
        budget(),
    )
    .unwrap();
    assert_eq!(gc.deleted, 1);
    assert!(!gc.remaining);
    let new_work = writer.prepare_parser(&next, progress()).unwrap();
    assert_ne!(new_work.token.work_id, work.token.work_id);
    assert!(matches!(
        writer.seal_parser(&work.token),
        Err(StoreError::StaleParserState)
    ));
}
#[test]
fn rejected_stage_or_apply_leaves_work_unchanged() {
    let (_dir, _root, mut writer, head) = fixture();
    let work = writer.prepare_parser(&head, progress()).unwrap();
    assert!(matches!(
        writer.stage_parser(
            &work.token,
            &[change("a", "one")],
            Budget {
                cells: 1,
                encoded_bytes: 3
            }
        ),
        Err(StoreError::ParserAdmission { .. })
    ));
    let reserved = CellChange {
        namespace: namespace(work.token.work_id),
        key: vec![],
        value: Some(vec![]),
    };
    assert!(matches!(
        writer.stage_parser(&work.token, &[reserved], budget()),
        Err(StoreError::Invariant(_))
    ));
    assert_eq!(
        writer
            .parser_work(&head.token)
            .unwrap()
            .unwrap()
            .token
            .staged,
        0
    );
    let work = writer
        .stage_parser(&work.token, &[change("a", "one")], budget())
        .unwrap();
    let work = writer.seal_parser(&work.token).unwrap();
    assert!(matches!(
        writer.stage_parser(&work.token, &[change("b", "two")], budget()),
        Err(StoreError::Invariant(_))
    ));
    assert!(matches!(
        writer.apply_parser(
            &work.token,
            Budget {
                cells: 1,
                encoded_bytes: 1
            }
        ),
        Err(StoreError::ParserAdmission { .. })
    ));
    assert_eq!(
        writer
            .parser_work(&head.token)
            .unwrap()
            .unwrap()
            .token
            .applied,
        0
    );
}

#[test]
fn unsealed_work_is_discarded_in_budgeted_steps() {
    let (_dir, root, mut writer, head) = fixture();
    let work = writer.prepare_parser(&head, progress()).unwrap();
    let work = writer
        .stage_parser(&work.token, &[change("a", "one")], budget())
        .unwrap();
    writer
        .stage_parser(&work.token, &[change("b", "two")], budget())
        .unwrap();
    drop(writer);
    let mut writer = reopen_after_drop(&root);
    let work = writer.parser_work(&head.token).unwrap().unwrap();
    let work = writer
        .discard_parser_preparation(&work.token, budget())
        .unwrap()
        .unwrap();
    assert_eq!(work.token.applied, 1);
    assert_eq!(work.phase, WorkPhase::Discarding);
    assert!(matches!(
        writer.seal_parser(&work.token),
        Err(StoreError::Invariant(_))
    ));
    drop(writer);
    let mut writer = reopen_after_drop(&root);
    let work = writer.parser_work(&head.token).unwrap().unwrap();
    assert!(
        writer
            .discard_parser_preparation(&work.token, budget())
            .unwrap()
            .is_none()
    );
    assert_eq!(
        writer
            .parser_snapshot(&head.token)
            .unwrap()
            .cell("provider", b"a", 100)
            .unwrap(),
        None
    );
    assert_eq!(
        writer
            .parser_head(head.token.input_pk, 1)
            .unwrap()
            .unwrap()
            .parsed_upto,
        0
    );
    let replacement = writer.prepare_parser(&head, progress()).unwrap();
    assert_ne!(replacement.token.work_id, work.token.work_id);
    assert!(matches!(
        writer.seal_parser(&work.token),
        Err(StoreError::StaleParserState)
    ));
}

#[test]
fn process_exit_before_publish_keeps_work_resumable() {
    let (_dir, root, mut writer, head) = fixture();
    let work = writer.prepare_parser(&head, progress()).unwrap();
    let work = writer
        .stage_parser(&work.token, &[change("a", "one")], budget())
        .unwrap();
    let work = writer
        .stage_parser(&work.token, &[change("b", "two")], budget())
        .unwrap();
    writer.seal_parser(&work.token).unwrap();
    drop(writer);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "parser_state::prepared::tests::prepared_crash_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_PREPARED_CRASH_ROOT", &root)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(73));
    let mut writer = crate::Writer::open(&root).unwrap();
    let resumed = writer.parser_work(&head.token).unwrap().unwrap();
    assert_eq!(resumed.token.applied, 1);
    assert_eq!(
        writer
            .parser_head(head.token.input_pk, 1)
            .unwrap()
            .unwrap()
            .parsed_upto,
        0
    );
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(StoreError::ParserWorkPending)
    ));
    let completed = writer.apply_parser(&resumed.token, budget()).unwrap();
    let head = writer.publish_parser(&completed.token).unwrap();
    assert_eq!(head.counts.framed, 1);
    assert_eq!(head.parsed_upto, 100);
}

#[test]
fn prepared_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_PREPARED_CRASH_ROOT") else {
        return;
    };
    let mut writer = crate::Writer::open(std::path::Path::new(&root)).unwrap();
    let input = writer
        ._index
        .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
        .unwrap();
    let head = writer.parser_head(input, 1).unwrap().unwrap();
    let work = writer.parser_work(&head.token).unwrap().unwrap();
    writer.apply_parser(&work.token, budget()).unwrap();
    std::process::exit(73);
}
