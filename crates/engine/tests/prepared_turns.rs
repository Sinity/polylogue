use polylogue_engine::converge::{PendingInstall, Recovery, Turn, recover_turn};
use polylogue_store::parser_state::{Accounting, Basis, Budget, CellChange, Head, Phase, Progress};
use polylogue_store::{Error, ImportObservation, InputFormat, Writer};

fn budget() -> Budget {
    Budget {
        cells: 1,
        encoded_bytes: 4096,
    }
}
fn fixture() -> (tempfile::TempDir, Writer, Head) {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(dir.path()).unwrap();
    let hash = writer.put_file_blob(&[b'x'; 128]).unwrap();
    let input = writer
        .store_immutable_input(
            hash,
            InputFormat::CodexRollout,
            &[],
            &ImportObservation {
                id: "test-import",
                machine: "test-machine",
                host: "test-host",
                path: "/synthetic/input.jsonl",
                seen_at: 0,
            },
            4096,
        )
        .unwrap();
    let head = writer
        .begin_parser(&Basis {
            input_pk: input.input_pk,
            chain: 1,
            parser_version: 1,
            digest: [7; 32],
            acquired_upto: 128,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    (dir, writer, head)
}
fn progress() -> Progress {
    Progress {
        inspected_upto: 100,
        parsed_upto: 100,
        phase: Phase::Finalizing,
        finalization_pending: true,
        counts: Accounting {
            framed: 1,
            used: 1,
            ..Accounting::default()
        },
    }
}
fn changes() -> Vec<CellChange> {
    (0u8..4)
        .map(|i| CellChange {
            namespace: "synthetic.provider".into(),
            key: vec![i],
            value: Some(vec![i; 64]),
        })
        .collect()
}

#[test]
fn provider_turns_allow_peer_reads_and_writes_and_eof_is_not_completion() {
    let (_dir, mut writer, head) = fixture();
    let pinned = writer.parser_snapshot(&head.token).unwrap();
    let reader = writer.read_handle().unwrap();
    let mut pending = PendingInstall::begin(&mut writer, &head, progress(), changes()).unwrap();
    let mut turns = 0;
    let published = loop {
        let result = pending.turn(&mut writer, budget()).unwrap();
        turns += 1;
        // Store peer operations.
        let blob = writer.put_file_blob(&[turns]).unwrap();
        assert_eq!(reader.read_blob_range(blob, 0, 1).unwrap(), [turns]);
        assert!(
            pinned
                .cell("synthetic.provider", &[0], 4096)
                .unwrap()
                .is_none()
        );
        match result {
            Turn::Yielded => {}
            Turn::Published(h) => break h,
        }
    };
    assert_eq!(turns, 10); // four staging, seal, four applying, publication.
    assert_eq!(published.parsed_upto, 100);
    assert_eq!(published.status, "finalizing");
    assert!(published.finalization_pending);
    let snapshot = writer.parser_snapshot(&published.token).unwrap();
    assert_eq!(
        snapshot
            .cell("synthetic.provider", &[3], 4096)
            .unwrap()
            .unwrap(),
        vec![3; 64]
    );
    assert!(matches!(
        pending.turn(&mut writer, budget()),
        Err(Error::StaleParserState)
    ));
    assert!(matches!(
        PendingInstall::begin(
            &mut writer,
            &published,
            Progress {
                phase: Phase::Done,
                finalization_pending: false,
                ..progress()
            },
            Vec::new()
        ),
        Err(Error::Invariant(_))
    ));
}

#[test]
fn unfinished_staging_discards_before_reparse_and_failed_admission_keeps_work() {
    let (dir, mut writer, head) = fixture();
    let mut pending = PendingInstall::begin(&mut writer, &head, progress(), changes()).unwrap();
    assert!(matches!(
        pending.turn(
            &mut writer,
            Budget {
                cells: 1,
                encoded_bytes: 1
            }
        ),
        Err(Error::ParserAdmission { .. })
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
    assert!(matches!(
        pending.turn(&mut writer, budget()).unwrap(),
        Turn::Yielded
    ));
    drop(pending);
    drop(writer);
    let mut writer = Writer::open(dir.path()).unwrap();
    loop {
        match recover_turn(&mut writer, &head.token, budget()).unwrap() {
            Recovery::Yielded => {}
            Recovery::Reparse => break,
            Recovery::Published(_) => panic!("unsealed delta published"),
        }
    }
    let snapshot = writer.parser_snapshot(&head.token).unwrap();
    assert!(
        snapshot
            .cell("synthetic.provider", &[0], 4096)
            .unwrap()
            .is_none()
    );
    assert_eq!(snapshot.head.parsed_upto, 0);
}

#[test]
fn crash_child() {
    let Ok(path) = std::env::var("POLYLOGUE_ENGINE_PREPARED_CRASH") else {
        return;
    };
    let mut writer = Writer::open(std::path::Path::new(&path)).unwrap();
    let head = writer.parser_head(1, 1).unwrap().unwrap();
    let mut pending = PendingInstall::begin(&mut writer, &head, progress(), changes()).unwrap();
    // Stage, seal, apply only the first member, then exit without destructors.
    // Parsing must never see that partial state.
    for _ in 0..6 {
        assert!(matches!(
            pending.turn(&mut writer, budget()).unwrap(),
            Turn::Yielded
        ));
    }
    std::process::exit(73);
}

#[test]
fn process_exit_during_apply_finishes_stored_members_without_reparse() {
    let (dir, writer, head) = fixture();
    drop(writer);
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child", "--nocapture"])
        .env("POLYLOGUE_ENGINE_PREPARED_CRASH", dir.path())
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(73),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut writer = Writer::open(dir.path()).unwrap();
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(Error::ParserWorkPending)
    ));
    let old = writer
        .parser_head(head.token.input_pk, head.token.chain)
        .unwrap()
        .unwrap();
    assert_eq!(old.parsed_upto, 0);
    let work = writer.parser_work(&head.token).unwrap().unwrap();
    assert_eq!(work.token.applied, 1);
    let published = loop {
        match recover_turn(&mut writer, &head.token, budget()).unwrap() {
            Recovery::Yielded => {}
            Recovery::Published(h) => break h,
            Recovery::Reparse => panic!("sealed delta was discarded"),
        }
    };
    assert_eq!(published.token.step, 1);
    assert_eq!(published.counts.framed, 1);
    assert!(published.finalization_pending);
    assert_ne!(published.status, "done");
    let snapshot = writer.parser_snapshot(&published.token).unwrap();
    for i in 0u8..4 {
        assert_eq!(
            snapshot
                .cell("synthetic.provider", &[i], 4096)
                .unwrap()
                .unwrap(),
            vec![i; 64]
        );
    }
}

#[path = "prepared_turns/batches.rs"]
mod batches;
