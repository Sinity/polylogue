use super::*;
use std::process::{Child, Command};
struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "owned child fixture, executed by killed_preparation_restarts_with_a_fresh_complete_view"]
fn parked_capture_child() {
    let source = PathBuf::from(std::env::var_os("POLYLOGUE_HISTORY_TURN_SOURCE").unwrap());
    let archive = PathBuf::from(std::env::var_os("POLYLOGUE_HISTORY_TURN_ARCHIVE").unwrap());
    let ready = PathBuf::from(std::env::var_os("POLYLOGUE_HISTORY_TURN_READY").unwrap());
    let writer = Writer::open(&archive).unwrap();
    writer
        .blob_capture()
        .unwrap()
        .capture_history_pages_in_turns(
            &source,
            None,
            &AtomicBool::new(false),
            turns::Budget {
                bytes: 65536,
                checkpoints: 2,
            },
            |progress| {
                if progress.provider_bytes_read >= 69632 {
                    std::fs::write(&ready, progress.provider_bytes_read.to_string()).unwrap();
                    // This owned fixture intentionally parks until the parent kills it.
                    loop {
                        std::thread::park();
                    }
                }
                Ok(())
            },
        )
        .unwrap();
    panic!("fixture should have parked before complete custody");
}

#[test]
fn killed_preparation_restarts_with_a_fresh_complete_view() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("provider.sqlite");
    let archive = temp.path().join("archive");
    let ready = temp.path().join("parked");
    let provider = Connection::open(&source).unwrap();
    provider.execute_batch("PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE left_field(value INTEGER);INSERT INTO left_field VALUES(10);CREATE TABLE padding(value BLOB);INSERT INTO padding VALUES(zeroblob(131072));CREATE TABLE right_field(value INTEGER);INSERT INTO right_field VALUES(10);").unwrap();
    let mut child = OwnedChild(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "history_pages::process_turn_tests::parked_capture_child",
                "--ignored",
                "--nocapture",
            ])
            .env("POLYLOGUE_HISTORY_TURN_SOURCE", &source)
            .env("POLYLOGUE_HISTORY_TURN_ARCHIVE", &archive)
            .env("POLYLOGUE_HISTORY_TURN_READY", &ready)
            .spawn()
            .unwrap(),
    );
    let until = std::time::Instant::now() + Duration::from_secs(15);
    while !ready.exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "child exited before its parked quantum"
        );
        assert!(
            std::time::Instant::now() < until,
            "child did not reach its owned barrier"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        std::fs::read_to_string(&ready)
            .unwrap()
            .parse::<u64>()
            .unwrap()
            >= 69632
    );
    provider
        .execute_batch(
            "BEGIN;UPDATE left_field SET value=20;UPDATE right_field SET value=20;COMMIT",
        )
        .unwrap();
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    let parts = std::fs::read_dir(archive.join("blobs/tmp"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect::<Vec<_>>();
    assert!(
        parts
            .iter()
            .any(|p| p.extension().is_some_and(|x| x == "part"))
    );
    // The encoder can still buffer its first frame, leaving an empty .part.
    // Existing file-layer .part names identify non-authoritative preparation;
    // the vanished SQLite view is never reconstructed from an integer offset.
    let mut writer = Writer::open(&archive).unwrap();
    assert_eq!(
        writer
            .archive
            .query_row("SELECT count(*) FROM input", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let cancel = AtomicBool::new(false);
    let fresh = writer
        .blob_capture()
        .unwrap()
        .capture_history_pages_in_turns(
            &source,
            None,
            &cancel,
            turns::Budget {
                bytes: 65536,
                checkpoints: 2,
            },
            |_| Ok(()),
        )
        .unwrap();
    let hash = fresh.hash();
    let retained = writer
        .retain_history_pages(
            fresh,
            &ImportObservation {
                id: "after-process-loss",
                machine: "synthetic",
                host: "synthetic",
                path: "copy:history",
                seen_at: 1,
            },
            131072,
        )
        .unwrap();
    let raw = writer
        .read_handle()
        .unwrap()
        .read_source_range(&retained.id, 1, hash, 0, retained.length)
        .unwrap();
    let selected = temp.path().join("selected.sqlite");
    std::fs::write(&selected, raw).unwrap();
    let selected = Connection::open(selected).unwrap();
    assert_eq!(
        selected
            .query_row(
                "SELECT (SELECT value FROM left_field),(SELECT value FROM right_field)",
                [],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
            )
            .unwrap(),
        (20, 20)
    );
    assert!(
        parts.iter().all(|p| p.exists()),
        "fresh acquisition must not repurpose earlier partial files"
    );
}
