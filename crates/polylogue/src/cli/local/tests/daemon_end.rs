use super::super::DaemonEnd;
use polylogue_store::storage_failure::StorageFailure;

fn failure() -> StorageFailure {
    StorageFailure {
        path: "/home/op/archive/archive.db-wal".into(),
        message: "Input/output error (os error 5)".into(),
        errno: Some(5),
    }
}

fn idle(errors: &[&str]) -> polylogue_engine::RunSummary {
    let mut summary = polylogue_engine::RunSummary::default();
    summary.work.errors = errors.iter().map(|error| (*error).to_owned()).collect();
    summary
}

/// Exit statuses of docs/operations.md: 0 clean stop, 1 failure, 74 storage
/// failure, 78 refused archive.
#[test]
fn each_way_a_run_ends_has_its_exit_status() {
    let ends = [
        (Ok(None), DaemonEnd::Stopped, 0),
        (Ok(Some(idle(&[]))), DaemonEnd::Idle { complete: true }, 0),
        (
            Ok(Some(idle(&["read input: synthetic"]))),
            DaemonEnd::Idle { complete: false },
            1,
        ),
        (
            Err(std::io::Error::other("synthetic worker failure")),
            DaemonEnd::Failed,
            1,
        ),
        (
            Err(std::io::Error::other(
                crate::daemon::storage_stop::StorageStopped("refusing to start".into()),
            )),
            DaemonEnd::StorageFailedEarlier,
            74,
        ),
        (
            Err(std::io::Error::other(
                polylogue_store::StoreError::OtherFormat {
                    database: "archive.db",
                    found: 1,
                },
            )),
            DaemonEnd::ArchiveRefused,
            78,
        ),
    ];
    for (served, end, exit) in ends {
        let found = DaemonEnd::of(&served, None);
        assert_eq!(found, end);
        assert_eq!(found.exit_code(), exit);
    }
}

/// A failed sync stops writing for the rest of the boot, so it decides the
/// exit however the run ended: a stop signal or an idle end that arrives
/// with the failure must not read as a clean stop.
#[test]
fn a_recorded_storage_failure_decides_the_exit_however_the_run_ended() {
    let ways = [
        Ok(None),
        Ok(Some(idle(&[]))),
        Ok(Some(idle(&["read input: synthetic"]))),
        Err(std::io::Error::other("the register failed a disk sync")),
    ];
    for served in ways {
        let end = DaemonEnd::of(&served, Some(failure()));
        assert_eq!(end, DaemonEnd::StorageFailed(failure()));
        assert_eq!(end.exit_code(), 74);
    }
}
