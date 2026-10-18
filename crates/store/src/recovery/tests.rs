use super::*;
use crate::Writer;
use std::sync::atomic::{AtomicBool, Ordering};
fn fixture() -> (tempfile::TempDir, PathBuf, u64) {
    crate::tests::killed_wal_fixture("stale-generation")
}
#[test]
fn recovered_copy_leaves_source_intact_and_keeps_recovery_record() {
    let (tmp, source, expected) = fixture();
    let cancel = AtomicBool::new(false);
    let original = inventory(&source, &cancel).unwrap();
    assert!(matches!(
        Writer::open(&source),
        Err(Error::RecoveryRequired { .. })
    ));
    let destination = tmp.path().join("recovered");
    let report = recover_copy(&source, &destination, &cancel).unwrap();
    assert_eq!(inventory(&source, &cancel).unwrap(), original);
    for entry in &original {
        let target = destination.join(&entry.path);
        assert_ne!(
            fs::metadata(source.join(&entry.path)).unwrap().ino(),
            fs::metadata(target).unwrap().ino()
        );
    }
    let manifest = fs::read(
        report
            .receipt
            .parent()
            .unwrap()
            .join("source-manifest.json"),
    )
    .unwrap();
    assert_eq!(
        BlobHash::of(&manifest),
        report.recovery.source_manifest_blake3
    );
    let mut writer = Writer::open(&destination).unwrap();
    assert_eq!(writer.status().unwrap().write_counter, expected + 1);
    assert_eq!(
        writer.status().unwrap().recovery.as_ref(),
        Some(&report.recovery)
    );
    writer
        .put_file_blob(b"ordinary post-recovery write")
        .unwrap();
    drop(writer);
    for suffix in ["", "-wal", "-shm"] {
        let path = destination.join(format!("index.db{suffix}"));
        if path.exists() {
            fs::remove_file(path).unwrap();
        }
    }
    let writer = Writer::open(&destination).unwrap();
    assert_eq!(
        writer.status().unwrap().recovery.as_ref(),
        Some(&report.recovery)
    );
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .session_catalog(65536)
            .unwrap()
            .recovery()
            .unwrap(),
        Some(report.recovery)
    );
    drop(writer);
    assert!(recover_copy(&source, &destination, &cancel).is_err());
    assert_eq!(inventory(&source, &cancel).unwrap(), original);
}
#[test]
fn recovery_refuses_damaged_wal_busy_overlap_and_symlinks() {
    let (tmp, source, _) = fixture();
    let cancel = AtomicBool::new(false);
    let wal = source.join("archive.db-wal");
    let original = fs::read(&wal).unwrap();
    let size = u32::from_be_bytes(original[8..12].try_into().unwrap()) as usize;
    let frame = size + 24;
    let first = original[16..24].to_vec();
    let boundary = (32..original.len())
        .step_by(frame)
        .find(|o| original[*o + 8..*o + 16] != first)
        .unwrap();
    for mutation in [
        "partial",
        "mixed",
        "current-checksum",
        "previous-checksum",
        "no-tail",
    ] {
        let mut data = original.clone();
        match mutation {
            "partial" => {
                data.pop();
            }
            "mixed" => {
                data[boundary + frame + 12] ^= 1;
            }
            "current-checksum" => {
                data[32 + 24 + 30] ^= 1;
            }
            "previous-checksum" => {
                data[boundary + frame + 24 + 30] ^= 1;
            }
            "no-tail" => {
                data.truncate(boundary);
            }
            _ => unreachable!(),
        }
        fs::write(&wal, &data).unwrap();
        let before = inventory(&source, &cancel).unwrap();
        let destination = tmp.path().join(mutation);
        assert!(
            recover_copy(&source, &destination, &cancel).is_err(),
            "{mutation}"
        );
        assert!(!destination.exists());
        assert_eq!(inventory(&source, &cancel).unwrap(), before);
    }
    fs::write(&wal, &original).unwrap();
    let held = crate::startup_lock(&source, false, false).unwrap();
    assert!(matches!(
        recover_copy(&source, &tmp.path().join("busy"), &cancel),
        Err(Error::WriterHeld { .. })
    ));
    drop(held);
    assert!(recover_copy(&source, &source.join("overlap"), &cancel).is_err());
    std::os::unix::fs::symlink(&wal, source.join("unsafe")).unwrap();
    assert!(recover_copy(&source, &tmp.path().join("unsafe"), &cancel).is_err());
}
#[test]
fn cancellation_before_publish_leaves_no_destination() {
    let (tmp, source, _) = fixture();
    for point in [
        Boundary::Copied,
        Boundary::Verified,
        Boundary::BeforePublish,
        Boundary::Published,
    ] {
        let cancel = AtomicBool::new(false);
        let destination = tmp.path().join(format!("at-{point:?}"));
        let mut observe = |at| {
            if at == point {
                cancel.store(true, Ordering::Release);
            }
            Ok(())
        };
        let result = recover_inner(&source, &destination, &cancel, &mut observe);
        if point == Boundary::Published {
            assert!(result.is_ok());
            assert!(
                ReadHandle::open(&destination)
                    .unwrap()
                    .status()
                    .unwrap()
                    .recovery
                    .is_some()
            );
        } else {
            assert!(result.is_err());
            assert!(!destination.exists());
        }
    }
    let cancel = AtomicBool::new(false);
    let destination = tmp.path().join("racing-destination");
    let mut race = |at| {
        if at == Boundary::BeforePublish {
            fs::create_dir(&destination).unwrap();
            fs::write(destination.join("sentinel"), b"unrelated").unwrap();
        }
        Ok(())
    };
    assert!(recover_inner(&source, &destination, &cancel, &mut race).is_err());
    assert_eq!(
        fs::read(destination.join("sentinel")).unwrap(),
        b"unrelated"
    );
}
#[test]
fn v1_migration_backs_up_files_without_recovery_record() {
    let (_tmp, source, expected) = crate::tests::killed_wal_fixture("legacy-normal");
    let cancel = AtomicBool::new(false);
    let before = inventory(&source, &cancel).unwrap();
    assert!(ReadHandle::open(&source).is_err());
    let writer = Writer::open(&source).unwrap();
    assert_eq!(writer.status().unwrap().archive_version, 4);
    assert_eq!(writer.status().unwrap().write_counter, expected + 1);
    assert!(writer.status().unwrap().recovery.is_none());
    drop(writer);
    let backups = fs::read_dir(source.join("migrations"))
        .unwrap()
        .collect::<std::io::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(backups.len(), 1);
    let backup = backups[0].path();
    for entry in before
        .iter()
        .filter(|e| e.path.ends_with(".db") || e.path.ends_with("-wal") || e.path.ends_with("-shm"))
    {
        assert_eq!(
            BlobHash::of(&fs::read(backup.join(&entry.path)).unwrap()),
            entry.blake3
        );
    }
}

#[test]
#[ignore = "child process for the kill test"]
fn crash_child() {
    let source = PathBuf::from(std::env::var_os("POLYLOGUE_RECOVERY_SOURCE").unwrap());
    let destination = PathBuf::from(std::env::var_os("POLYLOGUE_RECOVERY_DEST").unwrap());
    let point = std::env::var("POLYLOGUE_RECOVERY_BOUNDARY").unwrap();
    let ack = PathBuf::from(std::env::var_os("POLYLOGUE_RECOVERY_ACK").unwrap());
    recover_inner(&source, &destination, &AtomicBool::new(false), &mut |at| {
        if format!("{at:?}") == point {
            fs::write(&ack, b"boundary reached").unwrap();
            loop {
                std::thread::park();
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn killed_recovery_publishes_all_or_nothing() {
    let (tmp, source, _) = fixture();
    let cancel = AtomicBool::new(false);
    let before = inventory(&source, &cancel).unwrap();
    for point in [
        Boundary::Copied,
        Boundary::Verified,
        Boundary::BeforePublish,
        Boundary::Published,
    ] {
        let destination = tmp.path().join(format!("crash-{point:?}"));
        let ack = tmp.path().join(format!("ack-{point:?}"));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "recovery::tests::crash_child",
                "--ignored",
                "--nocapture",
            ])
            .env("POLYLOGUE_RECOVERY_SOURCE", &source)
            .env("POLYLOGUE_RECOVERY_DEST", &destination)
            .env("POLYLOGUE_RECOVERY_BOUNDARY", format!("{point:?}"))
            .env("POLYLOGUE_RECOVERY_ACK", &ack)
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !ack.exists() {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("child exited before the boundary: {status}");
            }
            if std::time::Instant::now() > deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("child did not reach the boundary in time");
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        if point == Boundary::Published {
            assert!(
                ReadHandle::open(&destination)
                    .unwrap()
                    .status()
                    .unwrap()
                    .recovery
                    .is_some()
            );
        } else {
            assert!(!destination.exists());
        }
        assert_eq!(inventory(&source, &cancel).unwrap(), before);
    }
}

#[test]
fn newer_format_and_missing_shm_refuse_without_changes() {
    let (tmp, source, _) = crate::tests::killed_wal_fixture("stale-newer");
    let cancel = AtomicBool::new(false);
    let before = inventory(&source, &cancel).unwrap();
    assert!(matches!(
        recover_copy(&source, &tmp.path().join("future"), &cancel),
        Err(Error::NewerSchema { found: 99, .. })
    ));
    assert_eq!(inventory(&source, &cancel).unwrap(), before);
    fs::remove_file(source.join("archive.db-shm")).unwrap();
    let before = inventory(&source, &cancel).unwrap();
    assert!(recover_copy(&source, &tmp.path().join("orphan"), &cancel).is_err());
    assert_eq!(inventory(&source, &cancel).unwrap(), before);
}

#[test]
fn source_change_or_corrupt_blob_refuses_and_leaves_source() {
    let (tmp, source, _) = fixture();
    let cancel = AtomicBool::new(false);
    let destination = tmp.path().join("changing");
    assert!(
        recover_inner(&source, &destination, &cancel, &mut |at| {
            if at == Boundary::Copied {
                fs::write(source.join("changed"), b"external write").unwrap();
            }
            Ok(())
        })
        .is_err()
    );
    assert!(!destination.exists());
    let blob = inventory(&source, &cancel)
        .unwrap()
        .into_iter()
        .find(|e| e.path.ends_with(".zst"))
        .unwrap();
    let path = source.join(&blob.path);
    let mut bytes = fs::read(&path).unwrap();
    bytes[12] ^= 1;
    fs::write(&path, bytes).unwrap();
    let before = inventory(&source, &cancel).unwrap();
    assert!(recover_copy(&source, &tmp.path().join("corrupt"), &cancel).is_err());
    assert_eq!(inventory(&source, &cancel).unwrap(), before);
    assert!(!tmp.path().join("corrupt").exists());
}

#[test]
fn empty_blob_still_needs_a_valid_file() {
    let (tmp, source, _) = crate::tests::killed_wal_fixture("stale-empty");
    let cancel = AtomicBool::new(false);
    let hash = BlobHash::of(b"");
    let hex = hash.to_string();
    let file = source
        .join("blobs")
        .join(&hex[..2])
        .join(format!("{hex}.zst"));
    let original = fs::read(&file).unwrap();
    assert!(
        !original.is_empty(),
        "an empty blob still has an encoded frame"
    );
    let report = recover_copy(&source, &tmp.path().join("valid-empty"), &cancel).unwrap();
    assert_eq!(report.checked_blobs, 4);
    let mut outcomes = vec![];
    for mutation in ["missing", "truncated", "damaged-frame"] {
        fs::write(&file, &original).unwrap();
        match mutation {
            "missing" => fs::remove_file(&file).unwrap(),
            "truncated" => fs::write(&file, b"").unwrap(),
            "damaged-frame" => {
                let mut damaged = original.clone();
                damaged[0] ^= 1; // Keep the seek table, break the zstd frame.
                fs::write(&file, damaged).unwrap();
            }
            _ => unreachable!(),
        }
        let before = inventory(&source, &cancel).unwrap();
        let destination = tmp.path().join(mutation);
        let refused = recover_copy(&source, &destination, &cancel).is_err();
        outcomes.push((mutation, refused));
        assert_eq!(inventory(&source, &cancel).unwrap(), before);
        if refused {
            assert!(!destination.exists());
        }
    }
    assert!(outcomes.iter().all(|(_, refused)| *refused), "{outcomes:?}");
}

#[test]
fn fifo_writer_lock_refuses_instead_of_blocking() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("unsafe-source");
    fs::create_dir(&source).unwrap();
    assert!(
        std::process::Command::new("mkfifo")
            .arg(source.join("writer.lock"))
            .status()
            .unwrap()
            .success()
    );
    assert!(
        recover_copy(
            &source,
            &tmp.path().join("destination"),
            &AtomicBool::new(false)
        )
        .is_err()
    );
}
