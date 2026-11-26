use super::*;
use serde_json::json;
use std::path::PathBuf;

impl PendingLines {
    pub(in crate::live_events) fn path(&self, spool: &Path, boot: &str) -> PathBuf {
        spool
            .join(self.route)
            .join(format!("{}.ndjson", self.file_name(boot)))
    }
}

#[test]
fn each_producer_keeps_failed_lines_and_their_original_bytes_until_retry() {
    for (route, stream) in [
        ("process", "process"),
        ("registry", "claude-registry"),
        ("registry", "codex-status"),
        ("git", "git"),
        ("registry", "host"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let spool = directory.path().join("spool");
        std::fs::write(&spool, b"blocked").unwrap();
        let status = SharedStatus::default();
        let mut pending = PendingLines::new(route, stream, Arc::clone(&status));
        let first = json!({"at_ns":"123", "event":"Status", "payload":{"status":"active"}});
        let second = json!({"at_ns":"456", "event":"Status", "payload":{"status":"idle"}});
        assert!(
            pending
                .append(&spool, "boot", std::slice::from_ref(&first))
                .is_err()
        );
        assert!(
            pending
                .append(&spool, "boot", std::slice::from_ref(&second))
                .is_err()
        );
        assert_eq!(status.locked().summary().lines, 2);
        assert!(status.locked().summary().error.unwrap().contains(stream));
        let expected = pending.bytes.clone();
        std::fs::remove_file(&spool).unwrap();
        pending.append(&spool, "boot", &[]).unwrap();
        pending.append(&spool, "boot", &[]).unwrap();
        let actual = std::fs::read(pending.path(&spool, "boot")).unwrap();
        assert_eq!(actual, expected);
        let lines: Vec<Value> = serde_json::Deserializer::from_slice(&actual)
            .into_iter()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(lines, [first, second]);
        assert_eq!(
            status.locked().summary(),
            PendingSummary {
                lines: 0,
                error: None
            }
        );
    }
}

#[test]
fn a_partial_registry_line_is_not_interleaved_with_another_producer() {
    // Shorter than the one pending line, so the write stops inside it.
    const PREFIX_BYTES: usize = 13;
    struct FullAfterPrefix {
        file: std::fs::File,
        remaining: usize,
    }
    impl std::io::Write for FullAfterPrefix {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Err(std::io::Error::from_raw_os_error(28));
            }
            let written = self.file.write(&bytes[..bytes.len().min(self.remaining)])?;
            self.remaining -= written;
            Ok(written)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.file.flush()
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let spool = directory.path().join("spool");
    std::fs::write(&spool, b"blocked").unwrap();
    let status = SharedStatus::default();
    let mut first = PendingLines::new("registry", "claude-registry", Arc::clone(&status));
    let mut other = PendingLines::new("registry", "codex-status", status);
    let observation = json!({"at_ns":"123", "event":"Status", "payload":{"status":"active"}});
    assert!(
        first
            .append(&spool, "boot", std::slice::from_ref(&observation))
            .is_err()
    );
    let expected = first.bytes.clone();
    std::fs::remove_file(&spool).unwrap();
    let path = first.path(&spool, "boot");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut writer = FullAfterPrefix {
        file: std::fs::File::create(&path).unwrap(),
        remaining: PREFIX_BYTES,
    };
    assert!(super::super::write_pending(&mut first.bytes, &mut writer).is_err());
    assert_eq!(first.bytes, expected[PREFIX_BYTES..]);
    drop(writer);
    other
        .append(&spool, "boot", &[json!({"event":"other"})])
        .unwrap();
    first.append(&spool, "boot", &[]).unwrap();
    assert_ne!(first.path(&spool, "boot"), other.path(&spool, "boot"));
    assert_eq!(std::fs::read(path).unwrap(), expected);
    assert_eq!(
        serde_json::from_slice::<Value>(&std::fs::read(other.path(&spool, "boot")).unwrap())
            .unwrap(),
        json!({"event":"other"})
    );
}

#[test]
fn status_reports_pending_live_lines_and_clears_the_error_after_recovery() {
    let daemon = crate::test_daemon::start("pending live line fixture");
    let directory = tempfile::tempdir().unwrap();
    let spool = directory.path().join("spool");
    std::fs::write(&spool, b"blocked").unwrap();
    let mut pending =
        PendingLines::new("registry", "fixture", Arc::clone(&daemon.daemon.live_spool));
    assert!(
        pending
            .append(&spool, "boot", &[json!({"event":"Status"})])
            .is_err()
    );
    let status = || {
        crate::protocol::into_value(
            crate::test_daemon::Call::new("status", json!({})).answer(&daemon.daemon),
        )
        .unwrap()
    };
    assert_eq!(status()["pending_writes"]["live_event_lines"], 1);
    assert!(
        status()["pending_writes"]["last_error"]
            .as_str()
            .unwrap()
            .contains("fixture")
    );
    std::fs::remove_file(&spool).unwrap();
    pending.append(&spool, "boot", &[]).unwrap();
    assert_eq!(status()["pending_writes"]["live_event_lines"], 0);
    assert!(status()["pending_writes"]["last_error"].is_null());
}

#[test]
fn writers_with_the_same_process_and_stream_own_different_files() {
    let directory = tempfile::tempdir().unwrap();
    let mut first = PendingLines::new("registry", "host", Default::default());
    let mut second = PendingLines::new("registry", "host", Default::default());
    first
        .append(directory.path(), "boot", &[json!({"event":"first"})])
        .unwrap();
    second
        .append(directory.path(), "boot", &[json!({"event":"second"})])
        .unwrap();
    let first_path = first.path(directory.path(), "boot");
    let second_path = second.path(directory.path(), "boot");
    assert_ne!(first_path, second_path);
    assert_eq!(
        serde_json::from_slice::<Value>(&std::fs::read(first_path).unwrap()).unwrap(),
        json!({"event":"first"})
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&std::fs::read(second_path).unwrap()).unwrap(),
        json!({"event":"second"})
    );
}

#[test]
fn instance_spool_uses_the_payload_boot_identity() {
    let directory = tempfile::tempdir().unwrap();
    let mut pending = PendingLines::new("registry", "host", Default::default());
    let observation = json!({"at_ns":"1790531445000000000", "route":"registry",
        "event":"Status", "provider":"claude-code", "session_id":"s1",
        "payload":{"status":"busy", "pid":4242,"start":77,"boot_id":"payload-boot"}});
    let spool = directory.path().join("spool");
    std::fs::write(&spool, b"blocked").unwrap();
    assert!(
        pending
            .append(
                &spool,
                "different-filename-boot",
                std::slice::from_ref(&observation)
            )
            .is_err()
    );
    std::fs::remove_file(&spool).unwrap();
    pending
        .append(&spool, "different-filename-boot", &[])
        .unwrap();
    let path = pending.path(&spool, "different-filename-boot");
    let bytes = std::fs::read(&path).unwrap();
    assert!(polylogue_sources::hook_spool::recognize(&path, &bytes));
    let parser = polylogue_sources::hook_spool::open(&path).unwrap();
    let (step, _) = parser.step(None, &bytes, 0).unwrap();
    assert_eq!(step.output.events.len(), 1);
    let event = &step.output.events[0];
    assert_eq!(event.kind, "registry_status");
    assert_eq!(
        event.session.as_ref().unwrap().to_string(),
        "claude-code:s1"
    );
    assert_eq!(event.data["boot_id"], "payload-boot");
    assert_eq!(event.data["status"], "busy");
    assert_eq!(event.range, Some((0, bytes.len() as u64 - 1)));
}
