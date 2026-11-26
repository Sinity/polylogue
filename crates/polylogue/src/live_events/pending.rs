//! Lines the live writers append to the hook spool, kept until the spool
//! accepts them. Each producer owns its file and retains exactly the bytes
//! not yet appended; a failed write is counted in the daemon's
//! `pending_writes` status.

use polylogue_model::LockExt as _;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(crate) struct SpoolStatus(BTreeMap<&'static str, (u64, String)>);

pub(crate) type SharedStatus = Arc<Mutex<SpoolStatus>>;

/// Lines waiting for a retry and the write errors that left them there.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PendingSummary {
    pub(crate) lines: u64,
    pub(crate) error: Option<String>,
}

impl SpoolStatus {
    pub(crate) fn summary(&self) -> PendingSummary {
        PendingSummary {
            lines: self.0.values().map(|(lines, _)| lines).sum(),
            error: self
                .0
                .iter()
                .map(|(stream, (_, error))| format!("{stream}: {error}"))
                .reduce(|left, right| format!("{left}; {right}")),
        }
    }

    fn record(&mut self, stream: &'static str, bytes: &[u8], result: &std::io::Result<()>) {
        let previously_failed = self.0.contains_key(stream);
        match result {
            Err(error) => {
                let lines = bytes.iter().filter(|byte| **byte == b'\n').count() as u64;
                if !previously_failed {
                    polylogue_store::log!(
                        warning,
                        "{stream} spool write failed; {lines} lines kept for retry: {error}"
                    );
                }
                self.0.insert(stream, (lines, error.to_string()));
            }
            Ok(()) => {
                self.0.remove(stream);
                if previously_failed {
                    polylogue_store::log!(info, "{stream} spool write recovered");
                }
            }
        }
    }
}

pub(crate) struct PendingLines {
    route: &'static str,
    stream: &'static str,
    pub(super) bytes: Vec<u8>,
    writer: uuid::Uuid,
    status: SharedStatus,
}

impl PendingLines {
    pub(crate) fn new(route: &'static str, stream: &'static str, status: SharedStatus) -> Self {
        Self {
            route,
            stream,
            bytes: Vec::new(),
            writer: uuid::Uuid::now_v7(),
            status,
        }
    }

    pub(super) fn file_name(&self, boot: &str) -> String {
        format!(
            "{boot}-{}-{}-{}",
            std::process::id(),
            self.stream,
            self.writer
        )
    }

    #[cfg(test)]
    pub(super) fn path(&self, spool: &Path, boot: &str) -> PathBuf {
        spool
            .join(self.route)
            .join(format!("{}.ndjson", self.file_name(boot)))
    }

    pub(crate) fn append(
        &mut self,
        spool: &Path,
        boot: &str,
        lines: &[Value],
    ) -> std::io::Result<()> {
        for line in lines {
            serde_json::to_writer(&mut self.bytes, line)?;
            self.bytes.push(b'\n');
        }
        let result = if self.bytes.is_empty() {
            Ok(())
        } else {
            let name = self.file_name(boot);
            super::spool_file(spool, self.route, &name)
                .and_then(|mut file| super::write_pending(&mut self.bytes, &mut file))
        };
        self.status
            .locked()
            .record(self.stream, &self.bytes, &result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn legacy_and_instance_spools_are_read_without_a_filename_boot_guess() {
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
        let new_path = pending.path(&spool, "different-filename-boot");
        let bytes = std::fs::read(&new_path).unwrap();
        let old_path = spool.join("registry/legacy-boot.ndjson");
        std::fs::write(&old_path, &bytes).unwrap();
        for path in [old_path, new_path] {
            let bytes = std::fs::read(&path).unwrap();
            assert!(polylogue_sources::hook_spool::recognize(
                &path,
                &polylogue_sources::document::Head::new(&bytes)
            ));
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
    }
}
