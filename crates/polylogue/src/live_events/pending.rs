//! Lines the live writers append to the hook spool, kept until the spool
//! accepts them. Each producer owns its file and retains exactly the bytes
//! not yet appended; a failed write is counted in the daemon's
//! `pending_writes` status.

use polylogue_model::LockExt as _;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
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
mod tests;
