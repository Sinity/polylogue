use super::{Daemon, Response, error, ok, store_error};
use polylogue_store::{Result, secret_scan};
use serde_json::json;

pub(super) fn handle(
    d: &Daemon,
    session: Option<String>,
    all: bool,
    status: bool,
    origin: Option<String>,
    limit: Option<u64>,
) -> Response {
    let run = || -> Result<serde_json::Value> {
        secret_scan::validate(session.as_deref(), all, status, origin.as_deref(), limit)?;
        let mut sessions_scanned = 0u64;
        let mut blocks_scanned = 0u64;
        let mut events_scanned = 0u64;
        let mut findings = Vec::new();
        let mut found = None;
        if !status {
            let Some(writes) = &d.writes else {
                return Err(polylogue_store::StoreError::Invalid(
                    "scan writer unavailable".into(),
                ));
            };
            if let Some(session) = session {
                let scanned = writes.scan_secrets(session)?;
                found = Some(scanned.found);
                sessions_scanned = u64::from(scanned.found);
                blocks_scanned = scanned.blocks_scanned;
                events_scanned = scanned.events_scanned;
                findings = scanned.findings;
            } else {
                let mut after = None;
                while limit.is_none_or(|n| sessions_scanned < n) {
                    if d.stopping.load(std::sync::atomic::Ordering::Acquire) {
                        break;
                    }
                    let pending = d
                        .reads
                        .reader()?
                        .pending_secret_scan(origin.as_deref(), after.as_deref())?;
                    let Some(session) = pending else {
                        break;
                    };
                    let scanned = writes.scan_secrets(session.clone())?;
                    after = Some(session);
                    sessions_scanned += u64::from(scanned.found);
                    blocks_scanned += scanned.blocks_scanned;
                    events_scanned += scanned.events_scanned;
                    findings.extend(scanned.findings);
                }
            }
        }
        let remaining = d
            .reads
            .reader()?
            .secret_scan_pending_count(origin.as_deref())?;
        Ok(
            json!({"status": if found == Some(false) { "not_found" } else { "ok" },
            "found": found, "scanner_version": secret_scan::version(),
            "sessions_scanned": sessions_scanned, "blocks_scanned": blocks_scanned, "events_scanned": events_scanned,
            "candidates_found": findings.len(), "findings": findings,
            "remaining_pending": remaining, "more_pending": remaining > 0}),
        )
    };
    if d.stopping.load(std::sync::atomic::Ordering::Acquire) {
        return error("daemon_stopping", "daemon is stopping");
    }
    match run() {
        Ok(data) => ok(data),
        Err(e) => store_error(&e),
    }
}
