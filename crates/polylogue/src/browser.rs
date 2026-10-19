//! The browser extension's side of Polylogue (spec
//! `surfaces/extension-and-clients.md`): the capture spool, the chat-action
//! queue, and the routes the extension calls through its native messaging
//! host, `polylogue browser-host`. Chrome starts the host only for the
//! extension IDs its manifest names, so no token or pairing is needed. The
//! host writes captures into the spool, which the daemon ingests like any
//! location, and asks the daemon everything else over its Unix socket.
//!
//! The chat-action conduit (operator 2026-07-16; polylogue-ptx, -b1n, -yqof)
//! is off by default behind two switches: the archive setting
//! `browser.actions` must be `on`, and the extension's own posting switch.
//! Setting `browser.actions` to anything else stops every claim (the kill
//! switch). Only the operator, through the CLI or the reader, enqueues.

pub mod actions;
pub mod native;
pub mod routes;

use crate::daemon::{Request, Response};
use polylogue_store::Archive;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The API identity the extension checks on every answer.
pub const API_SCHEMA: &str = "polylogue-browser-capture/v1";
/// The native messaging host's registered name.
pub const HOST_NAME: &str = "com.polylogue.browser_capture";

/// Where the extension's files live and how to reach the daemon.
pub struct Ctx {
    /// `<state>/browser-capture`: one file per captured conversation.
    pub spool: PathBuf,
    /// `<state>/browser-actions`: one file per action, `inputs/` for
    /// attachment bytes, `capture-health/` for the latest report per
    /// provider.
    pub actions: PathBuf,
    /// Stable per archive, so the extension can tell its receiver apart.
    pub receiver_id: String,
    /// The daemon's socket.
    pub socket: PathBuf,
}

impl Ctx {
    /// The extension's files under `state` (`$XDG_STATE_HOME/polylogue` as
    /// the daemon's loop sees it) for the archive at `archive`.
    pub fn at(state: &Path, archive: &Archive) -> std::io::Result<Self> {
        let spool = state.join("browser-capture");
        let actions = state.join("browser-actions");
        fs::create_dir_all(&spool)?;
        fs::create_dir_all(actions.join("inputs"))?;
        fs::create_dir_all(actions.join("capture-health"))?;
        Ok(Self {
            spool,
            actions,
            receiver_id: format!(
                "polylogue:{}",
                crate::web::access::archive_key(&archive.root)
            ),
            socket: archive.socket(),
        })
    }

    /// The running user's: the default archive and the state root its
    /// daemon's loop reads.
    pub fn for_user() -> std::io::Result<Self> {
        let config = polylogue_engine::Config::from_env();
        let state = config
            .root(polylogue_sources::browser_capture::ROOT)
            .join("polylogue");
        Self::at(&state, &Archive::at(&Archive::default_root()))
    }

    /// Asks the daemon. `None` when it is not running: callers report what
    /// they cannot know as unknown, never as absent.
    pub fn ask(&self, request: &Request) -> Option<Response> {
        crate::cli::send(&self.socket, request).ok()
    }

    /// Whether the operator switched the conduit on. Off when the daemon
    /// cannot say.
    pub fn actions_enabled(&self) -> bool {
        let request = Request::Setting {
            request_id: None,
            key: "browser.actions".into(),
            value: None,
            machine: false,
        };
        matches!(
            self.ask(&request),
            Some(Response::Ok { data }) if data["value"] == "on"
        )
    }
}

pub fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

pub fn now_iso() -> String {
    jiff::Timestamp::now().to_string()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A path segment from untrusted text: safe characters only, bounded.
pub fn safe_token(s: &str) -> String {
    let t: String = s
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let t = t.trim_matches(|c| c == '.' || c == '-');
    if t.is_empty() {
        "session".into()
    } else {
        t.chars().take(96).collect()
    }
}

/// Where a provider conversation's capture lives, relative to the spool:
/// one file per conversation, rewritten when it is captured again. The
/// archive keeps every distinct content as a revision.
pub fn artifact_ref(provider: &str, provider_session_id: &str) -> String {
    let hash = blake3::hash(format!("{provider}:{provider_session_id}").as_bytes());
    format!(
        "{}/{}-{}.json",
        safe_token(provider),
        safe_token(provider_session_id),
        &hash.to_hex()[..12]
    )
}

/// Writes `bytes` to `path` so a crash leaves the old file or the new one:
/// a temporary file, synced, renamed, then the directory synced.
pub fn write_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("no parent directory"))?;
    fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".{}.tmp", new_id()));
    let mut f = fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)?;
    fs::File::open(dir)?.sync_all()
}

#[cfg(test)]
mod tests;
