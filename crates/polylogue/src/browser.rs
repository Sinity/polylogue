//! The browser extension's side of Polylogue (spec
//! `surfaces/extension-and-clients.md`): the capture spool, the chat-action
//! queue, and the routes the extension calls through its native messaging
//! host, `polylogue browser-host`. Chrome starts the host only for the
//! extension IDs its manifest names, so no token or pairing is needed. The
//! host writes captures into the spool, which the daemon ingests like any
//! location, and asks the daemon everything else over its Unix socket.
//!
//! Chat actions (Polylogue asking the extension to start or reply to a
//! conversation in a chat web page) are off by default behind two switches: the archive setting
//! `browser.actions` must be `on`, and the extension's own posting switch.
//! Setting `browser.actions` to anything else stops every claim (the kill
//! switch). Only the operator, through the CLI or the reader, enqueues.

pub mod actions;
mod captures;
pub mod native;
mod notes;
pub mod routes;

use crate::daemon::{Request, Response};
use polylogue_engine::Config;
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
    /// The extension's files in Polylogue's state directory as the
    /// daemon's loop sees it (`config`), for the archive at `archive`.
    pub fn at(config: &Config, archive: &Archive) -> std::io::Result<Self> {
        let spool = config.spool_dir(polylogue_sources::browser_capture::ORIGIN);
        let actions = config.state_dir().join("browser-actions");
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
        Self::at(&Config::from_env(), &Archive::at(&Archive::default_root()))
    }

    /// Asks the daemon. `None` when it is not running: callers report what
    /// they cannot know as unknown, never as absent.
    pub fn ask(&self, request: &Request) -> Option<Response> {
        crate::cli::send(&self.socket, request).ok()
    }

    /// Whether the operator switched chat actions on. Off when the daemon
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
/// one append log per conversation. Its ordered envelopes retain every
/// distinct observation, including captures acknowledged before convergence.
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
    atomicwrites::AtomicFile::new(path, atomicwrites::AllowOverwrite)
        .write(|file| file.write_all(bytes))
        .map_err(Into::into)
}

#[cfg(test)]
mod tests;
