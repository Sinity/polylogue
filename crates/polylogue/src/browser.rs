//! The browser extension's side of Polylogue: the capture spool and the
//! routes the extension calls through its native messaging host,
//! `polylogue browser-host`. Chrome starts the host only for the extension
//! IDs its manifest names, so no token or pairing is needed. The host
//! writes captures into the spool, which the daemon ingests like any
//! location, and asks the daemon everything else over its Unix socket.
//!
//! Chat actions (Polylogue asking the extension to start or reply to a
//! conversation in a chat web page) are off by default: the archive setting
//! `browser.actions` must be `on`. Setting it to anything else stops every
//! claim. Only the operator queues them, through the CLI, MCP or the reader.
//! Backfill jobs (the extension capturing a web chat's older conversations)
//! are started the same way and run while the extension is open.

pub mod actions;
pub mod backfill;
mod captures;
pub mod native;
mod notes;
pub mod routes;

use crate::daemon::{Request, Response};
use polylogue_engine::Config;
use polylogue_store::Archive;
use std::fs;
use std::path::PathBuf;

/// The native messaging host's registered name.
pub const HOST_NAME: &str = "com.polylogue.browser_capture";

/// Where the extension's captures go and how to reach the daemon.
pub struct BrowserHost {
    /// `<state>/browser-capture`: one capture log per conversation.
    pub spool: PathBuf,
    /// Names the archive's runtime directory, where a running reader
    /// records its address.
    pub archive_key: String,
    /// The daemon's socket.
    pub socket: PathBuf,
}

impl BrowserHost {
    /// The spool in Polylogue's state directory as the daemon's loop sees
    /// it (`config`), for the archive at `archive`.
    pub fn at(config: &Config, archive: &Archive) -> std::io::Result<Self> {
        let spool = config.spool_dir(polylogue_sources::browser_capture::ORIGIN);
        fs::create_dir_all(&spool)?;
        Ok(Self {
            spool,
            archive_key: crate::runtime_dir::archive_key(&archive.root),
            socket: archive.socket(),
        })
    }

    /// Asks the daemon. `None` when it does not answer: callers report what
    /// they cannot know as unknown, never as absent.
    pub fn ask(&self, request: &Request) -> Option<Response> {
        crate::client::send(&self.socket, request).ok()
    }

    /// The chat actions setting, or `None` when the daemon does not answer.
    /// The answer also shows the daemon is serving: a cheap read, unlike
    /// `status`, which counts the whole archive.
    pub fn actions_setting(&self) -> Option<bool> {
        let request = Request::Setting {
            request_id: None,
            key: "browser.actions".into(),
            value: None,
            machine: false,
        };
        match self.ask(&request) {
            Some(Response::Ok { data, .. }) => Some(data["value"] == "on"),
            _ => None,
        }
    }

    /// The capture log of one provider conversation.
    pub fn capture_log(&self, provider: &str, conversation: &str) -> PathBuf {
        self.spool.join(capture_log_path(provider, conversation))
    }
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

/// Where a provider conversation's capture log lives, relative to the
/// spool. It keeps every distinct capture in order, including captures
/// stored before the loop parsed them.
pub fn capture_log_path(provider: &str, conversation: &str) -> String {
    let hash = blake3::hash(format!("{provider}:{conversation}").as_bytes());
    format!(
        "{}/{}-{}.json",
        safe_token(provider),
        safe_token(conversation),
        &hash.to_hex()[..12]
    )
}

#[cfg(test)]
mod tests;
