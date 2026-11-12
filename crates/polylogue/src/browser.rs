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
pub(crate) mod notes;
pub mod routes;

use command_surface::Answer;
use polylogue_engine::Config;
use polylogue_store::Archive;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

/// The native messaging host's registered name.
pub const HOST_NAME: &str = "com.polylogue.browser_capture";

/// Where the extension's captures go and how to reach the daemon.
pub struct BrowserHost {
    /// `<state>/browser-capture`: one capture log per conversation.
    pub spool: PathBuf,
    /// Where the daemon records its reader address.
    pub runtime_dir: PathBuf,
    /// The daemon's socket.
    pub socket: PathBuf,
}

impl BrowserHost {
    /// The spool in Polylogue's state directory as the daemon's loop sees
    /// it (`config`), for the archive at `archive`.
    pub fn at(
        config: &Config,
        archive: &Archive,
        paths: &crate::config::Paths,
    ) -> std::io::Result<Self> {
        let spool = config.spool_dir(polylogue_sources::browser_capture::ORIGIN);
        fs::create_dir_all(&spool)?;
        Ok(Self {
            spool,
            runtime_dir: paths.runtime_dir(&archive.root),
            socket: archive.socket(),
        })
    }

    /// Asks the daemon. `None` when it does not answer: callers report what
    /// they cannot know as unknown, never as absent.
    pub fn ask(&self, operation: &str, arguments: Value) -> Option<Answer<Value>> {
        self.ask_as(
            operation,
            arguments,
            polylogue_store::user_writes::new_request_id(),
        )
    }

    /// Asks the daemon with the caller's own retry identity.
    pub fn ask_as(
        &self,
        operation: &str,
        arguments: Value,
        request_id: String,
    ) -> Option<Answer<Value>> {
        crate::client::call(
            &self.socket,
            &crate::client::request(operation, arguments, request_id),
        )
        .ok()
    }

    /// The chat actions setting, or `None` when the daemon does not answer.
    /// The answer also shows the daemon is serving: a cheap read, unlike
    /// `status`, which counts the whole archive.
    pub fn actions_setting(&self) -> Option<bool> {
        let data = crate::protocol::into_value(
            self.ask("setting", serde_json::json!({ "key": "browser.actions" }))?,
        )
        .ok()?;
        let current: polylogue_model::settings::Current = serde_json::from_value(data).ok()?;
        match current.value {
            Some(polylogue_model::settings::Value::Boolean(enabled)) => Some(enabled),
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
