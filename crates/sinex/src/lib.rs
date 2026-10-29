//! Polylogue on Sinex. In `standalone` mode, the default, this crate is not
//! used. In `primary` mode Sinex holds the archive and Polylogue is one of its
//! producers:
//!
//! - each step of a captured file's appended bytes is a part: a material of
//!   its real input (origin, path, host), a growing file a chain of parts,
//!   named with the Polylogue parser that reads it ([`parts`]);
//! - a look at a file that found no new bytes, each request a person or agent
//!   makes is a small register material, with a typed event
//!   for notes, settings, kept results, deliveries
//!   ([`register`]);
//! - parsed sessions, messages, tool calls and turns are events keyed by
//!   Polylogue's IDs and anchored at their records ([`sessions`]).
//!
//! The local register and index are written only after Sinex acknowledged
//! what they describe, and are rebuilt from what Sinex holds.
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

mod client;
pub mod kinds;
pub(crate) mod native;
mod parts;
mod register;
mod replay;
pub use replay::Replay;
mod sessions;

use client::Client;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Sinex is unreachable: {0}")]
    Unreachable(String),
    #[error("Sinex refused {name}: {reason}")]
    Refused { name: String, reason: String },
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
    #[error(transparent)]
    Blobs(#[from] hashed_blobs::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for polylogue_store::StoreError {
    fn from(error: Error) -> Self {
        match error {
            Error::Store(error) => error,
            other => polylogue_store::StoreError::Sinex(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Standalone,
    Primary,
}

/// `$XDG_CONFIG_HOME/polylogue/sinex.toml`. It lives outside the archive,
/// so an archive can be rebuilt from Sinex into an empty directory.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub mode: Mode,
    /// NATS server URL, without credentials.
    pub nats: String,
    /// A file holding this producer's NATS password, when the server
    /// requires one. The NATS user is the producer's name, as Sinex
    /// declares its producers.
    pub password_file: Option<PathBuf>,
    /// Subject root (`sinex` when deployed).
    #[serde(default = "default_subject_root")]
    pub subject_root: String,
    /// This producer's name on the ingress.
    #[serde(default = "default_producer")]
    pub producer: String,
    /// Sinex's store root on this host. Polylogue writes the bytes it sends
    /// into frame directories under it and reads stored bytes from its
    /// content store.
    pub store: PathBuf,
}

fn default_subject_root() -> String {
    "sinex".into()
}

fn default_producer() -> String {
    "polylogue".into()
}

/// Authentication resolved once by the binary's configuration loader. This
/// value never reads the password file or environment during connection.
#[derive(Clone)]
pub enum Authentication {
    Anonymous,
    Password(String),
    Unavailable { reason: String },
}
impl std::fmt::Debug for Authentication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anonymous => f.write_str("Anonymous"),
            Self::Password(_) => f.write_str("Password([redacted])"),
            Self::Unavailable { reason } => f
                .debug_struct("Unavailable")
                .field("reason", reason)
                .finish(),
        }
    }
}
impl Config {
    /// Parse and validate supplied bytes. File selection and credential reads
    /// belong to the startup boundary.
    pub fn parse(text: &str) -> Result<Self> {
        let config: Self =
            toml::from_str(text).map_err(|error| Error::Invalid(error.to_string()))?;
        config.check().map_err(Error::Invalid)?;
        Ok(config)
    }
    /// A password in the URL would sit in the configuration file, and the
    /// NATS client ignores it there, so it is refused.
    fn check(&self) -> std::result::Result<(), String> {
        let authority = self
            .nats
            .split_once("://")
            .map_or(self.nats.as_str(), |(_, rest)| rest)
            .split('/')
            .next()
            .unwrap_or_default();
        if authority.contains('@') {
            return Err("`nats` holds credentials; name the producer's password file in `password_file` instead".into());
        }
        Ok(())
    }
}

/// A connection to Sinex as the producer `config.producer`, used by the
/// writer thread and the loop. One frame is in flight at a time.
pub struct Sinex {
    runtime: tokio::runtime::Runtime,
    /// `None` only while dropping.
    client: Mutex<Option<Client>>,
    /// This host's name, for the register materials of requests.
    host: String,
    /// Where the sessions last sent stand, beside the index they were read
    /// from ([`sessions`]).
    sent: PathBuf,
}

impl Sinex {
    /// Connects to Sinex for the archive at `root`.
    pub fn connect(
        config: &Config,
        authentication: &Authentication,
        root: &Path,
        host: &str,
    ) -> Result<Self> {
        if host.trim().is_empty() {
            return Err(Error::Invalid(
                "a configured host identity is required".into(),
            ));
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let client = runtime.block_on(Client::connect(config, authentication))?;
        Ok(Self {
            runtime,
            client: Mutex::new(Some(client)),
            host: host.into(),
            sent: root.join("sinex-sent.json"),
        })
    }

    /// Runs `work` on the connection; frames never interleave.
    fn with<T>(&self, work: impl AsyncFnOnce(&mut Client) -> Result<T>) -> Result<T> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| Error::Invalid("the Sinex connection panicked".into()))?;
        let client = client
            .as_mut()
            .ok_or_else(|| Error::Invalid("the Sinex connection is closed".into()))?;
        self.runtime.block_on(work(client))
    }

    /// Sinex's store root.
    pub fn store_root(&self) -> Result<PathBuf> {
        self.with(async |client| Ok(client.store_root.clone()))
    }

    /// Whether Sinex answers.
    pub fn reachable(&self) -> Result<()> {
        self.with(async |client| {
            client
                .recorded(&["polylogue-reachable".into()])
                .await
                .map(drop)
        })
    }
}

impl Drop for Sinex {
    /// The NATS client's subscriptions unsubscribe on a task of the runtime
    /// they were made in.
    fn drop(&mut self) {
        let _inside = self.runtime.enter();
        if let Ok(client) = self.client.get_mut() {
            drop(client.take());
        }
    }
}

impl polylogue_store::primary::Sinex for Sinex {
    fn send_part(
        &self,
        part: &polylogue_store::primary::Part,
        bytes: &mut dyn std::io::Read,
    ) -> polylogue_store::Result<polylogue_store::primary::ConfirmedBlob> {
        Ok(self.with(async |client| parts::send(client, part, bytes).await)?)
    }

    fn send_blob(
        &self,
        hash: polylogue_model::BlobHash,
        length: u64,
        bytes: &mut dyn std::io::Read,
    ) -> polylogue_store::Result<polylogue_store::primary::ConfirmedBlob> {
        Ok(self
            .with(async |client| parts::send_blob(client, &self.host, hash, length, bytes).await)?)
    }

    fn send_change(
        &self,
        change: &polylogue_store::primary::RegisterChange,
    ) -> polylogue_store::Result<String> {
        Ok(self.with(async |client| register::send(client, &self.host, change).await)?)
    }

    fn held(&self, after: i64) -> polylogue_store::Result<Vec<polylogue_store::primary::Held>> {
        Ok(self.with(async |client| {
            let page = client.materials(after).await?;
            let store = hashed_blobs::Reader::new(client.store_root.clone());
            page.into_iter()
                .map(|held| parts::held(&store, held))
                .collect::<Result<Vec<_>>>()
        })?)
    }

    fn send_sessions(&self, reads: &polylogue_store::ReadPool) -> polylogue_store::Result<()> {
        Ok(sessions::send(self, reads)?)
    }
}

#[cfg(test)]
mod tests;
