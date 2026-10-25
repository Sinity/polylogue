//! Polylogue on Sinex. In `standalone` mode, the default, this crate is not
//! used. In `primary` mode Sinex holds the archive and Polylogue is one of its
//! producers:
//!
//! - each piece of a captured file is a material of its real input (origin,
//!   path, host), a growing file a chain of parts, named with the Polylogue
//!   parser that reads it ([`parts`]);
//! - a look at a file that found no new bytes, and each request a person or
//!   agent makes, is a small record material, with a typed event for notes,
//!   settings, kept results and deliveries ([`records`]);
//! - parsed sessions, messages, tool calls and turns are events keyed by
//!   Polylogue's IDs and anchored at their records ([`sessions`]).
//!
//! The local register and index are written only after Sinex acknowledged
//! what they describe, and are rebuilt from what Sinex holds.
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

mod client;
mod parts;
mod records;
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
    /// NATS server URL, with the producer's credentials when required.
    pub nats: String,
    /// Subject root (`sinex` when deployed).
    #[serde(default = "default_subjects")]
    pub subjects: String,
    /// This producer's name on the ingress.
    #[serde(default = "default_producer")]
    pub producer: String,
    /// Sinex's store root on this host. Polylogue writes the bytes it sends
    /// into spool directories under it and reads stored bytes from its
    /// content store.
    pub store: PathBuf,
}

fn default_subjects() -> String {
    "sinex".into()
}

fn default_producer() -> String {
    "polylogue".into()
}

impl Config {
    pub fn path() -> PathBuf {
        let config = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        config.join("polylogue/sinex.toml")
    }

    /// The configuration, or `None` when there is no file (standalone). A
    /// file that does not parse is an error the caller shows; it never
    /// switches the mode silently.
    pub fn load() -> Result<Option<Self>> {
        Self::load_from(&Self::path())
    }

    /// Reads one explicitly selected configuration file.
    pub fn load_from(path: &Path) -> Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text)
                .map(Some)
                .map_err(|e| Error::Invalid(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(std::io::Error::new(e.kind(), format!("{}: {e}", path.display())).into()),
        }
    }
}

/// A connection to Sinex as the producer `config.producer`, used by the
/// writer thread and the loop. One frame is in flight at a time.
pub struct Sinex {
    runtime: tokio::runtime::Runtime,
    /// `None` only while dropping.
    client: Mutex<Option<Client>>,
    /// This host's name, for the records Polylogue writes.
    host: String,
    /// Where the sessions last sent stand, beside the index they were read
    /// from ([`sessions`]).
    sent: PathBuf,
}

impl Sinex {
    /// Connects to Sinex for the archive at `root`.
    pub fn connect(config: &Config, root: &Path) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let client = runtime.block_on(Client::connect(config))?;
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|h| h.trim().to_owned())
            .unwrap_or_default();
        Ok(Self {
            runtime,
            client: Mutex::new(Some(client)),
            host,
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
        self.with(async |client| client.have(&["polylogue-reachable".into()]).await.map(drop))
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
    ) -> polylogue_store::Result<String> {
        Ok(self.with(async |client| parts::send(client, part, bytes).await)?)
    }

    fn send_record(&self, record: &polylogue_store::primary::Record) -> polylogue_store::Result<()> {
        Ok(self.with(async |client| records::send(client, &self.host, record).await)?)
    }

    fn held(&self, after: i64) -> polylogue_store::Result<Vec<polylogue_store::primary::Held>> {
        Ok(self.with(async |client| {
            let page = client.materials(after).await?;
            let store = hashed_blobs::Reader::new(client.store_root.clone(), vec![]);
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
