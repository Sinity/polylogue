//! Polylogue on Sinex (architecture 7.1). In `standalone` mode, the default,
//! this crate is not used. In `primary` mode Sinex holds the archive: every
//! row batch of `archive.db` and every input blob its rows cite, so an empty
//! archive restores from Sinex and the loop rebuilds `index.db` from it.
//!
//! A row batch crosses as materials and one event: the blobs it cites, its
//! rows as JSON lines, and a `polylogue.rows` event naming both, sent under
//! the name `rows:<archive>:<position>` (its position in write order). Each
//! pass asks Sinex which names it holds and sends the rest in order, so Sinex
//! always holds a prefix of the archive. Sinex never deletes what it
//! holds; excision refuses once Sinex holds the archive.
use async_nats::jetstream;
use futures_util::StreamExt;
use serde::Deserialize;
use sinex_protocol::{Acknowledgment, Have, Ingress, Subjects};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

mod producer_bytes;
mod restore;
mod rows;
mod send;

pub use restore::Restored;
pub use send::Sent;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Standalone,
    Primary,
}

/// `$XDG_CONFIG_HOME/polylogue/sinex.toml`. It lives outside the archive,
/// so a restore can start from an empty one.
#[derive(Debug, Clone, Deserialize)]
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
    /// Sinex's store root on this host. Polylogue stages the bytes it sends
    /// under it and reads restored bytes from its content store.
    pub store: PathBuf,
    /// The archive Sinex holds for this machine: restored from Sinex when
    /// the local archive is empty.
    #[serde(default)]
    pub archive: Option<String>,
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

/// The last batch Sinex confirmed in this connection, so a pass starts after
/// it instead of asking about the whole archive again.
struct Confirmed {
    archive: String,
    position: u64,
    batch: polylogue_store::row_batches::RowBatch,
}

/// A connection to Sinex as the producer `config.producer`.
pub struct Sinex {
    client: async_nats::Client,
    subjects: Subjects,
    producer: String,
    max_message: usize,
    store_root: PathBuf,
    acknowledgments: async_nats::Subscriber,
    confirmed: Option<Confirmed>,
}

const ACKNOWLEDGMENT_WAIT: Duration = Duration::from_secs(120);

impl Sinex {
    pub async fn connect(config: &Config) -> Result<Self> {
        let client = async_nats::connect(&config.nats)
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let subjects = Subjects::new(config.subjects.clone());
        let acknowledgments = client
            .subscribe(subjects.ack(&config.producer))
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        client
            .flush()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let max_message = client.server_info().max_payload;
        std::fs::create_dir_all(&config.store)?;
        Ok(Self {
            client,
            subjects,
            producer: config.producer.clone(),
            max_message,
            store_root: config.store.clone(),
            acknowledgments,
            confirmed: None,
        })
    }

    /// Sinex's store root, an extra read root for Polylogue's blobs.
    pub fn store_root(&self) -> &Path {
        &self.store_root
    }

    /// What Sinex recorded under these names.
    pub async fn have(&self, names: &[String]) -> Result<HashMap<String, Acknowledgment>> {
        let mut out = HashMap::new();
        for chunk in names.chunks(500) {
            let request = Have {
                frames: chunk.to_vec(),
            };
            let reply = self
                .client
                .request(
                    self.subjects.have(&self.producer),
                    serde_json::to_vec(&request)?.into(),
                )
                .await
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let acknowledgments: Vec<Acknowledgment> = serde_json::from_slice(&reply.payload)?;
            out.extend(acknowledgments.into_iter().map(|a| (a.frame.clone(), a)));
        }
        Ok(out)
    }

    /// Sends one message and waits for its acknowledgment. Accepted by NATS
    /// is not confirmed: only the acknowledgment, which Sinex commits with
    /// the message's events, is.
    async fn transmit(&mut self, message: &Ingress) -> Result<Acknowledgment> {
        self.check_message(message)?;
        let js = jetstream::new(self.client.clone());
        js.publish(
            self.subjects.ingress(&self.producer),
            sinex_protocol::producer_bytes::encode_frame(message).into(),
        )
        .await
        .map_err(|e| Error::Unreachable(e.to_string()))?
        .await
        .map_err(|e| Error::Unreachable(e.to_string()))?;
        let name = message.frame();
        loop {
            let message = tokio::time::timeout(ACKNOWLEDGMENT_WAIT, self.acknowledgments.next())
                .await
                .map_err(|_| Error::Unreachable(format!("no acknowledgment of {name}")))?
                .ok_or_else(|| Error::Unreachable("acknowledgments closed".into()))?;
            let acknowledgment: Acknowledgment = serde_json::from_slice(&message.payload)?;
            if acknowledgment.frame == name {
                return checked(acknowledgment);
            }
        }
    }
}

fn checked(acknowledgment: Acknowledgment) -> Result<Acknowledgment> {
    match &acknowledgment.refusal {
        Some(reason) => Err(Error::Refused {
            name: acknowledgment.frame.clone(),
            reason: reason.clone(),
        }),
        None => Ok(acknowledgment),
    }
}

/// `blake3:<hex>`, the form Sinex names bytes by.
fn content_name(hash: &hashed_blobs::Hash) -> String {
    format!("blake3:{}", hash.hex())
}

fn parse_content_name(name: &str) -> Result<hashed_blobs::Hash> {
    name.strip_prefix("blake3:")
        .and_then(|hex| polylogue_model::BlobHash::parse(hex).ok())
        .map(|hash| hashed_blobs::Hash(hash.0))
        .ok_or_else(|| Error::Invalid(format!("{name} is not a BLAKE3 content name")))
}
