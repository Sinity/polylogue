//! Primary mode: durable prepared materials and complete product facts go to
//! Sinex through its ingress. One confirmed-stream consumer writes the local
//! read index. Captures and authored writes remain in the producer spool
//! through outages and retries, under their original identities.
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

mod client;
pub mod kinds;
mod material;
pub(crate) mod native;
mod producer;
mod products;
mod replay;
pub use replay::Replay;
mod spool;
mod status;
pub use status::Status;

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
            Ok(text) => {
                let config: Self = toml::from_str(&text)
                    .map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
                config
                    .check()
                    .map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
                Ok(Some(config))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(std::io::Error::new(e.kind(), format!("{}: {e}", path.display())).into()),
        }
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

    /// The NATS user and password, read from `password_file` when one is set.
    /// The file's trailing whitespace is not part of the password.
    pub(crate) fn credentials(&self) -> Result<Option<(String, String)>> {
        let Some(path) = &self.password_file else {
            return Ok(None);
        };
        let password = std::fs::read_to_string(path)
            .map_err(|e| Error::Invalid(format!("password file {}: {e}", path.display())))?;
        Ok(Some((
            self.producer.clone(),
            password.trim_end().to_owned(),
        )))
    }
}

/// A connection to Sinex as the producer `config.producer`, used by the
/// writer thread and the loop. One frame is in flight at a time.
pub struct Sinex {
    config: Config,
    health: Arc<Mutex<status::Health>>,
    spool: Arc<Mutex<spool::Spool>>,
    runtime: tokio::runtime::Runtime,
    /// `None` only while dropping.
    client: Mutex<Option<Client>>,
    /// This host's name, for the register materials of requests.
    host: String,
}

impl Sinex {
    /// Opens the durable producer spool before making any network connection.
    /// Capture can prepare writes while Sinex is unavailable.
    pub fn open(config: &Config) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let spool = Arc::new(Mutex::new(spool::Spool::open(
            &config.store,
            &config.producer,
        )?));
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|host| host.trim().to_owned())
            .unwrap_or_default();
        Ok(Self {
            config: config.clone(),
            health: Arc::new(Mutex::new(status::Health::default())),
            spool,
            runtime,
            client: Mutex::new(None),
            host,
        })
    }

    /// Opens this producer and requires Sinex to answer immediately.
    pub fn connect(config: &Config) -> Result<Self> {
        let sinex = Self::open(config)?;
        sinex.with(async |_client| Ok(()))?;
        Ok(sinex)
    }

    /// Runs `work` on the connection; frames never interleave.
    fn with<T>(&self, work: impl AsyncFnOnce(&mut Client) -> Result<T>) -> Result<T> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| Error::Invalid("the Sinex connection panicked".into()))?;
        if client.is_none() {
            let connected = self.runtime.block_on(Client::connect(
                &self.config,
                Arc::clone(&self.spool),
                Arc::clone(&self.health),
            ));
            match connected {
                Ok(connected) => *client = Some(connected),
                Err(error) => {
                    if let Ok(mut health) = self.health.lock() {
                        health.reachable = Some(!matches!(error, Error::Unreachable(_)));
                        health.error = Some(error.to_string());
                    }
                    return Err(error);
                }
            }
        }
        let result = self
            .runtime
            .block_on(work(client.as_mut().expect("connected client")));
        if let Ok(mut health) = self.health.lock() {
            health.reachable = Some(!matches!(result, Err(Error::Unreachable(_))));
            health.error = result.as_ref().err().map(ToString::to_string);
        }
        if matches!(result, Err(Error::Unreachable(_))) {
            // The immutable spool keeps every unacknowledged frame. A fresh
            // connection retries those identities after the next successful hello.
            let _inside = self.runtime.enter();
            client.take();
        }
        result
    }

    /// Sinex's store root.
    pub fn store_root(&self) -> Result<PathBuf> {
        Ok(self.config.store.clone())
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
    fn prepare_input(
        &self,
        captured: &polylogue_store::primary::CapturedInput,
        bytes: &mut dyn std::io::Read,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedInput> {
        Ok(self.prepare_input(captured, bytes)?)
    }
    fn prepare_blob(
        &self,
        hash: polylogue_model::BlobHash,
        length: u64,
        bytes: &mut dyn std::io::Read,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.prepare_blob(hash, length, bytes)?)
    }
    fn pending_captures(
        &self,
    ) -> polylogue_store::Result<Vec<polylogue_store::primary::PendingCapture>> {
        Ok(self.pending_captures()?)
    }
    fn capture(
        &self,
        identity: &str,
    ) -> polylogue_store::Result<polylogue_store::primary::Capture> {
        Ok(self.capture(identity)?)
    }
    fn complete_captures(
        &self,
        identities: &[String],
        products: &[polylogue_store::primary::ProductWrite],
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.complete_captures(identities, products)?)
    }
    fn prepare_products(
        &self,
        products: &[polylogue_store::primary::ProductWrite],
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.prepare_products(products)?)
    }
    fn prepare_request(
        &self,
        request: &polylogue_model::authored::PreparedRequest,
    ) -> polylogue_store::Result<polylogue_model::user::UserOutcome> {
        Ok(self.prepare_request(request)?)
    }
    fn canonical_publisher(&self) -> bool {
        self.health
            .lock()
            .is_ok_and(|health| health.canonical_publisher)
    }
    fn visit_pending_requests(
        &self,
        visit: &mut dyn FnMut(
            &polylogue_model::authored::PendingAuthored,
        ) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        self.visit_pending_requests(visit)
    }
    fn visit_pending_blobs(
        &self,
        visit: &mut dyn FnMut(&polylogue_store::NewBlob) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        self.visit_pending_blobs(visit)
    }
    fn prepare_named_products(
        &self,
        identity: &str,
        products: &[polylogue_store::primary::ProductWrite],
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.prepare_named_products(identity, products)?)
    }
    fn discard_confirmed(
        &self,
        identities: &[String],
        position: i64,
        applied: i64,
    ) -> polylogue_store::Result<()> {
        let receipts = self.with(async |client| client.recorded(identities).await)?;
        let queue = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
        let mut admitted = false;
        for receipt in receipts.into_values() {
            if let Some(entry) = queue.get(&receipt.frame)? {
                admitted |= matches!(
                    entry.kind,
                    spool::Kind::Events | spool::Kind::ReplaySequence
                ) && receipt.refusal.is_none()
                    && receipt.commit_seq.is_some();
                queue.acknowledge(receipt)?;
            }
        }
        if admitted {
            queue.retire(applied)?;
        } else {
            queue.discard_confirmed(identities, position, applied)?;
        }
        Ok(())
    }
    fn prepare_reading(
        &self,
        root: &str,
        version: &str,
        visit: &mut polylogue_store::primary::ReadingProducts<'_>,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        self.prepare_reading(root, version, visit)
    }
    fn confirmed(
        &self,
        after: i64,
        cache: &str,
        scope: Option<&str>,
    ) -> polylogue_store::Result<Vec<polylogue_store::primary::Confirmed>> {
        let page = self.with(async |client| client.confirmed(after, cache, scope).await)?;
        if !page.iter().any(|item| {
            matches!(
                item.change,
                polylogue_store::primary::ConfirmedChange::ScopeReset { .. }
            )
        }) {
            self.resume_publication();
        }
        Ok(page)
    }
    fn acknowledge(&self, position: i64) -> polylogue_store::Result<()> {
        Ok(self.with(async |client| client.acknowledge(position).await)?)
    }
}

#[cfg(test)]
mod tests;

/// A prepared material identity is determined by its immutable frame identity.
fn material_id(frame: &str) -> sinex_model::MaterialId {
    let hash = blake3::hash(frame.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hash.as_bytes()[..16]);
    sinex_model::MaterialId(uuid::Uuid::from_bytes(bytes))
}
