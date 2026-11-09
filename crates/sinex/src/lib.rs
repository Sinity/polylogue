//! Primary mode: captured bytes and complete product facts go to Sinex
//! through its ingress, and one confirmed-stream consumer writes the local
//! index. Captures and authored writes wait in the producer spool through
//! outages and retries, under their original identities.
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod client;
mod context;
pub use context::{ContextFailure, ContextFailureKind, ResolvedContext};
pub use sinex_protocol::PublicInterruptBudget;
pub mod kinds;
mod material;
pub(crate) mod native;
mod producer;
mod products;
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
    #[error("Sinex recorded different logical bytes for {frame}")]
    ReceiptConflict {
        frame: String,
        offered_hash: String,
        receipt: Box<sinex_protocol::Acknowledgment>,
    },
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

/// This adapter declaration is installed by the owner, never self-registered
/// from a producer's requested permissions.
pub fn source_reference() -> sinex_protocol::SourceRef {
    sinex_protocol::SourceRef {
        name: "polylogue".into(),
        version: 6,
    }
}

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
    config: Config,
    authentication: Authentication,
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
    pub fn open(config: &Config, authentication: &Authentication, host: &str) -> Result<Self> {
        if host.trim().is_empty() {
            return Err(Error::Invalid(
                "a configured host identity is required".into(),
            ));
        }
        // Backup control must remain responsive while the synchronous archive
        // loop is idle. One runtime worker also drives the NATS connection.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()?;
        let spool = Arc::new(Mutex::new(spool::Spool::open(
            &config.store,
            &config.producer,
        )?));
        Ok(Self {
            config: config.clone(),
            authentication: authentication.clone(),
            health: Arc::new(Mutex::new(status::Health::default())),
            spool,
            runtime,
            client: Mutex::new(None),
            host: host.to_owned(),
        })
    }

    /// Opens this producer and requires Sinex to answer immediately.
    pub fn connect(config: &Config, authentication: &Authentication, host: &str) -> Result<Self> {
        let sinex = Self::open(config, authentication, host)?;
        sinex.read_client(async |_client| Ok(()))?;
        Ok(sinex)
    }

    /// Reads and connection checks own their outcome independently of publication.
    fn read_client<T>(&self, work: impl AsyncFnOnce(&mut Client) -> Result<T>) -> Result<T> {
        let result = self.client_operation(work);
        if let Ok(mut health) = self.health.lock() {
            health.error = result.as_ref().err().map(ToString::to_string);
        }
        result
    }

    /// Executes one client operation under the connection lock; frames never interleave.
    fn client_operation<T>(&self, work: impl AsyncFnOnce(&mut Client) -> Result<T>) -> Result<T> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| Error::Invalid("the Sinex connection panicked".into()))?;
        if client.is_none() {
            let connected = self.runtime.block_on(Client::connect(
                &self.config,
                &self.authentication,
                Arc::clone(&self.spool),
                Arc::clone(&self.health),
            ));
            match connected {
                Ok(connected) => *client = Some(connected),
                Err(error) => {
                    if let Ok(mut health) = self.health.lock() {
                        health.reachable = Some(!matches!(error, Error::Unreachable(_)));
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
        self.read_client(async |client| {
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
    fn raw_acquisition_cursor(
        &self,
        query: &polylogue_store::primary::RawAcquisitionQuery,
    ) -> polylogue_store::Result<Option<polylogue_store::primary::RawAcquisitionCursor>> {
        Ok(self.raw_acquisition_cursor(query)?)
    }
    fn prepare_input_group(
        &self,
        plan: &polylogue_store::primary::CapturePlan,
        members: &mut polylogue_store::primary::CaptureMembers<'_>,
    ) -> polylogue_store::Result<polylogue_store::primary::CaptureReceipt> {
        Ok(self.prepare_input_group(plan, members)?)
    }
    fn pending_source(
        &self,
        source: &polylogue_store::primary::SourceSelector,
        reading_version: &str,
    ) -> polylogue_store::Result<polylogue_store::primary::PendingSource> {
        Ok(self.pending_source(source, reading_version)?)
    }
    fn capture_body(
        &self,
        identity: &str,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedBody> {
        Ok(self.capture_body(identity)?)
    }
    fn prepare_input(
        &self,
        captured: &polylogue_store::primary::CapturedInput,
        body: polylogue_store::primary::PreparedBody,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedInput> {
        Ok(self.prepare_input(captured, body)?)
    }
    fn prepare_blob(
        &self,
        hash: polylogue_model::BlobHash,
        length: u64,
        body: polylogue_store::primary::PreparedBody,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.prepare_blob(hash, length, body)?)
    }
    fn pending_group_headers(
        &self,
        group: &str,
        after: Option<u64>,
        limit: usize,
    ) -> polylogue_store::Result<polylogue_store::primary::PendingGroupPage> {
        Ok(self.pending_group_headers(group, after, limit)?)
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
    fn prepare_export_products(
        &self,
        key: &polylogue_store::primary::ExportReadingKey,
        products: &[polylogue_store::primary::ProductWrite],
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.prepare_export_products(key, products)?)
    }
    fn visit_pending_export_readings(
        &self,
        key: &polylogue_store::primary::ExportReadingKey,
        visit: &mut dyn FnMut(
            &polylogue_store::primary::CapturedReading,
        ) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        Ok(self.visit_pending_export_readings(key, visit)?)
    }
    fn prepare_request(
        &self,
        request: &polylogue_model::authored::PreparedRequest,
    ) -> polylogue_store::Result<polylogue_model::user::UserOutcome> {
        Ok(self.prepare_request(request)?)
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
    fn visit_named_products(
        &self,
        selector: &polylogue_store::primary::NamedProductSelector,
        visit: &mut dyn FnMut(&polylogue_store::primary::Product) -> polylogue_store::Result<()>,
    ) -> polylogue_store::Result<()> {
        Ok(self.visit_named_products(selector, visit)?)
    }
    fn prepare_named_products(
        &self,
        identity: &str,
        products: &[polylogue_store::primary::ProductWrite],
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        Ok(self.prepare_named_products(identity, products)?)
    }
    fn prepare_reading(
        &self,
        root: &str,
        key: &polylogue_store::primary::ExportReadingKey,
        visit: &mut polylogue_store::primary::ReadingProducts<'_>,
    ) -> polylogue_store::Result<polylogue_store::primary::PreparedWrite> {
        self.prepare_reading(root, key, visit)
    }
    fn confirmed(
        &self,
        state: &polylogue_store::primary::ReplayState,
        cache: &str,
    ) -> polylogue_store::Result<Vec<polylogue_store::primary::Confirmed>> {
        // Send durable preparations before reading the page that can confirm
        // them. The follower owns both this read and its later acknowledgment.
        self.send_pending();
        Ok(self.read_client(async |client| client.confirmed(state, cache).await)?)
    }
    fn acknowledge(
        &self,
        state: &polylogue_store::primary::ReplayState,
        applied: &dyn Fn(&str) -> polylogue_store::Result<bool>,
    ) -> polylogue_store::Result<()> {
        let applied = |event: &str| Ok(applied(event)?);
        Ok(self.read_client(async |client| client.acknowledge(state, &applied).await)?)
    }
}

#[cfg(test)]
mod tests;

/// Source-copy custody is scoped to the authenticated receiving producer.
fn material_id(producer: &str, frame: &str) -> sinex_model::MaterialId {
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue-material-1\0");
    hash.update(&(producer.len() as u64).to_le_bytes());
    hash.update(producer.as_bytes());
    hash.update(frame.as_bytes());
    let hash = hash.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hash.as_bytes()[..16]);
    sinex_model::MaterialId(uuid::Uuid::from_bytes(bytes))
}
