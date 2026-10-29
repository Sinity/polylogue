//! Durable primary producer intents and the confirmed cache contract.
pub mod authored;
pub mod products;
pub(crate) mod projection;
pub(crate) mod scope;
pub use products::{
    Capture, CaptureGroup, CapturedInput, CapturedReading, Confirmed, ConfirmedChange, Material,
    PendingCapture, PreparedInput, PreparedWrite, Product, ProductWrite, Reading, SourceLocation,
};

use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use polylogue_model::user::UserOutcome;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

pub type ProductEmitter<'a> = dyn FnMut(ProductWrite) -> Result<()> + 'a;
pub type ReadingProducts<'a> = dyn FnMut(&mut ProductEmitter<'_>) -> Result<()> + 'a;

/// A locally pinned encoded body. Its optional path is a same-host custody
/// optimization; the file and descriptor determine the bytes.
pub struct PreparedBody {
    pub file: std::fs::File,
    pub descriptor: hashed_blobs::EncodedBlob,
    pub path: Option<std::path::PathBuf>,
}

/// The connection to Sinex the writer and the loop send through.
pub trait Sinex: Send + Sync {
    fn prepare_input(&self, captured: &CapturedInput, body: PreparedBody) -> Result<PreparedInput>;
    fn prepare_blob(
        &self,
        hash: BlobHash,
        length: u64,
        body: PreparedBody,
    ) -> Result<PreparedWrite>;
    fn canonical_publisher(&self) -> bool {
        false
    }
    fn pending_captures(&self) -> Result<Vec<PendingCapture>>;
    /// Loads complete private facts for one ready capture, never the whole queue.
    fn capture(&self, identity: &str) -> Result<Capture>;
    fn complete_captures(
        &self,
        identities: &[String],
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn prepare_products(&self, products: &[ProductWrite]) -> Result<PreparedWrite>;
    fn prepare_request(
        &self,
        request: &polylogue_model::authored::PreparedRequest,
    ) -> Result<UserOutcome>;
    fn visit_pending_requests(
        &self,
        visit: &mut dyn FnMut(&polylogue_model::authored::PendingAuthored) -> Result<()>,
    ) -> Result<()>;
    fn visit_pending_blobs(
        &self,
        visit: &mut dyn FnMut(&crate::NewBlob) -> Result<()>,
    ) -> Result<()>;
    /// One complete reparse replaces its exact owner-confirmed material scope.
    fn prepare_reading(
        &self,
        root: &str,
        reading_version: &str,
        visit: &mut ReadingProducts<'_>,
    ) -> Result<PreparedWrite>;
    /// Retry identity comes from the same transaction as the applied position.
    fn prepare_named_products(
        &self,
        identity: &str,
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn discard_confirmed(
        &self,
        identities: &[String],
        tombstone_position: i64,
        applied: i64,
    ) -> Result<()>;
    fn confirmed(&self, after: i64, cache: &str, scope: Option<&str>) -> Result<Vec<Confirmed>>;
    fn acknowledge(&self, position: i64) -> Result<()>;
}

/// What an input is, in words Sinex keeps: the origin that wrote it, its
/// path (or `drive:<id>`, an import's file name, …) and the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InputName {
    pub origin: String,
    pub path: String,
    pub host: String,
}

/// The primary cache's persistent identity, independent of read cursor generations.
pub fn cache_identity(index: &Connection) -> Result<String> {
    index
        .query_row(
            "SELECT value FROM meta WHERE key='primary_cache'",
            [],
            |row| row.get(0),
        )
        .optional()?
        .filter(|value: &String| !value.is_empty())
        .ok_or_else(|| StoreError::Invalid("the primary cache has no identity".into()))
}
