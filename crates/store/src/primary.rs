//! Durable primary producer intents and the confirmed cache contract.
pub mod products;
pub(crate) mod projection;
pub use products::{
    Capture, CaptureGroup, CapturedInput, Confirmed, ConfirmedChange, Material, PendingCapture,
    PreparedInput, PreparedWrite, Product, ProductWrite, Reading, SourceLocation,
};

use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

/// The connection to Sinex the writer and the loop send through.
pub trait Sinex: Send + Sync {
    fn prepare_input(
        &self,
        captured: &CapturedInput,
        bytes: &mut dyn std::io::Read,
    ) -> Result<PreparedInput>;
    fn prepare_blob(
        &self,
        hash: BlobHash,
        length: u64,
        bytes: &mut dyn std::io::Read,
    ) -> Result<PreparedWrite>;
    fn canonical_publisher(&self) -> bool {
        false
    }
    fn pending_captures(&self) -> Result<Vec<PendingCapture>>;
    fn complete_captures(
        &self,
        identities: &[String],
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn prepare_products(&self, products: &[ProductWrite]) -> Result<PreparedWrite>;
    fn prepare_request(
        &self,
        request: &polylogue_model::authored::PreparedRequest,
    ) -> Result<polylogue_model::user::UserOutcome>;
    fn visit_pending_requests(
        &self,
        apply: &mut dyn FnMut(&polylogue_model::authored::PreparedRequest) -> Result<()>,
    ) -> Result<()>;
    fn visit_pending_blobs(
        &self,
        apply: &mut dyn FnMut(&crate::NewBlob) -> Result<()>,
    ) -> Result<()>;
    fn confirmed(&self, after: i64, cache: &str) -> Result<Vec<Confirmed>>;
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
