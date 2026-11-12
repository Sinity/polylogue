//! Durable primary producer intents and the confirmed cache contract.
pub use crate::source::SourceBoundary;
pub mod acquisition;
pub mod authored;
pub use acquisition::{
    AcquisitionReceipt, AcquisitionRoute, CaptureDisposition, CaptureEmitter, CaptureMembers,
    CaptureMode, CapturePlan, CaptureReceipt, ConfirmedSource, PendingCheckpoint, PendingGroupPage,
    PendingSource, RawCustodyReceipt, RawRange, SourceBasis, SourceSelector, SourceSnapshot,
    SourceStatus, SourceSubject,
};
pub mod products;
pub mod queue;
pub mod raw_acquisition;
pub mod replay;
pub use raw_acquisition::{
    DatabaseSnapshotToken, FileSnapshot, FileTime, RawAcquisitionCursor, RawAcquisitionProof,
    RawAcquisitionQuery, RawBodyRange, RawStream, RawStreamRole, SourceGeneration,
};
pub use replay::{
    ConfirmedOrigin, NativeDeliveryCursor, NativeFactContext, NativeKind, NativeSource,
    ProjectionIdentity, ReplayControl, ReplayState,
};
pub(crate) mod projection;
pub(crate) mod scope;
pub use products::{
    Capture, CaptureGroup, CapturedInput, CapturedReading, Confirmed, ConfirmedChange,
    ExportReadingKey, Material, PendingCapture, PreparedInput, PreparedWrite, Product,
    ProductWrite, Reading, SourceLocation,
};

use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use polylogue_model::user::UserOutcome;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

pub type ProductEmitter<'a> = dyn FnMut(ProductWrite) -> Result<()> + 'a;
pub type ReadingProducts<'a> = dyn FnMut(&mut ProductEmitter<'_>) -> Result<()> + 'a;

/// Logical names of immutable intents in the existing producer custody store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamedProductSelector {
    Exact(String),
    Prefix(String),
}

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
    /// Stages all member bodies and private facts before one durable group publication.
    fn prepare_input_group(
        &self,
        plan: &CapturePlan,
        members: &mut CaptureMembers<'_>,
    ) -> Result<CaptureReceipt>;
    /// Reads only local durable source headers and one verified selected control.
    fn pending_source(
        &self,
        source: &SourceSelector,
        reading_version: &str,
    ) -> Result<PendingSource>;
    /// Pages immutable accepted membership; headers contain no parsed outputs.
    /// Zero limit is invalid. Missing applied members do not prove source EOF.
    fn pending_group_headers(
        &self,
        group: &str,
        after_ordinal: Option<u64>,
        limit: usize,
    ) -> Result<PendingGroupPage>;
    /// Proves contiguous committed physical ranges in one exact generation.
    /// This local custody cursor does not acknowledge logical source acceptance.
    fn raw_acquisition_cursor(
        &self,
        query: &RawAcquisitionQuery,
    ) -> Result<Option<RawAcquisitionCursor>>;
    /// Pins the original encoded body for exact range comparison without
    /// network I/O: its one encoding, or its chunks' in order.
    fn capture_body(&self, identity: &str) -> Result<Vec<PreparedBody>>;
    fn pending_captures(&self) -> Result<Vec<PendingCapture>>;
    /// Loads complete private facts for one ready capture, never the whole queue.
    fn capture(&self, identity: &str) -> Result<Capture>;
    fn complete_captures(
        &self,
        identities: &[String],
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn prepare_products(&self, products: &[ProductWrite]) -> Result<PreparedWrite>;
    fn prepare_export_products(
        &self,
        key: &ExportReadingKey,
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn visit_pending_export_readings(
        &self,
        key: &ExportReadingKey,
        visit: &mut dyn FnMut(&CapturedReading) -> Result<()>,
    ) -> Result<()>;

    fn prepare_request(
        &self,
        request: &polylogue_model::authored::PreparedRequest,
    ) -> Result<UserOutcome>;
    /// Existing named intents, including published but unapplied frames.
    fn visit_named_products(
        &self,
        selector: &NamedProductSelector,
        visit: &mut dyn FnMut(&Product) -> Result<()>,
    ) -> Result<()>;
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
        key: &ExportReadingKey,
        visit: &mut ReadingProducts<'_>,
    ) -> Result<PreparedWrite>;
    /// Products under a caller-chosen write identity, so a retry reuses it.
    fn prepare_named_products(
        &self,
        identity: &str,
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn confirmed(&self, state: &ReplayState, cache: &str) -> Result<Vec<Confirmed>>;
    /// Acknowledges the applied page. `applied` answers whether `index.db`
    /// holds a fact for one Sinex event; producer operations retire only when
    /// all their admitted events are there.
    fn acknowledge(
        &self,
        state: &ReplayState,
        applied: &dyn Fn(&str) -> Result<bool>,
    ) -> Result<()>;
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
