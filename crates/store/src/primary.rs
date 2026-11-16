//! Primary mode: the producer operations Polylogue saves for Sinex, and the
//! confirmed cache.
pub use crate::source::SourceBoundary;
pub mod authored;
pub mod capture;
pub use capture::{
    AcquisitionReceipt, AcquisitionRoute, CaptureEmitter, CaptureMembers, CaptureMode,
    CaptureOutcome, CapturePlan, CaptureResult, ConfirmedSource, PendingGroupPage, PendingSource,
    RawRange, SourceBasis, SourceSelector, SourceSnapshot, SourceStatus, SourceSubject,
    SpooledGroup,
};
pub mod byte_cursor;
pub mod products;
pub mod queue;
pub mod replay;
pub use byte_cursor::{
    BodyRange, ByteCursor, ByteCursorCheck, ByteCursorQuery, DatabaseSnapshotToken, FileSnapshot,
    FileTime, RawStream, RawStreamRole, SourceGeneration,
};
pub use replay::{
    AppliedSnapshot, CompletedDelivery, ConfirmedOrigin, LastStreamMessage, PayloadIdentity,
    ReplayControl, ReplayState, SinexEventContext, SinexKind, SinexSource, SnapshotProgress,
    StreamCursor,
};
pub(crate) mod confirmed;
pub(crate) mod scope;
pub use products::{
    Capture, CaptureGroup, CapturedInput, Confirmed, ConfirmedChange, ExportReadingKey, Material,
    ParsedSource, PendingCapture, PreparedInput, PreparedWrite, Product, ProductWrite, Reading,
    SourceLocation, SourceRecord,
};

use crate::ConnExt as _;
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use polylogue_model::user::UserOutcome;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

pub type ProductEmitter<'a> = dyn FnMut(ProductWrite) -> Result<()> + 'a;
/// Emits every parsed event of a replacement reading.
pub type ReadingProducts<'a> = dyn FnMut(&mut ProductEmitter<'_>) -> Result<()> + 'a;

/// Names of saved producer operations in the spool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamedProductSelector {
    Exact(String),
    Prefix(String),
}

/// An encoded body pinned on this host. Its optional path is a shortcut for
/// readers on the same host; the file and descriptor determine the bytes.
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
    /// Stages member bodies and capture-read metadata before the group is
    /// spooled once.
    fn prepare_input_group(
        &self,
        plan: &CapturePlan,
        members: &mut CaptureMembers<'_>,
    ) -> Result<CaptureResult>;
    /// Reads the capture-read headers held in the spool.
    fn pending_source(&self, source: &SourceSelector) -> Result<PendingSource>;
    /// Pages immutable accepted membership; headers contain no parsed outputs.
    /// Zero limit is invalid. Missing applied members do not show the source ended.
    fn pending_group_headers(
        &self,
        group: &str,
        after_ordinal: Option<u64>,
        limit: usize,
    ) -> Result<PendingGroupPage>;
    /// Checks contiguous committed byte ranges in one observed state of the
    /// source. This spool cursor does not mean Sinex accepted the source.
    fn byte_cursor(&self, query: &ByteCursorQuery) -> Result<Option<ByteCursor>>;
    /// Pins the original encoded body for byte comparison of a range without
    /// network I/O: its one encoding, or its chunks' in order.
    fn capture_body(&self, identity: &str) -> Result<Vec<PreparedBody>>;
    fn pending_captures(&self) -> Result<Vec<PendingCapture>>;
    /// Loads the capture-read metadata of one capture.
    fn capture(&self, identity: &str) -> Result<Capture>;
    fn complete_captures(
        &self,
        identities: &[String],
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn prepare_products(&self, products: &[ProductWrite]) -> Result<PreparedWrite>;
    /// Prepares parsed events from a confirmed original chain in the spool,
    /// before the local rows of the same parse commit. The frames of one chain
    /// are sent in the order they were prepared.
    fn prepare_parsed(
        &self,
        source: &ParsedSource,
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn prepare_request(
        &self,
        request: &polylogue_model::authored::PreparedRequest,
    ) -> Result<UserOutcome>;
    /// Saved named operations, including frames sent but not yet applied.
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
    /// Prepares a producer replay that replaces Sinex's parsed events for one
    /// chain. `None` while an earlier frame of that chain is unacknowledged.
    fn prepare_reading(
        &self,
        root: &str,
        key: &ExportReadingKey,
        visit: &mut ReadingProducts<'_>,
    ) -> Result<Option<PreparedWrite>>;
    /// Products under a caller-chosen write identity, so a retry reuses it.
    fn prepare_named_products(
        &self,
        identity: &str,
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn confirmed(&self, state: &ReplayState, cache: &str) -> Result<Vec<Confirmed>>;
    /// Acknowledges the applied page. `applied` answers whether `index.db`
    /// holds a row for one Sinex event; producer operations retire only when
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
    crate::meta_get(&index, "primary_cache")?
        .filter(|value: &String| !value.is_empty())
        .ok_or_else(|| crate::StoreError::stored(0, "the primary cache has no identity"))
}
