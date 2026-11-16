//! Reading stored sessions and records as pages and streams.
mod error;
pub mod execution;
pub mod operation;
use command_surface::Deadline;
pub use error::Error;
use operation::Context;
use polylogue_model::{MessageId, freshness::ReadFreshness, read::*};
use polylogue_store::{Reader, WriterSender, selection::NearFn};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub target: String,
    pub options: ReadOptions,
    pub delivery: ReadDelivery,
    pub format: String,
    pub budget: Option<usize>,
    pub with_blobs: bool,
    pub cursor: Option<String>,
}
pub trait ReadServices {
    fn requests(&self) -> &polylogue_model::user::RequestMetadata;
    fn freshness(&self, reader: &Reader<'_>) -> polylogue_store::Result<ReadFreshness>;
    /// Compile options for `target` with this archive's ranking.
    fn compiled<T>(
        &self,
        reader: &Reader<'_>,
        operation: &Arc<Context>,
        target: &str,
        seed: u64,
        run: impl FnOnce(Compile<'_>) -> Result<T, Error>,
    ) -> Result<T, Error>;
    fn continuation(
        &self,
        request: &ReadRequest,
        cursor: String,
        deadline: Deadline,
    ) -> Result<ReadyCall, Error>;
    fn attachment_call(&self, at: &AttachmentPlace) -> Result<ReadyCall, Error>;
    fn writer(&self) -> Option<&WriterSender>;
}
pub struct Compile<'a> {
    pub options: polylogue_store::query::Options,
    pub ranking_identity: Option<String>,
    pub near: &'a NearFn<'a>,
}
pub struct AttachmentPlace {
    pub message: MessageId,
    pub block: u64,
    pub attachment: u64,
    pub input: Option<String>,
}
