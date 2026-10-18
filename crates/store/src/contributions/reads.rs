//! Reads of selected message bodies and sessions. Each view holds one SQLite
//! read transaction, so every read through it, including later cursor steps,
//! sees the same snapshot. Hiding or removing a message, or a new index
//! generation, makes the view fail with `StaleMessage` on its next read.
use crate::{Error, Result};
use rusqlite::Connection;
use std::{
    fs::File,
    sync::{Arc, Mutex, MutexGuard},
};

mod dependencies;
mod fields;
mod message;
mod ordered_extras;
mod session;
mod sources;
pub use dependencies::{SessionDependencyCursor, SessionInputDependency};
pub use fields::{BlockFieldAddress, BlockFieldEntry, FieldCursor, MessageFieldEntry};
pub(crate) use message::search_membership;
pub use message::{
    BlockView, FieldRead, FieldView, MessageHeader, MessageLookup, MessageView, ToolResultHeader,
};
pub use ordered_extras::{ExtraFieldEntry, OrderedExtras};
pub use session::{
    SessionCatalog, SessionCatalogCursor, SessionCoverage, SessionCursor, SessionHeader,
    SessionPosition, SessionStep, SessionView,
};
pub use sources::{SourceCursor, SourceSpan, SourceStep};

struct PinnedRead {
    connection: Connection,
    visibility: Connection,
    archive: Option<Connection>,
    archive_visibility: Option<Connection>,
    recovery: Option<polylogue_model::RecoveryRecord>,
    extra_orders: u64,
    _lock: Arc<File>,
}
type SharedRead = Arc<Mutex<PinnedRead>>;
fn locked(read: &SharedRead) -> Result<MutexGuard<'_, PinnedRead>> {
    read.lock()
        .map_err(|_| Error::Invariant("selected read snapshot mutex poisoned".into()))
}
fn metadata_admission(resource: &'static str, required: u64) -> Error {
    Error::ParserAdmission { resource, required }
}

#[cfg(test)]
mod tests;
