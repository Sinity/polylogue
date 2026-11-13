//! The only crate that opens SQLite. The archive is a directory of three
//! database files and a blob directory:
//!
//! - `archive.db`: what cannot be recomputed (inputs, revisions, notes,
//!   deliveries, settings);
//! - `index.db`: everything parsed and derived, rebuilt by the loop;
//! - `embeddings.db`: purchased vectors, never destroyed by a reindex;
//! - `segments/`: every input blob, in append-only shared frame segments.

mod admission;
pub mod archive;
pub mod assertions;
pub mod assets;
pub mod backfill_jobs;
pub mod blobs;
pub mod calendar;
mod chain_progress;
mod change_summary;
pub mod chat_actions;
pub mod completion;
mod container_members;
pub mod context;
pub mod coverage;
pub mod derive;
pub mod documents;
mod encoding;
mod events;
pub mod export_items;
pub mod fault;
pub(crate) mod first_build;
#[cfg(any(test, feature = "fixture"))]
pub mod fixture;
mod fragment;
pub mod input_rows;
pub mod items;
mod kept;
pub mod legacy_import;
pub mod lineage;
mod live;
pub use live::{
    FollowSnapshot, FollowedCall, FollowedMessage, LiveDetail, LiveFootprint, LiveRoot,
    LiveSession, ProcessRow, RunningSession,
};
pub mod measure;
pub mod older_archive;
mod ordered_blocks;
pub mod predecessor;
pub mod primary;
pub mod query;
pub mod query_objects;
pub mod read;
pub mod reference;
mod repo_lookup;
mod same_bytes;
pub use same_bytes::ReadingCopy;
pub mod schema;
pub mod schema_inputs;
pub mod scope_index;
pub mod search;
pub mod selection;
pub mod session_alias;
pub mod session_export;
mod session_row;
pub mod session_sql;
mod session_values;
pub mod signals;
pub(crate) mod source;
pub mod sql_functions;
pub mod sql_views;
pub mod staging;
pub mod storage_failure;
mod text_rows;
pub mod timeline;
pub mod timing;
pub mod tool_databases;
pub mod user_state;
pub mod user_writes;
pub mod variants;
pub mod vectors;
pub mod verify;
mod versions;
pub mod wire_shapes;
mod write_lock;
pub mod writer;

pub use archive::{Archive, ArchiveLock};
pub use blobs::{Change, Chunk, FileFacts, NewBlob};
pub use context::{DeliveryRow, OpenItems, Prompt, SessionSummary};
pub use read::{
    BehindChain, CapturedSession, KnownInput, ReadPool, Reader, ReadingContract, SqlValue, Status,
    StoppedChains, StoredSource, reserved_reads,
    scope::{ReadScopeToken, ScopeInvalidation},
};
pub use scope_index::{
    NearbySession, OpenBead, ScopeChanges, ScopeIndex, ScopeSession, ScopedLiveSession,
};
pub use writer::{
    Batch, DatabaseLog, ImportMember, InContainer, InFlight, InputKey, Observation, ParseStep,
    WriteReport, Writer, WriterSender,
};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Sqlite(rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Blobs(#[from] hashed_blobs::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Projection(#[from] polylogue_model::bundle::BundleError),
    #[error("archive {path} is held by another process")]
    Locked { path: std::path::PathBuf },
    #[error(
        "{database}.db is at version {found}, newer than this Polylogue build's {current}; \
         run a build at least as new"
    )]
    NewerSchema {
        database: &'static str,
        found: u32,
        current: u32,
    },
    #[error(
        "{database}.db at version {found} has tables this Polylogue build has no migration from"
    )]
    NoMigration { database: &'static str, found: u32 },
    /// `embeddings.db` holds paid vectors, so it is never set aside: a schema
    /// that adding tables cannot bring to this build's is refused.
    #[error(
        "embeddings.db has a schema this Polylogue build cannot reach by adding tables \
         ({difference}); it holds paid vectors and is left as it was"
    )]
    EmbeddingsSchemaDiffers { difference: String },
    /// A database written before the baseline (no Polylogue format identity).
    /// The importer lane matches this wording.
    #[error("older format: import with `polylogue import --from-archive PATH`")]
    OlderFormat { database: &'static str },
    /// The archive's open sets such an index aside and builds a new one.
    #[error("index format {found} is not {wanted}")]
    IndexVersion { found: u32, wanted: u32 },
    #[error("blob {0} failed its check")]
    BlobCorrupt(polylogue_model::BlobHash),
    #[error("parse step refused: {0}")]
    Counts(#[from] polylogue_model::CountMismatch),
    #[error("the writer has stopped")]
    WriterGone,
    /// Shutdown rolled back replay of retained bytes; its cursor did not move.
    #[error("retained-byte replay was cancelled")]
    WriteCancelled,
    /// The disk failed a sync of the register: writing stopped for the rest
    /// of this boot ([`storage_failure`]).
    #[error("{0}")]
    StorageFailed(storage_failure::StorageFailure),
    #[error("the read deadline was exceeded")]
    ReadDeadlineExceeded,
    #[error("the read was cancelled")]
    ReadCancelled,
    #[error("the read pool has closed")]
    ReadPoolClosed,
    #[error("read pool is full while this thread holds a reader; reuse or release it")]
    ReadPoolNested,
    #[error("the access scope changed while this read was in progress")]
    ScopeChanged,
    #[error("the index was replaced; acquire a new reader")]
    ReadIndexReplaced,
    #[error("request {0} was already used for different content")]
    Conflict(String),
    #[error("{reference} is stale; current revisions: {current:?}")]
    Stale {
        reference: String,
        current: Vec<String>,
    },
    #[error("{reference} is ambiguous: {candidates:?}")]
    Ambiguous {
        reference: String,
        candidates: Vec<String>,
    },
    #[error("the session's parents are incomplete: {0}")]
    IncompleteParents(Box<read::ParentIssue>),
    #[error("{0} not found")]
    NotFound(String),
    #[error("chat action refused: {0}")]
    ChatAction(&'static str),
    #[error("backfill job refused: {0}")]
    Backfill(&'static str),
    /// Sinex could not be reached, or refused what was sent (primary mode).
    #[error("capture {identity} is durable and waiting for Sinex confirmation")]
    PrimaryPending { identity: String },
    #[error("acquisition {custody:?} retained raw bytes after a source basis conflict")]
    UnresolvedBasisConflict { custody: primary::RawCustodyReceipt },
    #[error("Sinex: {0}")]
    Sinex(String),
    #[error("{0}")]
    Invalid(String),
    /// A batch's bytes were stored and its parsed rows refused for their
    /// content ([`Self::refuses_content`]): the bytes are in `stored`'s chain,
    /// the reading of them is not.
    #[error("{reason}")]
    RowsRefused {
        stored: staging::StoredInput,
        reason: String,
    },
    /// A value in the archive that this build cannot read back: corrupt or
    /// foreign stored data, never the caller's request.
    #[error("stored column {column} is unreadable: {reason}")]
    StoredValue { column: usize, reason: String },
    #[error(transparent)]
    Query(#[from] polylogue_query::QueryError),
    #[error("{reference} is unavailable: {reason}")]
    ReferenceUnavailable { reference: String, reason: String },
    /// A selection refusal with the product code that classifies it.
    #[error("{message}")]
    Selection { code: &'static str, message: String },
}

impl StoreError {
    /// Whether retrying a write can get past a transient storage condition.
    /// A content, permission or format failure needs its cause corrected first.
    pub fn retryable_write(&self) -> bool {
        match self {
            Self::Io(error) | Self::Blobs(hashed_blobs::Error::Io(error)) => retryable_io(error),
            Self::Sqlite(error) => matches!(
                error.sqlite_error_code(),
                Some(
                    rusqlite::ErrorCode::DatabaseBusy
                        | rusqlite::ErrorCode::DatabaseLocked
                        | rusqlite::ErrorCode::SystemIoFailure
                        | rusqlite::ErrorCode::DiskFull
                        | rusqlite::ErrorCode::OperationInterrupted
                )
            ),
            _ => false,
        }
    }

    /// Whether the writer refused the batch's content: writing the same
    /// rows to the same archive again is refused again. Every other write
    /// error is a failure to commit, which a later attempt can get past.
    pub fn refuses_content(&self) -> bool {
        match self {
            Self::Invalid(_)
            | Self::Counts(_)
            | Self::Conflict(_)
            | Self::Projection(_)
            | Self::RowsRefused { .. } => true,
            Self::Sqlite(error) => {
                error.sqlite_error_code() == Some(rusqlite::ErrorCode::ConstraintViolation)
            }
            _ => false,
        }
    }

    /// The error of a group member that a write of the whole group lost:
    /// the group's reason, refusing content exactly when the group's error
    /// did.
    pub(crate) fn for_group_member(&self) -> Self {
        if matches!(self, Self::WriteCancelled) {
            return Self::WriteCancelled;
        }
        let reason = self.to_string();
        if self.refuses_content() {
            Self::Invalid(reason)
        } else {
            Self::Io(std::io::Error::other(reason))
        }
    }

    /// A stored value at `column` of the row being decoded that this build
    /// cannot read back.
    pub(crate) fn stored(column: usize, reason: impl Into<String>) -> Self {
        Self::StoredValue {
            column,
            reason: reason.into(),
        }
    }

    /// The product failure code for this error. Every surface classifies a
    /// store error through this one mapping.
    pub fn code(&self) -> &'static str {
        match self {
            Self::ScopeChanged => "scope_changed",
            Self::Conflict(_) | Self::ReadIndexReplaced => "conflict",
            Self::Ambiguous { .. } => "ambiguous",
            Self::Stale { .. } => "stale",
            Self::ReferenceUnavailable { .. } => "not_yet_available",
            Self::Projection(issue) => match issue {
                polylogue_model::bundle::BundleError::ProjectionConflict
                | polylogue_model::bundle::BundleError::SourceVersionMismatch => "conflict",
                _ => "bad_request",
            },
            Self::ReadDeadlineExceeded => "deadline_exceeded",
            Self::ReadCancelled | Self::WriteCancelled => "cancelled",
            Self::ReadPoolClosed => "shutting_down",
            Self::NotFound(_) => "not_found",
            Self::Invalid(_) => "bad_request",
            Self::Query(error) => error.kind(),
            Self::Selection { code, .. } => code,
            Self::IncompleteParents(_) => "incomplete_parents",
            Self::PrimaryPending { .. } => "degraded",
            _ => "store",
        }
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        if matches!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::OperationInterrupted | rusqlite::ErrorCode::DatabaseBusy)
        ) && let Some(limit) = read::limit::ReadLimit::current()
            && let Err(stopped) = limit.check()
        {
            return stopped;
        }
        // A row value that does not decode is a fault in the stored data,
        // whichever reader met it.
        match error {
            rusqlite::Error::FromSqlConversionFailure(column, _, why) => Self::StoredValue {
                column,
                reason: why.to_string(),
            },
            rusqlite::Error::IntegralValueOutOfRange(column, value) => Self::StoredValue {
                column,
                reason: format!("{value} is out of range"),
            },
            error => Self::Sqlite(error),
        }
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;

pub mod parser_context;

/// Transient file failures may be retried with the original immutable bytes.
pub fn retryable_io(error: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    !matches!(
        error.kind(),
        ErrorKind::PermissionDenied
            | ErrorKind::ReadOnlyFilesystem
            | ErrorKind::InvalidInput
            | ErrorKind::Unsupported
            | ErrorKind::InvalidData
            | ErrorKind::NotFound
            | ErrorKind::IsADirectory
            | ErrorKind::NotADirectory
            | ErrorKind::AlreadyExists
    )
}

#[cfg(test)]
mod tests;
