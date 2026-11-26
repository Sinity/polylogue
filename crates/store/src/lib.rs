//! The only crate that opens SQLite. The archive is a directory of three
//! database files and the segment log:
//!
//! - `archive.db`: what cannot be recomputed (inputs, revisions, notes,
//!   deliveries, settings);
//! - `index.db`: everything parsed and derived, written by an index build;
//! - `embeddings.db`: purchased vectors, never destroyed by an index build;
//! - `segments/`: the segment log, every input blob in append-only shared
//!   frame segments.

mod helpers;
pub(crate) use helpers::write_counter;
pub use helpers::{ConnExt, OptionExt, ResultExt, meta_get, meta_set, to_i64, to_u64};

pub mod annotations;
pub mod archive;
pub mod assets;
pub mod blobs;
pub mod bytes;
pub mod calendar;
pub mod cancellation;
mod change_summary;
pub mod chat_actions;
mod commit_positions;
pub mod completion;
mod container_members;
mod content;
pub mod context;
pub mod coverage;
pub mod documents;
mod encoding;
mod events;
pub mod export_items;
pub mod fault;
#[cfg(any(test, feature = "fixture"))]
pub mod fixture;
mod fragment;
pub mod import_jobs;
pub mod index;
pub mod input_rows;
pub mod items;
mod kept;
pub mod lineage;
mod live;
pub mod log;
pub mod summary;
pub use live::{
    FollowSnapshot, FollowedCall, FollowedMessage, LiveDetail, LiveEditedFiles, LiveRoot,
    LiveSession, ProcessRow, RunningCandidates, RunningSession,
};
pub mod activity;
pub mod alternates;
pub mod authored_records;
pub mod capture_read;
pub mod follow;
pub mod measure;
mod ordered_blocks;
mod parse_state;
pub mod peak_memory;
pub mod python_polylogue;
pub mod query;
pub mod query_objects;
pub mod read;
pub mod reference;
mod repo_lookup;
pub mod row;
pub mod schema;
pub mod schema_inputs;
pub mod scope_index;
pub mod search;
pub mod selection;
pub mod session_bundle;
mod session_row;
pub mod session_sql;
pub mod signals;
pub(crate) mod sorted_runs;
pub(crate) mod source;
pub mod sql_functions;
pub mod sql_views;
pub mod staging;
pub mod storage_failure;
mod text_rows;
pub mod timing;
pub mod tool_databases;
pub mod user_state;
pub mod user_writes;
pub mod vectors;
pub mod verify;
pub mod wire_shapes;
mod write_lock;
pub mod writer;

pub use archive::{Archive, ArchiveLock};
pub use blobs::{Change, Chunk, FileFacts, NewBlob};
pub use context::{DeliveryRow, OpenItems, Prompt, SessionSummary};
pub use read::{
    BehindChain, CapturedSession, ChainCensus, KnownInput, Lane, LatestChain, ReadPool, Reader,
    ReadingContract, RevisionChains, SqlValue, Status, StoredSource,
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
    Bundle(#[from] polylogue_model::bundle::BundleError),
    #[error("archive {path} is held by another process{}", holder_suffix(.holder))]
    Locked {
        path: std::path::PathBuf,
        /// The pid the holder recorded in the lock file, when readable.
        holder: Option<u32>,
    },
    #[error(
        "{database}.db is at version {found}, newer than this Polylogue build's {current}; \
         run a build at least as new"
    )]
    NewerSchema {
        database: &'static str,
        found: u32,
        current: u32,
    },
    /// A database at an older version, or at this version with other tables.
    /// Nothing converts it.
    #[error(
        "{database}.db at version {found} is not in this Polylogue build's format; \
         it is left as it was"
    )]
    OtherFormat { database: &'static str, found: u32 },
    /// `embeddings.db` holds paid vectors, so it is never set aside: a schema
    /// that adding tables cannot bring to this build's is refused.
    #[error(
        "embeddings.db has a schema this Polylogue build cannot reach by adding tables \
         ({difference}); it holds paid vectors and is left as it was"
    )]
    EmbeddingsSchemaDiffers { difference: String },
    /// A database written before the baseline (no Polylogue format identity).
    /// Its history comes back by importing its original files.
    #[error(
        "{database}.db was written before this format's baseline and is left as it was; \
         import the original files it was built from with `polylogue import`"
    )]
    OlderFormat { database: &'static str },
    /// The archive's open builds a new index to replace it.
    #[error(
        "index.db at version {found} is not this build's version {wanted}; \
         a new index is built from the archive to replace it"
    )]
    IndexVersion { found: u32, wanted: u32 },
    #[error("blob {0} failed its check")]
    BlobCorrupt(polylogue_model::BlobHash),
    #[error("parse step refused: {0}")]
    Counts(#[from] polylogue_model::CountMismatch),
    #[error("the writer has stopped")]
    WriterGone,
    /// The process is stopping ([`cancellation`]): the read, wait or
    /// reconstructible write it ended left nothing half done, and is done
    /// again by the next start.
    #[error("the daemon is stopping")]
    Stopped,
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
    /// No index can be read until the rebuild in progress is installed
    /// ([`archive::index_build`]).
    #[error("{0}")]
    IndexBuilding(Box<archive::index_build::IndexBuild>),
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
    #[error("import job refused: {0}")]
    ImportJob(&'static str),
    /// A refusal still carried as text: only the writer and the index build
    /// construct it.
    #[error("{0}")]
    Invalid(String),
    /// A request the caller can correct, with its typed reason.
    #[error("{0}")]
    Refused(polylogue_model::Reason),
    /// A state no request can cause: a broken invariant in Polylogue.
    #[error("{0}; this is a Polylogue bug; report it with the request ID")]
    Internal(String),
    /// A batch's bytes were stored and its parsed rows refused, for a cause
    /// that writing them again would meet again: the bytes are in
    /// `stored`'s chain, the reading of them is not. `cause` is the
    /// refusal's own: only a `Content` one (the parser's reading of these
    /// bytes) stands as the chain's failure; a stale or inconsistent step
    /// (`Request`, `Bug`) leaves the chain due.
    #[error("{reason}")]
    RowsRefused {
        stored: staging::StoredInput,
        reason: String,
        cause: polylogue_model::Cause,
    },
    /// A value in the archive that this build cannot read back: corrupt or
    /// unrecognized stored data, never the caller's request.
    #[error("{}", stored_value_text(*.column, .reason))]
    StoredValue {
        /// The column of the row being decoded, when one value is at fault.
        column: Option<usize>,
        reason: String,
    },
    #[error(transparent)]
    Query(#[from] polylogue_query::QueryError),
    #[error("{reference} is unavailable: {reason}")]
    ReferenceUnavailable { reference: String, reason: String },
    /// A selection refusal with the product code that classifies it.
    #[error("{message}")]
    Selection {
        code: polylogue_model::Code,
        message: String,
    },
    /// A failure already classified where it happened, carried as a value
    /// with its cause and code: what a write of a whole group lost for one
    /// of its members, a write report's failures, a derivation's.
    #[error("{0}")]
    Failed(polylogue_model::Failure),
    /// A source file changed while it was read: it is read again.
    #[error("{0}")]
    SourceChanged(&'static str),
    /// A part of a chain's parse that does not start where the chain's
    /// stored parse ends: it was planned from an earlier state. Nothing is
    /// written, and the chain stays due from its stored state.
    #[error(
        "the part of {input} chain {chain} starts at byte {start} under {version}, where its stored parse ends at {parsed_to} under {}",
        stored_version.as_deref().unwrap_or("no parser version")
    )]
    StalePart {
        input: String,
        chain: i64,
        start: u64,
        version: String,
        parsed_to: u64,
        stored_version: Option<String>,
    },
}

impl StoreError {
    pub fn invalid(message: impl ToString) -> Self {
        Self::Invalid(message.to_string())
    }

    /// The error of a group member that a write of the whole group lost:
    /// the group's cause, code and sentence.
    pub(crate) fn for_group_member(&self) -> Self {
        match self {
            Self::Stopped => Self::Stopped,
            Self::Refused(reason) => Self::Refused(reason.clone()),
            error => Self::Failed(polylogue_model::Failure::of(error)),
        }
    }

    /// A stored value at `column` of the row being decoded that this build
    /// cannot read back.
    pub(crate) fn stored(column: usize, reason: impl Into<String>) -> Self {
        Self::StoredValue {
            column: Some(column),
            reason: reason.into(),
        }
    }

    /// Stored data this build cannot read back that no one column holds:
    /// extents, transcript lines, stored sums.
    pub fn stored_data(reason: impl ToString) -> Self {
        Self::StoredValue {
            column: None,
            reason: reason.to_string(),
        }
    }

    /// A broken invariant: no request causes it, so it is reported as a bug.
    pub fn internal(message: impl ToString) -> Self {
        Self::Internal(message.to_string())
    }

    /// The product failure code for this error. Every surface classifies a
    /// store error through this one mapping.
    pub fn code(&self) -> polylogue_model::Code {
        use polylogue_model::Code;
        match self {
            Self::Sqlite(error) => {
                if sqlite_busy(error) {
                    Code::Busy
                } else {
                    Code::Store
                }
            }
            Self::Io(_) => Code::Io,
            Self::Blobs(
                hashed_blobs::Error::Io(_)
                | hashed_blobs::Error::Locked(_)
                | hashed_blobs::Error::Unwrap(_),
            ) => Code::Io,
            Self::Blobs(hashed_blobs::Error::Corrupt { .. } | hashed_blobs::Error::Refused(_)) => {
                Code::BlobCorrupt
            }
            Self::Json(_) => Code::Store,
            Self::Bundle(issue) => bundle_code(issue),
            Self::Locked { .. } => Code::Conflict,
            Self::NewerSchema { .. } => Code::Refused,
            Self::OtherFormat { .. } => Code::Refused,
            Self::EmbeddingsSchemaDiffers { .. } => Code::Refused,
            Self::OlderFormat { .. } => Code::Refused,
            Self::IndexVersion { .. } => Code::IndexBuilding,
            Self::BlobCorrupt(_) => Code::BlobCorrupt,
            Self::Counts(_) => Code::Internal,
            Self::WriterGone => Code::NotRunning,
            Self::Stopped => Code::ShuttingDown,
            Self::StorageFailed(_) => Code::Io,
            Self::ReadDeadlineExceeded => Code::DeadlineExceeded,
            Self::ReadCancelled => Code::Cancelled,
            Self::ReadPoolClosed => Code::ShuttingDown,
            Self::ReadPoolNested => Code::Internal,
            Self::ScopeChanged => Code::ScopeChanged,
            Self::IndexBuilding(_) => Code::IndexBuilding,
            Self::Conflict(_) => Code::Conflict,
            Self::Stale { .. } => Code::Stale,
            Self::Ambiguous { .. } => Code::Ambiguous,
            Self::IncompleteParents(_) => Code::IncompleteParents,
            Self::NotFound(_) => Code::NotFound,
            Self::ChatAction(_) => Code::Conflict,
            Self::ImportJob(_) => Code::Conflict,
            Self::Invalid(_) => Code::BadRequest,
            Self::Refused(reason) => reason.code(),
            Self::Internal(_) => Code::Internal,
            Self::RowsRefused { .. } => Code::Import,
            Self::StoredValue { .. } => Code::Store,
            Self::Query(error) => error.kind(),
            Self::ReferenceUnavailable { .. } => Code::NotYetAvailable,
            Self::Selection { code, .. } => *code,
            Self::Failed(failure) => failure.code,
            Self::SourceChanged(_) => Code::Conflict,
            Self::StalePart { .. } => Code::Stale,
        }
    }
}

impl polylogue_model::Caused for StoreError {
    fn cause(&self) -> Cause {
        match self {
            Self::Sqlite(error) => sqlite_cause(error),
            Self::Io(error) => Self::io_cause(error),
            Self::Blobs(error) => Self::blob_cause(error),
            Self::Json(_) => Cause::StoredData,
            Self::Bundle(_) => Cause::Request,
            Self::Locked { .. } => Cause::Transient,
            Self::NewerSchema { .. } => Cause::StoredData,
            Self::OtherFormat { .. } => Cause::StoredData,
            Self::EmbeddingsSchemaDiffers { .. } => Cause::StoredData,
            Self::OlderFormat { .. } => Cause::StoredData,
            Self::IndexVersion { .. } => Cause::Transient,
            Self::BlobCorrupt(_) => Cause::StoredData,
            Self::Counts(_) => Cause::Content,
            Self::WriterGone => Cause::Stopping,
            Self::Stopped => Cause::Stopping,
            Self::StorageFailed(_) => Cause::Environment,
            Self::ReadDeadlineExceeded => Cause::Transient,
            Self::ReadCancelled => Cause::Stopping,
            Self::ReadPoolClosed => Cause::Stopping,
            Self::ReadPoolNested => Cause::Bug,
            Self::ScopeChanged => Cause::Transient,
            Self::IndexBuilding(_) => Cause::Transient,
            Self::Conflict(_) => Cause::Request,
            Self::Stale { .. } => Cause::Request,
            Self::Ambiguous { .. } => Cause::Request,
            Self::IncompleteParents(_) => Cause::Request,
            Self::NotFound(_) => Cause::Request,
            Self::ChatAction(_) => Cause::Request,
            Self::ImportJob(_) => Cause::Request,
            Self::Invalid(_) => Cause::Request,
            Self::Refused(_) => Cause::Request,
            Self::Internal(_) => Cause::Bug,
            Self::RowsRefused { cause, .. } => *cause,
            Self::StoredValue { .. } => Cause::StoredData,
            Self::Query(_) => Cause::Request,
            Self::ReferenceUnavailable { .. } => Cause::Transient,
            Self::Selection { .. } => Cause::Request,
            Self::Failed(failure) => failure.cause,
            Self::SourceChanged(_) => Cause::Transient,
            Self::StalePart { .. } => Cause::Transient,
        }
    }

    fn code(&self) -> polylogue_model::Code {
        StoreError::code(self)
    }
}

impl From<polylogue_model::Reason> for StoreError {
    fn from(reason: polylogue_model::Reason) -> Self {
        Self::Refused(reason)
    }
}

fn sqlite_busy(error: &rusqlite::Error) -> bool {
    matches!(
        error.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
    )
}

fn stored_value_text(column: Option<usize>, reason: &str) -> String {
    match column {
        Some(column) => format!("stored column {column} is unreadable: {reason}"),
        None => format!("stored data is unreadable: {reason}"),
    }
}

fn bundle_code(issue: &polylogue_model::bundle::BundleError) -> polylogue_model::Code {
    use polylogue_model::{Code, bundle::BundleError};
    match issue {
        BundleError::BlockSelectionConflict | BundleError::SourceVersionMismatch => Code::Conflict,
        BundleError::Annotation(_)
        | BundleError::TextFragment(_)
        | BundleError::WrongFormat(_)
        | BundleError::UnsupportedVersion { .. }
        | BundleError::InvalidQuery
        | BundleError::InvalidPosition
        | BundleError::BlobFlagMismatch
        | BundleError::ReadOptions(_)
        | BundleError::InvalidPath
        | BundleError::ReferenceMismatch
        | BundleError::InvalidLink(_)
        | BundleError::InvalidBlockSelection
        | BundleError::InvalidInjectionEvidence
        | BundleError::InvalidAttachmentBinding
        | BundleError::CanonicalMessage(_)
        | BundleError::Time(_)
        | BundleError::InvalidEventIdentity
        | BundleError::EventReferenceMismatch
        | BundleError::InvalidEventAssociation => Code::BadRequest,
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        if error.sqlite_error_code() == Some(rusqlite::ErrorCode::OperationInterrupted)
            && let Some(cause) = cancellation::interrupted_by()
        {
            return cause;
        }
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
            rusqlite::Error::FromSqlConversionFailure(column, _, why) => {
                Self::stored(column, why.to_string())
            }
            rusqlite::Error::IntegralValueOutOfRange(column, value) => {
                Self::stored(column, format!("{value} is out of range"))
            }
            error => Self::Sqlite(error),
        }
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;

use polylogue_model::Cause;

pub mod chain_names;

fn holder_suffix(holder: &Option<u32>) -> String {
    holder.map_or_else(String::new, |pid| format!(" (pid {pid})"))
}

impl StoreError {
    /// The cause of an I/O failure: a stop or cancelled read it carries, else
    /// its kind.
    pub fn io_cause(error: &std::io::Error) -> Cause {
        // An I/O failure a cancelled read or a stop reported.
        if let Some(inner) = error.get_ref()
            && let Some(store) = inner.downcast_ref::<StoreError>()
        {
            return polylogue_model::Caused::cause(store);
        }
        Cause::of_io(error)
    }

    /// The cause of a segment-log failure.
    pub fn blob_cause(error: &hashed_blobs::Error) -> Cause {
        match error {
            hashed_blobs::Error::Io(error) => Self::io_cause(error),
            hashed_blobs::Error::Locked(_) => Cause::Transient,
            hashed_blobs::Error::Unwrap(_)
            | hashed_blobs::Error::Corrupt { .. }
            | hashed_blobs::Error::Refused(_) => Cause::StoredData,
        }
    }
}

/// SQLite's result codes, by what ends them.
pub(crate) fn sqlite_cause(error: &rusqlite::Error) -> Cause {
    use rusqlite::ErrorCode;
    let Some(code) = error.sqlite_error_code() else {
        // A value that does not convert, a wrong parameter count, a
        // statement that returned no row: the calling code is at fault.
        return Cause::Bug;
    };
    match code {
        ErrorCode::DatabaseBusy
        | ErrorCode::DatabaseLocked
        | ErrorCode::SystemIoFailure
        | ErrorCode::OutOfMemory
        | ErrorCode::SchemaChanged
        | ErrorCode::FileLockingProtocolFailed => Cause::Transient,
        ErrorCode::OperationInterrupted | ErrorCode::OperationAborted => Cause::Stopping,
        ErrorCode::DiskFull
        | ErrorCode::CannotOpen
        | ErrorCode::ReadOnly
        | ErrorCode::PermissionDenied
        | ErrorCode::NoLargeFileSupport
        | ErrorCode::AuthorizationForStatementDenied => Cause::Environment,
        ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase => Cause::StoredData,
        _ => Cause::Bug,
    }
}

#[cfg(test)]
mod tests;
