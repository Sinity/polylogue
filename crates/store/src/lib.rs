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
pub mod session_export;
mod session_row;
pub mod session_sql;
pub mod signals;
pub(crate) mod source;
pub mod sql_functions;
pub mod sql_views;
pub mod staging;
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
pub use blobs::{Change, FileFacts, NewBlob};
pub use context::{DeliveryRow, OpenItems, Prompt, SessionSummary};
pub use read::{
    BehindChain, CapturedSession, CompletenessDimension, CompletenessReason, ImportedAs,
    KnownInput, ReadCompleteness, ReadPool, Reader, ReadingContract, SqlValue, Status,
    StoppedChains,
    scope::{ReadScopeToken, ScopeInvalidation},
};
pub use scope_index::{
    NearbySession, OpenBead, ScopeChanges, ScopeIndex, ScopeSession, ScopedLiveSession,
};
pub use writer::{
    Batch, ImportMember, InContainer, InFlight, InputKey, Observation, ParseStep, WriteReport,
    Writer, WriterSender,
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
    /// A database written before the baseline (no Polylogue format identity).
    /// The importer lane matches this wording.
    #[error("older format: import with `polylogue import --from-archive PATH`")]
    OlderFormat { database: &'static str },
    #[error(
        "index format {found} is not {wanted}; stop the daemon, move index.db and its WAL/SHM files aside, then restart the daemon to rebuild from stored bytes"
    )]
    IndexVersion { found: u32, wanted: u32 },
    #[error("blob {0} failed its check")]
    BlobCorrupt(polylogue_model::BlobHash),
    #[error("parse step refused: {0}")]
    Counts(#[from] polylogue_model::CountMismatch),
    #[error("the writer has stopped")]
    WriterGone,
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
    #[error(transparent)]
    Query(#[from] polylogue_query::QueryError),
    #[error("{reference} is unavailable: {reason}")]
    ReferenceUnavailable { reference: String, reason: String },
    /// A selection refusal with the product code that classifies it.
    #[error("{message}")]
    Selection { code: &'static str, message: String },
}

impl StoreError {
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
            Self::ReadCancelled => "cancelled",
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
        // A stored value the model cannot read back is invalid stored data,
        // whichever reader met it; the column error says which column and why.
        match error {
            rusqlite::Error::FromSqlConversionFailure(column, _, why) => {
                Self::Invalid(format!("stored column {column}: {why}"))
            }
            rusqlite::Error::IntegralValueOutOfRange(column, value) => {
                Self::Invalid(format!("stored column {column}: {value} is out of range"))
            }
            error => Self::Sqlite(error),
        }
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;

pub mod parser_context;
