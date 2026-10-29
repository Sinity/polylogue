//! The only crate that opens SQLite. The archive is a directory of three
//! database files and a blob directory:
//!
//! - `archive.db`: what cannot be recomputed (inputs, revisions, notes,
//!   deliveries, settings);
//! - `index.db`: everything parsed and derived, rebuilt by the loop;
//! - `embeddings.db`: purchased vectors, never destroyed by a reindex;
//! - `segments/`: every input blob, in append-only shared frame segments.

pub mod archive;
pub mod assets;
pub mod backfill_jobs;
pub mod blobs;
pub mod chat_actions;
pub mod check_runs;
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
pub mod input_rows;
pub mod legacy_import;
mod live;
pub mod measure;
pub mod notes;
mod ordered_blocks;
pub mod primary;
pub mod read;
mod repo_lookup;
pub mod schema;
pub mod schema_inputs;
pub mod search;
pub mod session_export;
mod session_row;
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
mod versions;
pub mod wire_shapes;
mod write_lock;
pub mod writer;

pub use archive::{Archive, ArchiveLock};
pub use blobs::{Change, FileFacts, NewBlob};
pub use context::{
    DeliveryRow, FailingCommand, NearbySession, OpenBead, OpenItems, Prompt, SessionWork,
};
pub use read::{
    BehindChain, CapturedSession, ImportedAs, KnownInput, ReadPool, Reader, SqlValue, Status,
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
    #[error("the index was replaced; acquire a new reader")]
    ReadIndexReplaced,
    #[error("request {0} was already used for different content")]
    Conflict(String),
    #[error("{reference} is ambiguous: {candidates:?}")]
    Ambiguous {
        reference: String,
        candidates: Vec<String>,
    },
    #[error("the session's parents are incomplete: {0}")]
    IncompleteParents(read::ParentIssue),
    #[error("{0} not found")]
    NotFound(String),
    #[error("chat action refused: {0}")]
    ChatAction(&'static str),
    #[error("backfill job refused: {0}")]
    Backfill(&'static str),
    /// Sinex could not be reached, or refused what was sent (primary mode).
    #[error("capture {identity} is durable and waiting for Sinex confirmation")]
    PrimaryPending { identity: String },
    #[error("Sinex: {0}")]
    Sinex(String),
    #[error("{0}")]
    Invalid(String),
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
        Self::Sqlite(error)
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;
