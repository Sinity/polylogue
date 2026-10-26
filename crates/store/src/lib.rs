//! The only crate that opens SQLite. The archive is a directory of three
//! database files and a blob directory:
//!
//! - `archive.db`: what cannot be recomputed (inputs, revisions, notes,
//!   excisions, deliveries, settings);
//! - `index.db`: everything parsed and derived, rebuilt by the loop;
//! - `embeddings.db`: purchased vectors, never destroyed by a reindex;
//! - `blobs/`: input bytes of 128 KiB and over.

pub mod archive;
pub mod assets;
pub mod backfill_jobs;
pub mod blobs;
pub mod chat_actions;
pub mod check_runs;
pub mod context;
pub mod derive;
pub mod documents;
mod encoding;
mod events;
pub mod excise;
pub mod export_items;
pub mod fault;
pub(crate) mod first_build;
#[cfg(any(test, feature = "fixture"))]
pub mod fixture;
pub mod input_rows;
mod live;
pub mod measure;
pub mod notes;
mod ordered_blocks;
pub mod primary;
pub mod read;
mod repo_lookup;
pub mod schema;
pub mod search;
pub mod session_export;
mod session_row;
pub mod sql_functions;
pub mod staging;
mod text_rows;
pub mod timing;
pub mod tool_databases;
pub mod user_state;
pub mod user_writes;
pub mod vectors;
mod write_lock;
pub mod writer;
mod zip_members;

pub use archive::{Archive, ArchiveLock};
pub use blobs::{Change, FileFacts, NewBlob};
pub use context::{
    DeliveryRow, FailingCommand, NearbySession, OpenBead, OpenItems, Prompt, SessionWork,
};
pub use read::{
    BehindChain, CapturedSession, ImportedAs, KnownInput, ReadPool, Reader, SqlValue, Status,
};
pub use user_writes::{Delivery, LeftOut, UserOutcome, UserRequest, UserWrite};
pub use writer::{
    Batch, ImportMember, InFlight, InZip, InputKey, Observation, ParseStep, WriteReport, Writer,
    WriterSender,
};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("input {input} is excluded by the archive's removal policy")]
    InputRefused { input: String },
    #[error("input {input} has bytes still being written: {reason}")]
    InputPending { input: String, reason: String },
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
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
    #[error("index format {found} is not {wanted}; it is rebuilt")]
    IndexVersion { found: u32, wanted: u32 },
    #[error("blob {0} failed its check")]
    BlobCorrupt(polylogue_model::BlobHash),
    #[error("parse step refused: {0}")]
    Counts(#[from] polylogue_model::CountMismatch),
    #[error("the writer has stopped")]
    WriterGone,
    #[error("the read pool has closed")]
    ReadPoolClosed,
    #[error("read pool is full while this thread holds a reader; reuse or release it")]
    ReadPoolNested,
    #[error("the index was replaced; acquire a new reader")]
    ReadIndexReplaced,
    #[error("blob {0} is still in use and cannot be deleted")]
    ExcisionBlobReferenced(polylogue_model::BlobHash),
    #[error("the excision stopped part way; reopen the archive to finish it: {0}")]
    ExcisionInterrupted(#[source] Box<StoreError>),
    #[error("request {0} was already used for different content")]
    Conflict(String),
    #[error("the session's parents are incomplete: {0}")]
    IncompleteParents(read::ParentIssue),
    #[error("{0} not found")]
    NotFound(String),
    #[error("excision refused for {target}: {reason}")]
    ExcisionRefused { target: String, reason: String },
    #[error(
        "{file} is an export ZIP; excision inside it is not supported, because removing it would also remove unrelated conversations"
    )]
    ExcisionInsideZip { input: String, file: String },
    #[error("chat action refused: {0}")]
    ChatAction(&'static str),
    #[error("backfill job refused: {0}")]
    Backfill(&'static str),
    /// Sinex could not be reached, or refused what was sent (primary mode).
    #[error("Sinex: {0}")]
    Sinex(String),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, StoreError>;
