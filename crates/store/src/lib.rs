//! The only crate that opens SQLite. The archive is a directory of three
//! database files and a blob directory (architecture 4.1):
//!
//! - `archive.db`: what cannot be recomputed (inputs, revisions, notes,
//!   excisions, deliveries, settings);
//! - `index.db`: everything parsed and derived, rebuilt by the loop;
//! - `embeddings.db`: purchased vectors, never destroyed by a reindex;
//! - `blobs/`: input bytes of 128 KiB and over.

pub mod blobs;
pub mod excise;
pub mod files;
pub mod notes;
pub mod read;
pub mod schema;
pub mod search;
pub mod writer;

pub use blobs::{Change, FileFacts, Piece};
pub use files::{Archive, ArchiveLock};
pub use read::{KnownInput, ReadPool, Reader, SessionSummary, Status};
pub use writer::{Batch, InputKey, Observation, ParseStep, WriteReport, Writer};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
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
    #[error("archive format {found} is newer than this Polylogue ({supported})")]
    TooNew { found: u32, supported: u32 },
    #[error("index format {found} is not {wanted}; it is rebuilt")]
    IndexVersion { found: u32, wanted: u32 },
    #[error("blob {0} failed its check")]
    BlobCorrupt(polylogue_model::BlobHash),
    #[error("parse step refused: {0}")]
    Counts(#[from] polylogue_model::CountMismatch),
    #[error("the writer has stopped")]
    WriterGone,
}

pub type Result<T> = std::result::Result<T, StoreError>;
