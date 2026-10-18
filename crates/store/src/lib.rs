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

pub use files::{Archive, ArchiveLock};
pub use read::{ReadPool, Reader};
pub use writer::{Batch, Writer};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("archive {path} is held by another process")]
    Locked { path: std::path::PathBuf },
    #[error("archive format {found} is newer than this Polylogue ({supported})")]
    TooNew { found: u32, supported: u32 },
    #[error("blob {0} failed its check")]
    BlobCorrupt(polylogue_model::BlobHash),
}

pub type Result<T> = std::result::Result<T, StoreError>;
