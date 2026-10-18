//! The convergence loop (architecture 5). One loop runs in the daemon:
//!
//! ```text
//! collect -> store -> parse -> derive -> excise -> publish (Sinex primary) -> wait
//! ```
//!
//! Each step finds its work by comparing what exists with what should
//! exist. There is no to-do list and no repair path; `status` calls the
//! same comparison.

pub mod collect;
pub mod derive;
pub mod pool;
pub mod publish;
pub mod stage;

use polylogue_store::{Archive, Writer};
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
    #[error(transparent)]
    Parse(#[from] polylogue_sources::ParseError),
    #[error(transparent)]
    Watch(#[from] notify::Error),
}

pub struct Config {
    /// The home directory origins' locations are relative to.
    pub home: PathBuf,
    /// Parse workers; defaults to the number of cores.
    pub workers: usize,
    /// Bytes admitted into the parse pool at once. The pool takes new work
    /// only while the bytes in flight stay under this.
    pub bytes_in_flight: u64,
}

pub struct Engine {
    pub config: Config,
    pub archive: Archive,
    pub writer: Writer,
}

/// What one pass of the loop found and did, for `status`.
#[derive(Debug, Default)]
pub struct Pass {
    pub inputs_seen: u64,
    pub inputs_changed: u64,
    pub bytes_parsed: u64,
    pub unreadable_roots: Vec<PathBuf>,
}

impl Engine {
    /// Runs one pass of the loop to completion.
    pub fn converge(&mut self) -> Result<Pass, EngineError> {
        // Architecture 5: collect, stage, parse pool -> writer, derive, excise, publish.
        todo!()
    }

    /// Runs passes until stopped, waking on file events and once a minute.
    pub fn run(&mut self, _stop: &std::sync::atomic::AtomicBool) -> Result<(), EngineError> {
        // Architecture 5.1, 6: notify events plus a periodic rescan.
        todo!()
    }
}
